# app/orchestrator/workflow.py
from __future__ import annotations
from typing import Dict, Any
from langgraph.graph import StateGraph, START, END
from app.orchestrator.state import WFState
from app.orchestrator.nodes import (
    NodeContext,
    make_node_ingest, make_node_build_graph, make_node_choose_route,
    make_node_retrieval, make_node_reasoning, make_node_verify, make_node_pack_result,
    node_claim_retrieval,
)
from app.telemetry.sinks import span

print("[workflow loaded from]", __file__)  # 加在 build_workflow 顶部


def build_workflow(ctx: NodeContext, dataset_cfg: Dict[str, Any] = None, dataset_loader=None):
    g = StateGraph(WFState)

    # ✅ InitExternal: 把样本拆成 external_context + meta
    def init_external_context(state: WFState) -> WFState:
        trace_id = state.get("trace_id", "trace-demo")
        if ctx.sink:
            with span("InitExternal", ctx.sink, trace_id):
                if dataset_loader:
                    samples = dataset_loader.load()
                    q = state.get("question", "").strip()

                    matched = None
                    for s in samples:
                        if s.get("question", "").strip() == q:
                            matched = s
                            break

                    if matched:
                        state["external_context"] = {"context": matched.get("context", [])}
                        state["meta"] = {
                            "_id": matched.get("_id"),
                            "answer": matched.get("answer"),
                            "supporting_facts": matched.get("supporting_facts", []),
                            "type": matched.get("type"),
                            "level": matched.get("level"),
                        }
                    elif samples:
                        s = samples[0]
                        state["external_context"] = {"context": s.get("context", [])}
                        state["meta"] = {
                            "_id": s.get("_id"),
                            "answer": s.get("answer"),
                            "supporting_facts": s.get("supporting_facts", []),
                            "type": s.get("type"),
                            "level": s.get("level"),
                        }
                return state
        # no sink fallback
        if dataset_loader:
            samples = dataset_loader.load()
            q = state.get("question", "").strip()
            matched = None
            for s in samples:
                if s.get("question", "").strip() == q:
                    matched = s
                    break
            if matched:
                state["external_context"] = {"context": matched.get("context", [])}
                state["meta"] = {
                    "_id": matched.get("_id"),
                    "answer": matched.get("answer"),
                    "supporting_facts": matched.get("supporting_facts", []),
                    "type": matched.get("type"),
                    "level": matched.get("level"),
                }
            elif samples:
                s = samples[0]
                state["external_context"] = {"context": s.get("context", [])}
                state["meta"] = {
                    "_id": s.get("_id"),
                    "answer": s.get("answer"),
                    "supporting_facts": s.get("supporting_facts", []),
                    "type": s.get("type"),
                    "level": s.get("level"),
                }
        return state

    # === 原有节点 ===
    g.add_node("InitExternal", init_external_context)
    g.add_node("Ingest", make_node_ingest(ctx))
    g.add_node("BuildGraph", make_node_build_graph(ctx))
    g.add_node("ChooseRoute", make_node_choose_route(ctx))
    g.add_node("Retrieval", make_node_retrieval(ctx))
    g.add_node("Reasoning", make_node_reasoning(ctx))
    g.add_node("Verify", make_node_verify(ctx))
    g.add_node("PackResult", make_node_pack_result(ctx))

    # === 新增回退节点 ===
    g.add_node("RetryRetrieval", lambda s: node_claim_retrieval(s, ctx.retriever, sink=ctx.sink))


    # === 主流程 ===
    g.add_edge(START, "InitExternal")
    g.add_edge("InitExternal", "Ingest")
    g.add_edge("Ingest", "BuildGraph")
    g.add_edge("BuildGraph", "ChooseRoute")

    def route_selector(state: WFState) -> str:
        rv = "Retrieval" if state.get("route") == "Retrieval" else "PackResult"
        print("[route_selector] route in state =", repr(state.get("route")), "-> return", rv)
        return rv

    g.add_conditional_edges(
        "ChooseRoute",
        route_selector,
        {"Retrieval": "Retrieval", "PackResult": "PackResult"},
    )

    g.add_edge("Retrieval", "Reasoning")
    g.add_edge("Reasoning", "Verify")

    # === Verify 分支控制 ===
    def verify_selector(state: WFState) -> str:
        verify_out = state.get("verification") or {}
        verdict = verify_out.get("verdict")
        status_detail = (verify_out.get("status_detail") or "").lower()
        final_score = float(verify_out.get("final_score") or 0.0)
        retries = int(verify_out.get("retry_round", 0))  # 直接取 verify 节点写回的值
        trace_id = state.get("trace_id", "trace-demo")

        # 打印 selector 的输入情况
        print(f"[verify_selector][{trace_id}] verdict={verdict}, "
            f"status_detail={status_detail}, score={final_score:.3f}, "
            f"retry_round(seen)={retries}")

        if (
            verdict in ("FAIL-UNSUPPORTED", "FAIL-CONTRADICTED", "INCONCLUSIVE")
            or (status_detail == "low_conf_pass" and final_score < 0.55)
        ) and retries < 1:
            print(f"[verify_selector][{trace_id}] -> RetryRetrieval")
            return "RetryRetrieval"

        print(f"[verify_selector][{trace_id}] -> PackResult")
        return "PackResult"

    g.add_conditional_edges(
        "Verify",
        verify_selector,
        {"RetryRetrieval": "RetryRetrieval", "PackResult": "PackResult"},
    )

    # === 回退闭环 ===
    g.add_edge("RetryRetrieval", "Reasoning")
    g.add_edge("Reasoning", "Verify")

    # === 收尾 ===
    g.add_edge("PackResult", END)

    return g.compile()