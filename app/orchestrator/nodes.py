# app/orchestrator/nodes.py
from __future__ import annotations
from typing import Dict, Any, Callable
import time

from app.orchestrator.state import WFState
from app.core.dto import GraphBuildIn, RetrievalIn, ReasoningIn, VerifyIn, RetrievalIn, RetrievalOut, VerifyOut
from app.core.interfaces import GraphConstruction, RetrievalAgent, ReasoningAgent, VerifierAgent
from app.telemetry.sinks import TelemetrySink, span

from app.telemetry.sinks import record_metrics

print("[nodes loaded from]", __file__)     # 加在文件顶部

class NodeContext:
    def __init__(self, graph_c: GraphConstruction, retriever: RetrievalAgent, reasoner: ReasoningAgent, verifier: VerifierAgent, sink: TelemetrySink | None = None):
        self.graph_c = graph_c
        self.retriever = retriever
        self.reasoner = reasoner
        self.verifier = verifier
        self.sink = sink

def _merge(state: WFState, extra: Dict[str, Any]) -> WFState:
    new_state = dict(state)
    new_state.update(extra)
    return new_state  # type: ignore

def make_node_ingest(ctx: NodeContext) -> Callable[[WFState], WFState]:
    def node_ingest(state: WFState) -> WFState:
        trace_id = state.get("trace_id", "trace-demo")
        if ctx.sink:
            with span("Ingest", ctx.sink, trace_id):
                q = (state.get("question") or "").strip()
                if not q:
                    raise ValueError("Empty question")
                ext = state.get("external_context", {})
                if "context" not in ext or not isinstance(ext["context"], list):
                    ext["context"] = []
                return _merge(state, {"question": q, "external_context": ext})
        # no sink fallback
        q = (state.get("question") or "").strip()
        if not q:
            raise ValueError("Empty question")
        ext = state.get("external_context", {})
        if "context" not in ext or not isinstance(ext["context"], list):
            ext["context"] = []
        return _merge(state, {"question": q, "external_context": ext})
    return node_ingest

def make_node_build_graph(ctx: NodeContext) -> Callable[[WFState], WFState]:
    def node_build_graph(state: WFState) -> WFState:
        trace_id = state.get("trace_id", "trace-demo")
        q = state["question"]
        context = (state.get("external_context") or {}).get("context", [])
        if ctx.sink:
            with span("BuildGraph", ctx.sink, trace_id):
                t0 = time.time()
                out = ctx.graph_c.build(GraphBuildIn(
                    trace_id=trace_id, question_text=q, context=context, extra={"meta": state.get("meta", {})}
                ))
                t1 = time.time()
                return _merge(state, {"graph": out.model_dump(), "t0": state.get("t0", t0), "t1": t1})
        # fallback
        t0 = time.time()
        out = ctx.graph_c.build(GraphBuildIn(
            trace_id=trace_id, question_text=q, context=context, extra={"meta": state.get("meta", {})}
        ))
        t1 = time.time()
        return _merge(state, {"graph": out.model_dump(), "t0": state.get("t0", t0), "t1": t1})
    return node_build_graph

def make_node_choose_route(ctx) -> Callable[[WFState], WFState]:
    """
    根据 policy.mode 选择路由：
      - "full"   -> "Retrieval"
      - 其他/缺省 -> "PackResult"

    注意：返回“增量”字典而非整份状态，交由 LangGraph 合并；
         这样只要 WFState 声明了 `route` 字段，就能被后续节点读到。
    """
    def node_choose_route(state: WFState) -> WFState:
        trace_id = state.get("trace_id", "trace-demo")

        # 1) 读取并归一化 mode（容忍大小写/空格）
        raw_mode = (state.get("policy") or {}).get("mode", "graph_only")
        mode = raw_mode.strip().lower() if isinstance(raw_mode, str) else "graph_only"

        # 2) 计算路由
        route = "Retrieval" if mode == "full" else "PackResult"
        log_msg = f"[ChooseRoute] mode={raw_mode!r} -> route={route!r}"

        # 3) 可观测性（有 sink 则包一层 span）
        if getattr(ctx, "sink", None):
            with span("ChooseRoute", ctx.sink, trace_id):
                print(log_msg)
                return {"route": route}

        print(log_msg)
        return {"route": route}

    return node_choose_route

def make_node_retrieval(ctx: NodeContext) -> Callable[[WFState], WFState]:
    def node_retrieval(state: WFState) -> WFState:
        if state.get("route") != "Retrieval":
            return state
        trace_id = state.get("trace_id", "trace-demo")
        if ctx.sink:
            with span("Retrieval", ctx.sink, trace_id):
                r = ctx.retriever.retrieve(RetrievalIn(
                    query=state.get("question", ""),
                    graph_id=(state.get("graph") or {}).get("graph_id", ""),
                    top_k=20,
                    trace_id=trace_id,
                ))
                return _merge(state, {"retrieval": r.model_dump()})
        r = ctx.retriever.retrieve(RetrievalIn(
            query=state.get("question", ""),
            graph_id=(state.get("graph") or {}).get("graph_id", ""),
            top_k=20,
            trace_id=trace_id,
        ))
        return _merge(state, {"retrieval": r.model_dump()})
    return node_retrieval

def make_node_reasoning(ctx: NodeContext) -> Callable[[WFState], WFState]:
    def node_reasoning(state: WFState) -> WFState:
        if state.get("route") != "Retrieval":
            return state
        trace_id = state.get("trace_id", "trace-demo")
        if ctx.sink:
            with span("Reasoning", ctx.sink, trace_id):
                q = state.get("question", "")
                rid = (state.get("graph") or {}).get("graph_id", "")
                hits = (state.get("retrieval") or {}).get("hits", [])
                r = ctx.reasoner.reason(ReasoningIn(question=q, hits=hits, graph_id=rid, trace_id=trace_id))
                return _merge(state, {"reasoning": r.model_dump()})
        q = state.get("question", "")
        rid = (state.get("graph") or {}).get("graph_id", "")
        hits = (state.get("retrieval") or {}).get("hits", [])
        r = ctx.reasoner.reason(ReasoningIn(question=q, hits=hits, graph_id=rid, trace_id=trace_id))
        return _merge(state, {"reasoning": r.model_dump()})
    return node_reasoning

def make_node_verify(ctx: NodeContext) -> Callable[[WFState], WFState]:
    def node_verify(state: WFState) -> WFState:
        if state.get("route") != "Retrieval":
            return state

        trace_id = state.get("trace_id", "trace-demo")
        retry_round = int(state.get("_verify_retries", 0))

        # 进入 verify 节点时打印
        print(f"[node_verify][{trace_id}] enter, retry_round(before)={retry_round}")

        ans = (state.get("reasoning") or {}).get("answer", "")
        hits = (state.get("retrieval") or {}).get("hits", [])
        rid = (state.get("graph") or {}).get("graph_id", "")

        verify_in = VerifyIn(
            answer=ans,
            evidence=hits,
            graph_id=rid,
            trace_id=trace_id,
            retry_round=retry_round,
            question=state.get("question"),
            query=state.get("question"),
        )

        v = ctx.verifier.verify(verify_in)
        v_dict = v.model_dump()

        verdict = v_dict.get("verdict")
        status_detail = (v_dict.get("status_detail") or "").lower()
        final_score = float(v_dict.get("final_score") or 0.0)

        # 条件满足时递增 retry_round
        if (
            verdict in ("FAIL-UNSUPPORTED", "FAIL-CONTRADICTED", "INCONCLUSIVE")
            or (status_detail == "low_conf_pass" and final_score < 0.55)
        ) and retry_round < 1:
            retry_round += 1

        retrieval = state.get("retrieval") or {}
        retrieval_source = retrieval.get("source", "default")

        # 退出 verify 节点时打印
        print(f"[node_verify][{trace_id}] exit, verdict={verdict}, "
              f"status_detail={status_detail}, score={final_score:.3f}, "
              f"retry_round(after)={retry_round}")

        # 把 retry_round 放进 verification，保证 selector 能读到
        v_dict["retry_round"] = retry_round

        return _merge(state, {
            "verification": v_dict,
            "_verify_retries": retry_round,
            "retry_round": retry_round,        # 顶层也保留一份
            "retrieval_source": retrieval_source,
        })

    return node_verify

def make_node_pack_result(ctx: NodeContext) -> Callable[[WFState], WFState]:
    def node_pack_result(state: WFState) -> WFState:
        trace_id = state.get("trace_id", "trace-demo")
        retry_round = int(state.get("_verify_retries", 0))

        retrieval = state.get("retrieval") or {}
        retrieval_source = retrieval.get("source", "default")  # ✅ 默认普通检索

        result = {
            "graph": state.get("graph"),
            "retrieval": retrieval,
            "reasoning": state.get("reasoning"),
            "verification": state.get("verification"),
            "metrics": {
                "t0": state.get("t0"),
                "t1": state.get("t1"),
                "t_end": time.time(),
                "retry_round": retry_round,
                "retrieval_source": retrieval_source,  # ✅ 新增
            },
            "retry_round": retry_round,
            "retrieval_source": retrieval_source,      # ✅ 顶层也保留一份
        }

        if ctx.sink:
            with span("PackResult", ctx.sink, trace_id):
                return _merge(state, {"result": result})
        return _merge(state, {"result": result})

    return node_pack_result

def node_claim_retrieval(state, retrieval_agent, sink=None):
    """
    基于 Claim-Check 的回退检索节点。
    从 Verifier 输出的 claim_check 里提取 claims，
    拼接为 query 再调用 retrieval agent。
    """
    verify_out = state.get("verification") or {}
    trace_id = state.get("trace_id", "trace-claim")

    claims = [
        c["claim"]
        for c in (verify_out.get("diagnostics", {}).get("claim_check", {}).get("results", []))
        if c.get("claim")
    ]

    if not claims:
        empty = RetrievalOut(hits=[], model="claim-fallback").model_dump()
        empty["source"] = "claim-retrieval"   # ✅ 标记回退检索
        return _merge(state, {"retrieval": empty})

    query = "; ".join(claims)
    retrieval_in = RetrievalIn(query=query, top_k=20, trace_id=f"{trace_id}-claim")

    if sink:
        with span("ClaimRetrieval", sink, trace_id):
            retrieval_out = retrieval_agent.retrieve(retrieval_in)
    else:
        retrieval_out = retrieval_agent.retrieve(retrieval_in)

    out_dict = retrieval_out.model_dump()

    # ✅ 给每条 hit 添加 source=claim-retrieval
    for h in out_dict.get("hits", []):
        if isinstance(h, dict):
            h["source"] = "claim-retrieval"

    out_dict["source"] = "claim-retrieval"   # ✅ 给整体结果也打上标记
    return _merge(state, {"retrieval": out_dict})
