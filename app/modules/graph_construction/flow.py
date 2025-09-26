# app/modules/graph_construction/flow.py
from __future__ import annotations
from typing import Dict, Any, TypedDict, List, Optional, Tuple, DefaultDict
from collections import defaultdict
import uuid, time

from langgraph.graph import StateGraph, START, END

from app.core.dto import GraphBuildIn, GraphBuildOut, RetrievalIn
from app.core.interfaces import GraphConstruction
from app.core.llm_router import LLMRouter
from app.di.factory import import_from_string
from app.modules.graph_construction.node_builder import NodeBuilder
from app.modules.graph_construction.edge_builder import EdgeBuilder
from app.modules.retrieval.flow import RetrievalAgentFlow
from app.telemetry.sinks import TelemetrySink, span


class _GCState(TypedDict, total=False):
    req: GraphBuildIn
    policy: Dict[str, Any]
    graph_id: str
    question_text: str
    context: List[Any]
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    out: GraphBuildOut
    t_build_start: float
    t_build_end: float


class GraphConstructionFlow(GraphConstruction):
    """
    Graph Construction Flow：
      Ingest -> BootstrapContext(如果context为空则检索生成) -> BuildNodes -> BuildEdges -> Assemble/Save -> Summarize
    """

    def __init__(
        self,
        impl,
        router: Optional[LLMRouter] = None,
        *,
        node_builder_kwargs: Dict[str, Any] | None = None,
        edge_builder_kwargs: Dict[str, Any] | None = None,
        sink: TelemetrySink | None = None,
        settings: Dict[str, Any] | None = None,
        bootstrap_top_k: int = 20,
        retriever=None,
    ):
        self.impl: GraphConstruction = impl
        self.router = router
        self.sink = sink

        # builders（沿用你现有的签名）
        self.node_builder = NodeBuilder(**(node_builder_kwargs or {}))
        self.edge_builder = EdgeBuilder(**(edge_builder_kwargs or {}))

        # bootstrap 配置
        self.settings = settings or {}
        self.bootstrap_top_k = int(bootstrap_top_k)
        self.retriever = retriever  # 预注入或延迟创建

        self._wf = self._compile_flow()

    @classmethod
    def from_settings(
        cls,
        settings: Dict[str, Any],
        router: Optional[LLMRouter] = None,
        sink: Optional[TelemetrySink] = None,
    ) -> "GraphConstructionFlow":
        modules_cfg = settings.get("modules", {}) or {}
        cfg = (modules_cfg.get("graph_construction") or settings.get("graph_construction") or {}) or {}

        impl_spec = cfg.get("impl") or cfg.get("type") or \
            "app.modules.graph_construction.impl_networkx:GraphConstructionNetworkX"
        impl_kwargs = dict(cfg.get("impl_kwargs") or {})

        # --- 修复点：剔除 node_builder / edge_builder ---
        node_builder_kwargs = impl_kwargs.pop("node_builder", cfg.get("node_builder", {}) or {})
        edge_builder_kwargs = impl_kwargs.pop("edge_builder", cfg.get("edge_builder", {}) or {})

        ImplCls = import_from_string(impl_spec)
        impl = ImplCls(**impl_kwargs)

        bootstrap_cfg = dict(cfg.get("bootstrap") or {})
        bootstrap_top_k = int(bootstrap_cfg.get("top_k", 20))

        retriever = None
        try:
            retriever = RetrievalAgentFlow.from_settings(settings, router=router)
        except Exception:
            retriever = None

        return cls(
            impl=impl,
            router=router,
            sink=sink,
            node_builder_kwargs=node_builder_kwargs,
            edge_builder_kwargs=edge_builder_kwargs,
            settings=settings,
            bootstrap_top_k=bootstrap_top_k,
            retriever=retriever,
        )


    # ---------------- flow nodes ----------------

    def _node_ingest(self, st: _GCState) -> _GCState:
        req = st["req"]
        st["question_text"] = req.question_text
        st["context"] = list(req.context or [])
        st["graph_id"] = req.graph_id or f"graph-{req.trace_id}-{uuid.uuid4().hex[:8]}"
        st["policy"] = (req.extra or {}).get("policy", {})
        return st

    def _node_bootstrap_context(self, st: _GCState) -> _GCState:
        # 已有 context 则跳过
        if st.get("context"):
            return st

        # 若还没有 retriever，则尝试基于当前 settings 构建一个
        if self.retriever is None and self.router is not None:
            try:
                self.retriever = RetrievalAgentFlow.from_settings(self.settings, router=self.router)
            except Exception:
                self.retriever = None

        if self.retriever is None:
            # 没有检索器，兜底只建 question 节点
            return st

        req = st["req"]
        trace_id = req.trace_id or "trace-gc-bootstrap"
        top_k = self.bootstrap_top_k

        if self.sink:
            with span("GC/BootstrapContext", self.sink, trace_id):
                ro = self.retriever.retrieve(
                    RetrievalIn(query=st["question_text"], graph_id="", top_k=top_k, trace_id=trace_id)
                )
        else:
            ro = self.retriever.retrieve(
                RetrievalIn(query=st["question_text"], graph_id="", top_k=top_k, trace_id=trace_id)
            )

        # 将 hits 聚合为 NodeBuilder 需要的 context 结构：[(doc, [sent...]), ...]
        by_doc: DefaultDict[str, List[Tuple[int, str]]] = defaultdict(list)
        for h in ro.hits:
            meta = h.meta or {}
            doc = str(meta.get("doc") or "default")
            text = str(meta.get("text") or "")
            sid = meta.get("sent_id")
            try:
                sid_int = int(sid) if sid is not None else 10**9  # 无 sent_id 放到末尾
            except Exception:
                sid_int = 10**9
            if text:
                by_doc[doc].append((sid_int, text))

        context: List[Tuple[str, List[str]]] = []
        for doc, pairs in by_doc.items():
            pairs_sorted = sorted(pairs, key=lambda x: x[0])
            sents: List[str] = []
            seen = set()
            for _, t in pairs_sorted:
                if t not in seen:
                    sents.append(t)
                    seen.add(t)
            if sents:
                context.append((doc, sents))

        st["context"] = context
        return st

    def _node_build_nodes(self, st: _GCState) -> _GCState:
        req = st["req"]; trace_id = req.trace_id or "trace-gc"
        def _do():
            base_nodes = self.node_builder.build(
                question=st["question_text"],
                context=st["context"],
                policy=st.get("policy", {}),
            )
            if req.nodes:
                seen_ids = {n["id"] for n in base_nodes}
                merged = base_nodes + [n for n in req.nodes if n.get("id") not in seen_ids]
                st["nodes"] = merged
            else:
                st["nodes"] = base_nodes
            return st
        if self.sink:
            with span("GC/BuildNodes", self.sink, trace_id):
                return _do()
        return _do()

    def _node_build_edges(self, st: _GCState) -> _GCState:
        req = st["req"]; trace_id = req.trace_id or "trace-gc"
        def _do():
            base_edges = self.edge_builder.build(
                nodes=st["nodes"],
                question=st["question_text"],
                policy=st.get("policy", {}),
            )
            st["edges"] = base_edges + (list(req.edges) if req.edges else [])
            return st
        if self.sink:
            with span("GC/BuildEdges", self.sink, trace_id):
                return _do()
        return _do()

    def _node_assemble_save(self, st: _GCState) -> _GCState:
        req = st["req"]
        trace_id = req.trace_id or "trace-gc"

        def _do():
            t0 = time.time()

            # ✅ 修复点：确保是 List[Dict]
            nodes = [n.model_dump() if hasattr(n, "model_dump") else n for n in st["nodes"]]
            edges = [e.model_dump() if hasattr(e, "model_dump") else e for e in st["edges"]]

            out = self.impl.build(
                GraphBuildIn(
                    trace_id=req.trace_id,
                    question_text=st["question_text"],
                    context=st["context"],
                    graph_id=st["graph_id"],
                    nodes=nodes,
                    edges=edges,
                    extra=req.extra,
                )
            )
            t1 = time.time()

            st["t_build_start"] = t0
            st["t_build_end"] = t1
            st["out"] = GraphBuildOut(
                graph_id=out.graph_id,
                node_count=out.node_count,
                edge_count=out.edge_count,
                nodes=nodes,
                edges=edges,
                provenance=(out.provenance or None),
                diagnostics={**(out.diagnostics or {}), "t_build_sec": t1 - t0},
                extra=out.extra,
            )
            return st

        if self.sink:
            with span("GC/AssembleSave", self.sink, trace_id):
                return _do()
        return _do()

    def _node_summarize(self, st: _GCState) -> _GCState:
        return st

    def _compile_flow(self):
        g = StateGraph(_GCState)
        g.add_node("Ingest", self._node_ingest)
        g.add_node("BootstrapContext", self._node_bootstrap_context)
        g.add_node("BuildNodes", self._node_build_nodes)
        g.add_node("BuildEdges", self._node_build_edges)
        g.add_node("AssembleSave", self._node_assemble_save)
        g.add_node("Summarize", self._node_summarize)
        g.add_edge(START, "Ingest")
        g.add_edge("Ingest", "BootstrapContext")
        g.add_edge("BootstrapContext", "BuildNodes")
        g.add_edge("BuildNodes", "BuildEdges")
        g.add_edge("BuildEdges", "AssembleSave")
        g.add_edge("AssembleSave", "Summarize")
        g.add_edge("Summarize", END)
        return g.compile()

    def build(self, req: GraphBuildIn) -> GraphBuildOut:
        st0: _GCState = {"req": req}
        st1 = self._wf.invoke(input=st0)
        return st1["out"]
