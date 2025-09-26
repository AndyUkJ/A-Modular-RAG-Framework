# app/modules/retrieval/flow.py
from __future__ import annotations
from typing import TypedDict, Any, List, Dict, Optional
from langgraph.graph import StateGraph, START, END

from app.core.interfaces import RetrievalAgent
from app.core.dto import RetrievalIn, RetrievalOut, Hit
from app.core.llm_router import LLMRouter
from app.telemetry.sinks import TelemetrySink, span
from app.di.factory import import_from_string

# 默认内置依赖，仅在未注入 backend 时使用
from .text_index import BM25LiteIndex
from .graph_utils import load_graph_json, build_index, expand_qmatch_neighbors


class _RState(TypedDict, total=False):
    req: RetrievalIn
    queries: List[str]
    bm25_hits: List[Hit]
    graph_hits: List[Hit]
    out: RetrievalOut


class RetrievalAgentFlow(RetrievalAgent):
    """
    Flow 适配层，支持两种模式：
      1) 外部注入 backend（如 HybridRetrievalBackend） -> 直接调用 backend.retrieve(req)
      2) 未注入 backend 时，使用内置 BM25 + 图邻域扩展的 LangGraph 流程
    """

    def __init__(
        self,
        router: LLMRouter,
        *,
        # Flow 自身参数（命中归一化时会用；保留扩展性）
        id_keys: Optional[List[str]] = None,
        score_keys: Optional[List[str]] = None,

        # 内置流程参数（只有 backend=None 时才会用到）
        index_path: str = "data/hotpotqa/docs.jsonl",
        graph_root: str = "data/graph",
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        graph_window: int = 1,
        alpha_text: float = 0.7,
        alpha_graph: float = 0.3,

        # 外部实现
        backend: Any = None,
        sink: Optional[TelemetrySink] = None,
    ):
        self.router = router
        self.sink = sink
        self.backend = backend

        # Flow 层参数（当前仅占位，适配器层/标准化可使用）
        self.id_keys = id_keys or ["id", "doc_id", "docId", "sid", "sent_id"]
        self.score_keys = score_keys or ["score", "relevance", "sim", "s"]

        # 内置流程参数
        self.index_path = index_path
        self.graph_root = graph_root
        self.graph_window = max(0, int(graph_window))
        self.alpha_text = float(alpha_text)
        self.alpha_graph = float(alpha_graph)

        if backend is None:
            # 默认 BM25-lite + 图邻域扩展
            self._bm25 = BM25LiteIndex(path=index_path, k1=bm25_k1, b=bm25_b)
            self._wf = self._compile()
        else:
            self._wf = None  # backend 模式不需要 LangGraph

    @classmethod
    def from_settings(
        cls,
        settings: Dict[str, Any],
        router: LLMRouter,
        sink: Optional[TelemetrySink] = None,
    ) -> "RetrievalAgentFlow":
        cfg = (settings.get("modules", {}) or {}).get("retrieval", {}) or {}

        # 1) Flow 自身参数
        flow_kwargs = dict(cfg.get("kwargs", {}) or {})

        # 2) Impl（backend）参数
        impl_spec = cfg.get("impl")
        impl_kwargs = dict(cfg.get("impl_kwargs", {}) or {})

        backend = None
        if impl_spec:
            ImplCls = import_from_string(impl_spec)

            # 过滤 impl_kwargs，只保留 Impl __init__ 可接收的字段；
            # 同时按需注入 router/sink（若构造函数声明了）
            import inspect
            sig = inspect.signature(ImplCls.__init__)
            valid_params = set(sig.parameters.keys())
            # 自动注入 router/sink
            if "router" in valid_params:
                impl_kwargs.setdefault("router", router)
            if "sink" in valid_params:
                impl_kwargs.setdefault("sink", sink)
            # 仅保留有效字段
            impl_kwargs = {k: v for k, v in impl_kwargs.items() if k in valid_params and k != "self"}

            backend = ImplCls(**impl_kwargs)

        return cls(router=router, backend=backend, sink=sink, **flow_kwargs)

    # ---------------- 内置默认流程（仅 backend=None 时使用） ----------------

    def _node_expand(self, st: _RState) -> _RState:
        req = st["req"]
        qe = self.router.complete(
            module="RetrievalAgent",
            purpose="query_expand",
            prompt=f"Expand 2-3 short queries (one per line) for: {req.query}",
            require={"context_window": 8000, "temperature": 0.2, "trace_id": req.trace_id},
        )
        # 尽量兼容各种 provider 返回结构
        text = ""
        if isinstance(qe, str):
            text = qe
        elif isinstance(qe, dict):
            text = qe.get("text") or qe.get("message", {}).get("content") or ""
        lines = [q.strip() for q in (text or "").splitlines() if q.strip()]
        st["queries"] = [req.query] + lines[:2]
        return st

    def _node_retrieve_text(self, st: _RState) -> _RState:
        req = st["req"]
        queries = st.get("queries") or [req.query]
        hits: List[Hit] = []
        if getattr(self, "_bm25", None) and self._bm25.N > 0:
            ranked = self._bm25.search(queries, top_k=req.top_k or 20, alpha_merge="max")
            for doc_idx, s in ranked:
                meta = self._bm25.doc_meta(doc_idx)
                hid = f"sent::{meta.get('doc_id') or meta.get('title')}"
                hits.append(
                    Hit(
                        id=hid,
                        score=float(s),
                        meta={
                            "kind": "sentence",
                            "text": meta.get("text"),
                            "doc": meta.get("title"),
                            "sent_id": meta.get("sent_id"),
                            "source": "bm25",
                        },
                    )
                )
        st["bm25_hits"] = hits
        return st

    def _node_graph_expand(self, st: _RState) -> _RState:
        req = st["req"]
        gid = req.graph_id or ""
        if not gid:
            st["graph_hits"] = []
            return st

        g = load_graph_json(self.graph_root, gid)
        nodes_by_id, next_f, next_b, node_texts, qmatch_list = build_index(g)

        expanded = expand_qmatch_neighbors(
            q_text=req.query,
            nodes_by_id=nodes_by_id,
            next_forward=next_f,
            next_backward=next_b,
            node_texts=node_texts,
            explicit_qmatch=qmatch_list,
            window=self.graph_window,
        )
        hits: List[Hit] = []
        for sid, (gs, meta) in expanded.items():
            hid = f"sent::{sid}"
            hits.append(Hit(id=hid, score=float(gs), meta={**meta, "source": "graph"}))
        st["graph_hits"] = sorted(hits, key=lambda h: h.score, reverse=True)[: (req.top_k or 20)]
        return st

    def _node_rank_select(self, st: _RState) -> _RState:
        alpha_t = self.alpha_text
        alpha_g = self.alpha_graph
        tmap: Dict[str, Hit] = {h.id: h for h in (st.get("bm25_hits") or [])}
        gmap: Dict[str, Hit] = {h.id: h for h in (st.get("graph_hits") or [])}

        ids = set(tmap.keys()) | set(gmap.keys())
        fused: List[Hit] = []
        for _id in ids:
            ts = tmap.get(_id).score if _id in tmap else 0.0
            gs = gmap.get(_id).score if _id in gmap else 0.0
            score = alpha_t * ts + alpha_g * gs
            meta = {}
            if _id in tmap:
                meta.update(tmap[_id].meta or {})
            if _id in gmap:
                meta.update(gmap[_id].meta or {})
            fused.append(Hit(id=_id, score=float(score), meta=meta))

        fused = sorted(fused, key=lambda h: h.score, reverse=True)[: (st["req"].top_k or 20)]
        st["out"] = RetrievalOut(
            hits=fused,
            diagnostics={
                "queries": st.get("queries", []),
                "bm25_candidates": len(st.get("bm25_hits") or []),
                "graph_candidates": len(st.get("graph_hits") or []),
                "alpha_text": alpha_t,
                "alpha_graph": alpha_g,
            },
        )
        return st

    def _compile(self):
        g = StateGraph(_RState)
        g.add_node("Expand", self._node_expand)
        g.add_node("RetrieveText", self._node_retrieve_text)
        g.add_node("GraphExpand", self._node_graph_expand)
        g.add_node("RankSelect", self._node_rank_select)
        g.add_edge(START, "Expand")
        g.add_edge("Expand", "RetrieveText")
        g.add_edge("RetrieveText", "GraphExpand")
        g.add_edge("GraphExpand", "RankSelect")
        g.add_edge("RankSelect", END)
        return g.compile()

    # 对外稳定接口
    def retrieve(self, req: RetrievalIn) -> RetrievalOut:
        trace_id = getattr(req, "trace_id", None) or "trace-retrieval"

        if self.backend is not None:
            # 外部实现模式
            if self.sink:
                with span("RetrievalAdapter/backend", self.sink, trace_id):
                    return self.backend.retrieve(req)
            return self.backend.retrieve(req)

        # 内置流程模式
        st0: _RState = {"req": req}
        if self.sink:
            with span("RetrievalAdapter/flow", self.sink, trace_id):
                st1 = self._wf.invoke(input=st0)
        else:
            st1 = self._wf.invoke(input=st0)
        return st1["out"]
