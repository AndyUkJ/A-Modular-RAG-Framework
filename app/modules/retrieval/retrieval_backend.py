# app/modules/retrieval/retrieval_backend.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import math, logging

from app.core.dto import RetrievalIn, RetrievalOut, Hit
from app.core.llm_router import LLMRouter
from app.telemetry.sinks import TelemetrySink, span

from .text_index import BM25LiteIndex
from .graph_utils import load_graph_json, build_index, expand_qmatch_neighbors

logger = logging.getLogger(__name__)


# ========== A & D ==========
@dataclass
class LLMQueryExpander:
    router: LLMRouter
    lines: int = 3
    enable_attribute_paraphrase: bool = True
    _attr_fallbacks: Dict[str, List[str]] = None

    def __post_init__(self):
        if self._attr_fallbacks is None:
            self._attr_fallbacks = {
                "nationality": ["citizen of", "from", "born in", "is an American", "is a British"],
                "spouse": ["married to", "husband", "wife"],
                "birth place": ["born in", "hails from"],
                "death place": ["died in", "passed away in"],
            }

    def _coerce_text(self, out: Any) -> str:
        if out is None: return ""
        if isinstance(out, str): return out
        if isinstance(out, dict):
            t = out.get("text")
            if isinstance(t, str): return t
            if isinstance(t, dict):
                if isinstance(t.get("text"), str): return t["text"]
                if isinstance(t.get("content"), str): return t["content"]
            msg = out.get("message")
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                return msg["content"]
            choices = out.get("choices")
            if isinstance(choices, list) and choices:
                ch0 = choices[0]
                if isinstance(ch0, dict):
                    if isinstance(ch0.get("text"), str): return ch0["text"]
                    if isinstance(ch0.get("message"), dict):
                        return ch0["message"].get("content", "")
        return ""

    def _build_prompt(self, query: str) -> str:
        if not self.enable_attribute_paraphrase:
            return f"Expand {self.lines} short queries (one per line) for: {query}"
        return (
            "You are improving recall for a retrieval system.\n"
            f"Task: Expand {self.lines} short search queries (one per line) for:\n{query}\n\n"
            "Rules:\n"
            "- Include paraphrases and synonyms.\n"
            "- Expand with related attributes or relations.\n"
            "  e.g., nationality → born in, citizen of, from, is an American\n"
            "- Keep each line short (<=8 words), no numbering.\n"
        )

    def _static_fallbacks(self, original_query: str) -> List[str]:
        ql = (original_query or "").lower()
        extras: List[str] = []
        for k, alts in (self._attr_fallbacks or {}).items():
            if k in ql:
                extras.extend(alts[:2])
        if len(original_query.split()) <= 10 and extras:
            subject = original_query
            extras = [f"{alt} {subject}" for alt in extras]
        return extras

    def expand(self, *, query: str, trace_id: str) -> List[str]:
        try:
            out = self.router.complete(
                module="RetrievalAgent",
                purpose="query_expand",
                prompt=self._build_prompt(query),
                require={"context_window": 8000, "temperature": 0.2, "trace_id": trace_id},
            )
            text = self._coerce_text(out)
            lines = [q.lstrip("-•").strip() for q in (text or "").splitlines() if q.strip()]

            fallbacks = self._static_fallbacks(query)
            uniq, seen = [], set()
            for q in lines + fallbacks:
                ql = q.lower()
                if ql and ql not in seen:
                    seen.add(ql)
                    uniq.append(q)

            logger.debug(f"[LLMQueryExpander] query={query}, expanded={uniq}")
            return uniq[: self.lines]
        except Exception as e:
            logger.error(f"[LLMQueryExpander] expand error: {e}")
            return self._static_fallbacks(query)[: self.lines]


# ========== 文本检索 ==========
@dataclass
class BM25TextSearcher:
    index: BM25LiteIndex

    def search(self, *, queries: List[str], top_k: int) -> List[Dict[str, Any]]:
        if self.index.N <= 0: return []
        ranked = self.index.search(queries, top_k=top_k, alpha_merge="max")
        out: List[Dict[str, Any]] = []
        for doc_idx, s in ranked:
            meta = self.index.doc_meta(doc_idx)
            doc_part = meta.get("doc_id") or meta.get("title") or "doc"
            sid_part = str(meta.get("sent_id") or "")
            out.append({
                "id": f"sent::{doc_part}::{sid_part}",
                "score": float(s),
                "meta": {
                    "kind": "sentence",
                    "text": meta.get("text"),
                    "doc": meta.get("title"),
                    "sent_id": meta.get("sent_id"),
                    "source": "bm25",
                }
            })
        return out


# ========== 图邻域扩展 ==========
@dataclass
class GraphNeighborExpander:
    graph_root: str
    window: int = 1  # 默认邻域深度

    def expand(
        self,
        *,
        query: str,
        graph_id: str,
        top_k: int,
        window_override: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        基于 q_match 的邻域扩展：
        - 支持 BFS 多跳扩展（依赖 graph_utils.expand_qmatch_neighbors）
        - window_override 可覆盖默认 self.window
        """
        if not graph_id:
            return []

        g = load_graph_json(self.graph_root, graph_id)
        nodes_by_id, next_f, next_b, node_texts, qmatch = build_index(g)
        win = max(0, int(window_override if isinstance(window_override, int) else self.window))

        expanded = expand_qmatch_neighbors(
            q_text=query,
            nodes_by_id=nodes_by_id,
            next_forward=next_f,
            next_backward=next_b,
            node_texts=node_texts,
            explicit_qmatch=qmatch,
            window=win,
        )

        hits: List[Dict[str, Any]] = [
            {
                "id": f"sent::{sid}",
                "score": float(gs),
                "meta": {**meta, "source": "graph"},
            }
            for sid, (gs, meta) in expanded.items()
        ]

        hits.sort(key=lambda h: h["score"], reverse=True)
        logger.debug(
            f"[GraphNeighborExpander] query={query}, graph_id={graph_id}, "
            f"window={win}, got={len(hits)}"
        )
        return hits[:top_k]


# ========== A & B ==========
@dataclass
class DenseReranker:
    router: LLMRouter
    max_pool: int = 200
    embed_batch: int = 50

    def _cosine(self, a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b): return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return float(dot / (na * nb)) if na and nb else 0.0

    def _resolve_embed_model(self) -> str:
        try:
            policy = getattr(self.router, "policy", {}) or {}
            emb = (policy.get("embedding") or [])
            if emb and isinstance(emb[0], dict) and emb[0].get("model"):
                return emb[0]["model"]
            provider_name = policy.get("embedding_provider")
            providers = getattr(self.router, "providers", {}) or {}
            prov = providers.get(provider_name) if provider_name else None
            if prov and hasattr(prov, "kwargs"):
                m = (getattr(prov, "kwargs") or {}).get("embed_model")
                if m: return m
        except Exception:
            pass
        return "text-embedding-3-large"

    def score(self, *, query: str, candidates: List[Dict[str, Any]], trace_id: str) -> Dict[str, float]:
        if not candidates: return {}
        texts, ids = [], []
        for h in candidates[: self.max_pool]:
            t = (h.get("meta") or {}).get("text") or ""
            if t: ids.append(h["id"]); texts.append(t)
        if not texts: return {}

        model_hint = self._resolve_embed_model()
        logger.info(f"[DenseReranker] model={model_hint}, batch_size={self.embed_batch}, total_texts={len(texts)}")

        try:
            q_emb = self.router.embed(model_hint=model_hint, texts=[query], require={"trace_id": trace_id})
            qv = (q_emb.get("vectors") if isinstance(q_emb, dict) else q_emb)[0]
        except Exception as e:
            logger.error(f"[DenseReranker] query embed error: {e}", exc_info=True)
            return {}

        all_vecs: List[List[float]] = []
        bs = max(8, int(self.embed_batch))
        for i in range(0, len(texts), bs):
            chunk = texts[i:i + bs]
            try:
                t_emb = self.router.embed(model_hint=model_hint, texts=chunk, require={"trace_id": trace_id})
                vecs = t_emb.get("vectors") if isinstance(t_emb, dict) else t_emb
                all_vecs.extend(vecs or [])
            except Exception as be:
                logger.error(f"[DenseReranker] batch embed error: {be}", exc_info=True)
                all_vecs.extend([[0.0] * len(qv) for _ in chunk])

        scores = {i: self._cosine(qv, v) for i, v in zip(ids, all_vecs)}
        logger.debug(f"[DenseReranker] DONE, scored={len(scores)}")
        return scores


# ========== 融合后端 ==========
@dataclass
class HybridRetrievalBackend:
    router: LLMRouter
    sink: Optional[TelemetrySink] = None

    index_path: str = "data/hotpotqa/docs.jsonl"
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    graph_root: str = "data/graph"
    graph_window: int = 2

    alpha_text: float = 0.4
    alpha_graph: float = 0.2
    alpha_dense: float = 0.4

    bm25_pool_k: int = 150
    default_top_k: int = 20
    qe_lines: int = 3
    qe_attr_paraphrase: bool = True
    embed_batch: int = 50

    def __post_init__(self):
        self.expander = LLMQueryExpander(self.router, self.qe_lines, self.qe_attr_paraphrase)
        self.text = BM25TextSearcher(index=BM25LiteIndex(path=self.index_path, k1=self.bm25_k1, b=self.bm25_b))
        self.graph = GraphNeighborExpander(self.graph_root, self.graph_window)
        self.dense = DenseReranker(self.router, max_pool=self.bm25_pool_k, embed_batch=self.embed_batch)
        logger.info(
            f"[HybridRetrievalBackend] init: bm25_pool_k={self.bm25_pool_k}, "
            f"alpha_text={self.alpha_text}, alpha_graph={self.alpha_graph}, alpha_dense={self.alpha_dense}, "
            f"graph_window={self.graph_window}, embed_batch={self.embed_batch}"
        )

    def _normalize_id(self, hit: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        _id = hit.get("id") or ""
        meta = hit.get("meta") or {}
        doc = meta.get("doc") or meta.get("title")
        sid = meta.get("sent_id") or meta.get("sid")
        if doc is not None and sid is not None:
            norm = f"sent::{doc}::{sid}"
        elif doc is not None:
            norm = f"sent::{doc}::{sid or ''}"
        else:
            norm = _id or "sent::unknown::"
        return norm, meta

    def _minmax_norm(self, values: Dict[str, float]) -> Dict[str, float]:
        if not values: return {}
        vs = list(values.values())
        vmin, vmax = min(vs), max(vs)
        if vmax <= vmin: return {k: 0.0 for k in values.keys()}
        return {k: (v - vmin) / (vmax - vmin) for k, v in values.items()}

    def run(self, req: RetrievalIn) -> Dict[str, Any]:
        trace_id = req.trace_id or "trace-demo"
        top_k = int(getattr(req, "top_k", None) or self.default_top_k)

        # 1) Query 扩展
        with (span("Backend/Expand", self.sink, trace_id) if self.sink else _null_ctx()):
            expanded = self.expander.expand(query=req.query, trace_id=trace_id)
            queries = [req.query] + expanded

        # 2) 文本召回
        with (span("Backend/TextSearch", self.sink, trace_id) if self.sink else _null_ctx()):
            t_hits_raw = self.text.search(queries=queries, top_k=max(top_k, self.bm25_pool_k))
            logger.debug(f"[HybridRetrievalBackend] TextSearch got {len(t_hits_raw)} candidates")

        # 3) 图邻域
        gw_override = getattr(req, "graph_window", None)
        with (span("Backend/GraphExpand", self.sink, trace_id) if self.sink else _null_ctx()):
            g_hits_raw = self.graph.expand(
                query=req.query,
                graph_id=req.graph_id or "",
                top_k=max(top_k, self.bm25_pool_k),
                window_override=gw_override if isinstance(gw_override, int) else None,
            )
            logger.debug(
                f"[HybridRetrievalBackend] GraphExpand got {len(g_hits_raw)} candidates "
                f"(graph_id={req.graph_id}, window={gw_override or self.graph_window})"
            )

        # 4) Dense 语义重排
        with (span("Backend/DenseRerank", self.sink, trace_id) if self.sink else _null_ctx()):
            dense_scores_raw = self.dense.score(query=req.query, candidates=t_hits_raw, trace_id=trace_id)
            logger.debug(f"[HybridRetrievalBackend] DenseRerank scored {len(dense_scores_raw)} candidates")

        def _norm_map(hits: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
            m: Dict[str, Dict[str, Any]] = {}
            for h in hits or []:
                nid, meta = self._normalize_id(h)
                prev = m.get(nid)
                if prev is None or float(h.get("score") or 0.0) > float(prev.get("score") or 0.0):
                    m[nid] = {"id": nid, "score": float(h.get("score") or 0.0), "meta": dict(meta or {})}
                else:
                    prev_meta = prev.get("meta") or {}
                    for k, v in (meta or {}).items():
                        prev_meta.setdefault(k, v)
                    prev["meta"] = prev_meta
            return m

        tmap = _norm_map(t_hits_raw)
        gmap = _norm_map(g_hits_raw)

        norm_text = self._minmax_norm({k: float(v["score"]) for k, v in tmap.items()})
        norm_graph = self._minmax_norm({k: float(v["score"]) for k, v in gmap.items()})
        norm_dense = self._minmax_norm(dense_scores_raw)

        ids = set(tmap.keys()) | set(gmap.keys()) | set(norm_dense.keys())
        fused: List[Dict[str, Any]] = []
        for nid in ids:
            ts = norm_text.get(nid, 0.0)
            gs = norm_graph.get(nid, 0.0)
            ds = norm_dense.get(nid, 0.0)
            score = self.alpha_text * ts + self.alpha_graph * gs + self.alpha_dense * ds

            meta: Dict[str, Any] = {}
            if nid in tmap and isinstance(tmap[nid].get("meta"), dict): meta.update(tmap[nid]["meta"])
            if nid in gmap and isinstance(gmap[nid].get("meta"), dict): meta.update(gmap[nid]["meta"])
            meta["score_text_norm"], meta["score_graph_norm"], meta["score_dense_norm"] = ts, gs, ds
            fused.append({"id": nid, "score": float(score), "meta": meta})

        fused.sort(key=lambda h: h["score"], reverse=True)
        fused = fused[: top_k]

        diagnostics = {
            "queries": queries,
            "bm25_candidates": len(t_hits_raw),
            "graph_candidates": len(g_hits_raw),
            "dense_scored": len(dense_scores_raw),
            "weights": {"alpha_text": self.alpha_text, "alpha_graph": self.alpha_graph, "alpha_dense": self.alpha_dense},
            "pool": {"bm25_pool_k": self.bm25_pool_k, "final_top_k": top_k},
            "graph_window_used": int(gw_override if isinstance(gw_override, int) else self.graph_window),
            "embed_batch": int(self.embed_batch),
            "resolved_embed_model": self.dense._resolve_embed_model(),
        }
        return {"hits": fused, "diagnostics": diagnostics}

    def retrieve(self, req: RetrievalIn) -> RetrievalOut:
        result = self.run(req)
        hits = [Hit(id=h["id"], score=h["score"], meta=h.get("meta")) for h in result["hits"]]
        return RetrievalOut(hits=hits, diagnostics=result.get("diagnostics", {}))


# —— 工具上下文 —— #
class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, a, b, c): return False
def _null_ctx(): return _NullCtx()