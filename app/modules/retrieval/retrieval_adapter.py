# app/modules/retrieval/retrieval_adapter.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from app.core.interfaces import RetrievalAgent
from app.core.dto import RetrievalIn, RetrievalOut, Hit
from app.core.llm_router import LLMRouter
from app.telemetry.sinks import TelemetrySink, span
from app.di.factory import import_from_string


@dataclass
class RetrievalAdapter(RetrievalAgent):
    """
    适配层：对 orchestrator 暴露统一接口（RetrievalAgent），
    内部委托给“真正实现（backend）”，并做格式归一化（Raw -> Hit）。
    """
    router: LLMRouter
    sink: Optional[TelemetrySink] = None

    # —— 后端装配参数（可在 settings 中覆盖） ——
    backend_impl: str = "app.modules.retrieval.retrieval_backend:HybridRetrievalBackend"
    backend_kwargs: Dict[str, Any] = None

    # —— 原始结果字段映射（兼容不同实现的字段命名差异） ——
    id_keys: List[str] = None
    score_keys: List[str] = None
    meta_key: Optional[str] = "meta"

    def __post_init__(self):
        self.backend_kwargs = self.backend_kwargs or {}
        self.id_keys = self.id_keys or ["id", "doc_id", "docId", "sid", "sent_id"]
        self.score_keys = self.score_keys or ["score", "relevance", "sim", "s"]

        # 实例化 backend；若支持注入 router/sink 就传入
        BackendCls = import_from_string(self.backend_impl)
        try:
            self.backend = BackendCls(router=self.router, sink=self.sink, **self.backend_kwargs)
        except TypeError:
            # 后端不需要 router/sink 的兼容分支
            self.backend = BackendCls(**self.backend_kwargs)

    @classmethod
    def from_settings(cls, settings: Dict[str, Any], router: LLMRouter) -> "RetrievalAdapter":
        """
        读取 modules.retrieval.kwargs.backend.* 配置（impl/kwargs），并构造适配层。
        """
        cfg = (settings.get("modules", {}) or {}).get("retrieval", {})
        kwargs: Dict[str, Any] = {}
        if isinstance(cfg, dict):
            # 读取 adapter 自身参数
            ak = (cfg.get("kwargs") or {}).copy()
            # 提取 backend 配置
            bspec = ak.pop("backend", {})
            backend_impl = bspec.get("impl") or "app.modules.retrieval.retrieval_backend:HybridRetrievalBackend"
            backend_kwargs = bspec.get("kwargs") or {}
            # 汇总
            kwargs.update(ak)
            kwargs["backend_impl"] = backend_impl
            kwargs["backend_kwargs"] = backend_kwargs
        return cls(router=router, **kwargs)

    # ---------- 字段归一化 ----------
    def _pick_first(self, obj: Dict[str, Any], keys: List[str]) -> Any:
        for k in keys:
            if k in obj and obj[k] is not None:
                return obj[k]
        return None

    def _normalize_hit(self, raw: Any) -> Optional[Hit]:
        """
        接受任意后端的“原始命中”，尽量归一化为 Hit。
        支持 dict 或带属性的对象；未知字段会放进 meta。
        """
        if raw is None:
            return None

        # 统一转字典视图
        if isinstance(raw, dict):
            d = dict(raw)
        else:
            d = {k: getattr(raw, k) for k in dir(raw) if not k.startswith("_") and hasattr(raw, k)}

        # 取 id / score
        _id = self._pick_first(d, self.id_keys)
        score = self._pick_first(d, self.score_keys)
        try:
            score = float(score) if score is not None else 0.0
        except Exception:
            score = 0.0

        # meta：优先取 meta_key；否则把其余字段塞进 meta
        meta = {}
        if self.meta_key and self.meta_key in d and isinstance(d[self.meta_key], dict):
            meta = dict(d[self.meta_key])
        else:
            for k, v in d.items():
                if k in self.id_keys or k in self.score_keys:
                    continue
                meta[k] = v

        if not _id:
            # 兜底构造稳定 id（尽量使用 doc/sent 信息）
            doc = meta.get("doc") or meta.get("title") or "doc"
            sid = meta.get("sent_id") or meta.get("sid") or ""
            _id = f"sent::{doc}::{sid}"

        return Hit(id=str(_id), score=score, meta=meta)

    # ---------- 对外接口 ----------
    def retrieve(self, req: RetrievalIn) -> RetrievalOut:
        trace_id = req.trace_id or "trace-demo"
        with (span("RetrievalAdapter", self.sink, trace_id) if self.sink else _nullctx()):
            # 后端输出允许是：
            # { "hits": [raw_hit, ...], "diagnostics": {...} }
            out = self.backend.run(req)  # 由后端完成真正的检索流程
            raw_hits = out.get("hits", []) if isinstance(out, dict) else []
            diagnostics = out.get("diagnostics", {}) if isinstance(out, dict) else {}

            # 统一归一化
            hits: List[Hit] = []
            for r in raw_hits:
                h = self._normalize_hit(r)
                if h:
                    hits.append(h)

            # 排序与截断（防止后端未处理）
            hits = sorted(hits, key=lambda x: x.score, reverse=True)
            if req.top_k:
                hits = hits[: req.top_k]

            return RetrievalOut(hits=hits, diagnostics=diagnostics)


# 兼容：无 sink 时的空上下文
class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): return False
def _nullctx(): return _NullCtx()