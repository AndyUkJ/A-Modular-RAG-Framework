# app/di/factory.py
from __future__ import annotations
from typing import Dict, Any, Tuple
import importlib, os

from app.core.llm_router import LLMRouter
from app.telemetry.sinks import TelemetrySink

import logging
logger = logging.getLogger(__name__)

def import_from_string(path: str):
    """支持 'pkg.mod:Class' 格式的动态导入"""
    mod, cls = path.split(":")
    m = importlib.import_module(mod)
    return getattr(m, cls)


def load_settings(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_env(v):
    """解析 "${ENV_VAR}" 为环境变量值"""
    if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
        return os.getenv(v[2:-1], "")
    return v


def build_providers(settings: Dict[str, Any]):
    """构建 provider 实例"""
    providers_cfg = settings.get("providers", {}) or {}
    providers = {}
    for name, cfg in providers_cfg.items():
        if isinstance(cfg, str):
            type_spec = cfg
            kwargs = {}
        else:
            type_spec = cfg.get("type")
            kwargs = (cfg.get("kwargs") or {}).copy()
        if not type_spec:
            continue

        kwargs = {k: _resolve_env(v) for k, v in kwargs.items()}
        Cls = import_from_string(type_spec)

        if hasattr(Cls, "from_settings"):
            try:
                inst = Cls.from_settings(settings)
                providers[name] = inst
                continue
            except Exception:
                pass

        providers[name] = Cls(**kwargs)
    return providers


def build_router(settings: Dict[str, Any], providers: Dict[str, Any], sink: TelemetrySink | None = None) -> LLMRouter:
    policy = settings.get("llm_policy", {}) or {}
    if not policy:
        logger.warning("[Factory] No llm_policy found in settings.")
    else:
        logger.info(f"[Factory] Loaded llm_policy with keys: {list(policy.keys())}")
        if "routes" in policy:
            logger.info(f"[Factory] Available routes: {list(policy['routes'].keys())}")
    return LLMRouter(providers=providers, policy=policy, sink=sink)


def _parse_module_spec(modules_cfg: Dict[str, Any], key: str, default_spec: str) -> Tuple[str, Dict[str, Any]]:
    """
    解析模块配置，支持三种写法：
      1) 字符串：
         modules.reasoning: "pkg.mod:Class"
      2) 简单字典：
         modules.reasoning: { impl: "pkg.mod:Class", kwargs: {...} }
      3) 完整字典（推荐）：
         modules.reasoning:
           type: "pkg.flow:FlowClass"
           kwargs: {...}         # 给 flow
           impl: "pkg.impl:ImplClass"
           impl_kwargs: {...}    # 给 impl
    返回: (flow_spec, flow_kwargs)
    """
    raw = (modules_cfg or {}).get(key)

    if isinstance(raw, str):
        return raw, {}

    if isinstance(raw, dict):
        # flow 层类（默认用 type，否则回退到 impl，否则 default_spec）
        spec = raw.get("type") or raw.get("impl") or default_spec
        kwargs = dict(raw.get("kwargs") or {})

        # 如果有 impl，就把 impl/impl_kwargs 注入到 flow.kwargs
        impl_spec = raw.get("impl")
        impl_kwargs = raw.get("impl_kwargs") or {}
        if impl_spec:
            kwargs["impl"] = impl_spec
            kwargs["impl_kwargs"] = impl_kwargs

        return spec, kwargs

    return default_spec, {}


def _instantiate(spec: str, kwargs: Dict[str, Any], settings: Dict[str, Any], router: LLMRouter, sink: TelemetrySink | None):
    """统一实例化逻辑：优先用 from_settings，否则直接构造"""
    Cls = import_from_string(spec)
    if hasattr(Cls, "from_settings"):
        return Cls.from_settings(settings, router=router, sink=sink)
    try:
        return Cls(router=router, sink=sink, **kwargs)
    except TypeError:
        try:
            return Cls(router=router, **kwargs)
        except TypeError:
            return Cls(**kwargs)


def build_modules(settings: Dict[str, Any], router: LLMRouter, sink: TelemetrySink | None = None):
    """构建四大模块（graph_construction / retrieval / reasoning / verification）"""
    modules_cfg = settings.get("modules", {}) or {}

    # Graph Construction
    gc_spec, gc_kwargs = _parse_module_spec(
        modules_cfg, "graph_construction", "app.modules.graph_construction.flow:GraphConstructionFlow"
    )
    graph_c = _instantiate(gc_spec, gc_kwargs, settings, router, sink)

    # Retrieval
    r_spec, r_kwargs = _parse_module_spec(
        modules_cfg, "retrieval", "app.modules.retrieval.flow:RetrievalAgentFlow"
    )
    retriever = _instantiate(r_spec, r_kwargs, settings, router, sink)

    # Reasoning
    s_spec, s_kwargs = _parse_module_spec(
        modules_cfg, "reasoning", "app.modules.reasoning.flow:ReasoningAgentFlow"
    )
    reasoner = _instantiate(s_spec, s_kwargs, settings, router, sink)

    # Verification
    v_spec, v_kwargs = _parse_module_spec(
        modules_cfg, "verification", "app.modules.verification.flow:VerifierAgentFlow"
    )
    verifier = _instantiate(v_spec, v_kwargs, settings, router, sink)

    from app.orchestrator.nodes import NodeContext
    return NodeContext(graph_c=graph_c, retriever=retriever, reasoner=reasoner, verifier=verifier, sink=sink)