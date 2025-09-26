# app/modules/verification/flow.py
from __future__ import annotations
from typing import Any, Dict, Optional
from app.core.interfaces import VerifierAgent
from app.core.dto import VerifyIn, VerifyOut
from app.core.llm_router import LLMRouter
from app.telemetry.sinks import TelemetrySink, span
from app.di.factory import import_from_string


class VerifierAgentFlow(VerifierAgent):
    """
    Flow 仅作适配层：
      - 从 settings 读取 impl / impl_kwargs（默认用 impl_rules_llm）
      - 实例化真正实现类（impl）
      - 对外暴露稳定接口：verify(req) -> VerifyOut
    """

    def __init__(
        self,
        impl: VerifierAgent,
        *,
        router: Optional[LLMRouter] = None,
        sink: Optional[TelemetrySink] = None,
        **_unused,  # 兼容占位
    ):
        self.impl = impl
        self.router = router
        self.sink = sink

    @classmethod
    def from_settings(
        cls,
        settings: Dict[str, Any],
        router: LLMRouter,
        sink: Optional[TelemetrySink] = None,
    ) -> "VerifierAgentFlow":
        cfg = (settings.get("modules", {}) or {}).get("verification", {}) or {}

        # 兼容两种写法：直接在 verification 下给 impl / impl_kwargs，
        # 或写在 kwargs.impl / kwargs.impl_kwargs 里
        impl_spec = (
            cfg.get("impl")
            or (isinstance(cfg.get("kwargs"), dict) and cfg["kwargs"].get("impl"))
            or "app.modules.verification.impl_rules_llm:VerifierAgentRulesLLM"
        )
        impl_kwargs = (
            cfg.get("impl_kwargs")
            or (isinstance(cfg.get("kwargs"), dict) and cfg["kwargs"].get("impl_kwargs"))
            or {}
        )

        # 反射过滤：仅传实现类 __init__ 接收的参数；若声明了 router/sink 则自动注入
        ImplCls = import_from_string(impl_spec)
        import inspect

        sig = inspect.signature(ImplCls.__init__)
        valid = set(sig.parameters.keys())
        if "router" in valid:
            impl_kwargs.setdefault("router", router)
        if "sink" in valid:
            impl_kwargs.setdefault("sink", sink)
        impl_kwargs = {k: v for k, v in impl_kwargs.items() if k in valid and k != "self"}

        impl = ImplCls(**impl_kwargs)
        return cls(impl=impl, router=router, sink=sink)

    # 对外稳定接口：委托给 impl，带可观测埋点
    def verify(self, req: VerifyIn) -> VerifyOut:
        trace_id = getattr(req, "trace_id", None) or "trace-verify"
        if self.sink:
            with span("VerifierAdapter", self.sink, trace_id):
                return self.impl.verify(req)
        return self.impl.verify(req)