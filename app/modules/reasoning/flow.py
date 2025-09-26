from __future__ import annotations
from typing import Any, Dict, Optional
import inspect

from app.core.interfaces import ReasoningAgent
from app.core.dto import ReasoningIn, ReasoningOut
from app.core.llm_router import LLMRouter
from app.telemetry.sinks import TelemetrySink, span
from app.di.factory import import_from_string


class ReasoningAgentFlow(ReasoningAgent):
    """
    Flow 仅作适配层：
      - 从 settings 读取 impl/impl_kwargs（默认使用 impl_planner_synth）
      - 负责实例化真实实现类（impl）
      - 对外暴露稳定接口：reason(req) -> ReasoningOut
    """

    def __init__(
        self,
        impl: ReasoningAgent,
        *,
        router: Optional[LLMRouter] = None,
        sink: Optional[TelemetrySink] = None,
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
    ) -> "ReasoningAgentFlow":
        cfg = (settings.get("modules", {}) or {}).get("reasoning", {}) or {}

        impl_spec = (
            cfg.get("impl")
            or (isinstance(cfg.get("kwargs"), dict) and cfg["kwargs"].get("impl"))
            or "app.modules.reasoning.impl_planner_synth:ReasoningAgentPlannerSynth"
        )
        impl_kwargs = (
            cfg.get("impl_kwargs")
            or (isinstance(cfg.get("kwargs"), dict) and cfg["kwargs"].get("impl_kwargs"))
            or {}
        )

        # 允许通过 settings 透传 coverage/refine 相关参数（若实现类支持）
        ImplCls = import_from_string(impl_spec)
        sig = inspect.signature(ImplCls.__init__)
        valid_params = set(sig.parameters.keys())

        # 注入 router/sink
        if "router" in valid_params:
            impl_kwargs.setdefault("router", router)
        if "sink" in valid_params:
            impl_kwargs.setdefault("sink", sink)

        # 过滤掉实现类未接受的参数，避免 TypeError
        impl_kwargs = {k: v for k, v in impl_kwargs.items() if k in valid_params and k != "self"}

        impl = ImplCls(**impl_kwargs)
        return cls(impl=impl, router=router, sink=sink)

    def reason(self, req: ReasoningIn) -> ReasoningOut:
        trace_id = getattr(req, "trace_id", None) or "trace-reason"
        if self.sink:
            with span("ReasoningAdapter", self.sink, trace_id):
                return self.impl.reason(req)
        return self.impl.reason(req)