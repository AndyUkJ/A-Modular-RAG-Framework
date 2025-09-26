from __future__ import annotations
from typing import Dict, Any, List, Optional
import time
import logging
import traceback

from app.core.providers.base import LLMProvider
from app.telemetry.sinks import TelemetrySink, record_llm_call

logger = logging.getLogger(__name__)


class LLMRouteDecision(dict):
    @property
    def model(self): return self.get("model")
    @property
    def provider(self): return self.get("provider")
    @property
    def reason(self): return self.get("reason")


class LLMRouter:
    def __init__(self, providers: Dict[str, LLMProvider], policy: Dict[str, Any], sink: Optional[TelemetrySink] = None):
        self.providers = providers
        self.policy = policy or {}
        self.sink = sink

    def select(self, module: str, purpose: str, require: Dict[str, Any] | None = None) -> LLMRouteDecision:
        routes = ((self.policy or {}).get("routes") or {}).get(module, {}) or {}
        cands: List[Dict[str, Any]] = []
        if purpose in routes:
            cands = routes[purpose] or []
        if not cands:
            cands = ((self.policy or {}).get("default") or [])
        if not cands:
            logger.warning(f"[Router] [NO_POLICY → MOCK] {module}/{purpose}")
            return LLMRouteDecision(model="mock", provider="mock", reason="no_policy")

        dec = LLMRouteDecision(**cands[0], reason=f"policy:{module}/{purpose}")
        logger.info(f"[Router] Route selected for {module}/{purpose}: provider={dec.provider}, model={dec.model}, reason={dec.reason}")
        return dec

    def complete(self, *, module: str, purpose: str, prompt: str, require: Dict[str, Any] | None = None) -> Dict[str, Any]:
        req = require or {}
        dec = self.select(module, purpose, req)
        provider_name = dec.provider
        model_name = dec.model
        provider = self.providers.get(provider_name)
        t0 = time.time()

        text, req_id, err, fb_reason = "", None, None, None
        try:
            if provider and provider_name != "mock":
                logger.info(f"[LLMRouter] → Calling provider={provider_name}, model={model_name}, module={module}, purpose={purpose}")
                logger.debug(f"[LLMRouter] Prompt preview: {prompt[:200]!r}")
                text = provider.complete(model=model_name, prompt=prompt, require=req)
            elif provider_name == "mock":
                fb_reason = "no_policy"
                text = f"[MOCK Router] {module}/{purpose}: {prompt[:100]}"
                logger.warning(f"[LLMRouter] [NO_POLICY → MOCK] {module}/{purpose}")
            else:
                fb_reason = "no_provider"
                text = f"[MOCK Router] Provider '{provider_name}' not found"
                logger.error(f"[LLMRouter] [NO_PROVIDER → MOCK] {module}/{purpose}")
        except Exception as e:
            err = repr(e)
            fb_reason = "error"
            stack = traceback.format_exc()
            logger.error(f"[LLMRouter] [ERROR → MOCK] provider={provider_name}, model={model_name}, err={err}\n{stack}")
            text = f"[MOCK Router Error] {err}"

        t1 = time.time()
        latency_ms = (t1 - t0) * 1000.0

        # Telemetry
        trace_id = req.get("trace_id") or ""
        if self.sink and trace_id:
            record_llm_call(self.sink, trace_id, {
                "provider": provider_name or "mock",
                "model": model_name or "mock",
                "tokens_in": req.get("tokens_in", 0),
                "tokens_out": len(text) // 4,
                "latency_ms": latency_ms,
                "cached": False,
                "temperature": req.get("temperature"),
                "max_tokens": req.get("max_tokens"),
                "module": module,
                "purpose": purpose,
                "error": err,
                "request_id": req_id,
            })

        return {
            "text": text,
            "_provider": provider_name,
            "_model": model_name,
            "_route_reason": dec.reason,
            "_latency_ms": latency_ms,
            "_error": err,
            "_fallback_reason": fb_reason,
        }

    def embed(self, *, model_hint: str, texts: List[str], require: Dict[str, Any] | None = None) -> List[List[float]]:
        provider_name = (self.policy or {}).get("embedding_provider") or "mock"
        provider = self.providers.get(provider_name)

        logger.info(f"[Router] Provider resolved = {provider_name}, available keys={list(self.providers.keys())}")

        t0 = time.time()
        vecs: List[List[float]] = []
        err, fb_reason = None, None
        try:
            if provider and provider_name != "mock":
                logger.info(f"[LLMRouter] → Embedding provider={provider_name}, model={model_hint}, texts={len(texts)}")
                vecs = provider.embed(model=model_hint, texts=texts, require=require or {})
            elif provider_name == "mock":
                fb_reason = "no_policy"
                logger.warning(f"[LLMRouter] [NO_POLICY → MOCK EMBED] using zeros, texts={len(texts)}")
                vecs = [[0.0] * 3 for _ in texts]
            else:
                fb_reason = "no_provider"
                logger.error(f"[LLMRouter] [NO_PROVIDER → MOCK EMBED] {provider_name}")
                vecs = [[0.0] * 3 for _ in texts]
        except Exception as e:
            err = repr(e)
            fb_reason = "error"
            stack = traceback.format_exc()
            logger.error(f"[LLMRouter] [ERROR → MOCK EMBED] provider={provider_name}, model={model_hint}, err={err}\n{stack}")
            vecs = [[0.0] * 3 for _ in texts]
        t1 = time.time()

        trace_id = (require or {}).get("trace_id") or ""
        latency_ms = (t1 - t0) * 1000.0
        if self.sink and trace_id:
            record_llm_call(self.sink, trace_id, {
                "provider": provider_name or "mock",
                "model": model_hint or "mock",
                "tokens_in": 0,
                "tokens_out": 0,
                "latency_ms": latency_ms,
                "cached": False,
                "module": "Embedding",
                "purpose": "embed",
                "error": err,
            })
        return vecs