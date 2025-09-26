from __future__ import annotations
from typing import List, Dict, Any, Optional
import os
import logging

logger = logging.getLogger(__name__)


class OpenAIProvider:
    """Provider for OpenAI API. Falls back to mock if SDK not installed or no API key."""

    # def __init__(self, api_key: Optional[str] = None, model_default: str = "gpt-4o-mini"):
    def __init__(self, kwargs: dict):
        # 优先用入参，其次用环境变量
        self.api_key = kwargs.get("api_key") or ""
        if self.api_key == "OPENAI_API_KEY":
            self.api_key = os.getenv(self.api_key, "")
        
        self.model_default = kwargs.get("api_key") or "gpt-4o-mini"
        self.proxy = kwargs.get("proxy") or ""

        try:
            import openai  # noqa: F401
            self._has_openai = True
        except Exception:
            self._has_openai = False
            logger.warning("[OpenAIProvider] openai SDK not available, will mock outputs.")

    # --- 从 settings.yaml 构造（推荐在 factory 调用） ---
    @classmethod
    def from_settings(cls, settings: Dict[str, Any]) -> "OpenAIProvider":
        prov = (settings.get("providers", {}) or {}).get("openai", {}) or {}
        kw = dict(prov.get("kwargs") or prov)

        def _resolve_env(v: Optional[str]) -> Optional[str]:
            if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
                return os.getenv(v[2:-1], "")
            return v

        api_key = _resolve_env(kw.get("api_key"))
        model_default = kw.get("model_default") or "gpt-4o-mini"
        return cls(kw)

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
        **kw
    ) -> Dict[str, Any]:
        model = kw.get("model", self.model_default)
        logger.info(f"[OpenAIProvider] complete() model={model}, len(prompt)={len(prompt)}")

        if self._has_openai and self.api_key:
            try:
                import httpx
                from openai import OpenAI

                logger.debug(f"[OpenAIProvider] - proxy={self.proxy}")
                http_client = None
                if self.proxy:
                    http_client = httpx.Client(
                        transport=httpx.HTTPTransport(proxy=self.proxy),
                        timeout=20.0,
                    )

                if self.proxy:
                    client = OpenAI(api_key=self.api_key, http_client=http_client)
                else:
                    client = OpenAI(api_key=self.api_key)

                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                text = resp.choices[0].message.content or ""
                usage = getattr(resp, "usage", None)
                tokens = getattr(usage, "total_tokens", 0) if usage else 0
                logger.debug(f"[OpenAIProvider] Response tokens={tokens}, text[:80]={text[:80]!r}")
                return {"text": text, "tokens": tokens}

            except Exception as e:
                logger.error(f"[OpenAIProvider] Error calling OpenAI: {e}", exc_info=True)
                return {
                    "text": f"[MOCK OpenAI due to error: {e}] {prompt[:200]}",
                    "tokens": len(prompt) // 4,
                }

        logger.warning("[OpenAIProvider] No API key or SDK missing, returning mock output.")
        return {"text": f"[MOCK OpenAI] {prompt[:200]}", "tokens": len(prompt) // 4}

    def embed(self, texts: List[str], **kw) -> Dict[str, Any]:
        model = kw.get("model", "text-embedding-3-small")
        logger.info(f"[OpenAIProvider] embed() model={model}, texts={len(texts)}")

        if self._has_openai and self.api_key:
            try:
                import httpx
                from openai import OpenAI

                # 正确初始化 OpenAI 客户端
                http_client = None
                if self.proxy:
                    http_client = httpx.Client(
                        transport=httpx.HTTPTransport(proxy=self.proxy),
                        timeout=20.0,
                    )

                if self.proxy:
                    client = OpenAI(api_key=self.api_key, http_client=http_client)
                else:
                    client = OpenAI(api_key=self.api_key)

                # 调用 embedding API
                resp = client.embeddings.create(model=model, input=texts)

                # 提取向量
                vectors = [d.embedding for d in resp.data]
                logger.debug(f"[OpenAIProvider] Got embeddings shape=({len(vectors)}, {len(vectors[0])})")
                return {"vectors": vectors}

            except Exception as e:
                logger.error(f"[OpenAIProvider] Error calling embeddings: {e}", exc_info=True)
                dim = kw.get("dim", 8)
                return {"vectors": [[0.0] * dim for _ in texts]}

        # 没有 openai 库或 API key → 兜底零向量
        logger.warning("[OpenAIProvider] Embedding fallback to zeros.")
        dim = kw.get("dim", 8)
        return {"vectors": [[0.0] * dim for _ in texts]}
