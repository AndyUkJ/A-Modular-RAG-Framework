from __future__ import annotations
from typing import List, Dict, Any
import requests

class OllamaProvider:
    def __init__(self, base_url: str = "http://localhost:11434", model_default: str = "llama3.1"):
        self.base_url = base_url.rstrip("/")
        self.model_default = model_default

    def complete(self, prompt: str, *, temperature: float = 0.2, max_tokens: int = 512, **kw) -> Dict[str, Any]:
        url = f"{self.base_url}/api/generate"
        model = kw.get("model", self.model_default)
        try:
            r = requests.post(url, json={
                "model": model,
                "prompt": prompt,
                "options": {"temperature": temperature, "num_predict": max_tokens}
            }, timeout=30)
            r.raise_for_status()
            # In real setup, parse streaming lines; here we just return last chunk
            text = r.text[-4000:]
            return {"text": text, "tokens": len(text)//4}
        except Exception as e:
            return {"text": f"[MOCK Ollama due to error: {e}] {prompt[:200]}", "tokens": len(prompt)//4}

    def embed(self, texts: List[str], **kw) -> Dict[str, Any]:
        dim = kw.get("dim", 8)
        return {"vectors": [[0.1]*dim for _ in texts]}
