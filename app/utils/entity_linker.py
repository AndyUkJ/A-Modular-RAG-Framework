from typing import List, Dict, Callable, Optional
import os
import re
import json

try:
    import requests  # 可选依赖；不存在也不影响 mock 模式
except Exception:
    requests = None


def _simple_ner(text: str) -> List[str]:
    """极轻量 NER：抽人名/专名，供真实 ELQ 回调或 HTTP 端点使用。"""
    return re.findall(r"[A-Z][a-z]+(?: [A-Z][a-z]+)*", text or "")


def _mock_elq(text: str) -> List[Dict[str, str]]:
    """默认 mock：稳定可用，并保证最小键 id/text 存在。"""
    return [
        {"id": "Q1", "text": "Barack Obama",
         "qid": "Q1", "mention": "Barack Obama", "canonical": "Barack Obama",
         "score": 0.95, "source": "mock"},
        {"id": "Q2", "text": "United States",
         "qid": "Q2", "mention": "United States", "canonical": "United States of America",
         "score": 0.92, "source": "mock"},
    ]


def elq_link_entities(
    text: str,
    *,
    use_real_elq: bool = False,
    max_entities: int = 8,
    provider: Optional[Callable[[List[str]], List[Dict[str, str]]]] = None,
) -> List[Dict[str, str]]:
    """
    统一实体链接接口（向后兼容）：
    - 返回的每条记录“至少”包含: id, text
    - 允许附加: qid, mention, canonical, score, source
    - use_real_elq=True 时：
        1) 如传入 provider 回调，优先用回调(mentions -> entity dicts)
        2) 否则若设置 ELQ_ENDPOINT 环境变量且安装了 requests，则POST到该端点
           期望返回形如 [{"id":"Qxxx","text":"xxx","score":0.9,...}, ...]
        3) 若上述失败，回退 mock
    """
    if not text:
        return []

    # --- mock 模式（默认） ---
    if not use_real_elq:
        return _mock_elq(text)[:max_entities]

    mentions = _simple_ner(text)[:max_entities] or []

    # --- 优先使用回调 provider ---
    if provider is not None:
        try:
            out = provider(mentions) or []
            fixed = []
            for e in out[:max_entities]:
                # 确保最小键存在
                eid = e.get("id") or e.get("qid") or f"ELQ::{e.get('canonical') or e.get('mention') or 'unknown'}"
                t = e.get("text") or e.get("canonical") or e.get("mention") or eid
                fixed.append({
                    "id": eid, "text": t,
                    **{k: v for k, v in e.items() if k not in {"id", "text"}}
                })
            return fixed
        except Exception:
            pass  # 回退下一方案

    # --- HTTP 端点（ELQ_ENDPOINT） ---
    endpoint = os.environ.get("ELQ_ENDPOINT")
    if endpoint and requests:
        try:
            payload = {"mentions": mentions, "text": text}
            resp = requests.post(endpoint, json=payload, timeout=10)
            resp.raise_for_status()
            arr = resp.json()
            fixed = []
            for e in (arr or [])[:max_entities]:
                eid = e.get("id") or e.get("qid") or f"ELQ::{e.get('canonical') or e.get('mention') or 'unknown'}"
                t = e.get("text") or e.get("canonical") or e.get("mention") or eid
                fixed.append({
                    "id": eid, "text": t,
                    **{k: v for k, v in e.items() if k not in {"id", "text"}}
                })
            if fixed:
                return fixed
        except Exception:
            pass  # 回退 mock

    # --- 兜底回退 ---
    return _mock_elq(text)[:max_entities]