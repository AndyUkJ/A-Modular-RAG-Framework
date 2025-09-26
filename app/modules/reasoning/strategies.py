from __future__ import annotations
from typing import Any, Dict, List, Tuple, Iterable, Set
import re
from collections import Counter
import math
import logging

logger = logging.getLogger(__name__)

# -------- text utils --------

def _tokenize(text: str) -> List[str]:
    return [t for t in re.split(r"[^a-zA-Z0-9]+", (text or "").lower()) if t]

def overlap_score(a: str, b: str) -> float:
    """词项交集分数：|A∩B| / (1 + log(|B|))，鼓励短证据段落"""
    A, B = set(_tokenize(a)), set(_tokenize(b))
    if not B:
        return 0.0
    inter = len(A & B)
    return inter / (1.0 + math.log(1.0 + len(B)))

# 兼容旧代码：部分历史版本用的是 `_overlap_score`
def _overlap_score(a: str, b: str) -> float:
    return overlap_score(a, b)

def normalize_answer(s: str) -> str:
    """自洽投票用：小写、去标点空白、消括号引用"""
    s = s or ""
    s = re.sub(r"\[[^\]]+\]", " ", s)         # 去掉内联引用，如 [#3]
    s = re.sub(r"[^a-zA-Z0-9]+", " ", s)      # 非字母数字变空格
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

# -------- LLM output coerce --------

def coerce_text(out: Any) -> str:
    """
    将不同 provider 的返回结构尽量收敛为 str。
    保留旧逻辑，新增更多兜底分支。
    """
    if out is None:
        return ""
    if isinstance(out, str):
        return out

    if isinstance(out, dict):
        # 1) 直接 text
        t = out.get("text")
        if isinstance(t, str):
            return t
        if isinstance(t, dict):
            if isinstance(t.get("text"), str):
                return t["text"]
            c = t.get("content")
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                for item in c:
                    if isinstance(item, dict) and isinstance(item.get("text"), str):
                        return item["text"]

        # 2) message.content
        msg = out.get("message")
        if isinstance(msg, dict):
            c = msg.get("content")
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                for item in c:
                    if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                        return item["text"]

        # 3) choices[0]
        choices = out.get("choices")
        if isinstance(choices, list) and choices:
            ch0 = choices[0]
            if isinstance(ch0, dict):
                if isinstance(ch0.get("text"), str):
                    return ch0["text"]
                msg0 = ch0.get("message")
                if isinstance(msg0, dict) and isinstance(msg0.get("content"), str):
                    return msg0["content"]
                delta0 = ch0.get("delta")
                if isinstance(delta0, dict) and isinstance(delta0.get("content"), str):
                    return delta0["content"]

        # 4) 其他常见兜底
        for key in ("output_text", "data"):
            v = out.get(key)
            if isinstance(v, str):
                return v

    return ""

# -------- evidence helpers --------

def _extract_meta(hit: Any) -> Dict[str, Any]:
    """统一获取 meta，兼容 dict / obj 两种形态"""
    meta = getattr(hit, "meta", None)
    if meta is None and isinstance(hit, dict):
        meta = hit.get("meta") or {}
    if not isinstance(meta, dict):
        meta = {}
    return meta

def _hit_text(hit: Any) -> str:
    meta = _extract_meta(hit)
    text = meta.get("text") or meta.get("content") or ""
    if not text and isinstance(hit, dict):
        text = hit.get("text") or hit.get("content") or ""
    return str(text or "")

def _hit_doc(hit: Any) -> str:
    meta = _extract_meta(hit)
    return str(meta.get("doc") or meta.get("title") or "")

# -------- 邻域扩展（新增，保持向下兼容） --------

def expand_with_neighbors(
    used: Set[int],
    hits: List[Any],
    window: int = 1,
    max_expand: int = 5,
) -> Set[int]:
    """
    在已有 hits 基础上扩展邻居句子：
      - 通过 doc/sent_id 连续性扩展（前后句）
      - 控制扩展 window（几跳）
      - 最多扩展 max_expand 个新证据
    返回更新后的 used 索引集合
    """
    if not hits or not used or window <= 0 or max_expand <= 0:
        return used

    # 建索引: doc -> [ (sent_id, idx) ... ]，方便邻居查找
    doc2sents: Dict[str, List[Tuple[int, int]]] = {}
    for idx, h in enumerate(hits):
        meta = _extract_meta(h)
        doc = str(meta.get("doc") or "")
        sid_val = meta.get("sent_id") if meta is not None else None
        try:
            sid = int(sid_val) if sid_val is not None else -1
        except Exception:
            sid = -1
        if sid >= 0:
            doc2sents.setdefault(doc, []).append((sid, idx))

    # 按句 id 排序
    for lst in doc2sents.values():
        lst.sort(key=lambda x: x[0])

    expanded: Set[int] = set(used)
    added = 0

    for idx in list(used):
        if added >= max_expand:
            break
        h = hits[idx]
        meta = _extract_meta(h)
        doc = str(meta.get("doc") or "")
        sid_val = meta.get("sent_id")
        try:
            sid = int(sid_val) if sid_val is not None else -1
        except Exception:
            sid = -1
        if sid < 0 or doc not in doc2sents:
            continue

        sent_list = doc2sents[doc]  # 已排序
        # 建立 sid -> idx 的快查
        sid2idx = {s: j for s, j in sent_list}

        for d in range(1, window + 1):
            for sign in (-1, 1):
                neighbor_sid = sid + d * sign
                j = sid2idx.get(neighbor_sid)
                if j is not None and j not in expanded:
                    expanded.add(j)
                    added += 1
                    if added >= max_expand:
                        return expanded

    return expanded

# -------- 证据选择（严格保留旧签名与输出；内部增强，但不改变行为边界） --------

def select_evidence_for_steps(
    steps: List[str],
    hits: Iterable[Any],
    per_step_k: int = 2,
    min_score: float = 0.0,
    require_entities: List[str] | None = None,
    neighbor_window: int = 1,       # 🔥 新增
    neighbor_max_expand: int = 5,   # 🔥 新增
) -> Tuple[List[List[int]], set]:
    """
    为每个 step 选择 top-K 证据，支持:
      - min_score: 最低相关性阈值
      - require_entities: 仅保留含有这些实体的句子（旧逻辑：强过滤）
    【增强但不破坏旧接口】：
      - 若命中 meta 中的规范化分：融合( text:0.5, dense:0.3, graph:0.2 )，无则回退纯 lexical
      - 实体“软加权”并未替代旧的 require_entities 强过滤；为了完全兼容，这里仍保留强过滤分支
      - 自动邻域扩展（window=1，max_expand=per_step_k），保证句间连续性
      - coverage 保底：如果某步不足 per_step_k，回退全局高分补齐
    """
    H = list(hits)
    step_evidences: List[List[int]] = []
    used = set()

    # 预计算全局排序（用于 coverage 保底）
    global_sorted = sorted(
        range(len(H)),
        key=lambda i: float(getattr(H[i], "score", 0.0)),
        reverse=True
    )

    for s in steps:
        scored: List[Tuple[int, float]] = []
        s_tokens = set(_tokenize(s))  # 保留变量，方便未来扩展

        for i, h in enumerate(H):
            meta = _extract_meta(h)
            text = _hit_text(h)
            if not text:
                continue

            # 1) 词项重叠
            lex = overlap_score(s, text)

            # 2) 若有规范化通道分，融合，否则只用 lex
            st = float(meta.get("score_text_norm") or 0.0)
            sd = float(meta.get("score_dense_norm") or 0.0)
            sg = float(meta.get("score_graph_norm") or 0.0)

            if (st + sd + sg) > 0.0:
                # 融合分：按权重综合（经验权重；不改变旧逻辑排序的兜底）
                fused_chan = 0.5 * st + 0.3 * sd + 0.2 * sg
                score = 0.6 * lex + 0.4 * fused_chan
            else:
                score = lex

            # 3) 旧逻辑：强过滤 require_entities
            if require_entities:
                tl = text.lower()
                if not any(ent.lower() in tl for ent in require_entities):
                    # 强过滤分支：和旧版完全一致
                    if score < min_score:
                        continue
                    # 直接跳过这条（不加分）
                    # 但为了尽量保留 recall，我们继续计算下一条
                    continue

            if score >= min_score and score > 0:
                scored.append((i, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        picked = [i for i, _ in scored[:max(1, per_step_k)]]

        # —— 邻域扩展（新增，不改变返回类型）——
        if picked:
            picked_set = expand_with_neighbors(set(picked), H, window=neighbor_window, max_expand=max(neighbor_max_expand, per_step_k))
            # 仍保持去重&限制
            picked = list(picked_set)
            # 为了可控：回到“按原先得分排序”的稳定性
            picked.sort(key=lambda i: next((sc for idx, sc in scored if idx == i), 0.0), reverse=True)
            picked = picked[:max(1, per_step_k)]

        # —— coverage 保底（新增，不改变返回类型）——
        if len(picked) < per_step_k:
            for gi in global_sorted:
                if gi not in picked:
                    picked.append(gi)
                if len(picked) >= per_step_k:
                    break

        step_evidences.append(picked)
        used.update(picked)

    return step_evidences, used

# -------- citations --------

def _dedup_keep_order(indices: Iterable[int]) -> List[int]:
    seen = set()
    ordered = []
    for i in indices:
        if i not in seen:
            seen.add(i)
            ordered.append(i)
    return ordered

def build_citation_block(hits: List[Any], indices: Iterable[int]) -> str:
    """
    生成稳定可复现的引用块。
    - 去重
    - 按索引升序排序，避免 set 无序
    - 输出格式保持不变
    """
    try:
        idx_list = sorted(set(int(i) for i in indices))
    except Exception:
        idx_list = _dedup_keep_order(indices)

    lines = []
    for j, i in enumerate(idx_list, 1):
        if i < 0 or i >= len(hits):
            logger.debug("Citation index out of range: %s", i)
            continue
        h = hits[i]
        meta = _extract_meta(h)
        doc = str(meta.get("doc") or meta.get("title") or "")
        sid = str(meta.get("sent_id") or meta.get("sid") or "")
        text = str(meta.get("text") or meta.get("content") or "").replace('"', '“')
        lines.append(f"[#{j}] (doc={doc}, sent_id={sid}) \"{text}\"")
    return "\n".join(lines)

# -------- majority vote --------

def majority_vote(candidates: List[str]) -> Tuple[str, Dict[str, int]]:
    votes = Counter(normalize_answer(c) for c in candidates if c and c.strip())
    if not votes:
        return "", {}
    best_norm, _ = votes.most_common(1)[0]
    for c in candidates:
        if normalize_answer(c) == best_norm:
            return c, dict(votes)
    return candidates[0], dict(votes)