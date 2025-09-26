from typing import List, Tuple, Callable, Optional
import re


def simple_rule_split(text: str) -> List[str]:
    """简单规则分句（按标点）。"""
    return [s.strip() for s in re.split(r"[。！？.!?]", text) if s.strip()]


def segment_context(
    ctx: List[Tuple[str, List[str]]],
    *,
    strategy: str = "rule",
    embed_fn: Optional[Callable[[str], List[float]]] = None,
    sim_threshold: float = 0.65,
) -> List[Tuple[str, List[str]]]:
    """
    对上下文进行语义切分（G1）:
    - 支持 rule: 简单标点分句
    - 支持 embed: 语义相似度切分
    - embed_fn: 可选的 embedding 函数 (text -> vector)，用于语义分割
    """
    out: List[Tuple[str, List[str]]] = []

    for title, sents in ctx:
        new_sents: List[str] = []

        if strategy == "rule":
            # rule 模式：对每个句子进一步做标点切分
            for s in sents:
                new_sents.extend(simple_rule_split(s))
        elif strategy == "embed" and embed_fn:
            # embed 模式：合并相似度高的句子
            batch: List[str] = []
            prev_vec = None
            for s in sents:
                vec = embed_fn(s) if embed_fn else None
                if prev_vec is not None and vec is not None:
                    # 计算余弦相似度
                    import numpy as np
                    va, vb = np.array(prev_vec), np.array(vec)
                    sim = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))
                    if sim < sim_threshold:
                        # 相似度低 → 切断
                        new_sents.append(" ".join(batch))
                        batch = []
                batch.append(s)
                prev_vec = vec
            if batch:
                new_sents.append(" ".join(batch))
        else:
            # 默认直接使用原句子
            new_sents = list(sents)

        out.append((title, new_sents))

    return out