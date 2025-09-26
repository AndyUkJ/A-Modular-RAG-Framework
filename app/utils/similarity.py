from difflib import SequenceMatcher
from typing import List, Callable, Optional, Tuple
import math

def compute_similarity_score(a: str, b: str) -> float:
    """
    与原版一致：字符级相似度（回退策略）
    """
    return SequenceMatcher(None, a or "", b or "").ratio()

def cosine(u: List[float], v: List[float]) -> float:
    if not u or not v: return 0.0
    du = math.sqrt(sum(x*x for x in u))
    dv = math.sqrt(sum(x*x for x in v))
    if du == 0 or dv == 0: return 0.0
    return sum(x*y for x, y in zip(u, v)) / (du * dv)

def embed_sim(
    a: str, b: str,
    embed: Optional[Callable[[str], List[float]]] = None,
    va: Optional[List[float]] = None,
    vb: Optional[List[float]] = None
) -> float:
    """
    嵌入相似度：优先使用传入向量；否则调用 embed()；两者都没有则回退 difflib
    """
    if va is None and embed: va = embed(a or "")
    if vb is None and embed: vb = embed(b or "")
    if va is None or vb is None:
        return compute_similarity_score(a, b)
    return cosine(va, vb)

def mmr_diversify(
    items: List[Tuple[str, float, Optional[List[float]]]],
    *, top_k: int = 20, lambda_weight: float = 0.7
) -> List[Tuple[str, float, Optional[List[float]]]]:
    """
    MMR 多样化工具（可用于检索/构图后处理）
    items: [(id, score, embed_vec)]
    """
    selected = []
    candidates = items[:]
    while candidates and len(selected) < top_k:
        best, best_val = None, -1e9
        for cid, cscore, cvec in candidates:
            if not selected:
                val = cscore
            else:
                sim_max = 0.0
                for sid, sscore, svec in selected:
                    if cvec is not None and svec is not None:
                        sim = cosine(cvec, svec)
                    else:
                        sim = 0.0
                    sim_max = max(sim_max, sim)
                val = lambda_weight * cscore - (1 - lambda_weight) * sim_max
            if val > best_val:
                best_val = val
                best = (cid, cscore, cvec)
        selected.append(best)
        candidates = [x for x in candidates if x[0] != best[0]]
    return selected