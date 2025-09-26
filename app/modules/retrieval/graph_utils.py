# app/modules/retrieval/graph_utils.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set
import re


def _tokenize(text: str) -> List[str]:
    return [t for t in re.split(r"[^a-zA-Z0-9]+", (text or "").lower()) if t]


def load_graph_json(graph_root: str, graph_id: str) -> Dict[str, Any]:
    """
    读取 GraphConstructionNetworkX 落盘的 graph.json：
    data/graph/<graph_id>/graph.json
    """
    p = Path(graph_root) / graph_id / "graph.json"
    if not p.exists():
        return {"nodes": [], "edges": []}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_index(graph: Dict[str, Any]):
    """
    构建若干便于检索的数据结构：
      - nodes_by_id
      - next_forward / next_backward（基于 next_in_doc 边）
      - node_texts（sentence 节点文本）
      - q_to_sent（q_match 指向的句子列表）
    """
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    nodes_by_id: Dict[str, Dict[str, Any]] = {n["id"]: n for n in nodes}
    next_forward: Dict[str, List[str]] = {}
    next_backward: Dict[str, List[str]] = {}
    q_to_sent: List[str] = []
    node_texts: Dict[str, str] = {}

    for n in nodes:
        if n.get("type") == "sentence":
            node_texts[n["id"]] = (n.get("props", {}) or {}).get("text", "")

    for e in edges:
        et = e.get("type")
        s, t = e.get("source"), e.get("target")
        if et == "next_in_doc":
            next_forward.setdefault(s, []).append(t)
            next_backward.setdefault(t, []).append(s)
        elif et == "q_match" and s == "q1":
            q_to_sent.append(t)

    return nodes_by_id, next_forward, next_backward, node_texts, q_to_sent


def expand_qmatch_neighbors(q_text: str,
                            nodes_by_id: Dict[str, Dict[str, Any]],
                            next_forward: Dict[str, List[str]],
                            next_backward: Dict[str, List[str]],
                            node_texts: Dict[str, str],
                            explicit_qmatch: List[str] | None = None,
                            window: int = 1) -> Dict[str, Tuple[float, Dict[str, Any]]]:
    """
    基于 q_match 的句子作为种子，做 +/- window 的邻域扩展，返回：
      sent_id -> (graph_score, meta)

    改进版：
    - 使用 BFS，保证 hop<=window 的邻居不会漏掉。
    - 分数衰减：d=0→1.0, d=1→0.7, d=2→0.5, d>=3→max(0.5-0.1*(d-2),0.1)。
    """
    q_terms = set(_tokenize(q_text))
    seeds: Set[str] = set(explicit_qmatch or [])

    if not seeds:
        # 回退：词交集匹配
        for sid, text in node_texts.items():
            if not text:
                continue
            s_terms = set(_tokenize(text))
            if q_terms & s_terms:
                seeds.add(sid)

    results: Dict[str, Tuple[float, Dict[str, Any]]] = {}

    def _decay(d: int) -> float:
        if d == 0:
            return 1.0
        if d == 1:
            return 0.7
        if d == 2:
            return 0.5
        return max(0.5 - 0.1 * (d - 2), 0.1)

    def _add(sid: str, score: float, dist: int):
        meta = {
            "kind": "sentence",
            "text": node_texts.get(sid, ""),
            "distance": dist,
            "doc": (nodes_by_id.get(sid, {}).get("props") or {}).get("doc"),
        }
        if sid in results:
            if score > results[sid][0]:
                results[sid] = (score, meta)
        else:
            results[sid] = (score, meta)

    # BFS 队列
    from collections import deque
    queue = deque([(sid, 0) for sid in seeds])
    visited: Set[str] = set(seeds)

    while queue:
        sid, dist = queue.popleft()
        score = _decay(dist)
        _add(sid, score, dist)

        if dist >= window:
            continue

        # 前后邻居
        neighs = next_forward.get(sid, []) + next_backward.get(sid, [])
        for nb in neighs:
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, dist + 1))

    return results
