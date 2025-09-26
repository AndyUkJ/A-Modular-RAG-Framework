from typing import List, Dict, Any, Optional, Callable
from collections import defaultdict
import itertools
import numpy as np
import re

from app.core.dto import GraphNode, GraphEdge, EdgeEvidence


class EdgeBuilder:
    """
    Edge 构建器（G2 + G3 + G4 默认启用，可通过 settings.yaml 调整）
    - 保留原有 5 类边：next_in_doc / in_doc / q_match / semantic_sim / mentions
    - G4: 多通道投票融合 + 稀疏化 (edge_min_vote, max_edges_per_node)
    - 支持 record_evidence (默认启用)，用于可解释性
    - 输出 last_diagnostics，供 impl_networkx 合并
    """

    def __init__(self,
                 use_adjacency: bool = True,
                 use_qmatch: bool = True,
                 use_doc_edges: bool = True,
                 use_entity_edges: bool = True,         # 默认开启实体边 (G3)
                 use_semantic_edges: bool = True,       # 默认开启语义边 (G2)
                 semantic_threshold: float = 0.9,       # 与原版一致
                 embed_fn: Optional[Callable[[str], List[float]]] = None,
                 record_evidence: bool = True,          # 默认开启 (G4)
                 assembly_policy: Optional[Dict[str, Any]] = None):
        self.use_adjacency = use_adjacency
        self.use_qmatch = use_qmatch
        self.use_doc_edges = use_doc_edges
        self.use_entity_edges = use_entity_edges
        self.use_semantic_edges = use_semantic_edges
        self.semantic_threshold = semantic_threshold
        self.embed_fn = embed_fn
        self.record_evidence = record_evidence
        # 默认投票融合策略
        self.assembly_policy = assembly_policy or {
            "channels": {"q_overlap": 1.0, "embed_sim": 1.0, "entity_link": 0.6, "position_prior": 0.2},
            "edge_min_vote": 0.6,
            "max_edges_per_node": 64
        }
        self.last_diagnostics: Dict[str, Any] = {}

    # ---------- 工具 ----------

    def _fake_embed(self, text: str) -> np.ndarray:
        return np.array([hash(text) % 1000], dtype=float) / 1000.0

    def _position_prior(self, a_meta: Dict[str, Any], b_meta: Dict[str, Any]) -> float:
        """句子相邻时给弱先验分数"""
        try:
            a_doc = a_meta.get("doc"); b_doc = b_meta.get("doc")
            a_sid = int(a_meta.get("sent_id", "-1")); b_sid = int(b_meta.get("sent_id", "-1"))
            if a_doc and b_doc and a_doc == b_doc and abs(a_sid - b_sid) == 1:
                return 0.8
        except Exception:
            pass
        return 0.0

    def _vote(self, evidences: List[EdgeEvidence]) -> float:
        """按通道权重聚合得分，限制在 [0,1]"""
        ch_w = (self.assembly_policy or {}).get("channels", {}) or {}
        score = 0.0
        for ev in evidences:
            score += float(ch_w.get(ev.channel, 0.0)) * float(ev.score)
        return max(0.0, min(1.0, score))

    def _add_edge(self, bag: List[GraphEdge], src: str, tgt: str, etype: str,
                  *, base_weight: float, evidence: Optional[List[EdgeEvidence]] = None,
                  meta: Optional[Dict[str, Any]] = None, use_vote: bool = True):
        """新增边，支持投票融合和 evidence"""
        weight = float(base_weight)
        ev = evidence or []
        if use_vote and ev:
            weight = self._vote(ev)
        ge = GraphEdge(source=src, target=tgt, type=etype,
                       weight=round(weight, 3), meta=meta or {})
        if self.record_evidence and ev:
            ge.evidence = ev
        bag.append(ge)

    # ---------- 主流程 ----------

    def build(self, nodes: List[Dict[str, Any]], question: str, policy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        保持原接口：
        - 第二参名为 question
        - 返回 dict 列表
        """
        edges: List[GraphEdge] = []
        gnodes = [n if isinstance(n, GraphNode) else GraphNode(**n) for n in nodes]

        q_node = next((n for n in gnodes if n.type == "question"), None)
        sent_nodes = [n for n in gnodes if n.type == "sentence"]
        doc_nodes = [n for n in gnodes if n.type == "document"]
        ent_nodes = [n for n in gnodes if n.type == "entity"]

        disabled = set(policy.get("disable_edges", [])) if isinstance(policy, dict) else set()

        ap = self.assembly_policy or {}
        use_fusion = True   # 一步到位默认启用
        min_vote = float(ap.get("edge_min_vote", 0.0))
        max_edges_per_node = int(ap.get("max_edges_per_node", 0))

        # 1) 文档内相邻（next_in_doc）
        if self.use_adjacency and "next_in_doc" not in disabled:
            doc_map: Dict[str, List[GraphNode]] = defaultdict(list)
            for s in sent_nodes:
                doc_map[s.meta.get("doc", "default")].append(s)
            for doc, sents in doc_map.items():
                sents_sorted = sorted(sents, key=lambda x: int(x.meta.get("sent_id", 0)))
                for i in range(len(sents_sorted) - 1):
                    a, b = sents_sorted[i], sents_sorted[i + 1]
                    ev = []
                    prior = self._position_prior(a.meta, b.meta)
                    if prior > 0:
                        ev.append(EdgeEvidence(channel="position_prior", score=prior, meta={"reason": "adjacent"}))
                    self._add_edge(edges, a.id, b.id, "next_in_doc",
                                   base_weight=1.0, evidence=ev, meta={"doc": doc}, use_vote=use_fusion)

        # 2) 句子归属文档（in_doc）
        if self.use_doc_edges and "in_doc" not in disabled:
            doc_ids = {d.id for d in doc_nodes}
            for s in sent_nodes:
                doc = s.meta.get("doc")
                doc_id = f"doc::{doc}"
                if doc_id in doc_ids:
                    ev = [EdgeEvidence(channel="position_prior", score=0.4, meta={"reason": "in_doc"})]
                    self._add_edge(edges, s.id, doc_id, "in_doc",
                                   base_weight=1.0, evidence=ev, meta={"doc": doc}, use_vote=use_fusion)

        # 3) Q 匹配（q_match）
        if self.use_qmatch and q_node and "q_match" not in disabled:
            q_words = set(re.findall(r"\w+", (q_node.text or "").lower()))
            for s in sent_nodes:
                s_words = set(re.findall(r"\w+", (s.text or "").lower()))
                overlap = q_words & s_words
                if overlap:
                    frac = len(overlap) / (len(q_words) + 1e-6)
                    ev = [EdgeEvidence(channel="q_overlap", score=float(min(1.0, frac)), meta={"overlap": list(overlap)})]
                    self._add_edge(edges, q_node.id, s.id, "q_match",
                                   base_weight=frac, evidence=ev, meta={"overlap": list(overlap)}, use_vote=use_fusion)

        # 4) 语义边（semantic_sim）
        if self.use_semantic_edges and "semantic_sim" not in disabled:
            def get_vec(text: str) -> np.ndarray:
                if self.embed_fn:
                    try:
                        return np.array(self.embed_fn(text or ""), dtype=float)
                    except Exception:
                        pass
                return self._fake_embed(text or "")

            vecs = {s.id: get_vec(s.text) for s in sent_nodes}
            for a, b in itertools.combinations(sent_nodes, 2):
                va, vb = vecs[a.id], vecs[b.id]
                if np.linalg.norm(va) == 0 or np.linalg.norm(vb) == 0:
                    sim = 0.0
                else:
                    sim = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))
                if sim >= float(self.semantic_threshold):
                    ev = [EdgeEvidence(channel="embed_sim", score=sim, meta={})]
                    # 加弱先验
                    prior = self._position_prior(a.meta, b.meta)
                    if prior > 0:
                        ev.append(EdgeEvidence(channel="position_prior", score=prior, meta={}))
                    self._add_edge(edges, a.id, b.id, "semantic_sim",
                                   base_weight=sim, evidence=ev, meta={"similarity": round(sim, 3)}, use_vote=use_fusion)

        # 5) mentions（句子->实体）
        if self.use_entity_edges and "mentions" not in disabled:
            for s in sent_nodes:
                for e in ent_nodes:
                    if e.text and s.text and e.text in s.text:
                        ev = [EdgeEvidence(channel="entity_link", score=0.6, meta={"reason": "substring"})]
                        self._add_edge(edges, s.id, e.id, "mentions",
                                       base_weight=1.0, evidence=ev, meta={"entity": e.text}, use_vote=use_fusion)

        # ---------- 稀疏化 ----------
        min_before = len(edges)
        if use_fusion:
            edges = [e for e in edges if e.weight >= min_vote]
            if max_edges_per_node and max_edges_per_node > 0:
                per_node: Dict[str, List[GraphEdge]] = defaultdict(list)
                for e in edges:
                    per_node[e.source].append(e)
                    per_node[e.target].append(e)
                kept = []
                for _, lst in per_node.items():
                    lst_sorted = sorted(lst, key=lambda x: x.weight, reverse=True)[:max_edges_per_node]
                    kept.extend(lst_sorted)
                uniq = {}
                for e in kept:
                    key = (e.source, e.target, e.type)
                    if key not in uniq or e.weight > uniq[key].weight:
                        uniq[key] = e
                edges = list(uniq.values())
        min_after = len(edges)

        # ---------- Diagnostics ----------
        type_cnt = defaultdict(int)
        for e in edges: type_cnt[e.type] += 1
        self.last_diagnostics = {
            "config": {
                "use_adjacency": self.use_adjacency,
                "use_qmatch": self.use_qmatch,
                "use_doc_edges": self.use_doc_edges,
                "use_entity_edges": self.use_entity_edges,
                "use_semantic_edges": self.use_semantic_edges,
                "semantic_threshold": self.semantic_threshold,
                "fusion_enabled": use_fusion,
                "assembly_policy": self.assembly_policy,
            },
            "edge_counts": dict(type_cnt),
            "total_edges": len(edges),
            "total_edges_before_prune": min_before,
            "total_edges_after_prune": min_after,
        }

        # 返回 dict 列表
        return [e.model_dump() for e in edges]