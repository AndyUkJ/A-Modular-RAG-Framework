from typing import List, Dict, Any, Union
import re
from app.core.dto import GraphNode
from app.utils.entity_linker import elq_link_entities

try:
    from app.modules.graph_construction.segmenter import segment_context
except Exception:
    segment_context = None


class NodeBuilder:
    """
    Node 构建器（G1 + G3 默认启用，可在 settings.yaml 配置关闭）
    - G1: 语义切分
    - G3: 实体节点
    """

    def __init__(self,
                 enable_segmentation: bool = True,
                 segmentation_strategy: str = "rule",
                 segmentation_sim_threshold: float = 0.65,
                 use_entity_nodes: bool = True,
                 use_doc_nodes: bool = True):
        self.enable_segmentation = enable_segmentation
        self.segmentation_strategy = segmentation_strategy
        self.segmentation_sim_threshold = segmentation_sim_threshold
        self.use_entity_nodes = use_entity_nodes
        self.use_doc_nodes = use_doc_nodes
        self.last_diagnostics: Dict[str, Any] = {}

    def _normalize_context(self, context: List[Union[str, tuple, list, dict]]) -> List[tuple]:
        out = []
        for item in context:
            if isinstance(item, str):
                out.append(("default", [item]))
            elif isinstance(item, (tuple, list)) and len(item) == 2:
                out.append((str(item[0]), [str(x) for x in item[1]]))
            elif isinstance(item, dict) and "title" in item and "sentences" in item:
                out.append((str(item["title"]), [str(x) for x in item["sentences"]]))
        return out

    def build(self, question: str, context: List[Union[str, tuple, list, dict]], policy: Dict[str, Any]) -> List[GraphNode]:
        nodes: List[GraphNode] = []

        if question:
            nodes.append(GraphNode(id="q1", type="question", text=question, meta={"source": "question"}))

        norm_ctx_before = self._normalize_context(context)
        norm_ctx = norm_ctx_before

        seg_applied = False
        if self.enable_segmentation and segment_context:
            norm_ctx = segment_context(norm_ctx,
                                       strategy=self.segmentation_strategy,
                                       embed_fn=policy.get("embed_fn"),
                                       sim_threshold=self.segmentation_sim_threshold)
            seg_applied = True

        sent_idx = 0
        doc_titles = set()
        for doc_title, sentences in norm_ctx:
            doc_titles.add(doc_title)
            for j, sent in enumerate(sentences):
                node_id = f"{doc_title}::sent{j}" if doc_title != "default" else f"sent{sent_idx}"
                nodes.append(GraphNode(id=node_id, type="sentence", text=sent,
                                       meta={"doc": doc_title, "sent_id": j if doc_title != "default" else sent_idx,
                                             "source": "context"}))
                sent_idx += 1

        if self.use_doc_nodes:
            for doc in doc_titles:
                nodes.append(GraphNode(id=f"doc::{doc}", type="document", text=doc, meta={"source": "context"}))

        entity_count = 0
        if self.use_entity_nodes:
            entity_set = set()
            for n in nodes:
                if n.type == "sentence":
                    entity_set.update(re.findall(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)*\b", n.text or ""))
            linked = elq_link_entities(" ".join([n.text for n in nodes if n.type == "sentence"]))
            for ent in linked:
                if "text" in ent:
                    entity_set.add(ent["text"])
            for e in sorted(entity_set):
                nodes.append(GraphNode(id=f"ent::{e.replace(' ', '_')}", type="entity", text=e, meta={"source": "elq"}))
                entity_count += 1

        self.last_diagnostics = {
            "segment": {
                "enabled": seg_applied,
                "strategy": self.segmentation_strategy if seg_applied else None,
                "sim_threshold": self.segmentation_sim_threshold if seg_applied else None,
                "sent_count_before": sum(len(s) for _, s in norm_ctx_before),
                "sent_count_after": sum(len(s) for _, s in norm_ctx),
            },
            "node_counts": {
                "question": 1 if question else 0,
                "document": len(doc_titles) if self.use_doc_nodes else 0,
                "sentence": len([n for n in nodes if n.type == "sentence"]),
                "entity": entity_count,
            }
        }
        return nodes