import uuid
import re
from typing import Dict, Any
from app.schemas.graph_request_v2 import AssembleGraphRequestV2, Inputs, Sentence


def normalize_title(title: str) -> str:
    """
    将文档标题规范化为可用的 ID 片段
    - 去掉前后空格
    - 把非字母数字字符替换为下划线
    
    type=directed → 图的方向性（GEXF 标准字段）
    kind=doc2sent → 表示句子属于哪个文档
    kind=next_sent → 表示句子之间的先后顺序
    kind=q2doc → 表示问题和文档的弱连接
    """
    return re.sub(r"\W+", "_", title.strip())


def upgrade_to_v2(raw: Dict[str, Any], *, default_trace_id: str) -> AssembleGraphRequestV2:
    """
    通用 v1 → v2 适配器（保持原逻辑）
    """
    raw_inputs = raw.get("inputs") or {}
    nodes = raw_inputs.get("nodes", raw.get("nodes", [])) or []
    edges = raw_inputs.get("edges", raw.get("edges", [])) or []

    # sentences 可以来自 inputs.sentences / sentences / question
    sents = raw_inputs.get("sentences") or raw.get("sentences")
    if sents is None and "question" in raw:
        sents = [raw["question"]]

    sentences = []
    if isinstance(sents, list):
        for i, t in enumerate(sents):
            sid = f"sent:{i}"
            sentences.append(Sentence(id=sid, text=t))
    elif isinstance(sents, str):
        sentences.append(Sentence(id="sent:0", text=sents))

    graph_id = f"graph-{default_trace_id}-{uuid.uuid4().hex[:8]}"
    inputs = Inputs(sentences=sentences, nodes=list(nodes), edges=list(edges))
    return AssembleGraphRequestV2(graph_id=graph_id, inputs=inputs)


def hotpotqa_to_v2(external_context: Dict[str, Any], trace_id: str = "trace-demo") -> AssembleGraphRequestV2:
    """
    将 HotpotQA 的 context 转换成 AssembleGraphRequestV2
    external_context = { "context": [ [doc_title, [sent0, sent1, ...]], ... ] }
    """

    graph_id = f"graph-{trace_id}-{uuid.uuid4().hex[:8]}"
    context = external_context.get("context", [])

    nodes: list[Dict[str, Any]] = []
    edges: list[Dict[str, Any]] = []
    sentences: list[Sentence] = []

    # question 节点（占位，后续由 state["question"] 替换）
    q_node_id = "question:0"
    nodes.append({"id": q_node_id, "label": "__USER_QUESTION__", "kind": "question"})
    sentences.append(Sentence(id=q_node_id, text="__USER_QUESTION__"))

    for doc_idx, (doc_title, sents) in enumerate(context):
        # ✅ 用标题生成 doc 节点 ID
        norm_title = normalize_title(doc_title)
        doc_id = f"doc:{norm_title}"
        nodes.append({"id": doc_id, "label": doc_title, "kind": "doc"})

        # question → doc 边
        edges.append({
            "source": q_node_id,
            "target": doc_id,
            "type": "directed",   # GEXF 保留字段
            "kind": "q2doc",
            "label": "q2doc"
        })

        prev_sent_id = None
        for sent_idx, sent_text in enumerate(sents):
            # ✅ 用文档标题 + 句子编号生成句子节点 ID
            sent_id = f"{doc_id}::sent{sent_idx}"
            nodes.append({"id": sent_id, "label": sent_text, "kind": "sentence"})
            sentences.append(Sentence(id=sent_id, text=sent_text))

            # doc → sent 边
            edges.append({
                "source": doc_id,
                "target": sent_id,
                "type": "directed",
                "kind": "doc2sent",
                "label": "doc2sent"
            })

            # 顺序边 sent[i] → sent[i+1]
            if prev_sent_id is not None:
                edges.append({
                    "source": prev_sent_id,
                    "target": sent_id,
                    "type": "directed",
                    "kind": "next_sent",
                    "label": "next_sent"
                })

            prev_sent_id = sent_id

    inputs = Inputs(sentences=sentences, nodes=nodes, edges=edges)
    return AssembleGraphRequestV2(graph_id=graph_id, inputs=inputs)
