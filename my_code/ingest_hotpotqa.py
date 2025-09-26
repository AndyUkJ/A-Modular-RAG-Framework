from __future__ import annotations
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

from app.modules.graph_construction.impl_networkx import GraphConstructionNetworkX
from app.core.dto import GraphBuildIn


def _mk_node(title: str) -> Dict[str, Any]:
    return {"id": title, "type": "page", "props": {"title": title}}


def _mk_edge(a: str, b: str, etype: str = "supporting") -> Dict[str, Any]:
    return {"source": a, "target": b, "type": etype, "props": {}}


def build_raw_request(sample: Dict[str, Any]) -> Dict[str, Any]:
    sid = sample.get("_id") or sample.get("id")
    context = sample["context"]  # [[title, [sent0, sent1,...]], ...]

    # nodes
    nodes = [_mk_node(title) for title, _ in context]

    # edges based on supporting_facts
    sf_pairs = sample.get("supporting_facts", [])
    sf_titles = list({t for (t, _) in sf_pairs})
    edges = []
    for i in range(len(sf_titles)):
        for j in range(i + 1, len(sf_titles)):
            a, b = sf_titles[i], sf_titles[j]
            edges.append(_mk_edge(a, b))
            edges.append(_mk_edge(b, a))

    return {
        "api_version": "v1",
        "request_id": f"req-{sid}",
        "graph_id": f"hotpotqa-{sid}",
        "nodes": nodes,
        "edges": edges,
        "options": {"dedup": True},
    }


def ingest(input_path: Path, graph_root: Path, docs_out: Path, limit: int = 500):
    if not input_path.exists():
        raise FileNotFoundError(f"HotpotQA file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 初始化 GraphConstruction
    gc = GraphConstructionNetworkX(root_dir=str(graph_root))

    docs_out.parent.mkdir(parents=True, exist_ok=True)
    fout = open(docs_out, "w", encoding="utf-8")

    for i, sample in enumerate(data):
        if limit and i >= limit:
            break

        raw_request = build_raw_request(sample)
        req = GraphBuildIn(
            graph_id=raw_request["graph_id"],
            nodes=raw_request["nodes"],
            edges=raw_request["edges"],
            trace_id=f"trace-hotpot-{i}",
        )
        out = gc.build(req)

        # 写入句子索引
        for title, sentences in sample["context"]:
            for sid, sent in enumerate(sentences):
                doc = {
                    "doc_id": f"{title}#{sid}",
                    "title": title,
                    "sent_id": sid,
                    "text": sent,
                }
                fout.write(json.dumps(doc, ensure_ascii=False) + "\n")

        if (i + 1) % 50 == 0:
            print(f"[{i+1}] graphs built, last graph_id={out.graph_id}")

    fout.close()
    print(f"Finished. Graphs saved under {graph_root}, docs in {docs_out}")


def main():
    ap = argparse.ArgumentParser(description="Ingest HotpotQA into Graph + Docs")
    ap.add_argument("--input", type=str, default="data/hotpotqa/hotpot_dev_distractor_v1.json",
                    help="Path to hotpotqa json file")
    ap.add_argument("--graph_root", type=str, default="data/graph/hotpotqa",
                    help="Root dir to store graph files")
    ap.add_argument("--docs_out", type=str, default="data/hotpotqa/docs.jsonl",
                    help="Output path for flattened docs jsonl")
    ap.add_argument("--limit", type=int, default=500,
                    help="Limit number of samples to ingest (0=all)")
    args = ap.parse_args()

    ingest(
        input_path=Path(args.input),
        graph_root=Path(args.graph_root),
        docs_out=Path(args.docs_out),
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
