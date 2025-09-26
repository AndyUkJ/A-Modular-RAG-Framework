import json
import networkx as nx
from pathlib import Path
from typing import Dict, Any
from collections import Counter

from app.core.dto import GraphBuildIn, GraphBuildOut
from app.core.interfaces import GraphConstruction
from app.utils.graph_analyzer import analyze_graph_file


def _sanitize_attrs(attrs: Dict[str, Any]) -> Dict[str, Any]:
    """将复杂属性序列化为 JSON 字符串，确保可写入 GEXF/JSON。"""
    clean = {}
    for k, v in attrs.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            clean[k] = v
        elif isinstance(v, (list, dict)):
            clean[k] = json.dumps(v, ensure_ascii=False)
        else:
            clean[k] = str(v)
    return clean


class GraphConstructionNetworkX(GraphConstruction):
    def __init__(self, root_dir: str = "data/graph"):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def build(self, req: GraphBuildIn) -> GraphBuildOut:
        G = nx.DiGraph()

        # ---- 节点 ----
        for n in req.nodes:
            node_id = n["id"]
            # 保留所有字段，做序列化
            G.add_node(node_id, **_sanitize_attrs({**n}))

        # ---- 边 ----
        for e in req.edges:
            src, tgt = e["source"], e["target"]
            G.add_edge(src, tgt, **_sanitize_attrs({**e}))

        # ---- 落盘 ----
        graph_id = req.graph_id or "graph-unknown"
        out_dir = self.root_dir / graph_id
        out_dir.mkdir(parents=True, exist_ok=True)
        gexf_path = out_dir / "graph.gexf"
        json_path = out_dir / "graph.json"
        manifest_path = out_dir / "manifest.json"
        analysis_dir = out_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)

        nx.write_gexf(G, gexf_path)

        nodes = [{"id": nid, **attrs} for nid, attrs in G.nodes(data=True)]
        edges = [{"source": s, "target": t, **attrs} for s, t, attrs in G.edges(data=True)]
        summary = {
            "graph_id": graph_id,
            "node_count": G.number_of_nodes(),
            "edge_count": G.number_of_edges(),
            "nodes": nodes,
            "edges": edges,
        }
        json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        manifest = {
            "graph_id": graph_id,
            "node_count": G.number_of_nodes(),
            "edge_count": G.number_of_edges(),
            "paths": {
                "dir": str(out_dir),
                "gexf": str(gexf_path),
                "json": str(json_path),
                "manifest": str(manifest_path),
            },
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        # ---- 图分析 ----
        try:
            analysis = analyze_graph_file(str(json_path), str(analysis_dir))
        except Exception as e:
            analysis = {"error": f"{e.__class__.__name__}: {e}"}

        # ---- 诊断信息聚合 ----
        diag = {
            "node_types": dict(Counter([dat.get("type") for _, dat in G.nodes(data=True)])),
            "edge_types": dict(Counter([dat.get("type") for _, _, dat in G.edges(data=True)])),
            "analysis": analysis,
        }

        if isinstance(req.extra, dict):
            # 合并 NodeBuilder/EdgeBuilder diagnostics
            for k in ("node_builder_diagnostics", "edge_builder_diagnostics", "diagnostics"):
                v = req.extra.get(k)
                if isinstance(v, dict) and v:
                    diag[k] = v

            # 统计 evidence channel（如果存在）
            ev_counts = Counter()
            for e in req.edges:
                evs = e.get("evidence") or []
                if isinstance(evs, list):
                    for ev in evs:
                        ch = (ev.get("channel") if isinstance(ev, dict) else None) or ""
                        if ch:
                            ev_counts[ch] += 1
            if ev_counts:
                diag["evidence_channels"] = dict(ev_counts)

        provenance = {"impl": "networkx", "graph_id": graph_id}
        if isinstance(req.extra, dict) and "policy" in req.extra:
            provenance["policy"] = req.extra["policy"]

        return GraphBuildOut(
            graph_id=graph_id,
            node_count=G.number_of_nodes(),
            edge_count=G.number_of_edges(),
            nodes=nodes,
            edges=edges,
            provenance=provenance,
            diagnostics=diag,
            extra={"paths": manifest["paths"]},
        )