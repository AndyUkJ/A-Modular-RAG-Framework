# 📁 graph_analyzer.py - 精细版图分析工具
import json
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter, defaultdict


def analyze_graph_file(json_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    nodes = data["nodes"]
    edges = data["edges"]

    # === 构图 ===
    G = nx.DiGraph()
    G.add_nodes_from((n["id"], n) for n in nodes)
    G.add_edges_from((e["source"], e["target"]) for e in edges)

    # === 统计：边类型分布 ===
    edge_type_counts = Counter(e["type"] for e in edges)
    with open(output_dir / "edge_type_stats.json", "w") as f:
        json.dump(edge_type_counts, f, indent=2)

    # === 统计：节点度分布（Top 10） ===
    degree = defaultdict(int)
    for e in edges:
        degree[e["source"]] += 1
        degree[e["target"]] += 1
    top_nodes = sorted(degree.items(), key=lambda x: x[1], reverse=True)[:10]
    with open(output_dir / "top_nodes.json", "w") as f:
        json.dump(top_nodes, f, indent=2)

    # === 连通性分析 ===
    weakly_conn = nx.is_weakly_connected(G)
    num_components = nx.number_weakly_connected_components(G)
    components = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
    comp_sizes = [len(c) for c in components]
    with open(output_dir / "connectivity.json", "w") as f:
        json.dump({
            "is_weakly_connected": weakly_conn,
            "num_components": num_components,
            "component_sizes": comp_sizes[:5]  # 前5个子图大小
        }, f, indent=2)

    # === 中心性分析（Degree Centrality） ===
    centrality = nx.degree_centrality(G)
    top_cent = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    with open(output_dir / "top_centrality.json", "w") as f:
        json.dump(top_cent, f, indent=2)

    # === 可视化：边类型直方图 ===
    plt.figure(figsize=(8, 4))
    plt.bar(edge_type_counts.keys(), edge_type_counts.values(), color="skyblue")
    plt.title("Edge Type Distribution")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(output_dir / "edge_distribution.png")

    # === 可视化：子图大小分布 ===
    if len(comp_sizes) > 1:
        plt.figure(figsize=(6, 4))
        plt.bar(range(1, len(comp_sizes) + 1), comp_sizes, color="lightcoral")
        plt.title("Top Component Sizes")
        plt.xlabel("Component Rank")
        plt.ylabel("Node Count")
        plt.tight_layout()
        plt.savefig(output_dir / "component_sizes.png")