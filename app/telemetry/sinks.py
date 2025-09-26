# app/telemetry/sinks.py
from __future__ import annotations
from typing import Protocol, Dict, Any, TypedDict, Optional, List
from pathlib import Path
import json, time, threading, contextlib

# ---------- 事件与指标 Schema ----------

class LLMUsage(TypedDict, total=False):
    provider: str                # openai / ollama / ...
    model: str                   # gpt-4o / llama3:8b-instruct / ...
    tokens_in: int
    tokens_out: int
    latency_ms: float
    cached: bool
    temperature: Optional[float]
    max_tokens: Optional[int]
    error: Optional[str]
    module: Optional[str]        # RetrievalAgent / ReasoningAgent / VerifierAgent / Embedding
    purpose: Optional[str]       # query_expand / plan / synthesize / factcheck / embed ...
    request_id: Optional[str]    # provider 请求 ID（如有）

class CoverageMetrics(TypedDict, total=False):
    gold_facts_total: int
    gold_facts_hit: int
    coverage_rate: float
    details: List[Dict[str, Any]]

class PathMatchMetrics(TypedDict, total=False):
    gold_paths_total: int
    matched_paths: int
    match_rate: float
    details: List[Dict[str, Any]]

class LatencyBreakdown(TypedDict, total=False):
    by_node: Dict[str, float]    # {"BuildGraph": 0.153, "Retrieval": 0.421, ...}
    total_sec: float

class VerifierMetrics(TypedDict, total=False):
    coverage_score: float
    consistency_score: float
    hallucination_risk: float
    final_score: float
    verdict: str
    agreement_rate: float
    issues_count: int

class TelemetryEvent(TypedDict, total=False):
    trace_id: str
    ts: float
    event: str                    # run_start/run_end/node_start/node_end/error/llm_call/metrics
    node: Optional[str]
    status: Optional[str]         # ok/error/running
    duration_sec: Optional[float]
    error: Optional[str]
    payload: Dict[str, Any]       # 对 llm_call: {"llm": LLMUsage}; 对 metrics: {"coverage"/"path_match"/"latency"/"verifier": ...}

# ---------- Sink 抽象 ----------

class TelemetrySink(Protocol):
    def record(self, evt: TelemetryEvent) -> None: ...
    def flush_run(self, trace_id: str, result: Dict[str, Any]) -> None: ...

class NullSink:
    def record(self, evt: TelemetryEvent) -> None: ...
    def flush_run(self, trace_id: str, result: Dict[str, Any]) -> None: ...

class LocalJsonlSink:
    """
    本地落盘：
      runs/<trace_id>/events.jsonl  —— 一行一个 TelemetryEvent
      runs/<trace_id>/run.json      —— 最终结果快照（含 metrics/telemetry/mermaid）
      runs/<trace_id>/assets/flow.mmd —— Mermaid diagram（基于 events.jsonl 生成）
    """
    def __init__(self, root_dir: str = "runs"):
        self.root = Path(root_dir); self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _dir(self, trace_id: str) -> Path:
        d = self.root / trace_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "assets").mkdir(parents=True, exist_ok=True)
        return d

    def record(self, evt: TelemetryEvent) -> None:
        trace_id = evt.get("trace_id") or "trace-unknown"
        d = self._dir(trace_id)
        line = json.dumps(evt, ensure_ascii=False)
        with self._lock:
            with open(d / "events.jsonl", "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def flush_run(self, trace_id: str, result: Dict[str, Any]) -> None:
        d = self._dir(trace_id)
        snap = {"trace_id": trace_id, "created_at": time.time(), "result": result}
        with self._lock:
            with open(d / "run.json", "w", encoding="utf-8") as f:
                json.dump(snap, f, ensure_ascii=False, indent=2)

# ---------- 实用：计时器 / LLM 埋点 / 指标埋点 / Run 事件 ----------

def now() -> float:
    return time.time()

@contextlib.contextmanager
def span(node: str, sink: TelemetrySink, trace_id: str):
    t0 = now()
    sink.record({"trace_id": trace_id, "ts": t0, "event": "node_start", "node": node, "status": "running", "payload": {}})
    try:
        yield
        t1 = now()
        sink.record({"trace_id": trace_id, "ts": t1, "event": "node_end", "node": node, "status": "ok", "duration_sec": t1 - t0, "payload": {}})
    except Exception as e:
        t1 = now()
        sink.record({"trace_id": trace_id, "ts": t1, "event": "error", "node": node, "status": "error", "duration_sec": t1 - t0, "error": repr(e), "payload": {}})
        raise

def record_llm_call(sink: TelemetrySink, trace_id: str, usage: LLMUsage):
    sink.record({
        "trace_id": trace_id,
        "ts": now(),
        "event": "llm_call",
        "node": None,
        "status": "ok" if not usage.get("error") else "error",
        "payload": {"llm": usage},
    })

def record_metrics(
    sink: TelemetrySink,
    trace_id: str, *,
    coverage: CoverageMetrics | None = None,
    path_match: PathMatchMetrics | None = None,
    latency: LatencyBreakdown | None = None,
    verifier: VerifierMetrics | None = None,   # 新增：Verifier 关键指标
):
    payload: Dict[str, Any] = {}
    if coverage: payload["coverage"] = coverage
    if path_match: payload["path_match"] = path_match
    if latency: payload["latency"] = latency
    if verifier: payload["verifier"] = verifier
    if payload:
        sink.record({"trace_id": trace_id, "ts": now(), "event": "metrics", "node": None, "status": "ok", "payload": payload})

def record_run_start(sink: TelemetrySink, trace_id: str, payload: Dict[str, Any] | None = None):
    sink.record({"trace_id": trace_id, "ts": now(), "event": "run_start", "node": None, "status": "running", "payload": payload or {}})

def record_run_end(sink: TelemetrySink, trace_id: str, payload: Dict[str, Any] | None = None):
    sink.record({"trace_id": trace_id, "ts": now(), "event": "run_end", "node": None, "status": "ok", "payload": payload or {}})

# ---------- 事件解析与 Mermaid 生成 ----------

def _read_events(trace_dir: Path) -> List[TelemetryEvent]:
    p = trace_dir / "events.jsonl"
    if not p.exists():
        return []
    evts: List[TelemetryEvent] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                evts.append(json.loads(line))
            except Exception:
                pass
    return evts

def build_latency_breakdown_from_events(evts: List[TelemetryEvent]) -> LatencyBreakdown:
    by_node = {}
    for e in evts:
        if e.get("event") == "node_end" and e.get("node"):
            by_node[e["node"]] = by_node.get(e["node"], 0.0) + float(e.get("duration_sec", 0.0))
    total = sum(by_node.values())
    return {"by_node": by_node, "total_sec": total}

def build_mermaid_from_events(evts: List[TelemetryEvent]) -> str:
    """
    依据 node_start/node_end 顺序生成一个简单的 Mermaid 流程图。
    注意：这里不是严格的 DAG 抽取，而是基于时间序列的“执行轨迹”展示。
    """
    # 按时间排序
    evts_sorted = sorted(evts, key=lambda x: x.get("ts", 0.0))
    nodes_seen: List[str] = []
    seq_edges: List[tuple[str, str]] = []

    # 收集执行顺序中的节点（只取 node_start）
    for e in evts_sorted:
        if e.get("event") == "node_start" and e.get("node"):
            nodes_seen.append(e["node"])

    # 串成顺序边
    for i in range(len(nodes_seen) - 1):
        seq_edges.append((nodes_seen[i], nodes_seen[i+1]))

    # 生成 Mermaid
    lines = ["flowchart TD"]
    # 节点声明
    uniq_nodes = []
    for n in nodes_seen:
        if n not in uniq_nodes: uniq_nodes.append(n)
    for n in uniq_nodes:
        safe = n.replace(" ", "_").replace("-", "_")
        lines.append(f'  {safe}["{n}"]')

    # 边
    for a, b in seq_edges:
        sa = a.replace(" ", "_").replace("-", "_")
        sb = b.replace(" ", "_").replace("-", "_")
        lines.append(f"  {sa} --> {sb}")

    return "\n".join(lines) if len(lines) > 1 else "flowchart TD\n  A[Start] --> B[End]"

def write_mermaid(trace_dir: Path, mermaid_text: str):
    assets = trace_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    with open(assets / "flow.mmd", "w", encoding="utf-8") as f:
        f.write(mermaid_text)

def finalize_trace_artifacts(root_dir: str, trace_id: str, sink: TelemetrySink):
    """
    - 读取 events.jsonl
    - 生成 latency breakdown 指标事件（补发）
    - 生成 Mermaid diagram 并落地
    """
    if not isinstance(sink, LocalJsonlSink):
        return  # 只有本地 Sink 才做离线文件
    trace_dir = Path(sink.root) / trace_id
    evts = _read_events(trace_dir)
    if not evts:
        return
    # 1) latency breakdown
    latency = build_latency_breakdown_from_events(evts)
    record_metrics(sink, trace_id, latency=latency)
    # 2) Mermaid
    mmd = build_mermaid_from_events(evts)
    write_mermaid(trace_dir, mmd)