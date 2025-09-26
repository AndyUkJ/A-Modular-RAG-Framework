# app/system.py
import time, uuid
from typing import Dict, Any
from app.di.factory import load_settings, build_providers, build_router, build_modules
from app.orchestrator.workflow import build_workflow
from app.core.dataset_loader import build_dataset_loader   # ✅ 已有
from app.telemetry.sinks import LocalJsonlSink, record_run_start, record_run_end, finalize_trace_artifacts

def new_trace_id() -> str:
    ts = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    return f"trace-{ts}-{uuid.uuid4().hex[:8]}"

def init_system():
    """
    初始化系统：加载配置、构建 provider/router/modules/workflow
    """
    settings = load_settings("config/settings.yaml")
    sink = LocalJsonlSink(root_dir="runs")  # ✅ 统一 Telemetry Sink

    providers = build_providers(settings)
    router = build_router(settings, providers, sink=sink)
    node_ctx = build_modules(settings, router, sink=sink)

    # ✅ dataset loader 配置
    dataset_cfg = settings.get("dataset", {})
    dataset_loader = build_dataset_loader(dataset_cfg) if dataset_cfg else None

    # ✅ 把 dataset 相关传给 workflow
    wf = build_workflow(node_ctx, dataset_cfg=dataset_cfg, dataset_loader=dataset_loader)
    return wf, sink

def answer_question(question: str, *, mode: str = "full") -> Dict[str, Any]:
    """
    系统对外暴露的统一入口
    """
    wf, sink = init_system()
    trace_id = new_trace_id()

    init_state = {
        "external_context": {},
        "question": question,
        "trace_id": trace_id,
        "policy": {"mode": mode},
    }

    # --- Run start
    record_run_start(sink, trace_id, {"question": question, "mode": mode})

    # 执行
    final_state = wf.invoke(input=init_state)
    result = final_state["result"]

    # --- Run end + 生成离线可视化与指标补发
    record_run_end(sink, trace_id, {"status": "completed"})
    finalize_trace_artifacts(root_dir="runs", trace_id=trace_id, sink=sink)

    # --- Flush run snapshot
    sink.flush_run(trace_id, result)
    return result
