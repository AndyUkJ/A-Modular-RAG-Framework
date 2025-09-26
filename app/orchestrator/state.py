from typing import TypedDict, Any, Dict
from app.schemas.graph_request_v2 import AssembleGraphRequestV2


class WFState(TypedDict, total=False):
    """
    全局工作流状态（由 LangGraph 在各节点间传递与合并）。
    注意：必须显式声明 `route`，否则 ChooseRoute 返回的路由值会在
    结构化合并时被丢弃，导致后续条件分支拿不到。
    """

    # 外部上下文/输入
    external_context: Dict[str, Any]
    request_v2: AssembleGraphRequestV2
    question: str
    trace_id: str
    policy: Dict[str, Any]  # e.g. {"mode": "full"}

    # ★ 路由（必须声明）
    route: str  # "Retrieval" | "PackResult"

    # 各阶段产物
    graph: Dict[str, Any]
    retrieval: Dict[str, Any]
    reasoning: Dict[str, Any]
    verification: Dict[str, Any]

    # 计时与最终结果
    t0: float
    t1: float
    result: Dict[str, Any]
