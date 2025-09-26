# app/core/dto.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


# ========= Graph In/Out =========

class GraphBuildIn(BaseModel):
    trace_id: str
    question_text: str = ""
    context: List = Field(default_factory=list)

    graph_id: Optional[str] = None
    nodes: List[Dict[str, Any]] = Field(default_factory=list)
    edges: List[Dict[str, Any]] = Field(default_factory=list)

    # 原版 extra 保留，用于 provenance/policy/meta
    extra: Dict[str, Any] = Field(default_factory=dict)


class GraphBuildOut(BaseModel):
    graph_id: str
    node_count: int
    edge_count: int

    # 原版是可选字段
    nodes: Optional[List[Dict[str, Any]]] = None
    edges: Optional[List[Dict[str, Any]]] = None
    provenance: Optional[Dict[str, Any]] = None
    diagnostics: Optional[Dict[str, Any]] = None

    extra: Dict[str, Any] = Field(default_factory=dict)


# ========= Retrieval =========

class RetrievalIn(BaseModel):
    # 必须叫 query（原版如此）
    query: str
    graph_id: str
    top_k: int = 20
    trace_id: str


class Hit(BaseModel):
    # 保持原版最小字段
    id: str
    score: float
    meta: Dict[str, Any] = Field(default_factory=dict)


class RetrievalOut(BaseModel):
    hits: List[Hit] = Field(default_factory=list)
    diagnostics: Dict[str, Any] = Field(default_factory=dict)


# ========= Reasoning =========

class ReasoningIn(BaseModel):
    question: str
    hits: List[Hit] = Field(default_factory=list)
    graph_id: str
    trace_id: str


class ReasoningOut(BaseModel):
    answer: str
    evidence_used: List[Hit] = Field(default_factory=list)
    steps: List[Dict[str, Any]] = Field(default_factory=list)
    model: Optional[str] = None


# ========= Verification =========

class VerifyIn(BaseModel):
    answer: str
    evidence: List[Hit] = Field(default_factory=list)
    question: Optional[str] = None
    query: Optional[str] = None
    graph_id: Optional[str] = None
    trace_id: Optional[str] = None
    retry_round: int = 0   # 标记当前是第几次验证（默认 0）


class VerifyOut(BaseModel):
    """
    Output schema for VerifierAgent (验证器输出格式)

    📌 字段语义说明：

    ---- 核心状态字段 ----
    status: str
        主状态（兼容旧逻辑）
        - "pass" | "fail" | "warn"
        - 建议逐步弃用，仅保留向后兼容

    status_detail: Optional[str]
        细粒度状态分类（枚举值，详见 StatusDetail）
        - "fail"               → 明确失败（存在矛盾或缺失关键证据）
        - "high_conf_pass"     → 高置信度通过（核心事实直接被证据强支撑）
        - "low_conf_pass"      → 低置信度通过（部分依赖间接证据 / 存在噪声）
        - "unknown_pass"       → 模糊通过（未检测到矛盾，但支持力度不足）

    status_detail_label: Optional[str]
        人类可读标签（英文，适合 UI 直接展示）
        - "Fail"
        - "High Confidence Pass"
        - "Low Confidence Pass"
        - "Unknown Confidence Pass"

    ok: Optional[bool]
        二值判定（兼容老逻辑）：
        - True  → 通过（包括高/低置信度）
        - False → 未通过

    ---- 评分与诊断指标 ----
    score: Optional[float]
        最终综合评分（融合规则、LLM、一致性、自洽性）

    coverage_score: Optional[float]
        引用覆盖率（0~1），反映答案中引用证据的覆盖程度

    consistency_score: Optional[float]
        一致性评分（0~1），反映 LLM 判断下事实支持度

    hallucination_risk: Optional[float]
        幻觉风险（0~1），值越高风险越大

    final_score: Optional[float]
        最终分数（等价于 score，用于兼容历史字段）

    ---- 结构化诊断信息 ----
    issues: List[str]
        验证过程中发现的问题列表（纯字符串）

    findings: List[Dict[str, Any]]
        结构化发现（如 contradiction, partial_support 等）

    diagnostics: Dict[str, Any]
        详细诊断结果（包含 rule/llm/self-consistency/claim-check/citations 等子结构）

    self_consistency: Optional[Dict[str, Any]]
        自洽性信息（投票结果、majority verdict、agreement_rate 等）

    ---- 上游编排/用户提示 ----
    recommended_action: Optional[str]
        推荐动作（供 Orchestrator 或 UI 使用）
        - "Reject and re-run"
        - "Retry retrieval / claim-check"
        - "Accept (high confidence)"
        - "Accept; prune noisy citations"
        - "Review required (uncertain evidence)"
        - "Review recommended (low confidence)"
    """

    # ---- 核心状态 ----
    status: str  # "pass" | "fail" | "warn"
    findings: List[Dict[str, Any]] = Field(default_factory=list)
    model: Optional[str] = None

    # ---- 新增：可选丰富字段 ----
    ok: Optional[bool] = None
    score: Optional[float] = None
    issues: List[str] = Field(default_factory=list)
    diagnostics: Dict[str, Any] = Field(default_factory=dict)

    # ---- 指标字段 ----
    coverage_score: Optional[float] = None
    consistency_score: Optional[float] = None
    hallucination_risk: Optional[float] = None
    final_score: Optional[float] = None

    # ---- 细粒度 verdict ----
    verdict: Optional[str] = None
    self_consistency: Optional[Dict[str, Any]] = None

    # ---- 推荐动作 ----
    recommended_action: Optional[str] = None

    # ---- 新增：细粒度状态 ----
    status_detail: Optional[str] = None
    status_detail_label: Optional[str] = None


# ========= Graph Node / Edge =========

class EdgeEvidence(BaseModel):
    channel: str
    score: float
    meta: Dict[str, Any] = Field(default_factory=dict)


class GraphNode(BaseModel):
    id: str
    type: str
    text: str
    meta: Dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    source: str
    target: str
    type: str
    weight: float = 1.0
    meta: Dict[str, Any] = Field(default_factory=dict)

    # 新增：兼容增强的可选证据
    evidence: List[EdgeEvidence] = Field(default_factory=list)