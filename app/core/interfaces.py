from __future__ import annotations
from typing import Protocol
from app.core.dto import (
    GraphBuildIn, GraphBuildOut,
    RetrievalIn, RetrievalOut,
    ReasoningIn, ReasoningOut,
    VerifyIn, VerifyOut,
)

class GraphConstruction(Protocol):
    """
    职责：从 question_text + context 出发，构建节点/边，生成 graph_id，组装/落盘图，并返回 GraphBuildOut。
    - Orchestration 仅传入 GraphBuildIn（不要求提前构建 nodes/edges，也可兼容传入）
    """
    def build(self, req: GraphBuildIn) -> GraphBuildOut: ...

class RetrievalAgent(Protocol):
    def retrieve(self, req: RetrievalIn) -> RetrievalOut: ...

class ReasoningAgent(Protocol):
    def reason(self, req: ReasoningIn) -> ReasoningOut: ...

class VerifierAgent(Protocol):
    def verify(self, req: VerifyIn) -> VerifyOut: ...
