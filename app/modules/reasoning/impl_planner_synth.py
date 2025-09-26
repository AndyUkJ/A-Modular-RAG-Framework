from __future__ import annotations
from typing import Any, Dict, List, Tuple
import logging

from app.core.interfaces import ReasoningAgent
from app.core.dto import ReasoningIn, ReasoningOut, Hit
from app.core.llm_router import LLMRouter
from app.telemetry.sinks import TelemetrySink, span
from app.modules.reasoning import strategies

logger = logging.getLogger(__name__)


class ReasoningAgentPlannerSynth(ReasoningAgent):
    """
    增强版 Reasoning Agent：
      1) PLAN 阶段：把问题拆分为 step list
      2) EVIDENCE 阶段：为每个 step 选择相关证据（支持实体约束；融合 lexical + dense/graph 规范化分）
      3) SYNTH 阶段：结合证据合成答案，支持多草稿 + 自洽投票
      4) REACT-REFINE 阶段：coverage 过低时，基于邻域扩展自动补证据并再合成
    """

    def __init__(self,
                 router: LLMRouter,
                 *,
                 sink: TelemetrySink | None = None,
                 max_hops: int = 3,
                 temperature: float = 0.6,
                 n_drafts: int = 1,
                 sc_runs: int = 3,
                 max_refine_rounds: int = 1,
                 coverage_threshold: float = 0.2,
                 refine_window: int = 2,
                 max_expand: int = 5):
        self.router = router
        self.sink = sink
        self.max_hops = int(max_hops)
        self.temperature = float(temperature)
        self.n_drafts = max(1, int(n_drafts))
        self.sc_runs = max(1, int(sc_runs))
        self.max_refine_rounds = max(0, int(max_refine_rounds))
        self.coverage_threshold = float(coverage_threshold)
        # 新增：可配置的邻域扩展参数
        self.refine_window = max(0, int(refine_window))
        self.max_expand = max(0, int(max_expand))

    # ---------------- internals ----------------

    def _plan_steps(self, question: str, trace_id: str) -> List[str]:
        """调用 LLM 把问题拆分为步骤"""
        prompt = (
            "You are a decomposition planner for multi-hop QA.\n"
            f"Question: {question}\n"
            f"Decompose into at most {self.max_hops} concise steps. "
            "Return one step per line with a leading number like '1) ...'. "
            "Steps should be atomic and verifiable."
        )
        logger.info(f"[ReasoningAgentPlannerSynth] PLAN call, trace_id={trace_id}")

        out = self.router.complete(
            module="ReasoningAgent", purpose="plan",
            prompt=prompt,
            require={"context_window": 16000, "temperature": 0.2, "trace_id": trace_id}
        )

        text = strategies.coerce_text(out)
        steps: List[str] = []
        for line in (text or "").splitlines():
            s = (line or "").strip()
            if not s:
                continue
            s = s.lstrip("-•").strip()
            s = s.split(")", 1)[-1].strip() if ")" in s[:4] else s
            if s:
                steps.append(s)

        logger.info(f"[PLAN Parsed Steps]: {steps}")
        return steps[: self.max_hops] or [question]

    def _synthesize_once(self, *, question: str, steps: List[str], citations: str, trace_id: str) -> str:
        """单次合成答案"""
        sys_hint = (
            "Synthesize a final answer using ONLY the provided citations. "
            "Cite evidence inline using [#k] where k is the citation number. "
            "Be concise and factual."
        )
        plan_block = "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(steps))
        context = f"{sys_hint}\n\nPlan:\n{plan_block}\n\nCitations:\n{citations}\n"

        out = self.router.complete(
            module="ReasoningAgent", purpose="synthesize",
            prompt=f"{context}\nQuestion: {question}\nAnswer:",
            require={"context_window": 32000,
                     "temperature": self.temperature,
                     "trace_id": trace_id}
        )

        text = strategies.coerce_text(out) or ""
        logger.info(f"[SYNTH Coerced Text]: {text[:200]!r}")
        return text

    # ---------------- main interface ----------------

    def reason(self, req: ReasoningIn) -> ReasoningOut:
        trace_id = req.trace_id or "trace-reason"
        logger.info(f"[ReasoningAgentPlannerSynth] Start reasoning, trace_id={trace_id}, question={req.question!r}")

        # === Step 1: PLAN ===
        if self.sink:
            with span("Reasoning/Plan", self.sink, trace_id):
                steps = self._plan_steps(req.question, trace_id)
        else:
            steps = self._plan_steps(req.question, trace_id)

        # === Step 2: EVIDENCE SELECTION ===
        hits = list(req.hits or [])
        # 与旧版一致：从问题里抽取大写开头 token 作为实体硬过滤条件
        require_entities = [w for w in (req.question or "").split() if w and w[0].isupper()]

        step_evidences, used = strategies.select_evidence_for_steps(
            steps,
            hits,
            per_step_k=2,
            min_score=0.05,
            require_entities=require_entities,
            neighbor_window=self.refine_window,        # 🔥 从 settings 透传
            neighbor_max_expand=self.max_expand,       # 🔥 从 settings 透传
        )
        citations = strategies.build_citation_block(hits, used)

        # === Step 3: SYNTHESIS (multi-draft + self-consistency) ===
        drafts: List[str] = []
        # 兼容旧版：草稿数使用 max(n_drafts, sc_runs) 以支持自洽投票
        for _ in range(max(self.n_drafts, self.sc_runs)):
            drafts.append(self._synthesize_once(
                question=req.question, steps=steps, citations=citations, trace_id=trace_id
            ))

        if len(drafts) > 1:
            answer, votes = strategies.majority_vote(drafts)
        else:
            answer, votes = (drafts[0] if drafts else ""), {}

        # === Step 4: COVERAGE CHECK + REACT-REFINE ===
        coverage = len(set(used)) / max(1, len(hits))
        refine_rounds: List[Dict[str, Any]] = []

        if coverage < self.coverage_threshold and self.max_refine_rounds > 0:
            logger.info(
                "[ReasoningAgentPlannerSynth] Coverage=%.2f < %.2f, triggering refinement "
                "(window=%d, max_expand=%d, rounds=%d)",
                coverage, self.coverage_threshold, self.refine_window, self.max_expand, self.max_refine_rounds
            )
            for r in range(self.max_refine_rounds):
                # 基于“已用证据”的 doc/sent 邻域扩展，控制窗口与最大扩展数
                new_used_set = strategies.expand_with_neighbors(
                    set(used), hits, window=self.refine_window, max_expand=self.max_expand
                )
                new_used = sorted(new_used_set)
                new_citations = strategies.build_citation_block(hits, new_used)
                new_draft = self._synthesize_once(
                    question=req.question, steps=steps,
                    citations=new_citations, trace_id=f"{trace_id}-ref{r}"
                )
                refine_rounds.append({"round": r, "draft": new_draft})
                # 采用最新稿作为当前答案；并更新 used/citations 以便下一轮迭代
                answer = new_draft
                used = new_used
                citations = new_citations

        # === Step 5: OUTPUT ===
        return ReasoningOut(
            answer=answer,
            evidence_used=[hits[i] for i in sorted(used) if 0 <= i < len(hits)],
            steps=[
                {"plan": "\n".join(steps)},
                {"evidence_map": step_evidences},
                {"citations": citations},
                {"drafts": drafts, "votes": votes},
                {"refine_rounds": refine_rounds},
            ],
            model="planner+synth+react",
        )