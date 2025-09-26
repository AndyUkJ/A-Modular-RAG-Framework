# app/modules/verification/impl_rules_llm.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import json
import re

from app.core.interfaces import VerifierAgent
from app.core.dto import VerifyIn, VerifyOut, Hit
from app.core.llm_router import LLMRouter
from app.telemetry.sinks import TelemetrySink, span, record_metrics

from enum import Enum


class StatusDetail(str, Enum):
    """
    VerifierAgent 的细粒度状态分类（补充在 status="pass"/"fail" 之上）

    ✅ 枚举值说明：
        FAIL
            - 明确失败（存在矛盾或缺失关键证据）
            - 场景：答案与证据冲突，或关键事实无支撑
            - 推荐动作：
                * Orchestrator：触发“回退检索 → 再推理 → 再验证”闭环
                * UI：标红展示，提示“答案存在错误或缺失关键证据”

        HIGH_CONF_PASS
            - 高置信度通过（核心事实直接被证据有力支持）
            - 场景：核心事实被直接证据强支撑，无明显噪声
            - 推荐动作：
                * Orchestrator：直接接受答案，进入下游环节
                * UI：绿色展示，提示“验证通过（高置信度）”

        LOW_CONF_PASS
            - 低置信度通过（部分依赖间接证据，或存在噪声/引用不足）
            - 场景：部分证据间接支持，或引用覆盖不足，或存在冗余/噪声
            - 推荐动作：
                * Orchestrator：可选择接受，但建议触发二次 claim-check 或再检索补充
                * UI：黄色展示，提示“验证通过，但存在不确定性/噪声”

        UNKNOWN_PASS
            - 模糊通过（未检测到矛盾，但支持力度不足，不足以高置信度通过）
            - 场景：证据不矛盾，但整体覆盖不足，自洽性低
            - 推荐动作：
                * Orchestrator：进入人工复核队列，或触发 claim-check / 重新验证
                * UI：灰色展示，提示“验证结果不确定，需人工确认”

    ⚙️ 调用方使用建议：
        - 逻辑分支时避免写死字符串，统一用 StatusDetail 枚举：
            if status_detail is StatusDetail.HIGH_CONF_PASS: ...
        - UI 渲染可用颜色映射：
            FAIL → 红色，HIGH_CONF_PASS → 绿色，LOW_CONF_PASS → 黄色，UNKNOWN_PASS → 灰色
        - Orchestrator 策略：
            FAIL → 回退检索
            HIGH_CONF_PASS → 接受
            LOW_CONF_PASS → 接受+补检索
            UNKNOWN_PASS → 人工复核/二次验证
    """

    FAIL = "fail"
    HIGH_CONF_PASS = "high_conf_pass"
    LOW_CONF_PASS = "low_conf_pass"
    UNKNOWN_PASS = "unknown_pass"

# ---------------- helpers ----------------

def _coerce_text(out: Any) -> str:
    """尽力从多种返回结构中抽出文本内容"""
    if out is None:
        return ""
    if isinstance(out, str):
        return out
    if isinstance(out, dict):
        t = out.get("text")
        if isinstance(t, str):
            return t
        if isinstance(t, dict):
            if isinstance(t.get("text"), str):
                return t["text"]
            c = t.get("content")
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                for item in c:
                    if isinstance(item, dict) and isinstance(item.get("text"), str):
                        return item["text"]
        msg = out.get("message")
        if isinstance(msg, dict):
            c = msg.get("content")
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                for item in c:
                    if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                        return item["text"]
        choices = out.get("choices")
        if isinstance(choices, list) and choices:
            ch0 = choices[0]
            if isinstance(ch0, dict):
                if isinstance(ch0.get("text"), str):
                    return ch0["text"]
                msg0 = ch0.get("message")
                if isinstance(msg0, dict) and isinstance(msg0.get("content"), str):
                    return msg0["content"]
    return ""


def _safe_json_parse(s: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
        return None


def _extract_citation_ids(answer: str) -> List[int]:
    """从答案中提取 [#k] 引用编号"""
    if not answer:
        return []
    ids: List[int] = []
    for m in re.finditer(r"\[#(\d+)\]", answer):
        try:
            ids.append(int(m.group(1)))
        except Exception:
            pass
    return ids


def _mk_evidence_block(evs: List[Hit]) -> str:
    """把命中证据整理成 LLM 可读块，并固定编号"""
    lines = []
    for i, h in enumerate(evs, 1):
        meta = h.meta or {}
        doc = str(meta.get("doc") or meta.get("title") or "")
        sid = str(meta.get("sent_id") or meta.get("sid") or "")
        text = str(meta.get("text") or "")
        text = text.replace('"', '“')
        lines.append(f"[#{i}] (doc={doc}, sent_id={sid}) \"{text}\"")
    return "\n".join(lines)


def _bounded(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(v)))


def _compute_hallucination_risk(verdict: str, consistency_score: float) -> float:
    """
    简单映射：被判 contradicted → 高风险；insufficient → 中风险；supported → 低风险
    同时考虑一致性分，分数越低风险越高。
    """
    v = (1.0 - consistency_score)
    if verdict == "contradicted":
        base = 0.9
    elif verdict == "insufficient":
        base = 0.6
    else:
        base = 0.2
    return _bounded(0.5 * base + 0.5 * v)


def _vote_majority(items: List[str]) -> Tuple[str, float]:
    """返回众数与占比"""
    if not items:
        return "", 0.0
    from collections import Counter
    c = Counter(items)
    k, n = c.most_common(1)[0]
    return k, n / max(1, len(items))


def _map_to_fine_verdict(core_supported: bool, core_missing: bool, contradicted: bool,
                         noisy: bool, agreement_rate: float, core_indirect: bool = False) -> str:
    """
    Verdict 逻辑优化：
    - contradicted (explicit) → FAIL-CONTRADICTED
    - unsupported → PARTIAL
    - indirect-only → PARTIAL
    - noisy → PASS-WITH-NOISE
    - supported clean → PASS
    """
    if contradicted and not core_indirect:  # 只在有明确反例时 fail
        return "FAIL-CONTRADICTED"
    if agreement_rate < 0.5:
        return "INCONCLUSIVE"
    if core_supported:
        if core_indirect and not core_missing:
            return "PARTIAL"
        return "PASS-WITH-NOISE" if noisy else "PASS"
    if core_missing:  # ✅ 改这里：缺失证据 → PARTIAL 而不是 fail
        return "PARTIAL"
    return "PARTIAL"

# ---------------- 可选 Claim-Check 接口 ----------------
ExternalClaimRetriever = Callable[[str, List[str], str], List[Hit]]


# ---------------- Impl: 规则 + LLM ----------------

@dataclass
class VerifierAgentRulesLLM(VerifierAgent):
    """
    结合「启发式规则」与「LLM 事实一致性评估」的验证器（增强版）
    """

    router: LLMRouter
    sink: Optional[TelemetrySink] = None

    # 规则阈值
    min_citations: int = 1
    min_coverage_ratio: float = 0.2
    require_citation_in_answer: bool = True

    # LLM 参数
    temperature: float = 0.0
    ctx: int = 64000

    # 打分权重
    weight_rules: float = 0.4
    weight_llm: float = 0.6
    weight_risk: float = 0.0
    decision_threshold: float = 0.6

    # 自洽性配置
    sc_runs: int = 5
    sc_agreement_threshold: float = 0.7

    # Claim-Check
    enable_claim_check: bool = True
    external_claim_retriever: Optional[ExternalClaimRetriever] = None
    max_claims: int = 5

    # ---- 规则检查 ----
    def _rule_check(self, question: str, answer: str, evidence: List[Hit]) -> Tuple[float, List[str], Dict[str, Any]]:
        issues: List[str] = []
        score = 1.0
        diag: Dict[str, Any] = {}

        if not (answer and answer.strip()):
            issues.append("Empty answer.")
            return 0.0, issues, diag

        if self.require_citation_in_answer:
            cited_ids = _extract_citation_ids(answer)
            if len(cited_ids) == 0:
                issues.append("No inline citations like [#k] found in answer.")
                score *= 0.7
            if self.min_citations > 0 and len(cited_ids) < self.min_citations:
                issues.append(f"Not enough citations: found {len(cited_ids)}, require ≥ {self.min_citations}.")
                score *= 0.85

            uniq = set(i for i in cited_ids if 1 <= i <= len(evidence))
            coverage = (len(uniq) / max(1, len(evidence))) if evidence else 0.0
            diag["coverage"] = coverage
            if coverage < self.min_coverage_ratio:
                issues.append(f"Citation coverage low: {coverage:.2f} < {self.min_coverage_ratio:.2f}.")
                score *= 0.9

        if len(evidence) == 0:
            issues.append("No evidence provided.")
            score *= 0.8

        if len(answer) > 400 and (len(_extract_citation_ids(answer)) == 0):
            issues.append("Long answer without citations.")
            score *= 0.9

        return _bounded(score), issues, diag

    # ---- 单轮 LLM ----
    def _llm_fact_check_once(
        self, question: str, answer: str, evidence: List[Hit], trace_id: str
    ) -> Tuple[float, Dict[str, Any]]:
        citations = _mk_evidence_block(evidence)
        sys = (
            "You are a strict but fair fact-checker.\n"
            "- Judge only using the GIVEN citations.\n"
            "- Extract CORE facts vs SECONDARY facts from the answer.\n"
            "- Distinguish citations into three groups:\n"
            "    * valid_citations: relevant and supportive\n"
            "    * noisy_citations: irrelevant but not contradicting\n"
            "    * misleading_citations: irrelevant AND misleading/contradicting\n"
            "- IMPORTANT:\n"
            "    * If evidence does NOT mention the fact, mark as \"unsupported\".\n"
            "    * If evidence IMPLIES the fact indirectly (e.g. nationality inferred from birthplace), mark as \"indirect\".\n"
            "    * Mark as \"contradicted\" ONLY if evidence EXPLICITLY states the opposite.\n"
            "- FAIL only if CRITICAL/CORE facts are CONTRADICTED (explicit conflict).\n"
            "- If a fact is IMPLIED but not directly stated, mark it as \"indirect\", not \"unsupported\".\n"
            "Return pure JSON keys with mandatory fields:\n"
            "{\n"
            '  "core_facts": {...},\n'
            '  "secondary_facts": {...},\n'
            '  "valid_citations": [...],\n'
            '  "noisy_citations": [...],\n'
            '  "misleading_citations": [...],\n'
            '  "verdict": "supported|partial|refuted|insufficient",\n'
            '  "score": 0.0 ~ 1.0\n'
            "}\n"
        )

        usr = f"Question:\n{question}\n\nAnswer:\n{answer}\n\nCitations:\n{citations}\n"
        out = self.router.complete(
            module="VerifierAgent",
            purpose="factcheck",
            prompt=f"{sys}\n\n{usr}",
            require={
                "context_window": self.ctx,
                "temperature": self.temperature,
                "trace_id": trace_id,
            },
        )

        text = _coerce_text(out)
        data = _safe_json_parse(text) or {}

        # ✅ 保证 score 一定有值
        verdict = (data.get("verdict") or "").lower()
        raw_score = data.get("score")
        if isinstance(raw_score, (int, float)):
            score = _bounded(float(raw_score))
        else:
            # verdict → score 兜底映射
            if verdict == "supported":
                score = 0.9
            elif verdict in ("partial", "insufficient"):
                score = 0.5
            elif verdict == "refuted":
                score = 0.1
            else:
                score = 0.3
            data["score"] = score

        print(f"[Verifier] fact-check raw LLM text: {text[:200]}")
        return score, data

    # ---- 自洽性 ----
    def _llm_fact_check(self, question: str, answer: str, evidence: List[Hit], trace_id: str) -> Tuple[float, List[str], Dict[str, Any]]:
        scores: List[float] = []
        verdicts: List[str] = []
        runs_data: List[Dict[str, Any]] = []

        for _ in range(max(1, self.sc_runs)):
            s, d = self._llm_fact_check_once(question, answer, evidence, trace_id)
            scores.append(_bounded(s))
            verdicts.append(str(d.get("verdict") or "insufficient"))
            runs_data.append(d)

        maj_verdict, agreement = _vote_majority(verdicts)
        avg_score = _bounded(sum(scores) / max(1, len(scores)))

        issues, used_union, noisy_union = [], [], []
        facts_agg = {"core": [], "secondary": []}
        for d in runs_data:
            issues.extend([str(x) for x in (d.get("issues") or [])])
            for u in (d.get("used") or []):
                if u not in used_union:
                    used_union.append(u)
            for n in (d.get("noisy_citations") or []):
                if n not in noisy_union:
                    noisy_union.append(n)
            for k in ("core", "secondary"):
                for item in (d.get("facts") or {}).get(k, [])[:8]:
                    if isinstance(item, dict):
                        facts_agg[k].append(item)

        diag = {
            "verdict": maj_verdict,
            "agreement_rate": float(agreement),
            "used": used_union,
            "noisy": noisy_union,
            "facts": facts_agg,
            "runs": len(runs_data),
            "runs_raw": runs_data[:3],
        }
        return avg_score, issues, diag

    # ---- Claim-Check ----
    def _claim_check(self, question: str, answer: str, trace_id: str, base_facts: Dict[str, Any]) -> Dict[str, Any]:
        claims: List[str] = []
        for k in ("core", "secondary"):
            for item in (base_facts.get(k) or []):
                fact = (item.get("fact") or "").strip()
                if fact:
                    claims.append(fact)
        claims = claims[: self.max_claims]

        results: List[Dict[str, Any]] = []
        for c in claims:
            results.append({"claim": c, "label": "not_enough_info", "rationale": "", "evidence": []})
        return {"results": results, "summary": {"supported": 0, "refuted": 0, "not_enough_info": len(results)}}

    # ---- 主接口 ----
    def verify(self, req: VerifyIn) -> VerifyOut:
        question = getattr(req, "question", "") or getattr(req, "query", "") or ""
        answer = getattr(req, "answer", "") or ""
        evidence: List[Hit] = list(getattr(req, "evidence", None) or getattr(req, "evidence_used", None) or [])
        trace_id = getattr(req, "trace_id", None) or "trace-verify"

        # 1) 规则
        if self.sink:
            with span("Verifier/Rules", self.sink, trace_id):
                r_score, r_issues, r_diag = self._rule_check(question, answer, evidence)
        else:
            r_score, r_issues, r_diag = self._rule_check(question, answer, evidence)

        # 2) LLM 核验
        if self.sink:
            with span("Verifier/LLM", self.sink, trace_id):
                l_score, l_issues, l_diag = self._llm_fact_check(question, answer, evidence, trace_id)
        else:
            l_score, l_issues, l_diag = self._llm_fact_check(question, answer, evidence, trace_id)

        maj_verdict = str(l_diag.get("verdict") or "insufficient")
        agreement_rate = float(l_diag.get("agreement_rate") or 0.0)

        # 3) Claim-Check
        claim_diag: Dict[str, Any] = {}
        if self.enable_claim_check:
            facts = l_diag.get("facts") or {}
            if self.sink:
                with span("Verifier/ClaimCheck", self.sink, trace_id):
                    claim_diag = self._claim_check(question, answer, trace_id, facts)
            else:
                claim_diag = self._claim_check(question, answer, trace_id, facts)

        # 4) 打分
        coverage_score = float(r_diag.get("coverage") or 0.0)
        consistency_score = float(l_score)

        # 次要事实的惩罚：若有 unsupported/contradicted，则轻度降低一致性分
        sec_items = (l_diag.get("facts") or {}).get("secondary") or []
        if any(it.get("status") in ("unsupported", "contradicted") for it in sec_items):
            consistency_score *= 0.9

        hallucination_risk = _compute_hallucination_risk(maj_verdict, consistency_score)

        # ✅ 使用 r_score 保留原始逻辑
        final_score = (
            self.weight_rules * float(r_score) +
            self.weight_llm * consistency_score +
            self.weight_risk * (1.0 - hallucination_risk)
        )
        final_score = _bounded(final_score)

        core_items = (l_diag.get("facts") or {}).get("core") or []
        core_supported = any(it.get("status") in ("supported", "indirect") for it in core_items) if core_items else (maj_verdict == "supported")
        core_missing = any(it.get("status") in ("unsupported",) for it in core_items)
        contradicted = (maj_verdict == "contradicted") or any(it.get("status") == "contradicted" for it in core_items)
        noisy = bool(l_diag.get("noisy_citations"))
        core_indirect = any(it.get("status") == "indirect" for it in core_items)

        fine_verdict = _map_to_fine_verdict(core_supported, core_missing, contradicted, noisy, agreement_rate, core_indirect=core_indirect)

        # ✅ Verdict 主导，通过分数调节推荐动作，而不是直接 fail
        ok = fine_verdict in ("PASS", "PASS-WITH-NOISE", "PARTIAL")
        if not ok:
            status = "fail"
            status_detail = StatusDetail.FAIL
        elif fine_verdict == "PASS":
            status = "pass"
            status_detail = StatusDetail.HIGH_CONF_PASS
        elif fine_verdict in ("PASS-WITH-NOISE", "PARTIAL"):
            status = "pass"
            status_detail = StatusDetail.LOW_CONF_PASS
        else:
            status = "pass"
            status_detail = StatusDetail.UNKNOWN_PASS

        issues: List[str] = []
        issues.extend(r_issues)
        issues.extend(l_issues)

        findings: List[Dict[str, Any]] = []
        if contradicted:
            findings.append({"type": "contradiction", "severity": "high"})
        if fine_verdict == "PASS-WITH-NOISE" and noisy:
            findings.append({"type": "redundant_citation", "severity": "low"})
        if fine_verdict == "PARTIAL":
            findings.append({"type": "partial_support", "severity": "medium"})
        if fine_verdict == "INCONCLUSIVE":
            findings.append({"type": "inconclusive", "severity": "medium"})

        diagnostics = {
            "rule_score": r_score,
            "llm_score": consistency_score,
            "rule_diag": r_diag,
            "llm_diag": l_diag,
            "claim_check": claim_diag,
            "final_score_formula": {
                "weights": {"rules": self.weight_rules, "llm": self.weight_llm, "risk": self.weight_risk},
                "threshold": self.decision_threshold
            },
            "citations": {
                "valid": l_diag.get("valid_citations") or [],
                "noisy": l_diag.get("noisy_citations") or [],
                "misleading": l_diag.get("misleading_citations") or [],
            },
            "status_detail": status_detail.value,
            "status_detail_label": _status_label(status_detail),
        }

        if self.sink:
            retry_round = getattr(req, "retry_round", 0)
            record_metrics(
                self.sink,
                trace_id,
                verifier={
                    "coverage_score": coverage_score,
                    "consistency_score": consistency_score,
                    "hallucination_risk": hallucination_risk,
                    "final_score": final_score,
                    "verdict": fine_verdict,
                    "agreement_rate": agreement_rate,
                    "issues_count": len(issues),
                    "status": status,
                    "status_detail": status_detail.value,
                    "retry_round": retry_round,  # ✅ 新增字段
                }
            )

        if status_detail == StatusDetail.FAIL:
            if fine_verdict == "FAIL-CONTRADICTED":
                recommended = "Reject and re-run"
            else:
                recommended = "Retry retrieval / claim-check"
        elif status_detail == StatusDetail.LOW_CONF_PASS:
            if fine_verdict == "PASS-WITH-NOISE":
                recommended = "Accept; prune noisy citations"
            else:
                recommended = "Review recommended (low confidence)"
        elif status_detail == StatusDetail.UNKNOWN_PASS:
            recommended = "Review required (uncertain evidence)"
        elif status_detail == StatusDetail.HIGH_CONF_PASS:
            recommended = "Accept (high confidence)"
        else:
            recommended = "Accept."

        return VerifyOut(
            status=status,
            findings=findings,
            model="llm+rules",
            ok=ok,
            score=final_score,
            issues=issues,
            diagnostics={**diagnostics, "retry_round": retry_round},
            coverage_score=coverage_score,
            consistency_score=consistency_score,
            hallucination_risk=hallucination_risk,
            final_score=final_score,
            verdict=fine_verdict,
            self_consistency={
                "runs": int(l_diag.get("runs") or self.sc_runs),
                "agreement_rate": agreement_rate,
                "majority_verdict": maj_verdict,
            },
            recommended_action=recommended,
            status_detail=status_detail.value,
            status_detail_label=_status_label(status_detail),
        )
    
def _status_label(status: StatusDetail) -> str:
    """Map StatusDetail enum to human-readable English label"""
    mapping = {
        StatusDetail.FAIL: "Fail",
        StatusDetail.HIGH_CONF_PASS: "High Confidence Pass",
        StatusDetail.LOW_CONF_PASS: "Low Confidence Pass",
        StatusDetail.UNKNOWN_PASS: "Unknown Confidence Pass",
    }
    return mapping.get(status, "Unknown")