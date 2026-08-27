from __future__ import annotations

import json
import re

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from app.models.responses import (
    AnalysisResult,
    Change,
    ScoringResult,
    ValidationResult,
)
from app.worker.llm.graph.state import GraphState
from app.worker.llm.provider import get_llm

logger = structlog.get_logger(__name__)

# One shared client for all four nodes. The backend (Ollama or Anthropic) is
# chosen in get_llm(); nodes use only the BaseChatModel .invoke() contract.
llm = get_llm()


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of an LLM response.

    Handles plain JSON, ```json fenced blocks, and JSON embedded in prose.
    """
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))

    obj_match = re.search(r"\{.*\}", text, re.DOTALL)
    if obj_match:
        return json.loads(obj_match.group(0))

    raise ValueError(f"No JSON object found in LLM response: {text[:200]!r}")


# ---------------------------------------------------------------------------
# Analyzer node
# ---------------------------------------------------------------------------

_ANALYSIS_SCHEMA = """{
  "keywords_in_jd_not_in_resume": ["list of specific keyword strings"],
  "relevant_sections": ["experience", "skills", "summary", "projects"],
  "experience_level_signal": "senior | mid-level | entry-level",
  "tone_signal": "e.g. technical and precise | startup casual | corporate formal",
  "priority_changes": ["ordered list from highest to lowest impact"]
}"""


def analyzer_node(state: GraphState) -> dict:
    """Gap analysis between resume and JD; produces a brief, makes no edits.

    Analysis and editing are split across nodes so each gets a focused prompt.
    """
    log = logger.bind(node="analyzer")

    try:
        system = SystemMessage(
            content=(
                "You are a technical recruiter analyzing fit between a candidate's "
                "resume and a job description. Your job is to identify the gap — "
                "what the JD requires that the resume does not currently show.\n\n"
                "Return ONLY valid JSON matching the schema below. No prose, no "
                "markdown, no explanation — just the JSON object.\n\n"
                "Be specific. Generic observations like 'improve skills section' "
                "are not useful. Name concrete keywords and concrete changes.\n\n"
                f"Schema:\n{_ANALYSIS_SCHEMA}"
            )
        )
        human = HumanMessage(
            content=(
                f"RESUME LATEX:\n{state['resume_latex']}\n\n"
                f"JOB DESCRIPTION:\n{state['job_description']}"
            )
        )

        response = llm.invoke([system, human])
        raw = _extract_json(str(response.content))
        analysis = AnalysisResult.model_validate(raw)

        log.info(
            "analyzer.complete",
            keywords_found=len(analysis.keywords_in_jd_not_in_resume),
            priority_changes=len(analysis.priority_changes),
            experience_level=analysis.experience_level_signal,
        )
        return {"analysis": analysis}

    except Exception as e:
        log.exception("analyzer.failed", error=str(e))
        return {"error": f"Analyzer failed: {e}"}


# ---------------------------------------------------------------------------
# Tailor node
# ---------------------------------------------------------------------------

_TAILOR_SCHEMA = """{
  "modified_latex": "complete LaTeX source — every \\\\documentclass to \\\\end{document}",
  "changes": [
    {
      "section": "which resume section was changed",
      "action": "added | modified | reordered | removed",
      "content": "what specifically changed",
      "reason": "why this change improves alignment with the JD"
    }
  ]
}"""


def tailor_node(state: GraphState) -> dict:
    """Rewrite the resume from the analyzer's brief.

    On a retry, the validator's feedback is injected into the prompt so the
    model fixes the named issues. Increments the retry count each pass so the
    conditional edge can cap it.
    """
    log = logger.bind(
        node="tailor",
        retry=state.get("tailor_retry_count", 0),
    )

    if state.get("error"):
        return {"error": state["error"]}

    try:
        analysis = state["analysis"]
        analysis_brief = (
            f"Gap analysis findings:\n"
            f"Priority changes: {analysis.priority_changes}\n"
            f"Keywords to add: {analysis.keywords_in_jd_not_in_resume}\n"
            f"Relevant sections: {analysis.relevant_sections}\n"
            f"Experience level signal: {analysis.experience_level_signal}\n"
            f"Tone to match: {analysis.tone_signal}"
        )

        retry_instruction = ""
        if state.get("validation_feedback"):
            retry_instruction = (
                f"\n\nPREVIOUS ATTEMPT REJECTED — fix these specific issues:\n"
                f"{state['validation_feedback']}\n"
                "Address every point above before returning your response."
            )

        system = SystemMessage(
            content=(
                "You are an expert resume writer making targeted edits to improve "
                "keyword alignment with a specific job description.\n\n"
                "Rules:\n"
                "- Never fabricate experience, skills, or achievements not present "
                "in the original resume. Only surface and reframe what exists.\n"
                "- Preserve all LaTeX commands, environments, and structure exactly. "
                "Do not add, remove, or rename any LaTeX section commands.\n"
                "- Make the minimum number of changes that achieve maximum impact.\n"
                "- Return the COMPLETE modified LaTeX source — not a diff, not a "
                "snippet — the entire document from \\documentclass to \\end{document}.\n\n"
                f"Return ONLY valid JSON matching this schema:\n{_TAILOR_SCHEMA}"
            )
        )
        human = HumanMessage(
            content=(
                f"ORIGINAL RESUME LATEX:\n{state['resume_latex']}\n\n"
                f"JOB DESCRIPTION:\n{state['job_description']}\n\n"
                f"{analysis_brief}{retry_instruction}"
            )
        )

        response = llm.invoke([system, human])
        raw = _extract_json(str(response.content))

        modified_latex: str = raw["modified_latex"]
        changes = [Change(**c) for c in raw["changes"]]
        retry_count = state.get("tailor_retry_count", 0) + 1

        log.info("tailor.complete", changes_made=len(changes), retry_count=retry_count)
        return {
            "modified_latex": modified_latex,
            "changes": changes,
            "tailor_retry_count": retry_count,
        }

    except Exception as e:
        log.exception("tailor.failed", error=str(e))
        return {"error": f"Tailor failed: {e}"}


# ---------------------------------------------------------------------------
# Validator node
# ---------------------------------------------------------------------------

_VALIDATION_SCHEMA = """{
  "passed": true | false,
  "fabrication_detected": true | false,
  "latex_structure_intact": true | false,
  "changes_verified": true | false,
  "feedback": "null if passed=true; otherwise specific instructions on what to fix"
}"""


def validator_node(state: GraphState) -> dict:
    """Quality gate: check the rewrite for fabrication, structure, and accuracy.

    Failure routes back to the tailor with feedback; success routes to scoring.
    Exceptions become a failed validation (not a crash) so the tailor can retry.
    """
    log = logger.bind(node="validator")

    if state.get("error"):
        return {"validation_passed": False, "validation_feedback": state["error"]}

    try:
        changes_json = json.dumps(
            [c.model_dump() for c in (state.get("changes") or [])],
            indent=2,
        )

        system = SystemMessage(
            content=(
                "You are a strict quality reviewer for AI-generated resume edits. "
                "Check three things in order and return ONLY valid JSON.\n\n"
                "1. Fabrication check: does the modified resume contain any "
                "experience, skills, job titles, companies, dates, or achievements "
                "that are NOT present in the original resume?\n"
                "2. Structure check: are ALL LaTeX section commands from the original "
                "resume present and unchanged in the modified version?\n"
                "3. Changes accuracy: does EVERY Change object in the changes list "
                "correspond to an actual textual difference between original and "
                "modified LaTeX? Are there any changes that are claimed but not made, "
                "or changes made but not claimed?\n\n"
                "If any check fails, set passed=false and write specific, actionable "
                "feedback — exactly what needs to be fixed, not vague direction.\n\n"
                f"Schema:\n{_VALIDATION_SCHEMA}"
            )
        )
        human = HumanMessage(
            content=(
                f"ORIGINAL RESUME LATEX:\n{state['resume_latex']}\n\n"
                f"MODIFIED RESUME LATEX:\n{state.get('modified_latex', '')}\n\n"
                f"CHANGES LIST:\n{changes_json}"
            )
        )

        response = llm.invoke([system, human])
        raw = _extract_json(str(response.content))
        result = ValidationResult.model_validate(raw)

        log.info(
            "validator.complete",
            passed=result.passed,
            fabrication_detected=result.fabrication_detected,
            latex_structure_intact=result.latex_structure_intact,
            changes_verified=result.changes_verified,
        )
        return {
            "validation_passed": result.passed,
            "validation_feedback": result.feedback,
        }

    except Exception as e:
        log.exception("validator.error", error=str(e))
        # Never raise from here — turn the error into a retryable failure.
        return {
            "validation_passed": False,
            "validation_feedback": f"Validator error: {e} — retry tailoring",
        }


# ---------------------------------------------------------------------------
# Scorer node
# ---------------------------------------------------------------------------

_SCORING_SCHEMA = """{
  "ats_score": 0-100,
  "ats_score_delta": "integer — improvement over original (can be negative)",
  "confidence": 0.0-1.0,
  "remaining_gaps": ["keywords or skills the JD requires that could not be honestly added"]
}"""


def scorer_node(state: GraphState) -> dict:
    """Score the result (ATS score, delta, confidence, remaining gaps).

    Last node by design: a scoring failure returns zero-confidence defaults
    instead of an error, so the user still receives the tailored resume.
    """
    log = logger.bind(node="scorer")

    if state.get("error"):
        return {
            "scoring": ScoringResult(
                ats_score=0,
                ats_score_delta=0,
                confidence=0.0,
                remaining_gaps=["scoring unavailable — graph error"],
            )
        }

    try:
        analysis = state.get("analysis")
        analysis_context = (
            f"Original gap analysis:\n"
            f"Keywords missing: {analysis.keywords_in_jd_not_in_resume}\n"
            f"Priority changes requested: {analysis.priority_changes}"
            if analysis
            else "No gap analysis available."
        )

        system = SystemMessage(
            content=(
                "You are an ATS (Applicant Tracking System) analyst and resume "
                "quality expert. Score the tailored resume against the job description "
                "and return ONLY valid JSON.\n\n"
                "Calibration:\n"
                "- ats_score of 100 means every keyword and requirement matches "
                "perfectly — this almost never happens.\n"
                "- Most well-tailored resumes score 65-85.\n"
                "- ats_score_delta must reflect genuine improvement over the original "
                "resume, not how good the modified version is in isolation. It can be "
                "negative if the tailor made the resume worse.\n"
                "- confidence reflects your certainty in the tailoring quality overall, "
                "not just ATS score.\n\n"
                f"Schema:\n{_SCORING_SCHEMA}"
            )
        )
        human = HumanMessage(
            content=(
                f"ORIGINAL RESUME LATEX:\n{state['resume_latex']}\n\n"
                f"MODIFIED RESUME LATEX:\n{state.get('modified_latex', '')}\n\n"
                f"JOB DESCRIPTION:\n{state['job_description']}\n\n"
                f"{analysis_context}"
            )
        )

        response = llm.invoke([system, human])
        raw = _extract_json(str(response.content))
        result = ScoringResult.model_validate(raw)

        log.info(
            "scorer.complete",
            ats_score=result.ats_score,
            ats_score_delta=result.ats_score_delta,
            confidence=result.confidence,
        )
        return {"scoring": result}

    except Exception as e:
        # Scoring failure must not fail the job — return zeroed defaults.
        log.exception("scorer.failed", error=str(e))
        return {
            "scoring": ScoringResult(
                ats_score=0,
                ats_score_delta=0,
                confidence=0.0,
                remaining_gaps=["scoring unavailable"],
            )
        }
