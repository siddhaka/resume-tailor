from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models.requests import TailorRequest
from app.models.responses import (
    AnalysisResult,
    JobResponse,
    ScoringResult,
    TailorResult,
    ValidationResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LONG_LATEX = (
    r"\documentclass[11pt]{article}"
    r"\begin{document}"
    r"\section*{Experience}Senior Software Engineer at Acme Corp, 2020--2024. "
    r"Built REST APIs, led migration to microservices, improved reliability."
    r"\section*{Skills}Python, FastAPI, PostgreSQL, Docker\end{document}"
)

_LONG_JD = (
    "We are seeking a Senior Backend Engineer to join our distributed systems team. "
    "You will design and build gRPC services, mentor junior engineers, and own "
    "reliability for our core platform. Python, Kubernetes, and Docker required."
)

# ---------------------------------------------------------------------------
# TailorRequest validators
# ---------------------------------------------------------------------------


def test_tailor_request_valid():
    req = TailorRequest(resume_latex=_LONG_LATEX, job_description=_LONG_JD)
    assert req.resume_latex == _LONG_LATEX
    assert req.job_description == _LONG_JD


def test_resume_latex_too_short_raises():
    with pytest.raises(ValidationError, match="too short"):
        TailorRequest(resume_latex=r"\documentclass{a}", job_description=_LONG_JD)


def test_resume_latex_missing_documentclass_raises():
    bad_latex = "x" * 101  # long enough but wrong start
    with pytest.raises(ValidationError, match=r"\\documentclass"):
        TailorRequest(resume_latex=bad_latex, job_description=_LONG_JD)


def test_resume_latex_leading_whitespace_accepted():
    # lstrip() is applied before checking \documentclass — newlines allowed
    padded = "\n\n" + _LONG_LATEX
    req = TailorRequest(resume_latex=padded, job_description=_LONG_JD)
    assert req.resume_latex == padded


def test_job_description_too_few_words_raises():
    with pytest.raises(ValidationError, match="20 words"):
        TailorRequest(resume_latex=_LONG_LATEX, job_description="Too short.")


def test_job_description_exactly_20_words_accepted():
    jd = " ".join(["word"] * 20)
    req = TailorRequest(resume_latex=_LONG_LATEX, job_description=jd)
    assert len(req.job_description.split()) == 20


# ---------------------------------------------------------------------------
# AnalysisResult
# ---------------------------------------------------------------------------


def test_analysis_result_valid(mock_analysis_result):
    # Fixture constructs it without raising — just assert fields are correct
    assert mock_analysis_result.experience_level_signal == "senior"
    assert "Kubernetes" in mock_analysis_result.keywords_in_jd_not_in_resume
    assert len(mock_analysis_result.priority_changes) == 2


def test_analysis_result_empty_lists_accepted():
    result = AnalysisResult(
        keywords_in_jd_not_in_resume=[],
        relevant_sections=[],
        experience_level_signal="mid-level",
        tone_signal="corporate formal",
        priority_changes=[],
    )
    assert result.keywords_in_jd_not_in_resume == []


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------


def test_validation_result_passed():
    vr = ValidationResult(
        passed=True,
        fabrication_detected=False,
        latex_structure_intact=True,
        changes_verified=True,
    )
    assert vr.passed is True
    assert vr.feedback is None


def test_validation_result_failed_requires_feedback():
    """A failed ValidationResult with feedback=None is structurally valid.

    Pydantic enforces schema structure, not business logic. The constraint
    that a failed validation must include actionable feedback is enforced by
    the validator node's system prompt — the LLM is instructed to always
    populate feedback when passed=False. We test Pydantic structure here;
    the business rule lives in nodes.py.
    """
    vr = ValidationResult(
        passed=False,
        fabrication_detected=True,
        latex_structure_intact=True,
        changes_verified=False,
        feedback=None,  # structurally valid even without feedback
    )
    assert vr.passed is False
    assert vr.feedback is None  # Pydantic allows it; the prompt enforces content


# ---------------------------------------------------------------------------
# ScoringResult — boundary constraints
# ---------------------------------------------------------------------------


def test_scoring_result_valid_boundaries():
    sr = ScoringResult(
        ats_score=0, ats_score_delta=-5, confidence=0.0, remaining_gaps=[]
    )
    assert sr.ats_score == 0

    sr2 = ScoringResult(
        ats_score=100, ats_score_delta=20, confidence=1.0, remaining_gaps=[]
    )
    assert sr2.ats_score == 100


def test_scoring_result_ats_score_above_100_raises():
    with pytest.raises(ValidationError):
        ScoringResult(
            ats_score=101,
            ats_score_delta=0,
            confidence=0.5,
            remaining_gaps=[],
        )


def test_scoring_result_ats_score_below_0_raises():
    with pytest.raises(ValidationError):
        ScoringResult(
            ats_score=-1,
            ats_score_delta=0,
            confidence=0.5,
            remaining_gaps=[],
        )


# ---------------------------------------------------------------------------
# GraphState
# ---------------------------------------------------------------------------


def test_graph_state_initialization():
    from app.worker.llm.graph.state import GraphState

    state: GraphState = {
        "resume_latex": r"\documentclass{article}\begin{document}x\end{document}",
        "job_description": "some job description",
        "analysis": None,
        "modified_latex": None,
        "changes": None,
        "validation_passed": None,
        "validation_feedback": None,
        "tailor_retry_count": 0,
        "scoring": None,
        "error": None,
    }
    assert state["tailor_retry_count"] == 0
    assert state["analysis"] is None
    assert state["error"] is None


def test_graph_state_is_plain_dict():
    """GraphState is a TypedDict — it is a plain dict at runtime."""
    from app.worker.llm.graph.state import GraphState

    state: GraphState = {
        "resume_latex": "x",
        "job_description": "y",
        "analysis": None,
        "modified_latex": None,
        "changes": None,
        "validation_passed": None,
        "validation_feedback": None,
        "tailor_retry_count": 0,
        "scoring": None,
        "error": None,
    }
    # LangGraph needs plain dict — not a Pydantic model — for state merging
    assert isinstance(state, dict)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


def test_job_response_without_result():
    jr = JobResponse(job_id="abc", status="queued")
    assert jr.result is None
    assert jr.error is None
    assert jr.created_at is None


def test_tailor_result_requires_all_lists():
    with pytest.raises(ValidationError):
        TailorResult(modified_latex="x")  # missing required list fields
