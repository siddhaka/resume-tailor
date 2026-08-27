from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from app.models.responses import AnalysisResult, Change, ScoringResult
from app.worker.llm.graph.nodes import (
    analyzer_node,
    scorer_node,
    tailor_node,
    validator_node,
)
from app.worker.llm.graph.state import GraphState

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_RESUME = r"""\documentclass[11pt]{article}
\begin{document}
\section*{Skills}Python, FastAPI, Docker
\end{document}"""

_JD = "We need a Python engineer with Kubernetes and Docker experience for microservices."

_ANALYSIS_JSON = json.dumps(
    {
        "keywords_in_jd_not_in_resume": ["Kubernetes"],
        "relevant_sections": ["skills"],
        "experience_level_signal": "mid-level",
        "tone_signal": "technical",
        "priority_changes": ["Add Kubernetes to skills"],
    }
)

_MODIFIED_LATEX = r"""\documentclass[11pt]{article}
\begin{document}
\section*{Skills}Python, FastAPI, Docker, Kubernetes
\end{document}"""

_TAILOR_JSON = json.dumps(
    {
        "modified_latex": _MODIFIED_LATEX,
        "changes": [
            {
                "section": "skills",
                "action": "added",
                "content": "Added Kubernetes",
                "reason": "Required by JD",
            }
        ],
    }
)

_VALIDATION_JSON = json.dumps(
    {
        "passed": True,
        "fabrication_detected": False,
        "latex_structure_intact": True,
        "changes_verified": True,
        "feedback": None,
    }
)

_SCORING_JSON = json.dumps(
    {
        "ats_score": 75,
        "ats_score_delta": 10,
        "confidence": 0.8,
        "remaining_gaps": [],
    }
)


def _make_state(**overrides) -> GraphState:
    base: GraphState = {
        "resume_latex": _RESUME,
        "job_description": _JD,
        "analysis": None,
        "modified_latex": None,
        "changes": None,
        "validation_passed": None,
        "validation_feedback": None,
        "tailor_retry_count": 0,
        "scoring": None,
        "error": None,
    }
    base.update(overrides)  # type: ignore[typeddict-item]
    return base


def _mock_response(content: str) -> MagicMock:
    """Return a fake ChatAnthropic response with the given content string."""
    response = MagicMock()
    response.content = content
    return response


# ---------------------------------------------------------------------------
# Analyzer node
# ---------------------------------------------------------------------------


def test_analyzer_node_returns_analysis():
    with patch("app.worker.llm.graph.nodes.llm") as mock_llm:
        mock_llm.invoke.return_value = _mock_response(_ANALYSIS_JSON)
        result = analyzer_node(_make_state())

    assert "analysis" in result
    assert isinstance(result["analysis"], AnalysisResult)
    assert "Kubernetes" in result["analysis"].keywords_in_jd_not_in_resume


def test_analyzer_node_handles_llm_error():
    with patch("app.worker.llm.graph.nodes.llm") as mock_llm:
        mock_llm.invoke.side_effect = RuntimeError("API timeout")
        result = analyzer_node(_make_state())

    assert "error" in result
    assert "Analyzer failed" in result["error"]


def test_analyzer_node_handles_invalid_json():
    with patch("app.worker.llm.graph.nodes.llm") as mock_llm:
        mock_llm.invoke.return_value = _mock_response("not json at all")
        result = analyzer_node(_make_state())

    assert "error" in result
    assert "Analyzer failed" in result["error"]


# ---------------------------------------------------------------------------
# Tailor node
# ---------------------------------------------------------------------------


def test_tailor_node_returns_modified_latex(mock_analysis_result):
    with patch("app.worker.llm.graph.nodes.llm") as mock_llm:
        mock_llm.invoke.return_value = _mock_response(_TAILOR_JSON)
        result = tailor_node(_make_state(analysis=mock_analysis_result))

    assert "modified_latex" in result
    assert result["modified_latex"] == _MODIFIED_LATEX
    assert isinstance(result["changes"][0], Change)
    assert result["tailor_retry_count"] == 1


def test_tailor_node_includes_feedback_on_retry(mock_analysis_result):
    """The validation_feedback string must appear in the human message sent on retry."""
    feedback = "The skills section still missing Kubernetes — add it explicitly."

    with patch("app.worker.llm.graph.nodes.llm") as mock_llm:
        mock_llm.invoke.return_value = _mock_response(_TAILOR_JSON)
        tailor_node(
            _make_state(
                analysis=mock_analysis_result,
                validation_feedback=feedback,
                tailor_retry_count=1,
            )
        )

    # llm.invoke is called with [SystemMessage, HumanMessage]
    call_args = mock_llm.invoke.call_args
    messages = call_args[0][0]
    human_content = messages[1].content  # HumanMessage is second
    assert feedback in human_content


def test_tailor_node_skips_on_error_state():
    """tailor_node must return early without touching the LLM when state has error."""
    with patch("app.worker.llm.graph.nodes.llm") as mock_llm:
        result = tailor_node(_make_state(error="upstream analyzer failed"))
        mock_llm.invoke.assert_not_called()

    assert result["error"] == "upstream analyzer failed"


def test_tailor_node_increments_retry_count(mock_analysis_result):
    with patch("app.worker.llm.graph.nodes.llm") as mock_llm:
        mock_llm.invoke.return_value = _mock_response(_TAILOR_JSON)
        result = tailor_node(_make_state(analysis=mock_analysis_result, tailor_retry_count=1))

    assert result["tailor_retry_count"] == 2


# ---------------------------------------------------------------------------
# Validator node
# ---------------------------------------------------------------------------


def test_validator_node_returns_passed(mock_analysis_result):
    state = _make_state(
        analysis=mock_analysis_result,
        modified_latex=_MODIFIED_LATEX,
        changes=[Change(section="skills", action="added", content="Kubernetes", reason="JD req")],
    )
    with patch("app.worker.llm.graph.nodes.llm") as mock_llm:
        mock_llm.invoke.return_value = _mock_response(_VALIDATION_JSON)
        result = validator_node(state)

    assert result["validation_passed"] is True
    assert result["validation_feedback"] is None


def test_validator_node_returns_failed_with_feedback(mock_analysis_result):
    failed_json = json.dumps(
        {
            "passed": False,
            "fabrication_detected": True,
            "latex_structure_intact": True,
            "changes_verified": False,
            "feedback": "Kubernetes was not in the original resume",
        }
    )
    state = _make_state(
        analysis=mock_analysis_result,
        modified_latex=_MODIFIED_LATEX,
        changes=[],
    )
    with patch("app.worker.llm.graph.nodes.llm") as mock_llm:
        mock_llm.invoke.return_value = _mock_response(failed_json)
        result = validator_node(state)

    assert result["validation_passed"] is False
    assert result["validation_feedback"] == "Kubernetes was not in the original resume"


def test_validator_node_skips_on_error_state():
    with patch("app.worker.llm.graph.nodes.llm") as mock_llm:
        result = validator_node(_make_state(error="tailor crashed"))
        mock_llm.invoke.assert_not_called()

    assert result["validation_passed"] is False


def test_validator_node_returns_failed_on_llm_error():
    """A validator exception must surface as a failed validation, not a hard error."""
    state = _make_state(modified_latex=_MODIFIED_LATEX, changes=[])
    with patch("app.worker.llm.graph.nodes.llm") as mock_llm:
        mock_llm.invoke.side_effect = ConnectionError("timeout")
        result = validator_node(state)

    assert result["validation_passed"] is False
    assert result["validation_feedback"] is not None
    assert "error" not in result  # must NOT set graph error — tailor should retry


# ---------------------------------------------------------------------------
# Scorer node
# ---------------------------------------------------------------------------


def test_scorer_node_returns_scoring_result(mock_analysis_result):
    state = _make_state(
        analysis=mock_analysis_result,
        modified_latex=_MODIFIED_LATEX,
        validation_passed=True,
    )
    with patch("app.worker.llm.graph.nodes.llm") as mock_llm:
        mock_llm.invoke.return_value = _mock_response(_SCORING_JSON)
        result = scorer_node(state)

    assert "scoring" in result
    assert isinstance(result["scoring"], ScoringResult)
    assert result["scoring"].ats_score == 75


def test_scorer_node_returns_defaults_on_error(mock_analysis_result):
    """Scorer failures must NOT propagate as graph errors — scoring is informational."""
    state = _make_state(analysis=mock_analysis_result, modified_latex=_MODIFIED_LATEX)
    with patch("app.worker.llm.graph.nodes.llm") as mock_llm:
        mock_llm.invoke.side_effect = RuntimeError("LLM unreachable")
        result = scorer_node(state)

    assert "scoring" in result
    assert result["scoring"].ats_score == 0
    assert result["scoring"].confidence == 0.0
    assert "error" not in result  # scorer errors must NEVER set graph error
