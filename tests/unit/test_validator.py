from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from app.models.responses import Change, TailorResult
from app.worker.llm.graph.nodes import _extract_json

# ---------------------------------------------------------------------------
# _extract_json — JSON extraction from LLM responses
# ---------------------------------------------------------------------------


def test_extract_json_plain():
    payload = {"key": "value", "number": 42}
    result = _extract_json(json.dumps(payload))
    assert result == payload


def test_extract_json_fenced_with_json_tag():
    payload = {"passed": True, "score": 78}
    text = f"```json\n{json.dumps(payload)}\n```"
    result = _extract_json(text)
    assert result == payload


def test_extract_json_fenced_without_tag():
    payload = {"modified_latex": r"\documentclass{a}\begin{document}\end{document}"}
    text = f"```\n{json.dumps(payload)}\n```"
    result = _extract_json(text)
    assert result == payload


def test_extract_json_embedded_in_prose():
    payload = {"ats_score": 72}
    text = f"Here is the result: {json.dumps(payload)} — hope that helps!"
    result = _extract_json(text)
    assert result == payload


def test_extract_json_no_json_raises():
    with pytest.raises(ValueError, match="No JSON object found"):
        _extract_json("This is plain text with no JSON at all.")


def test_extract_json_prefers_fenced_over_embedded():
    """Fenced JSON should be extracted before the fallback regex."""
    fenced_payload = {"source": "fenced"}
    other_payload = {"source": "embedded"}
    text = (
        f"Some text {json.dumps(other_payload)} more text "
        f"```json\n{json.dumps(fenced_payload)}\n```"
    )
    result = _extract_json(text)
    assert result == fenced_payload


# ---------------------------------------------------------------------------
# TailorResult — final output validation
# ---------------------------------------------------------------------------


def test_tailor_result_assembles_correctly(mock_tailor_output, mock_scoring_result):
    changes = [Change(**c) for c in mock_tailor_output["changes"]]
    result = TailorResult(
        modified_latex=mock_tailor_output["modified_latex"],
        changes=changes,
        ats_keywords_added=["Kubernetes", "distributed systems"],
        ats_keywords_missing=mock_scoring_result.remaining_gaps,
    )
    assert result.modified_latex.startswith(r"\documentclass")
    assert len(result.changes) == 2
    assert "Kubernetes" in result.ats_keywords_added
    assert "gRPC" in result.ats_keywords_missing


def test_tailor_result_accepts_empty_keyword_lists(mock_tailor_output):
    changes = [Change(**c) for c in mock_tailor_output["changes"]]
    result = TailorResult(
        modified_latex=mock_tailor_output["modified_latex"],
        changes=changes,
        ats_keywords_added=[],
        ats_keywords_missing=[],
    )
    assert result.ats_keywords_added == []
    assert result.ats_keywords_missing == []


def test_tailor_result_missing_required_field_raises():
    with pytest.raises(ValidationError):
        TailorResult(
            # modified_latex omitted
            changes=[],
            ats_keywords_added=[],
            ats_keywords_missing=[],
        )


def test_change_all_fields_required():
    with pytest.raises(ValidationError):
        Change(section="skills", action="added")  # missing content and reason


def test_change_valid():
    ch = Change(
        section="skills",
        action="added",
        content="Added Kubernetes",
        reason="Required by the job description",
    )
    assert ch.section == "skills"
    assert ch.action == "added"
