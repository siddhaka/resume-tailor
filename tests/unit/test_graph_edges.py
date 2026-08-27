from __future__ import annotations

from app.worker.llm.graph.edges import should_retry_tailor
from app.worker.llm.graph.state import GraphState


def make_state(**overrides) -> GraphState:
    """Create a GraphState with sensible defaults, overridden by kwargs."""
    base: GraphState = {
        "resume_latex": r"\documentclass{a}\begin{document}\end{document}",
        "job_description": "some job description text",
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


def test_routes_to_scorer_on_validation_pass():
    state = make_state(validation_passed=True, tailor_retry_count=1)
    assert should_retry_tailor(state) == "scorer"


def test_routes_to_tailor_on_validation_fail():
    state = make_state(validation_passed=False, tailor_retry_count=0)
    assert should_retry_tailor(state) == "tailor"


def test_routes_to_tailor_on_first_retry():
    """After 1 tailor call (count=1), a second attempt (retry 1) is still allowed."""
    state = make_state(validation_passed=False, tailor_retry_count=1)
    assert should_retry_tailor(state) == "tailor"


def test_routes_to_scorer_when_max_retries_reached():
    """After 2 tailor calls (count=2), we must not loop forever — proceed to scorer."""
    state = make_state(validation_passed=False, tailor_retry_count=2)
    assert should_retry_tailor(state) == "scorer"


def test_routes_to_scorer_on_second_retry_exhausted():
    """Equivalent to max_retries_reached; tests that the third attempt is not allowed."""
    state = make_state(validation_passed=False, tailor_retry_count=2)
    result = should_retry_tailor(state)
    assert result == "scorer"
    assert result != "tailor"


def test_routes_to_end_on_error():
    """Error state bypasses the retry loop entirely — retrying without analysis is useless."""
    state = make_state(error="Analyzer failed: API timeout", validation_passed=False)
    assert should_retry_tailor(state) == "end_with_error"


def test_error_takes_priority_over_validation_pass():
    """Even if validation somehow passed, an error routes to end_with_error."""
    state = make_state(error="some error", validation_passed=True)
    assert should_retry_tailor(state) == "end_with_error"


def test_routes_to_scorer_at_zero_retries_if_passed():
    """If validation passes on the first attempt, no retries needed."""
    state = make_state(validation_passed=True, tailor_retry_count=1)
    assert should_retry_tailor(state) == "scorer"
