from __future__ import annotations

from typing_extensions import TypedDict

from app.models.responses import AnalysisResult, Change, ScoringResult

# TypedDict, not a Pydantic model: LangGraph merges each node's returned dict
# into the running state, which fits a plain dict better than immutable models,
# while the annotations still catch field-name typos. Inputs stay untouched so
# the validator can diff original vs. modified; tailor_retry_count is an int so
# the conditional edge can compare it against MAX_RETRIES.


class GraphState(TypedDict):
    # Inputs — set at graph entry, never modified by any node
    resume_latex: str
    job_description: str

    # Analyzer node output
    analysis: AnalysisResult | None

    # Tailor node output — overwritten on each retry
    modified_latex: str | None
    changes: list[Change] | None

    # Validator node output
    validation_passed: bool | None
    validation_feedback: str | None
    tailor_retry_count: int

    # Scorer node output
    scoring: ScoringResult | None

    # Error tracking — set by any node on unrecoverable failure
    error: str | None
