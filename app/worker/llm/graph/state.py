from __future__ import annotations

from typing_extensions import TypedDict

from app.models.responses import AnalysisResult, Change, ScoringResult

# Why TypedDict rather than a Pydantic model
# ------------------------------------------
# LangGraph passes state between nodes as a plain dict and applies partial
# updates by merging the dict each node returns into the accumulated state.
# Pydantic models are immutable by default — assigning a new value to a field
# creates a new object rather than mutating in place, which conflicts with
# LangGraph's update-accumulation pattern. TypedDict gives us static type
# annotations (so mypy and editors catch field name typos) while remaining a
# plain dict at runtime that LangGraph can freely merge.
#
# Why inputs are never modified
# -----------------------------
# Both the tailor node and the validator node need the *original* resume to
# compare against — the tailor uses it as the source to edit from, and the
# validator diffs original vs. modified to check for fabrication. If any node
# were allowed to overwrite resume_latex, a buggy node could corrupt the
# baseline that every downstream node depends on, making it impossible to
# detect fabrication or measure improvement accurately.
#
# Why tailor_retry_count is int, not bool
# ----------------------------------------
# A bool would only distinguish "has retried" from "has not retried". We want
# up to MAX_RETRIES=2 retries, so we need a counter the conditional edge can
# compare against a threshold. The edge reads tailor_retry_count >= MAX_RETRIES
# to decide when to give up and proceed to scoring rather than looping again.


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
