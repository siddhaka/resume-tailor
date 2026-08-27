from __future__ import annotations

import structlog

from app.worker.llm.graph.state import GraphState

logger = structlog.get_logger(__name__)

# Cap retries so a systematic failure can't loop (and bill) forever; past the
# cap we still hand the last output to the scorer rather than failing hard, so
# the user gets a result plus a low score signalling manual review. Node
# exceptions route straight to END — retrying the tailor with no analysis would
# only produce worse output.
MAX_RETRIES = 2


def should_retry_tailor(state: GraphState) -> str:
    """Route after the validator node.

    Returns one of three node names:
    - "tailor"         — validation failed and retries remain
    - "scorer"         — validation passed, or max retries exhausted
    - "end_with_error" — an upstream node raised an unrecoverable exception
    """
    if state.get("error"):
        return "end_with_error"

    if state.get("validation_passed"):
        return "scorer"

    retry_count = state.get("tailor_retry_count", 0)
    if retry_count >= MAX_RETRIES:
        logger.warning(
            "retry.max_reached",
            retry_count=retry_count,
            max_retries=MAX_RETRIES,
            action="proceeding_to_scorer_with_last_output",
        )
        return "scorer"

    return "tailor"
