from __future__ import annotations

import structlog

from app.worker.llm.graph.state import GraphState

logger = structlog.get_logger(__name__)

# MAX_RETRIES is 2 rather than unlimited for two reasons:
#
# Cost: each retry is a full LLM round-trip on a potentially long prompt.
# Three total attempts (1 original + 2 retries) already triples the per-job
# cost. Unlimited retries on a systematic prompt failure would loop forever
# and bill indefinitely until the task is killed by a Celery time limit.
#
# Signal: if the tailor cannot pass validation in three attempts, this is
# almost certainly a prompt or model capability issue — not a randomness
# issue that more retries would fix. The right fix is improving the prompt,
# not spinning endlessly.
#
# Why we proceed to scorer even at max retries rather than failing hard:
# The tailor produced *something* — a modified resume that may still be
# useful to the user even if the validator had objections. Returning a
# result with a low confidence score is more useful than returning an error:
# the user gets their resume back and the low score signals manual review.
# A hard error gives the user nothing and forces them to resubmit.
#
# Why error state routes to END rather than retrying:
# Node exceptions are categorically different from validation failures.
# A validation failure means "the output was produced but has quality issues —
# try again with different instructions." An exception in the analyzer means
# "we have no analysis object at all." Retrying the tailor without analysis
# would mean sending it an empty brief — the output would be worse, not
# better. The only honest thing to do is surface the error immediately.

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
