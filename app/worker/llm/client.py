from __future__ import annotations

import time

import structlog
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.models.responses import TailorResult
from app.worker.llm.graph.edges import should_retry_tailor
from app.worker.llm.graph.nodes import (
    analyzer_node,
    scorer_node,
    tailor_node,
    validator_node,
)
from app.worker.llm.graph.state import GraphState

logger = structlog.get_logger(__name__)


def build_graph() -> CompiledStateGraph:
    """Build and compile the tailoring graph.

    analyzer -> tailor -> validator, then a conditional edge routes to a retry,
    the scorer, or END. Built per call rather than at module level so call_llm()
    stays self-contained and easy to test in isolation.
    """
    graph: StateGraph = StateGraph(GraphState)

    graph.add_node("analyzer", analyzer_node)
    graph.add_node("tailor", tailor_node)
    graph.add_node("validator", validator_node)
    graph.add_node("scorer", scorer_node)

    graph.set_entry_point("analyzer")

    graph.add_edge("analyzer", "tailor")
    graph.add_edge("tailor", "validator")

    graph.add_conditional_edges(
        "validator",
        should_retry_tailor,
        {
            "tailor": "tailor",
            "scorer": "scorer",
            "end_with_error": END,
        },
    )

    graph.add_edge("scorer", END)

    return graph.compile()


def call_llm(resume_latex: str, job_description: str) -> TailorResult:
    """Run the pipeline and return a structured TailorResult.

    The single entry point the Celery task calls — it knows nothing about the
    nodes or graph state.
    """
    graph = build_graph()

    initial_state: GraphState = {
        "resume_latex": resume_latex,
        "job_description": job_description,
        "analysis": None,
        "modified_latex": None,
        "changes": None,
        "validation_passed": None,
        "validation_feedback": None,
        "tailor_retry_count": 0,
        "scoring": None,
        "error": None,
    }

    log = logger.bind(pipeline="resume_tailor")
    start = time.monotonic()

    final_state: GraphState = graph.invoke(initial_state)  # type: ignore[assignment]

    elapsed = round(time.monotonic() - start, 2)
    retries = max(0, final_state.get("tailor_retry_count", 1) - 1)
    scoring = final_state.get("scoring")

    log.info(
        "pipeline.complete",
        elapsed_seconds=elapsed,
        tailor_retries=retries,
        validation_passed=final_state.get("validation_passed"),
        ats_score=scoring.ats_score if scoring else None,
        ats_score_delta=scoring.ats_score_delta if scoring else None,
    )

    if final_state.get("error"):
        raise RuntimeError(final_state["error"])

    if not final_state.get("modified_latex"):
        raise RuntimeError("Graph completed but modified_latex is empty")

    analysis = final_state.get("analysis")
    modified = final_state["modified_latex"]

    ats_keywords_added = (
        [k for k in analysis.keywords_in_jd_not_in_resume if k in modified]
        if analysis
        else []
    )
    ats_keywords_missing = scoring.remaining_gaps if scoring else []

    return TailorResult(
        modified_latex=modified,
        changes=final_state.get("changes") or [],
        ats_keywords_added=ats_keywords_added,
        ats_keywords_missing=ats_keywords_missing,
    )
