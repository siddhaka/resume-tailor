from __future__ import annotations

from app.worker.llm.client import build_graph


def test_graph_compiles_without_error():
    graph = build_graph()
    assert graph is not None


def test_graph_has_expected_nodes():
    graph = build_graph()
    nodes = graph.get_graph().nodes
    assert "analyzer" in nodes
    assert "tailor" in nodes
    assert "validator" in nodes
    assert "scorer" in nodes


def test_graph_entry_point_is_analyzer():
    """The first real node (after __start__) must be 'analyzer'."""
    graph = build_graph()
    g = graph.get_graph()

    # LangGraph represents the entry point as edges from the reserved
    # __start__ node. We confirm the target is 'analyzer'.
    start_edges = [e for e in g.edges if e.source == "__start__"]
    assert len(start_edges) == 1, "Expected exactly one edge from __start__"
    assert start_edges[0].target == "analyzer"


def test_graph_scorer_leads_to_end():
    graph = build_graph()
    g = graph.get_graph()

    scorer_edges = [e for e in g.edges if e.source == "scorer"]
    assert len(scorer_edges) == 1
    assert scorer_edges[0].target == "__end__"


def test_graph_analyzer_leads_to_tailor():
    graph = build_graph()
    g = graph.get_graph()

    analyzer_edges = [e for e in g.edges if e.source == "analyzer"]
    assert len(analyzer_edges) == 1
    assert analyzer_edges[0].target == "tailor"


def test_graph_tailor_leads_to_validator():
    graph = build_graph()
    g = graph.get_graph()

    tailor_edges = [e for e in g.edges if e.source == "tailor"]
    assert len(tailor_edges) == 1
    assert tailor_edges[0].target == "validator"


def test_graph_validator_has_three_conditional_targets():
    """Validator must branch to: tailor (retry), scorer (pass/max), __end__ (error)."""
    graph = build_graph()
    g = graph.get_graph()

    validator_targets = {e.target for e in g.edges if e.source == "validator"}
    assert "tailor" in validator_targets
    assert "scorer" in validator_targets
    assert "__end__" in validator_targets
