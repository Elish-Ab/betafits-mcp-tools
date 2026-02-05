"""Tests for workflow routing helpers."""

from workflows.langgraph.orchestrator.graph import route_after_brain
from lib.state import WorkflowState
from langgraph.graph import END  # type: ignore


def test_route_after_brain_returns_code_generator() -> None:
    state: WorkflowState = {
        "message": "",
        "next_node": "code_generator",
        "code_generator_output": None,
        "pcf_parser_output": None,
        "error": None,
        "pcf_record_id": None,
        "pcf_table": None,
    }
    assert route_after_brain(state) == "code_generator"


def test_route_after_brain_returns_code_reviewer() -> None:
    state: WorkflowState = {
        "message": "review",
        "next_node": "code_reviewer",
        "code_generator_output": None,
        "pcf_parser_output": None,
        "error": None,
        "pcf_record_id": None,
        "pcf_table": None,
    }
    assert route_after_brain(state) == "code_reviewer"


def test_route_after_brain_returns_pcf_parser() -> None:
    state: WorkflowState = {
        "message": "parse rec1234567890abcd",
        "next_node": "pcf_parser",
        "code_generator_output": None,
        "pcf_parser_output": None,
        "error": None,
        "pcf_record_id": None,
        "pcf_table": None,
    }
    assert route_after_brain(state) == "pcf_parser"


def test_route_after_brain_returns_end_for_unknown_node() -> None:
    state: WorkflowState = {
        "message": "",
        "next_node": "unknown",
        "code_generator_output": None,
        "pcf_parser_output": None,
        "error": None,
        "pcf_record_id": None,
        "pcf_table": None,
    }
    assert route_after_brain(state) == END
