"""LangGraph orchestrator that routes messages through the workflow."""
from typing import Literal

from langgraph.graph import END, StateGraph #type: ignore

from lib.state import WorkflowState
from workflows.langgraph.orchestrator.nodes.brain_node import mcp_brain_node
from workflows.langgraph.orchestrator.nodes.code_generator import code_generator_node
from workflows.langgraph.orchestrator.nodes.code_reviewer import code_reviewer_node
from workflows.langgraph.orchestrator.nodes.pcf_parser import pcf_parser_node


def route_after_brain(state: WorkflowState) -> Literal["code_generator", "code_reviewer", "pcf_parser", "END"]:
    """
    Conditional routing function that routes based on the brain's LLM decision.

    Args:
        state: The current workflow state with next_node set by brain

    Returns:
        The name of the next node to execute, or END
    """
    next_node = state.get("next_node")

    if next_node == "code_generator":
        return "code_generator"
    if next_node == "code_reviewer":
        return "code_reviewer"
    if next_node == "pcf_parser":
        return "pcf_parser"

    # If no valid node or unknown node, end the workflow
    return END


def create_graph() -> StateGraph:
    """
    Creates and configures the LangGraph workflow.

    The graph starts with the MCP brain node that analyzes the message using LLM
    and routes to appropriate handler nodes.
    """
    # Create the graph with WorkflowState
    graph = StateGraph(WorkflowState)

    # Add the MCP brain node as the entry point
    graph.add_node("mcp_brain", mcp_brain_node)

    # Add the code generator node
    graph.add_node("code_generator", code_generator_node)

    # Add the code reviewer node
    graph.add_node("code_reviewer", code_reviewer_node)
    # Add the PCF parser node
    graph.add_node("pcf_parser", pcf_parser_node)

    # Set the entry point to the MCP brain
    graph.set_entry_point("mcp_brain")

    # Use conditional routing from brain based on LLM decision
    graph.add_conditional_edges(
        "mcp_brain",
        route_after_brain,
        {
            "code_generator": "code_generator",
            "code_reviewer": "code_reviewer",
            "pcf_parser": "pcf_parser",
            END: END,
        },
    )

    # Code generator routes to END
    graph.add_edge("code_generator", END)
    # Code reviewer routes to END (for now)
    graph.add_edge("code_reviewer", END)
    # PCF parser routes to END (for now)
    graph.add_edge("pcf_parser", END)

    return graph.compile()


# Create the compiled graph instance
workflow = create_graph()
