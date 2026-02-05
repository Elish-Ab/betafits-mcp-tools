"""State definition for the LangGraph workflow."""

from typing import Any, Dict, Optional, TypedDict


class WorkflowState(TypedDict):
    """State that flows through the LangGraph graph."""

    # The input message that invokes the graph (e.g., Slack text, CLI prompt)
    message: str

    # The next node to route to, as decided by the MCP brain.
    # Examples: "code_generator", "code_reviewer", or None to end.
    next_node: Optional[str]

    # Output from the code generator node (hierarchical repo JSON, etc.)
    code_generator_output: Optional[Dict[str, Any]]

    # Output from the PCF parser node (PCF context + documents).
    pcf_parser_output: Optional[Dict[str, Any]]

    # Generic error string if any node encounters an error.
    error: Optional[str]

    # Optional PCF context identifiers.
    pcf_record_id: Optional[str]
    pcf_table: Optional[str]

    # Optional repository context identifiers.
    repo_id: Optional[str]
    repo_name: Optional[str]
    record_id: Optional[str]

    # Optional meeting context for PCF parser workflow.
    meeting_record_id: Optional[str]
    transcript: Optional[str]

    # Optional source context from external triggers (e.g., Airtable buttons).
    source_table: Optional[str]
    source_record_id: Optional[str]
    repo_url: Optional[str]
    repo_github_id: Optional[str]

    # Optional run context for logging.
    _run_context: Optional[Dict[str, Any]]

    # Optional context files passed from CLI.
    context_files: Optional[list[str]]
