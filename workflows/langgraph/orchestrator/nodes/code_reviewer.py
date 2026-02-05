"""Code reviewer node that handles code review tasks."""

from lib.state import WorkflowState
from lib.lg_run_logger import log_snapshot
from services.code_reviewer.run import run
from lib.context_engine import DEFAULT_PCF_TABLE


def code_reviewer_node(state: WorkflowState) -> WorkflowState:
    """
    Code reviewer node that processes code review requests.

    This node delegates to the `services.code_reviewer.run` entrypoint, which is
    responsible for performing the actual review (LLM, prompts, etc.).

    Args:
        state: The current workflow state.

    Returns:
        Updated state after code review processing.
    """
    log_snapshot(state, "code_reviewer", "Pre-Step")
    message = state.get("message", "")
    pcf_record_id = state.get("pcf_record_id")
    pcf_table = state.get("pcf_table") or DEFAULT_PCF_TABLE
    repo_name = state.get("repo_name")
    repo_id = state.get("repo_id") or state.get("record_id")
    repo_url = state.get("repo_url")

    try:
        # Call the code reviewer service. This can later accept richer inputs
        # (e.g., code snippets, repo JSON) encoded in the message/state.
        if not message:
            parts = ["Review repository"]
            if repo_name:
                parts.append(f"name={repo_name}")
            if repo_id:
                parts.append(f"id={repo_id}")
            if repo_url:
                parts.append(f"url={repo_url}")
            message = " | ".join(parts)
        result = run(message, pcf_record_id=pcf_record_id, pcf_table=pcf_table)

        # Attach result to state; extend `WorkflowState` later if you want
        # a dedicated field instead of overloading `code_generator_output`.
        state["code_generator_output"] = result  # TODO: add dedicated code_reviewer_output
        state["error"] = None
    except Exception as exc:  # pragma: no cover - defensive
        state["error"] = str(exc)

    log_snapshot(state, "code_reviewer", "Post-Step")
    return state

