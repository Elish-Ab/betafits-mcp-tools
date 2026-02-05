"""PCF parser node that loads PCF context and documents."""
from lib.state import WorkflowState
from lib.lg_run_logger import log_snapshot
from services.pcf_parser.run import run
from lib.context_engine import DEFAULT_PCF_TABLE


def pcf_parser_node(state: WorkflowState) -> WorkflowState:
    """
    PCF parser node that processes PCF context requests.

    Args:
        state: The current workflow state

    Returns:
        Updated state after PCF parsing
    """
    log_snapshot(state, "pcf_parser", "Pre-Step")
    message = state.get("message", "")
    pcf_record_id = state.get("pcf_record_id")
    pcf_table = state.get("pcf_table") or DEFAULT_PCF_TABLE
    transcript = state.get("transcript")
    meeting_record_id = state.get("meeting_record_id")

    try:
        result = run(
            message=message,
            pcf_record_id=pcf_record_id,
            pcf_table=pcf_table,
            transcript=transcript,
            meeting_record_id=meeting_record_id,
        )
        state["pcf_parser_output"] = result
        state["error"] = None
    except Exception as exc:  # pragma: no cover - defensive
        state["pcf_parser_output"] = None
        state["error"] = str(exc)

    log_snapshot(state, "pcf_parser", "Post-Step")
    return state
