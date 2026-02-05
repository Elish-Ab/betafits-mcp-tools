"""Code generator node that handles code generation tasks."""
from lib.state import WorkflowState
from lib.lg_run_logger import log_snapshot
from services.code_generator.run import run
from lib.context_engine import DEFAULT_PCF_TABLE


def code_generator_node(state: WorkflowState) -> WorkflowState:
    """
    Code generator node that processes code generation requests.
    
    Args:
        state: The current workflow state
        
    Returns:
        Updated state after code generation processing
    """
    log_snapshot(state, "code_generator", "Pre-Step")
    message = state.get("message", "")
    context_files = state.get("context_files")
    pcf_record_id = state.get("pcf_record_id")
    pcf_table = state.get("pcf_table") or DEFAULT_PCF_TABLE
    source_record_id = state.get("source_record_id")
    record_id = state.get("record_id")
    
    try:
        # Call the code generator service
        if not message:
            parts = ["Generate code"]
            if pcf_record_id or source_record_id or record_id:
                parts.append(f"pcf_id={pcf_record_id or source_record_id or record_id}")
            if pcf_table:
                parts.append(f"pcf_table={pcf_table}")
            message = " | ".join(parts)
        result = run(
            message,
            use_chain=True,
            context_files=context_files,
            pcf_record_id=pcf_record_id,
            pcf_table=pcf_table,
        )
        
        # Update state with output
        state["code_generator_output"] = result
        state["error"] = None
        
    except Exception as e:
        # Handle errors
        state["code_generator_output"] = None
        state["error"] = str(e)
    
    log_snapshot(state, "code_generator", "Post-Step")
    return state
