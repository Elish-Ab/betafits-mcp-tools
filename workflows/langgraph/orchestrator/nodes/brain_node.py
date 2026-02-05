"""MCP Brain node that analyzes the message and routes to appropriate nodes."""
from lib.state import WorkflowState
from lib.lg_run_logger import log_snapshot
from lib.llm_client import call_llm


# Available nodes that the brain can route to
AVAILABLE_NODES = [
    "code_generator",
    "pcf_parser",
    "pcf_writer",
    "code_refactor",
    "code_reviewer",
]


def mcp_brain_node(state: WorkflowState) -> WorkflowState:
    """
    MCP Brain node that analyzes the incoming message using LLM and determines
    which node should handle it next.
    
    Args:
        state: The current workflow state containing the message
        
    Returns:
        Updated state with next_node determined by LLM
    """
    log_snapshot(state, "mcp_brain", "Pre-Step")
    message = state.get("message", "")
    pcf_record_id = state.get("pcf_record_id")
    pcf_table = state.get("pcf_table")
    repo_id = state.get("repo_id")
    repo_name = state.get("repo_name")
    record_id = state.get("record_id")
    source_table = state.get("source_table")
    source_record_id = state.get("source_record_id")
    repo_url = state.get("repo_url")
    repo_github_id = state.get("repo_github_id")
    meeting_record_id = state.get("meeting_record_id")
    transcript = state.get("transcript")

    if not message and not pcf_record_id and not meeting_record_id and not source_record_id and not repo_id and not repo_name and not repo_url and not repo_github_id:
        state["next_node"] = None
        log_snapshot(state, "mcp_brain", "Post-Step")
        return state
    
    # Create prompt for LLM to decide routing
    routing_prompt = f"""Analyze the following message and determine which node should handle it.

Available nodes:
- code_generator: For generating new code
- pcf_parser: For parsing/loading PCF context
- pcf_writer: For writing/creating PCF files
- code_refactor: For refactoring existing code
- code_reviewer: For reviewing code

Decision guidance:
- If meeting_record_id is provided or source_table is Meetings, choose pcf_parser.
- If source_table is Repositories or repo_url/repo_github_id is provided, choose code_reviewer.
- If source_table is PCFs and a pcf_record_id is provided, choose code_generator.
- If repo_id or repo_name is provided, choose code_reviewer (unless the message clearly requests generation).
- If a PCF record id is provided (pcf_record_id) and there is no meeting_record_id, choose code_generator.
- If the message asks to parse/load PCF context, choose pcf_parser.

Context:
- message: {message or "None provided"}
- pcf_record_id: {pcf_record_id or "None provided"}
- pcf_table: {pcf_table or "None provided"}
- repo_id: {repo_id or "None provided"}
- repo_name: {repo_name or "None provided"}
- record_id: {record_id or "None provided"}
- source_table: {source_table or "None provided"}
- source_record_id: {source_record_id or "None provided"}
- repo_url: {repo_url or "None provided"}
- repo_github_id: {repo_github_id or "None provided"}
- meeting_record_id: {meeting_record_id or "None provided"}
- transcript_present: {"Yes" if transcript else "No"}

Message: {message or ""}

Respond with ONLY the node name (e.g., "code_generator") or "END" if no node matches.
Do not include any explanation or additional text."""
    
    try:
        # Call LLM to determine routing
        llm_response = call_llm(routing_prompt)
        
        # Clean the response and validate
        next_node = llm_response.strip().lower()
        print("node",next_node)
        
        # Validate that the response is a valid node name
        if next_node in AVAILABLE_NODES:
            state["next_node"] = next_node
        elif next_node == "end":
            state["next_node"] = None
        else:
            # If LLM returns something unexpected, default to None (END)
            state["next_node"] = None
            
    except Exception as e:
        # On error, default to None (END)
        error_msg = str(e)
        if "invalid_api_key" in error_msg or "401" in error_msg or "API_KEY_INVALID" in error_msg:
            print(f"❌ OpenRouter API Key Error: Please check your OPENROUTER_API_KEY in .env file")
            print(f"   Get your API key at: https://openrouter.ai/keys")
        elif "404" in error_msg or "not found" in error_msg.lower() or "not supported" in error_msg.lower():
            print(f"❌ OpenRouter Model Error: The model name may be incorrect")
            print(f"   Using model: google/gemini-2.0-flash-001")
            print(f"   Error details: {e}")
        else:
            print(f"Error in brain node LLM call: {e}")
        state["next_node"] = None
    
    log_snapshot(state, "mcp_brain", "Post-Step")
    return state
