"""Entry point for the code reviewer service."""

from typing import Any, Dict, Optional

from services.code_reviewer.chain import run_code_reviewer_chain
from lib.context_engine import DEFAULT_PCF_TABLE
from lib.validation import validate_user_message


def run(
    message: str,
    persist: bool = True,
    pcf_record_id: Optional[str] = None,
    pcf_table: str = DEFAULT_PCF_TABLE,
) -> Dict[str, Any]:
    """Run the code reviewer service.

    Current behavior:
    - Uses a multi-step chain to:
      1) Match the message to repositories in Airtable.
      2) Collect repositories/folders/files context.
      3) Grade repositories using the Betafits code grading rubric.
    - Returns JSON only. Optionally writes review fields back to Airtable.
    - Optional PCF context can be supplied to enrich reviewer context.
    """
    validated_message = validate_user_message(message)
    return run_code_reviewer_chain(
        validated_message,
        persist_to_airtable=persist,
        pcf_record_id=pcf_record_id,
        pcf_table=pcf_table,
    )
