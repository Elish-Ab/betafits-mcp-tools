import logging
from state.state import PCFParserState
from chains.workflow_chains import (
    summarize_meeting_chain,
    create_record_chain,
    save_log_chain
)

logger = logging.getLogger(__name__)

def extract_meeting_title(text: str) -> str | None:
    """Return the first non-empty line of the transcript verbatim."""
    if not text:
        return None
    for raw in text.splitlines():
        line = raw.strip().strip('"')
        if line:
            return line
    return None

def process_pcfs_node(state: PCFParserState) -> PCFParserState:
    """
    Node to process each mapped PCF: summarize and create parser record.
    """
    mapped = state.get("mapped_pcfs", [])
    transcript = state.get("transcript", "")
    record_type = state.get("type", "Meeting")
    pcf_records = state.get("pcf_records", [])
    
    # Create a lookup dictionary for PCF records by ID
    pcf_records_by_id = {rec["id"]: rec for rec in pcf_records}
    
    meeting_title_global = extract_meeting_title(transcript)
    if meeting_title_global and len(meeting_title_global) > 120:
        meeting_title_global = meeting_title_global[:117] + "..."

    created_parser_records = []
    
    logger.info(f"📝 STEP 4: Creating summaries and PCF Parser records for {len(mapped)} PCF(s)")
    
    for idx, pcf in enumerate(mapped, 1):
        try:
            pcf_id = pcf.get("pcf_id")
            pcf_name = pcf.get("name") or f"PCF {pcf_id}"
            
            # Get meeting IDs from the full PCF record
            meeting_ids = None
            if pcf_id and pcf_id in pcf_records_by_id:
                pcf_record = pcf_records_by_id[pcf_id]
                meeting_ids = pcf_record.get("fields", {}).get("Link to Meetings", [])

            logger.info(f"  [{idx}/{len(mapped)}] Processing: {pcf_name}")

            # Summarize
            summary_result = summarize_meeting_chain.invoke(
                {
                    "meeting_transcript": transcript,
                    "pcf_id": pcf_id,
                    "pcf_name": pcf_name,
                    "pcf_type": pcf.get("pcf_type", "Unknown"),
                    "pcf_summary": pcf.get("llm_reason") or "No description available",
                    "meeting_id": None,
                    "project_name": None,
                    "component_name": None,
                    "feature_name": None,
                }
            )
            summary_text = summary_result["summary"]

            # Compose record name
            record_name = pcf_name
            meeting_name = meeting_title_global

            if not meeting_name and pcf_id and pcf_id in pcf_records_by_id:
                _fields = pcf_records_by_id[pcf_id].get("fields", {})
                expanded_meetings = _fields.get("Meetings", [])
                if isinstance(expanded_meetings, list) and expanded_meetings:
                    first_meeting = expanded_meetings[0] or {}
                    meeting_name = first_meeting.get("Name") or None

            if meeting_name:
                combined = f"{pcf_name} — {meeting_name}".strip()
                if len(combined) > 160:
                    combined = combined[:157] + "..."
                record_name = combined

            # Create Record
            created = create_record_chain.invoke(
                {
                    "name": record_name,
                    "type": record_type,
                    "summary": summary_text,
                    "link_to_pcfs": [pcf_id] if pcf_id else None,
                    "link_to_meetings": meeting_ids,
                }
            )

            created_parser_records.append({
                "pcf_id": pcf_id, 
                "created_record": created, 
                "summary_text" : summary_text
            })
            
            record_id = created.get("id")
            logger.info(f"    ✓ Created PCF Parser record: {record_id}")
            
            save_log_chain.invoke(
                {
                    "log_info": f"Created PCF Parser record for {pcf_id}: {record_id}",
                    "origin": "pcf_parser_workflow",
                }
            )

        except Exception as e:
            logger.error(f"    ❌ Failed creating parser record for {pcf.get('pcf_id')}: {e}")
            save_log_chain.invoke(
                {
                    "log_info": f"Failed creating parser record for {pcf.get('pcf_id')}: {e}",
                    "origin": "pcf_parser_workflow",
                }
            )
            
    return {"created_parser_records": created_parser_records}
