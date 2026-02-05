"""Entry point for the PCF parser service."""
from __future__ import annotations

import re
from typing import Any, Dict, Optional
import os

import requests
from pyairtable import Api

from lib.context_engine import (
    DEFAULT_PCF_TABLE,
    PCF_PARSER_API_KEY,
    PCF_PARSER_BASE_ID,
    PCF_PARSER_TABLE_ID,
    build_pcf_context,
)
from agents.pcf_parser_workflow import run_pcf_parser_workflow


PCF_RECORD_ID_RE = re.compile(r"\brec[a-zA-Z0-9]{14}\b")


def _extract_pcf_record_id(message: str) -> Optional[str]:
    if not message:
        return None
    match = PCF_RECORD_ID_RE.search(message)
    if not match:
        return None
    return match.group(0)


def _fetch_meeting_transcript(record_id: str) -> str:
    api_key = os.getenv("AIR_TABLE_ACCESS_TOKEN")
    base_id = os.getenv("AIR_TABLE_BASE_ID")
    if not api_key or not base_id:
        raise ValueError("AIR_TABLE_ACCESS_TOKEN and AIR_TABLE_BASE_ID must be set.")

    api = Api(api_key)
    table = api.table(base_id, "Meetings")
    record = table.get(record_id)
    fields = record.get("fields", {})
    files = fields.get("File", [])
    if not files:
        raise ValueError(f"No transcript file found for meeting record {record_id}")
    file_info = files[0]
    url = file_info.get("url")
    if not url:
        raise ValueError(f"Transcript file URL missing for meeting record {record_id}")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.text


def run(
    message: str = "",
    pcf_record_id: Optional[str] = None,
    pcf_table: str = DEFAULT_PCF_TABLE,
    transcript: Optional[str] = None,
    meeting_record_id: Optional[str] = None,
    top_k: int = 5,
    record_type: str = "Meeting",
) -> Dict[str, Any]:
    """Run PCF parser workflow or load PCF context."""
    if transcript or meeting_record_id:
        if not transcript and meeting_record_id:
            transcript = _fetch_meeting_transcript(meeting_record_id)
        return run_pcf_parser_workflow(transcript=transcript or "", top_k=top_k, type=record_type)

    record_id = pcf_record_id or _extract_pcf_record_id(message)
    if PCF_PARSER_API_KEY:
        masked = f"{PCF_PARSER_API_KEY[:4]}...{PCF_PARSER_API_KEY[-4:]}"
    else:
        masked = "None"
    table_label = pcf_table
    if PCF_PARSER_TABLE_ID:
        table_label = f"{pcf_table} (table_id={PCF_PARSER_TABLE_ID})"
    print(f"[pcf_parser] using token={masked} base={PCF_PARSER_BASE_ID or 'None'} table={table_label} record={record_id or 'None'}")
    context = build_pcf_context(record_id, pcf_table=pcf_table)
    if not record_id:
        context["pcf_context_error"] = "pcf_record_id not provided."
    context["pcf_record_id"] = record_id
    return context
