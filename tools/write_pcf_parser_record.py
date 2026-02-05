import os
from typing import List, Optional, Dict
from datetime import datetime
from dotenv import load_dotenv
from pyairtable import Api
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

import json  # ADD THIS LINE

# Load .env file
load_dotenv()


# ✅ Input schema for structured tool
class PCFParserRecordInput(BaseModel):
    name: str = Field(..., description="Name of the record.")
    date: Optional[str] = Field(None, description="Date in ISO format (YYYY-MM-DD).")
    type: Optional[str] = Field(
        None, description="Type value, e.g. 'Slack', 'Meeting', or 'Phone Call'."
    )
    summary: Optional[str] = Field(
        None,
        description="Long text summary in Markdown format containing bullets e.g some description .... \n- bullet 1\n - bullet2 and so on.",
    )
    link_to_pcfs: Optional[List[str]] = Field(
        None, description="List of linked record IDs from PCFs table."
    )
    link_to_meetings: Optional[List[str]] = Field(
        None, description="List of linked record IDs from Meetings table."
    )


def create_pcf_parser_record(
    name: str,
    date: Optional[str] = None,
    type: Optional[str] = None,
    summary: Optional[str] = None,
    link_to_pcfs: Optional[List[str]] = None,
    link_to_meetings: Optional[List[str]] = None,
) -> Dict:
    """
    Creates a new record in the PCF Parser table using field IDs (recommended).
    If date is not provided, defaults to today's date in ISO format (YYYY-MM-DD).
    """
    # Default date to today if not provided
    if date is None:
        date = datetime.now().isoformat()
        #date = datetime.now().strftime("%Y-%m-%d")

    # Ensure environment setup
    for var in [
        "AIR_TABLE_ACCESS_TOKEN",
        "AIR_TABLE_BASE_ID",
        "AIR_TABLE_PCF_PARSER_TABLE_ID",
    ]:
        if not os.getenv(var):
            raise EnvironmentError(f"Missing environment variable: {var}")

    # Initialize Airtable API
    api = Api(os.getenv("AIR_TABLE_ACCESS_TOKEN"))
    table = api.table(
        os.getenv("AIR_TABLE_BASE_ID"), os.getenv("AIR_TABLE_PCF_PARSER_TABLE_ID")
    )

    # Prepare record payload using field names (more maintainable than IDs)
    fields = {
        "Name": name,
    }

    if date:
        fields["Date Processed"] = date
    if type:
        fields["Type"] = type
    
    if summary:
       fields["Summary"] = summary

    
    if link_to_pcfs:
        fields["Link to PCFs"] = link_to_pcfs
    
    # Handle link_to_meetings: only add if there are meetings
    if link_to_meetings and len(link_to_meetings) > 0:
        fields["Link to Meetings"] = link_to_meetings

    print(f"Creating PCF Parser record: {fields}")

    # Create record (typecast=True allows auto-creation of new single-select options if needed)
    created = table.create(fields, typecast=True)

    record_id = created["id"]
    print(f"✅ Record created successfully: {record_id}")
    return created


def update_pcf_parser_record_supabase_id(
    record_id: str,
    supabase_id: str,
) -> Dict:
    """
    Updates an existing PCF Parser record with its Supabase ID.
    
    Args:
        record_id: The Airtable record ID to update
        supabase_id: The Supabase RAG ID to store
    
    Returns:
        Updated Airtable record
    """
    # Ensure environment setup
    for var in [
        "AIR_TABLE_ACCESS_TOKEN",
        "AIR_TABLE_BASE_ID",
        "AIR_TABLE_PCF_PARSER_TABLE_ID",
    ]:
        if not os.getenv(var):
            raise EnvironmentError(f"Missing environment variable: {var}")

    # Initialize Airtable API
    api = Api(os.getenv("AIR_TABLE_ACCESS_TOKEN"))
    table = api.table(
        os.getenv("AIR_TABLE_BASE_ID"), os.getenv("AIR_TABLE_PCF_PARSER_TABLE_ID")
    )

    # Update only the Supabase ID field (using field name)
    fields = {
        "Supabase ID": supabase_id,
    }

    print(f"Updating PCF Parser record {record_id} with Supabase ID: {supabase_id}")

    # Update record
    updated = table.update(record_id, fields)

    print(f"✅ Record updated successfully: {record_id}")
    return updated


# ✅ Define as LangChain StructuredTool
create_pcf_parser_record_tool = StructuredTool.from_function(
    func=create_pcf_parser_record,
    name="create_pcf_parser_record",
    description="Creates a new record in the PCF Parser Airtable table using field IDs. Returns the created record details.",
    args_schema=PCFParserRecordInput,
    return_direct=True,
)

# ✅ Standalone test runner
if __name__ == "__main__":
    # Test 1: No meetings (will write "NA")
    print("\n=== Test 1: No meeting links (should write 'NA') ===")
    result1 = create_pcf_parser_record(
        name="ML Model & Visualization - No Meetings",
        date="2024-04-01",
        type="Slack",
        summary="Discussion focused on refining the softmax model and adding GBM.\n - A\n - B",
        link_to_pcfs=[],
        link_to_meetings=None,  # This will write "NA"
    )
    print(result1)
    
    # Test 2: With meeting links (example record IDs)
    print("\n=== Test 2: With meeting links ===")
    result2 = create_pcf_parser_record(
        name="ML Model & Visualization - With Meetings",
        date="2024-04-02",
        type="Meeting",
        summary="Follow-up discussion with action items.\n - Review model\n - Update docs",
        link_to_pcfs=[],
        link_to_meetings=["recExampleMeetingID1", "recExampleMeetingID2"],  # Example meeting record IDs
    )
    print(result2)
