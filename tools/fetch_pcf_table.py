import os
from typing import List, Dict, Any
from pyairtable import Api
from pyairtable.formulas import match
from dotenv import load_dotenv
from langchain_core.tools import StructuredTool

load_dotenv()


def fetch_pcf_table() -> List[Dict[str, Any]]:
    """
    Fetches all PCF Parser records and automatically expands linked records.
    Returns a list of PCF dicts with all related data (projects, features, etc.).
    """
    # Get credentials from environment
    api_key = os.getenv("AIR_TABLE_ACCESS_TOKEN")
    base_id = os.getenv("AIR_TABLE_BASE_ID")

    if not api_key or not base_id:
        raise ValueError(
            "AIR_TABLE_ACCESS_TOKEN and AIR_TABLE_BASE_ID must be set in environment"
        )

    # Initialize API and tables using the recommended Api.table() method
    api = Api(api_key)
    tables = {
        "pcf": api.table(base_id, "PCFs"),
        "team": api.table(base_id, "Team"),  # Corrected from "Team Members"
        "tools": api.table(base_id, "Tools"),
        "meetings": api.table(base_id, "Meetings"),
    }

    # Recursive self-link strategy for Project/Component/Feature
    # "Projects", "Components", "Features" are links to the "PCFs" table itself.
    
    # Simple in-memory cache to avoid repeated fetches
    _cache: Dict[str, Any] = {}

    def expand_ids(ids: List[str], tablename: str) -> List[Dict[str, Any]]:
        """Fetch and expand linked record IDs into full record fields."""
        if not ids:
            return []
        
        # Type check: Ensure ids is a list, not a string
        if isinstance(ids, str):
            print(f"⚠️ Warning: '{tablename}' field returned a string instead of a list: '{ids}'")
            return []
        
        results: List[Dict[str, Any]] = []
        target_table = tables.get(tablename)
        
        # Special case for self-linked PCF records
        if tablename in ["projects", "components", "features"]:
             target_table = tables["pcf"]

        if not target_table:
             print(f"⚠️ Warning: Table '{tablename}' not defined in table map.")
             return []

        for rid in ids:
            if rid not in _cache:
                try:
                    _cache[rid] = target_table.get(rid)
                except Exception as e:
                    print(f"⚠️ Could not expand {tablename} record {rid}: {e}")
                    continue
            results.append(_cache[rid]["fields"])
        return results

    # Fetch only PCF Parser records where Current is True
    records = tables["pcf"].all(formula=match({"Current": True}))
    expanded_records = []

    for rec in records:
        f = rec.get("fields", {})
        # Map fields to correct logical table buckets
        f["Projects"] = expand_ids(f.get("Link to Projects"), "projects")
        f["Components"] = expand_ids(f.get("Link to Components"), "components") 
        f["Features"] = expand_ids(f.get("Link to Features"), "features")
        
        # Correct field name for Team Members found in inspection
        f["Team Members"] = expand_ids(f.get("Team Members (Link to Team)"), "team")
        
        f["Tools"] = expand_ids(f.get("Link to Tools"), "tools")
        f["Meetings"] = expand_ids(f.get("Link to Meetings"), "meetings")

        # Keep the full record structure with id and createdTime
        expanded_records.append(
            {"id": rec.get("id"), "createdTime": rec.get("createdTime"), "fields": f}
        )

    return expanded_records


fetch_pcf_table_tool = StructuredTool.from_function(
    func=fetch_pcf_table,
    name="fetch_pcf_table",
    description="Fetches records from an Airtable table PCF tables.",
    return_direct=True,
)


if __name__ == "__main__":
    print(fetch_pcf_table_tool.name)
    print(fetch_pcf_table_tool.description)
    print("invoke dummy tool call")
    records = fetch_pcf_table_tool.invoke({})
    # print("Schema: ", table.schema())

    # print(f"{type(table)}, {type(records)},{type(count)}")
    for i in records:
        for key, value in i.items():
            print(key, value)

        print("=" * 50)
