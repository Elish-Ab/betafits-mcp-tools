import os
import sys
from dotenv import load_dotenv
from pyairtable import Api

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

load_dotenv()

def inspect_record(record_id: str):
    token = os.getenv("AIR_TABLE_ACCESS_TOKEN")
    base_id = os.getenv("AIR_TABLE_BASE_ID")
    table_id = os.getenv("AIR_TABLE_PCF_PARSER_TABLE_ID")

    if not all([token, base_id, table_id]):
        print("❌ ERROR: Missing Airtable configuration in .env")
        return

    api = Api(token)
    table = api.table(base_id, table_id)

    try:
        print(f"🔍 Fetching details for record: {record_id}...")
        record = table.get(record_id)
        fields = record.get("fields", {})
        
        print("\n" + "="*50)
        print(f"RECORD ID: {record_id}")
        print(f"NAME:      {fields.get('Name', 'N/A')}")
        print(f"TYPE:      {fields.get('Type', 'N/A')}")
        print(f"DATE:      {fields.get('Date Processed', 'N/A')}")
        print("-"*50)
        print("SUMMARY PREVIEW:")
        summary = fields.get('Summary', '')
        print(summary[:500] + ("..." if len(summary) > 500 else ""))
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"❌ FAILURE: Could not find record {record_id}.")
        print(f"   Error details: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/inspect_pcf_record.py <RECORD_ID>")
    else:
        inspect_record(sys.argv[1])
