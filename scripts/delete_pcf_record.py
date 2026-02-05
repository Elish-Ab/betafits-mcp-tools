import os
import sys
from dotenv import load_dotenv
from pyairtable import Api

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

load_dotenv()

def delete_record(record_id: str):
    token = os.getenv("AIR_TABLE_ACCESS_TOKEN")
    base_id = os.getenv("AIR_TABLE_BASE_ID")
    table_id = os.getenv("AIR_TABLE_PCF_PARSER_TABLE_ID")

    if not all([token, base_id, table_id]):
        print("❌ ERROR: Missing Airtable configuration in .env")
        return

    print(f"Connecting to Airtable Table: {table_id}...")
    api = Api(token)
    table = api.table(base_id, table_id)

    try:
        print(f"🗑️ Deleting record {record_id}...")
        table.delete(record_id)
        print(f"✅ SUCCESS: Record {record_id} deleted.")
    except Exception as e:
        print(f"❌ FAILURE: Could not delete record.")
        print(f"   Error details: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python delete_pcf_record.py <RECORD_ID>")
        print("Example: python delete_pcf_record.py rec123abcd456")
    else:
        delete_record(sys.argv[1])
