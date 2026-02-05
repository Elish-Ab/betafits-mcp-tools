import os
from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()

# supabase credentials
supabase_url: str = os.getenv("SUPABASE_URL")
supabase_key: str = os.getenv("SUPABASE_SERVICE_KEY")
SUPABASE_CLIENT_: Client = create_client(
    supabase_url, supabase_key
)  # client for supabase connection


if __name__ == "__main__":
    print("Supabase client initialized.")
    try:
        response = SUPABASE_CLIENT_.rpc("list_tables", {}).execute()
        print("✅ Tables:", response.data)
    except Exception as e:
        print("❌ Error:", e)
