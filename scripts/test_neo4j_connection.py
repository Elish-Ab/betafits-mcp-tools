import os
import socket
from urllib.parse import urlparse
from dotenv import load_dotenv
from neo4j import GraphDatabase
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

load_dotenv()

def test_connection():
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")

    print(f"--- Diagnostic Report ---")
    print(f"Target URI: {uri}")
    print(f"User: {user}")
    
    if not uri:
        print("❌ ERROR: NEO4J_URI is not set in .env")
        return

    # 1. Parse URI
    parsed = urlparse(uri)
    host = parsed.hostname
    port = parsed.port or 7687
    
    print(f"Parsed Host: {host}")
    print(f"Parsed Port: {port}")

    # 2. DNS Check
    print(f"\n1. Checking DNS Resolution for {host}...")
    try:
        ip = socket.gethostbyname(host)
        print(f"✅ DNS Success: {host} resolves to {ip}")
    except socket.gaierror as e:
        print(f"❌ DNS FAILURE: Could not resolve {host}.")
        print(f"   Tip: Check for typos in .env or verify the instance exists in Neo4j Aura console.")
        print(f"   Error details: {e}")
        return

    # 3. Socket Check
    print(f"\n2. Checking Port {port} Connectivity...")
    try:
        with socket.create_connection((host, port), timeout=5):
            print(f"✅ Port Success: Port {port} is open and reachable.")
    except Exception as e:
        print(f"❌ PORT FAILURE: Could not connect to {host}:{port}.")
        print(f"   Tip: This might be a firewall or VPN blocking the connection.")
        print(f"   Error details: {e}")
        return

    # 4. Driver Check
    print(f"\n3. Checking Neo4j Login...")
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            result = session.run("RETURN 1 AS one").single()
            if result and result["one"] == 1:
                print("✅ LOGIN SUCCESS: Authenticated successfully!")
        driver.close()
    except Exception as e:
        print(f"❌ LOGIN FAILURE: Connection established but login failed.")
        print(f"   Tip: Check your NEO4J_USER and NEO4J_PASSWORD.")
        print(f"   Error details: {e}")

if __name__ == "__main__":
    test_connection()
