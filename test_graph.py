"""
Quick test — verifies your Azure credentials and Graph API connection.
Run: python test_graph.py
"""
import os
import httpx
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID     = os.getenv("AZURE_CLIENT_ID")
CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET")
TENANT_ID     = os.getenv("AZURE_TENANT_ID")

def get_token():
    url = f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token"
    data = {
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "scope": "https://graph.microsoft.com/.default",
    }
    resp = httpx.post(url, data=data)
    resp.raise_for_status()
    return resp.json()["access_token"]

def test_connection(token):
    headers = {"Authorization": f"Bearer {token}"}
    resp = httpx.get("https://graph.microsoft.com/v1.0/organization", headers=headers)
    resp.raise_for_status()
    org = resp.json()["value"][0]
    print(f"✅ Connected to tenant: {org['displayName']}")
    print(f"   Tenant ID: {org['id']}")
    return True

if __name__ == "__main__":
    print("Testing Azure Graph API connection...")
    token = get_token()
    print("✅ Token acquired.")
    test_connection(token)
    print("\nGraph API is working. Ready to execute fixes.")
