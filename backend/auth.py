"""
AzureAutoFix — Token Manager
Acquires and caches MS Graph API tokens using client credentials flow.
"""

import os
import time
import httpx
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID     = os.getenv("AZURE_CLIENT_ID")
CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET")
TENANT_ID     = os.getenv("AZURE_TENANT_ID")

_token_cache = {"token": None, "expires_at": 0}


def get_app_token() -> str:
    """Returns a valid app-level MS Graph token, refreshing if expired."""
    if _token_cache["token"] and time.time() < _token_cache["expires_at"] - 60:
        return _token_cache["token"]

    url = f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token"
    resp = httpx.post(url, data={
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "scope": "https://graph.microsoft.com/.default",
    })
    resp.raise_for_status()
    data = resp.json()
    _token_cache["token"] = data["access_token"]
    _token_cache["expires_at"] = time.time() + data["expires_in"]
    return _token_cache["token"]
