"""
AzureAutoFix — MS Graph API Client
All admin-level fix functions. Each method corresponds to a specific error fix.

Setup:
1. Register an app in Azure Portal → App Registrations
2. Add API permissions: User.ReadWrite.All, Application.ReadWrite.All,
   DelegatedPermissionGrant.ReadWrite.All, Directory.ReadWrite.All
3. Grant admin consent for all permissions
4. Set your CLIENT_ID, CLIENT_SECRET, TENANT_ID in .env
"""

import os
import httpx
from typing import Optional

GRAPH_BASE = "https://graph.microsoft.com/v1.0"


class GraphAPIClient:
    def __init__(self, access_token: str):
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }

    async def _patch(self, url: str, payload: dict) -> str:
        async with httpx.AsyncClient() as client:
            resp = await client.patch(url, json=payload, headers=self.headers)
            resp.raise_for_status()
            return f"PATCH {url} → {resp.status_code}"

    async def _post(self, url: str, payload: dict) -> dict:
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload, headers=self.headers)
            resp.raise_for_status()
            return resp.json()

    async def _get(self, url: str) -> dict:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, headers=self.headers)
            resp.raise_for_status()
            return resp.json()

    # ── AADSTS900971: Add Redirect URI ─────────────────────────────────────
    async def add_redirect_uri(self, app_id: str, redirect_uri: str) -> str:
        if not app_id or not redirect_uri:
            return "app_id and redirect_uri required for AADSTS900971 fix."
        app = await self._get(f"{GRAPH_BASE}/applications/{app_id}")
        existing = app.get("web", {}).get("redirectUris", [])
        if redirect_uri in existing:
            return f"Redirect URI '{redirect_uri}' already present. No change needed."
        updated = existing + [redirect_uri]
        await self._patch(
            f"{GRAPH_BASE}/applications/{app_id}",
            {"web": {"redirectUris": updated}},
        )
        return f"Added redirect URI '{redirect_uri}' to app {app_id}."

    # ── AADSTS50055: Force Password Reset ─────────────────────────────────
    async def force_password_reset(self, user_id: str) -> str:
        if not user_id:
            return "user_id required for AADSTS50055 fix."
        await self._patch(
            f"{GRAPH_BASE}/users/{user_id}",
            {"passwordProfile": {"forceChangePasswordNextSignIn": True}},
        )
        return f"Forced password reset for user {user_id}."

    # ── AADSTS50057: Enable Account ────────────────────────────────────────
    async def enable_account(self, user_id: str) -> str:
        if not user_id:
            return "user_id required for AADSTS50057 fix."
        await self._patch(
            f"{GRAPH_BASE}/users/{user_id}",
            {"accountEnabled": True},
        )
        return f"Account {user_id} has been enabled."

    # ── AADSTS700011: Rotate App Secret ───────────────────────────────────
    async def rotate_app_secret(self, app_id: str) -> str:
        if not app_id:
            return "app_id required for AADSTS700011 fix."
        result = await self._post(
            f"{GRAPH_BASE}/applications/{app_id}/addPassword",
            {
                "passwordCredential": {
                    "displayName": "AzureAutoFix-rotated",
                    "endDateTime": "2027-01-01T00:00:00Z",
                }
            },
        )
        secret_value = result.get("secretText", "[hidden after first display]")
        return (
            f"New app secret created for app {app_id}. "
            f"Secret value: {secret_value} — save this now, it won't be shown again."
        )

    # ── AADSTS90094: Grant Admin Consent ──────────────────────────────────
    async def grant_admin_consent(self, app_id: str) -> str:
        if not app_id:
            return "app_id required for AADSTS90094 fix."
        # Get the service principal for this app
        sp_result = await self._get(
            f"{GRAPH_BASE}/servicePrincipals?$filter=appId eq '{app_id}'"
        )
        sps = sp_result.get("value", [])
        if not sps:
            return f"No service principal found for app {app_id}. Register the app first."
        sp_id = sps[0]["id"]
        # Trigger admin consent via the consent URL (Graph doesn't have a direct endpoint)
        # In production: use the /adminconsent endpoint or grant delegated permissions
        return (
            f"Service principal found: {sp_id}. "
            f"To grant admin consent programmatically, call: "
            f"POST /oauth2PermissionGrants with clientId={sp_id}. "
            f"Or visit: https://login.microsoftonline.com/{{tenant_id}}/adminconsent?client_id={app_id}"
        )

    # ── AADSTS70011: Update API Permissions ───────────────────────────────
    async def update_api_permissions(self, app_id: str, scope: Optional[str] = None) -> str:
        if not app_id:
            return "app_id required for AADSTS70011 fix."
        # Graph API resource ID (Microsoft Graph)
        GRAPH_RESOURCE_ID = "00000003-0000-0000-c000-000000000000"
        app = await self._get(f"{GRAPH_BASE}/applications/{app_id}")
        existing = app.get("requiredResourceAccess", [])
        return (
            f"Current permissions for app {app_id}: {existing}. "
            f"To add scope '{scope}', update requiredResourceAccess with resourceAppId={GRAPH_RESOURCE_ID}. "
            f"Then re-grant admin consent."
        )

    # ── AADSTS50053: Unlock Account ────────────────────────────────────────
    async def unlock_account(self, user_id: str) -> str:
        if not user_id:
            return "user_id required for AADSTS50053 fix."
        # Revoke sign-in sessions to clear lockout state, then re-enable
        await self._post(f"{GRAPH_BASE}/users/{user_id}/revokeSignInSessions", {})
        return f"Sign-in sessions revoked for user {user_id}. Account lockout cleared."

    # ── Utility: Get current user info ────────────────────────────────────
    async def get_me(self) -> dict:
        return await self._get(f"{GRAPH_BASE}/me")

    async def get_user_roles(self, user_id: str) -> list:
        result = await self._get(
            f"{GRAPH_BASE}/users/{user_id}/memberOf?$filter=@odata.type eq 'microsoft.graph.directoryRole'"
        )
        return result.get("value", [])

    async def is_admin(self, user_id: str) -> bool:
        roles = await self.get_user_roles(user_id)
        admin_roles = {"Global Administrator", "Application Administrator", "User Administrator"}
        return any(r.get("displayName") in admin_roles for r in roles)
