"""
AzureAutoFix — FastAPI Backend
Run: uvicorn backend.main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import Optional
import sys, os
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.inference import classify, load_model
from backend.graph_api import GraphAPIClient
from backend.escalation import generate_escalation_message
from backend.auth import get_app_token
from monitoring.metrics import compute_metrics
from monitoring.middleware import LatencyTracingMiddleware
from monitoring import writer as trace_writer

# -- Production safety switches ----------------------------------------------
# DEMO_API_KEY: if set, /fix requires a matching `X-API-Key` header.
# ALLOW_APP_TOKEN_FALLBACK: if NOT "true", /fix will never fall back to the
#   app-level (client-credentials) Graph token and instead requires the
#   caller to supply their own access_token. Defaults to false so a public
#   deployment can never use the stored Azure app credentials to mutate
#   the tenant. Local dev can opt in via .env.
DEMO_API_KEY = os.getenv("DEMO_API_KEY")
ALLOW_APP_TOKEN_FALLBACK = os.getenv("ALLOW_APP_TOKEN_FALLBACK", "false").lower() == "true"


def require_api_key(x_api_key: Optional[str] = Header(default=None)):
    """Gate /fix behind an API key when DEMO_API_KEY is configured."""
    if DEMO_API_KEY and x_api_key != DEMO_API_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header")
    return True


app = FastAPI(
    title="AzureAutoFix API",
    description="Agentic Azure AD error resolution via LLM + MS Graph API",
    version="2.0.0",
)

from backend.analyze_sequence import router as sequence_router
app.include_router(sequence_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(LatencyTracingMiddleware)

# Load model on startup
@app.on_event("startup")
async def startup_event():
    try:
        load_model()
        print("[Startup] Model loaded successfully.")
    except Exception as e:
        print(f"[Startup] WARNING: Could not load model ({type(e).__name__}: {e}). Run model/train_local.py to train.")

    # Warm the retrieval index. Building it costs ~150ms for 350 documents --
    # small, but it would otherwise land on whichever unlucky user sends the
    # first non-curated error after a cold start, which on a free-tier
    # container is a real user fairly often.
    try:
        import time as _t
        from model.retrieval import get_retriever
        _t0 = _t.perf_counter()
        n = len(get_retriever().catalog)
        print(f"[Startup] Retrieval index warm: {n} AADSTS codes "
              f"in {(_t.perf_counter()-_t0)*1000:.0f}ms.")
    except FileNotFoundError:
        print("[Startup] WARNING: catalog missing. Run: python model/build_catalog.py")
    except Exception as e:
        print(f"[Startup] WARNING: retrieval unavailable ({type(e).__name__}: {e}).")


@app.on_event("shutdown")
async def shutdown_event():
    """Flush any buffered trace records before the process exits."""
    trace_writer.shutdown()


# -- Request/Response Models --------------------------------------------------

class AnalyzeRequest(BaseModel):
    error_input: str                      # Raw error code or description
    access_token: Optional[str] = None    # MS Graph access token (if user is signed in)
    user_id: Optional[str] = None         # Azure AD user object ID (for admin fixes)
    app_id: Optional[str] = None          # App registration object ID (for app fixes)
    redirect_uri: Optional[str] = None    # URI to add for AADSTS900971

class FixRequest(BaseModel):
    error_code: str
    fix_category: str
    access_token: str
    user_id: Optional[str] = None
    app_id: Optional[str] = None
    redirect_uri: Optional[str] = None
    new_scope: Optional[str] = None

class Citation(BaseModel):
    title: str
    url: str
    score: float = 0.0

class AnalyzeResponse(BaseModel):
    error_code: str
    fix_category: str
    user_or_admin: str
    explanation: str
    reasoning: str
    action: str
    action_detail: str
    user_message: str
    confidence: float
    source: str                       # lookup | retrieval | model | abstain
    # True when the system declined to commit to a remediation. The UI shows
    # this prominently -- an unactioned error is much cheaper than a wrong
    # automated write against a live tenant.
    abstained: bool = False
    citations: list[Citation] = []

class FixResponse(BaseModel):
    success: bool
    fix_applied: str
    details: str
    error_code: str

class EscalateResponse(BaseModel):
    subject: str
    body: str
    teams_message: str


# -- Routes --------------------------------------------------------------------

@app.get("/")
def root():
    return {"status": "AzureAutoFix API running", "docs": "/docs"}


@app.get("/status")
def status():
    """
    Honest health. The old version hardcoded model="loaded" and reported it
    even when the checkpoint had failed to load at startup, which made a
    degraded deployment look healthy.
    """
    from model import inference
    from model.retrieval import CATALOG_PATH

    return {
        "status": "ok",
        "version": "2.1.0",
        "tiers": {
            "lookup":    bool(inference._error_lookup) or Path(ROOT_DATA).exists(),
            "retrieval": CATALOG_PATH.exists(),
            "model":     inference._model is not None,
        },
    }


# -- Frontend + sign-in config -------------------------------------------------
# The Streamlit app has been replaced by a single static page (frontend/index.html)
# that signs the admin in with MSAL.js and calls this API directly.
#
# Why the change: the old UI never actually implemented Microsoft sign-in -- it
# had a "paste your MS Graph access token" text box. Streamlit reruns its whole
# script on every interaction and has no routing, which makes holding an OAuth
# redirect round-trip painful; MSAL.js in a normal page does it in one call.

_FRONTEND = Path(__file__).parent.parent / "frontend" / "index.html"
ROOT_DATA = Path(__file__).parent.parent / "data" / "azure_errors.json"

# Delegated Graph scopes the signed-in admin must consent to for /fix to work.
# Overridable so a deployment can request less than the full set.
AUTH_SCOPES = [
    s.strip() for s in os.getenv(
        "AZURE_AUTH_SCOPES",
        "User.ReadWrite.All,Application.ReadWrite.All,Directory.ReadWrite.All",
    ).split(",") if s.strip()
]


@app.get("/app", response_class=HTMLResponse)
def frontend():
    """Serve the demo UI."""
    if not _FRONTEND.exists():
        raise HTTPException(status_code=404, detail="frontend/index.html not found")
    return FileResponse(_FRONTEND, media_type="text/html")


@app.get("/auth/config")
def auth_config():
    """
    Public, non-secret config the browser needs to start an MSAL sign-in.

    Deliberately returns only the client (application) ID and tenant ID, both
    of which are public identifiers that appear in the sign-in URL anyway. The
    client *secret* is never sent here -- a single-page app has nowhere safe to
    put one, which is exactly why the app registration must list this page as a
    "Single-page application" redirect URI rather than a "Web" one.
    """
    client_id = os.getenv("AZURE_CLIENT_ID")
    tenant_id = os.getenv("AZURE_TENANT_ID")

    if not client_id or not tenant_id:
        return {
            "configured": False,
            "reason": (
                "AZURE_CLIENT_ID / AZURE_TENANT_ID are not set on this deployment, "
                "so interactive sign-in is disabled. /analyze and /escalate still work."
            ),
        }

    return {
        "configured": True,
        "client_id": client_id,
        "tenant_id": tenant_id,
        "scopes": AUTH_SCOPES,
        "demo_api_key_required": bool(DEMO_API_KEY),
    }


@app.get("/privacy", response_class=HTMLResponse)
def privacy_policy():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <title>AzureAutoFix — Privacy Policy</title>
      <style>
        body { font-family: 'Segoe UI', Tahoma, sans-serif; max-width: 720px; margin: 40px auto; padding: 0 20px; color: #222; line-height: 1.6; }
        h1 { color: #0078d4; }
        h2 { color: #0078d4; margin-top: 32px; font-size: 18px; }
        code { background: #f0f7ff; padding: 2px 6px; border-radius: 4px; }
        .updated { color: #777; font-size: 13px; }
      </style>
    </head>
    <body>
      <h1>⚡ AzureAutoFix — Privacy Policy</h1>
      <p class="updated">Last updated: June 2026</p>

      <p>AzureAutoFix is a Chrome extension and API that detects Azure Active
      Directory (Entra ID) sign-in errors (codes like <code>AADSTS900971</code>),
      explains them in plain English, and — for admin-level issues — can apply
      a fix directly via the Microsoft Graph API.</p>

      <h2>What the extension reads</h2>
      <p>The content script only scans the visible text, title, and URL of
      pages on <code>*.microsoft.com</code>, <code>*.azure.com</code>, and
      <code>*.microsoftonline.com</code> for AADSTS error codes
      (e.g. <code>AADSTS900971</code>). It does not read passwords, form
      fields, cookies, or any other page data.</p>

      <h2>What is sent to the backend</h2>
      <p>When an AADSTS error code is detected, only that error code is sent
      to the AzureAutoFix API (<code>/analyze</code>) for classification. If
      you choose to use the "Fix Now" feature, the extension additionally
      sends the access token, object IDs, and/or redirect URI <em>you
      manually enter</em> in the popup, so the backend can call the Microsoft
      Graph API on your behalf. Nothing is sent unless you click "Fix Now".</p>

      <h2>What is stored</h2>
      <p>The extension uses <code>chrome.storage.local</code> to remember the
      most recently detected error and any values you typed into the popup
      (access token, object IDs, redirect URI) so the popup can repopulate
      them. This data stays in your browser and is never synced or shared.</p>

      <h2>What the backend stores</h2>
      <p>The AzureAutoFix API is stateless — it does not log, store, or
      retain error codes, access tokens, or any request data beyond the
      lifetime of a single request. No analytics, tracking, or third-party
      data sharing is used.</p>

      <h2>Your MS Graph access token</h2>
      <p>If you provide an access token to use "Fix Now", it is used only to
      make the specific Microsoft Graph API call needed to resolve the
      detected error (e.g. adding a redirect URI), and is not stored or
      logged by the backend.</p>

      <h2>Contact</h2>
      <p>Questions about this policy can be directed via the
      <a href="https://github.com/aminabk99/AzureAutoFix">GitHub repository</a>.</p>
    </body>
    </html>
    """


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest, request: Request):
    """
    Step 1: Classify the error and determine fix path.
    Returns structured analysis including fix_category and action.
    """
    if not req.error_input.strip():
        raise HTTPException(status_code=400, detail="error_input cannot be empty")
    result = classify(req.error_input)

    # Annotate for the tracing middleware rather than having it re-parse the
    # request body -- see monitoring/middleware.py.
    request.state.error_code = result.get("error_code")
    request.state.source = result.get("source")
    request.state.confidence = result.get("confidence")
    return result


@app.post("/fix", response_model=FixResponse, dependencies=[Depends(require_api_key)])
async def fix_error(req: FixRequest):
    """
    Step 2 (admin only): Execute the fix via MS Graph API.
    Requires a valid admin access token.
    """
    # Use the caller-supplied token. Falling back to the app-level
    # client-credentials token is opt-in (ALLOW_APP_TOKEN_FALLBACK=true)
    # and intended for local/dev use only -- never enable it on a public
    # deployment, since it would let any caller mutate this Azure tenant.
    token = req.access_token
    if not token:
        if ALLOW_APP_TOKEN_FALLBACK:
            token = get_app_token()
        else:
            raise HTTPException(
                status_code=400,
                detail=(
                    "access_token is required for /fix on this deployment. "
                    "Set ALLOW_APP_TOKEN_FALLBACK=true in your local .env to "
                    "use app-level credentials for testing."
                ),
            )
    client = GraphAPIClient(token)
    fix_category = req.fix_category

    try:
        if fix_category == "admin_auto":
            result = await _dispatch_admin_fix(client, req)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Fix category '{fix_category}' cannot be auto-fixed. Use /escalate for admin_escalate errors."
            )
        return FixResponse(
            success=True,
            fix_applied=fix_category,
            details=result,
            error_code=req.error_code,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _dispatch_admin_fix(client: GraphAPIClient, req: FixRequest) -> str:
    """Route to the correct Graph API action based on error code."""
    code = req.error_code

    dispatch = {
        "AADSTS900971": lambda: client.add_redirect_uri(req.app_id, req.redirect_uri),
        "AADSTS50055":  lambda: client.force_password_reset(req.user_id),
        "AADSTS50057":  lambda: client.enable_account(req.user_id),
        "AADSTS700011": lambda: client.rotate_app_secret(req.app_id),
        "AADSTS90094":  lambda: client.grant_admin_consent(req.app_id),
        "AADSTS70011":  lambda: client.update_api_permissions(req.app_id, req.new_scope),
        "AADSTS50053":  lambda: client.unlock_account(req.user_id),
    }

    if code not in dispatch:
        return f"No automated fix registered for {code}. Manual review required."

    return await dispatch[code]()


@app.post("/escalate", response_model=EscalateResponse)
def escalate(req: AnalyzeRequest):
    """
    For errors requiring admin action when the current user is NOT an admin.
    Returns a pre-written email + Teams message the user can send.
    """
    result = classify(req.error_input)
    msg = generate_escalation_message(
        error_code=result["error_code"],
        explanation=result["explanation"],
        action_detail=result["action_detail"],
    )
    return msg


# -- GET /metrics -------------------------------------------------------------

@app.get("/metrics")
def get_metrics(last_n: int = 1000):
    """
    Returns p50/p95/p99 latency, per-endpoint breakdown, and error code
    frequency computed from the last N requests in monitoring/traces.jsonl.
    """
    try:
        return compute_metrics(last_n=last_n)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Metrics computation failed: {exc}") from exc
