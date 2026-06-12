"""
AzureAutoFix — FastAPI Backend
Run: uvicorn backend.main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.inference import classify, load_model
from backend.graph_api import GraphAPIClient
from backend.escalation import generate_escalation_message
from backend.auth import get_app_token

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
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model on startup
@app.on_event("startup")
async def startup_event():
    try:
        load_model()
        print("[Startup] Model loaded successfully.")
    except FileNotFoundError:
        print("[Startup] WARNING: Model file not found. Train the model first (model/train.py).")


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
    source: str

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
    return {"status": "ok", "model": "loaded", "version": "1.0.0"}


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
def analyze(req: AnalyzeRequest):
    """
    Step 1: Classify the error and determine fix path.
    Returns structured analysis including fix_category and action.
    """
    if not 