"""
AzureAutoFix — FastAPI Backend
Run: uvicorn backend.main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.inference import classify, load_model
from backend.graph_api import GraphAPIClient
from backend.escalation import generate_escalation_message
from backend.auth import get_app_token

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


# ── Request/Response Models ─────────────────────────────────────────────────

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


# ── Routes ──────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "AzureAutoFix API running", "docs": "/docs"}


@app.get("/status")
def status():
    return {"status": "ok", "model": "loaded", "version": "1.0.0"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    """
    Step 1: Classify the error and determine fix path.
    Returns structured analysis including fix_category and action.
    """
    if not req.error_input.strip():
        raise HTTPException(status_code=400, detail="error_input cannot be empty")
    result = classify(req.error_input)
    return result


@app.post("/fix", response_model=FixResponse)
async def fix_error(req: FixRequest):
    """
    Step 2 (admin only): Execute the fix via MS Graph API.
    Requires a valid admin access token.
    """
    # Use provided token or fall back to app-level token from .env
    token = req.access_token or get_app_token()
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
