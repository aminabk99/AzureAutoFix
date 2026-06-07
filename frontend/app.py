"""
AzureAutoFix — Streamlit Frontend
Run: streamlit run frontend/app.py
"""

import streamlit as st
import requests
import time
import json
import re

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="AzureAutoFix",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Styles ───────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main { padding-top: 2rem; }
    .stButton > button {
        background-color: #0078d4;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
    }
    .stButton > button:hover { background-color: #106ebe; }
    .fix-card {
        background: #f0f7ff;
        border-left: 4px solid #0078d4;
        border-radius: 6px;
        padding: 1rem 1.25rem;
        margin: 0.75rem 0;
    }
    .reasoning-box {
        background: #1e1e1e;
        color: #d4d4d4;
        font-family: monospace;
        font-size: 0.85rem;
        padding: 1rem;
        border-radius: 6px;
        white-space: pre-wrap;
    }
    .badge-user   { background:#e6f4ea; color:#1e7e34; padding:2px 10px; border-radius:12px; font-size:0.8rem; }
    .badge-admin  { background:#fff3cd; color:#856404; padding:2px 10px; border-radius:12px; font-size:0.8rem; }
    .badge-auto   { background:#cce5ff; color:#004085; padding:2px 10px; border-radius:12px; font-size:0.8rem; }
    .badge-escalate { background:#f8d7da; color:#721c24; padding:2px 10px; border-radius:12px; font-size:0.8rem; }
    h1 { font-size: 2rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────

st.markdown("# ⚡ AzureAutoFix")
st.markdown("**AI-powered Azure AD error resolution.** Paste an error code and let the agent fix it.")

tab_analyze, tab_history, tab_about = st.tabs(["Analyze & Fix", "Fix Log", "About"])

# ── Session state ─────────────────────────────────────────────────────────────

if "fix_history" not in st.session_state:
    st.session_state.fix_history = []

# ── Tab 1: Analyze & Fix ──────────────────────────────────────────────────────

with tab_analyze:
    st.markdown("### Paste your error")
    error_input = st.text_input(
        "",
        placeholder="e.g. AADSTS900971 or 'User cannot sign in — admin consent required'",
        label_visibility="collapsed",
    )

    with st.expander("Optional: provide context for auto-fix"):
        access_token = st.text_input("MS Graph Access Token (admin only)", type="password")
        user_id = st.text_input("User Object ID (for user account fixes)")
        app_id = st.text_input("App Registration Object ID (for app fixes)")
        redirect_uri = st.text_input("Redirect URI to add (for AADSTS900971)")

    analyze_btn = st.button("Analyze Error", use_container_width=True)

    if analyze_btn and error_input.strip():
        with st.spinner("Analyzing with AI..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/analyze",
                    json={"error_input": error_input, "access_token": access_token or None},
                    timeout=10,
                )
                resp.raise_for_status()
                result = resp.json()
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to backend. Make sure FastAPI is running on port 8000.")
                st.stop()
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        # ── Results ──────────────────────────────────────────────────────────

        fc = result["fix_category"]
        st.markdown("---")
        st.markdown(f"**Error:** `{result['error_code']}`")

        # Badge
        if fc == "user":
            st.markdown('<span class="badge-user">User Action</span>', unsafe_allow_html=True)
        elif fc in ("admin_auto",):
            st.markdown('<span class="badge-auto">Auto-Fix Available</span>', unsafe_allow_html=True)
        elif fc == "admin_escalate":
            st.markdown('<span class="badge-escalate">Admin Required</span>', unsafe_allow_html=True)
        elif fc == "retry":
            st.markdown('<span class="badge-user">Retry</span>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="fix-card">
        <b>What happened:</b> {result['explanation']}<br><br>
        <b>Fix path:</b> {result['action_detail']}<br><br>
        <b>Message to user:</b> {result['user_message']}
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Show reasoning"):
            st.markdown(f"""
            <div class="reasoning-box">Classified as: {fc}
Confidence: {result['confidence']:.0%}
Source: {result['source']}

Reasoning: {result['reasoning']}</div>
            """, unsafe_allow_html=True)

        # ── State-specific UI ─────────────────────────────────────────────

        if fc == "admin_auto" and access_token:
            st.markdown("### 🔄 Auto-Fix")
            fix_btn = st.button("Fix Now (via MS Graph API)", use_container_width=True)
            if fix_btn:
                progress = st.progress(0, text="Connecting to MS Graph API...")
                for i, (pct, msg) in enumerate([
                    (25, "Authenticating..."),
                    (50, f"Applying fix for {result['error_code']}..."),
                    (80, "Verifying change..."),
                    (100, "Done!"),
                ]):
                    time.sleep(0.6)
                    progress.progress(pct, text=msg)
                try:
                    fix_resp = requests.post(
                        f"{API_BASE}/fix",
                        json={
                            "error_code": result["error_code"],
                            "fix_category": fc,
                            "access_token": access_token,
                            "user_id": user_id or None,
                            "app_id": app_id or None,
                            "redirect_uri": redirect_uri or None,
                        },
                        timeout=15,
                    )
                    fix_resp.raise_for_status()
                    fix_result = fix_resp.json()
                    st.success(f"Fix applied: {fix_result['details']}")
                    st.session_state.fix_history.append({
                        "error": result["error_code"],
                        "fix": fix_result["details"],
                        "category": fc,
                    })
                except Exception as e:
                    st.error(f"Fix failed: {e}")

        elif fc == "admin_auto" and not access_token:
            st.info("Provide an admin MS Graph access token above to auto-fix this error.")

        elif fc == "user":
            st.markdown("### 📋 Steps to fix")
            steps = _get_user_steps(result["action"], result["error_code"])
            for i, step in enumerate(steps, 1):
                st.markdown(f"**{i}.** {step}")

        elif fc == "admin_escalate":
            st.markdown("### 📧 Escalate to Admin")
            try:
                esc_resp = requests.post(
                    f"{API_BASE}/escalate",
                    json={"error_input": error_input},
                    timeout=10,
                )
                esc_resp.raise_for_status()
                esc = esc_resp.json()
                st.markdown(f"**Subject:** `{esc['subject']}`")
                st.text_area("Email body (copy and send to your IT admin)", esc["body"], height=200)
                st.text_area("Teams message", esc["teams_message"], height=80)
            except Exception as e:
                st.error(f"Escalation failed: {e}")

        elif fc == "retry":
            st.markdown("### 🔄 Retry")
            st.info("This was a transient Azure AD error. Wait 10 seconds and try your action again.")


def _get_user_steps(action: str, error_code: str) -> list:
    steps_map = {
        "prompt_reauth": [
            "Click **Sign out** from the application.",
            "Clear your browser cookies for `login.microsoftonline.com`.",
            "Sign in again with your correct username and password.",
            "If you've forgotten your password, click **Forgot my password** on the sign-in screen.",
        ],
        "trigger_interactive_login": [
            "Close all browser tabs for the application.",
            "Open a new private/incognito window.",
            "Navigate back to the application and sign in again.",
            "Complete any MFA prompt if one appears.",
        ],
        "trigger_consent_flow": [
            "When the application asks for permissions, click **Accept**.",
            "If you don't see a prompt, try: add `?prompt=consent` to the app URL.",
            "If you're blocked from consenting, your admin may have restricted user consent — use the escalation option.",
        ],
        "redirect_correct_tenant": [
            "Check you're signed into the correct Microsoft account (work vs personal).",
            "Sign out and sign back in with your **work or school account** (e.g. `you@yourcompany.com`).",
            "If the problem persists, your admin needs to register the application in your tenant.",
        ],
    }
    return steps_map.get(action, ["Follow the instructions provided by your application or IT admin."])


# ── Tab 2: Fix Log ────────────────────────────────────────────────────────────

with tab_history:
    st.markdown("### Fix Log")
    if not st.session_state.fix_history:
        st.info("No fixes applied yet this session.")
    else:
        for entry in reversed(st.session_state.fix_history):
            st.markdown(f"""
            <div class="fix-card">
            <b>Error:</b> <code>{entry['error']}</code> &nbsp;
            <span class="badge-auto">{entry['category']}</span><br>
            <b>Fix:</b> {entry['fix']}
            </div>
            """, unsafe_allow_html=True)

# ── Tab 3: About ──────────────────────────────────────────────────────────────

with tab_about:
    st.markdown("""
### AzureAutoFix

An agentic Azure AD error resolution system built with:
- **From-scratch transformer** trained on Azure AD error taxonomy
- **MS Graph API** for automated admin-level fixes
- **FastAPI** backend for reasoning + execution
- **Streamlit** frontend with 3 resolution modes

#### Supported Errors
| Code | Issue | Fix |
|------|-------|-----|
| AADSTS900971 | Missing redirect URI | Auto (Graph API) |
| AADSTS50057 | Account disabled | Auto (Graph API) |
| AADSTS50055 | Password expired | Auto (Graph API) |
| AADSTS700011 | Invalid client creds | Auto (Graph API) |
| AADSTS90094 | Admin consent required | Auto (Graph API) |
| AADSTS50053 | Account locked | Auto (Graph API) |
| AADSTS65001 | No user consent | User-guided |
| AADSTS50126 | Invalid credentials | User-guided |
| AADSTS50058 | Silent sign-in failed | User-guided |
| AADSTS700016 | App not in tenant | User-guided |
| AADSTS20050 | External user not found | Escalate to admin |

#### Architecture
```
Chrome Extension → FastAPI Backend → LLM Classifier → MS Graph API
                                   ↓
                             Streamlit Frontend
```
    """)
