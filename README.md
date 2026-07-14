<div align="center">

# ⚡ AzureAutoFix
### Agentic Azure AD Error Resolution — Detect, Diagnose, and Fix in Under 10 Seconds

An agentic system that detects Azure AD authentication errors, classifies them with a **from-scratch PyTorch transformer**, and resolves them automatically via the **MS Graph API**. Exposed via a **FastAPI** backend, a **Streamlit** web app, and a **Chrome extension** for live error detection.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-From--Scratch_Transformer-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Azure](https://img.shields.io/badge/Azure-MS_Graph_API-0078D4?style=for-the-badge&logo=microsoftazure&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Railway_Deploy-2496ED?style=for-the-badge&logo=docker&logoColor=white)

</div>

---

**🔗 Live API:** [azureautofix-production.up.railway.app](https://azureautofix-production.up.railway.app) · [API docs (Swagger)](https://azureautofix-production.up.railway.app/docs)

![Demo: AzureAutoFix detects an AADSTS900971 error and resolves it live via the MS Graph API](assets/demo.gif)

---

## How It Works

1. **Detect** — the Chrome extension watches for Azure AD error codes (e.g. `AADSTS900971`) automatically, or you paste an error into the Streamlit web app
2. **Classify** — a from-scratch PyTorch transformer, trained on the Azure AD error taxonomy, identifies the error type and root cause
3. **Route** — each error is routed to one of three paths: auto-fix, user-guided steps, or admin escalation with a pre-drafted message
4. **Fix** — for admin-resolvable errors, the backend calls the MS Graph API directly and resolves the issue in seconds

---

## Architecture

<div align="center">
  <img src="assets/architecture.svg" alt="AzureAutoFix architecture: Chrome Extension detects errors, FastAPI backend classifies via a from-scratch transformer and resolves via MS Graph API" width="650">
</div>

---

## Supported Errors

| Error Code | Issue | Resolution |
|---|---|---|
| `AADSTS900971` | Missing redirect URI | ✅ Auto-fix via Graph API |
| `AADSTS50057` | Account disabled | ✅ Auto-fix via Graph API |
| `AADSTS50055` | Password expired | ✅ Auto-fix via Graph API |
| `AADSTS700011` | Invalid client credentials | ✅ Auto-fix via Graph API |
| `AADSTS90094` | Admin consent required | ✅ Auto-fix via Graph API |
| `AADSTS70011` | Invalid scope | ✅ Auto-fix via Graph API |
| `AADSTS50053` | Account locked out | ✅ Auto-fix via Graph API |
| `AADSTS50126` | Invalid credentials | 📋 User-guided steps |
| `AADSTS50058` | Silent sign-in failed | 📋 User-guided steps |
| `AADSTS65001` | User consent missing | 📋 User-guided steps |
| `AADSTS700016` | App not in tenant | 📋 User-guided steps |
| `AADSTS50076` | MFA required | 📋 User-guided steps |
| `AADSTS700082` | Refresh token expired | 📋 User-guided steps |
| `AADSTS20050` | External user not found | 📧 Escalate to admin |
| `AADSTS90033` | Transient error | 🔄 Retry |

---

## Setup

**Requirements:** Python 3.11+ · Azure AD tenant with admin access

**1. Clone & install**

```bash
git clone https://github.com/aminabk99/AzureAutoFix
cd AzureAutoFix
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**2. Train the model**

Open `model/train.py` in [Google Colab](https://colab.research.google.com/) (free GPU), run all cells, then download into `model/`:
- `azure_error_model.pt`
- `vocab.json`

**3. Configure Azure**

```bash
cp .env.example .env
# Fill in AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID
```

Register an app in Azure Portal with these Graph API permissions (grant admin consent):
- `User.ReadWrite.All`
- `Application.ReadWrite.All`
- `Directory.ReadWrite.All`
- `DelegatedPermissionGrant.ReadWrite.All`

**4. Run**

```bash
# Backend
uvicorn backend.main:app --reload --port 8000

# Frontend (new terminal)
streamlit run frontend/app.py
```

Or with Docker:

```bash
docker-compose up --build
```

**5. Load Chrome extension**

1. Open `chrome://extensions`
2. Enable **Developer mode**
3. Click **Load unpacked** → select the `extension/` folder

---

## Project Structure

```
AzureAutoFix/
├── model/
│   ├── train.py          # From-scratch transformer training (run in Colab)
│   ├── inference.py      # Model loader + classify() function
│   ├── azure_error_model.pt  # Trained weights (after training)
│   └── vocab.json        # Token vocabulary (after training)
├── backend/
│   ├── main.py           # FastAPI routes
│   ├── graph_api.py      # MS Graph API client
│   └── escalation.py     # Admin message generator
├── frontend/
│   └── app.py            # Streamlit web app
├── extension/
│   ├── manifest.json     # Chrome extension config
│   ├── content.js        # DOM watcher (error detection)
│   ├── background.js     # Service worker (API calls)
│   ├── popup.html        # Extension popup UI
│   └── popup.js          # Popup logic
├── data/
│   └── azure_errors.json # Training dataset (15 errors with causes + fixes)
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Stack

| Layer | Tech |
|---|---|
| LLM | From-scratch PyTorch transformer (trained on Azure AD taxonomy) |
| Backend | FastAPI + httpx |
| Graph API | MS Graph API v1.0 |
| Frontend | Streamlit |
| Extension | Chrome Manifest V3 |
| Deploy | Docker + Railway |

---

## Hardest Part

**Mapping free-text Azure AD errors to structured fixes.** Azure AD error messages bury the actionable signal inside verbose, inconsistent strings (`AADSTS900971: ...`). The from-scratch transformer was trained on a hand-built taxonomy of 15 error codes — each tagged with cause, severity, and resolution path — so classification output maps directly to a Graph API call or escalation template, not just a label.

## Most Interesting

**The `/fix` security model.** Because `/fix` can perform real write operations against a live Azure AD tenant, the public deployment requires both a caller-supplied `access_token` (no silent fallback to app-level credentials) and an `X-API-Key` header. `/analyze` and `/escalate` stay open and read-only, so the live demo is safe to share without exposing tenant write access.

---

## Security

- `/fix` requires a valid `access_token` from the caller — there is **no** silent fallback to app-level Graph credentials
- `/fix` requires an `X-API-Key` header matching `DEMO_API_KEY`
- `/analyze` and `/escalate` are read-only (no Graph writes) and open on the live deployment
- `ALLOW_APP_TOKEN_FALLBACK` must stay unset/`false` on any public deployment — it's local/dev only

| Variable | Required for public deploy? | Purpose |
|---|---|---|
| `DEMO_API_KEY` | Recommended | If set, `/fix` requires a matching `X-API-Key` header |
| `ALLOW_APP_TOKEN_FALLBACK` | Leave unset/`false` | If `true`, `/fix` falls back to app-level Graph credentials when no `access_token` is supplied. **Local/dev only** |
| `AZURE_CLIENT_ID` / `AZURE_CLIENT_SECRET` / `AZURE_TENANT_ID` | Local only | App registration creds for the client-credentials fallback above |

---

## Deployment

The backend is deployed on [Railway](https://railway.app) (free tier) directly from this GitHub repo, built via `Dockerfile.backend`.

To deploy your own instance: create a Railway project from this repo, set the build's Dockerfile path to `Dockerfile.backend`, expose port `8000`, and generate a domain.

---

## Resume Bullet

> Built an agentic Azure AD error resolution system featuring a from-scratch transformer trained on AD error taxonomies, MS Graph API integration for automated fixes, and role-aware remediation — resolving admin-level issues in under 10 seconds with full reasoning transparency

---

<div align="center">
  <sub>Built by <a href="https://github.com/aminabk99">Amina Bilal</a> · <a href="https://linkedin.com/in/amina-bilal-926340382">LinkedIn</a></sub>
</div>
