<div align="center">

# ⚡ AzureAutoFix
### Agentic Azure AD Error Resolution — Detect, Diagnose, and Fix in Under 10 Seconds

An agentic system that detects Azure AD (Entra ID) authentication errors, classifies them using a **three-paper AIOps research pipeline**, and resolves them automatically via the **Microsoft Graph API** — exposed as a FastAPI backend with a Chrome extension for live detection.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-LogBERT_+_DeepLog-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Azure](https://img.shields.io/badge/Azure-MS_Graph_API-0078D4?style=for-the-badge&logo=microsoftazure&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Railway_Deploy-2496ED?style=for-the-badge&logo=docker&logoColor=white)

**🔗 Live API:** [azureautofix-production.up.railway.app](https://azureautofix-production.up.railway.app) · [Swagger docs](https://azureautofix-production.up.railway.app/docs)

</div>

---

## What It Does

When a user hits an Azure AD error, AzureAutoFix detects it, figures out what's wrong, and either fixes it automatically or tells the user exactly what to do — in under 10 seconds.

**Three outcomes depending on error type:**
- **User-fixable** — plain English instructions for the end user (e.g., wrong password → reset it)
- **Auto-fix** — backend calls the Graph API and resolves the issue without manual steps (e.g., unlocks account, adds redirect URI)
- **Escalate** — generates a pre-written email + Teams message the user can send directly to IT

---

## Features

- **Chrome extension** watches for AADSTS error codes live on any page
- **Drain log parser** strips trace IDs and noise from raw Azure AD error strings, extracting the structured AADSTS code (He et al., ICWS 2017)
- **LogBERT classifier** uses a bidirectional Transformer encoder to predict the fix category — works on error codes it hasn't seen before, not just a lookup table (Guo et al., IJCNN 2021)
- **DeepLog anomaly detector** detects attack patterns like credential stuffing across a session by modeling normal error sequences with a 2-layer LSTM (Du et al., CCS 2017)
- **Microsoft Graph API integration** auto-resolves 7 admin-level errors (account unlock, password reset, redirect URI registration, etc.)
- **Auto-escalation** generates a pre-written IT message with error details and recommended action

### Supported Errors

| Error Code | Issue | Resolution |
|---|---|---|
| `AADSTS50053` | Account locked out | ✅ Auto-fix |
| `AADSTS50057` | Account disabled | ✅ Auto-fix |
| `AADSTS50055` | Password expired | ✅ Auto-fix |
| `AADSTS900971` | Redirect URI not registered | ✅ Auto-fix |
| `AADSTS90094` | Admin consent not granted | ✅ Auto-fix |
| `AADSTS70011` | Invalid scope | ✅ Auto-fix |
| `AADSTS700011` | Client secret expired | ✅ Auto-fix |
| `AADSTS50126` | Wrong credentials | 📋 User-guided |
| `AADSTS50076` | MFA required | 📋 User-guided |
| `AADSTS65001` | Permissions misconfigured | 📧 Escalate to admin |
| `AADSTS20050` | External user not found | 📧 Escalate to admin |
| Unknown codes | Any other AADSTS error | LogBERT infers fix category |

---

## Architecture

```
User hits Azure AD error
        │
        ▼
Chrome Extension detects AADSTS code on page
        │
        ▼
POST /analyze
  ├─ Drain parser      →  extracts log key from raw string
  └─ LogBERT encoder   →  classifies fix_category
        │
        ├── "user"           →  Show user what to do
        ├── "retry"          →  Tell user to retry
        ├── "admin_escalate" →  Generate IT email + Teams message
        └── "admin_auto"     →  Show "Fix Now" button
                                      │
                                      ▼
                               POST /fix (admin token required)
                                 └─ MS Graph API call → resolved ✅

[Optional] POST /analyze_sequence
  └─ DeepLog LSTM  →  scores anomaly across session error sequence
```

---

## Setup

**Requirements:** Python 3.11+ · Azure AD tenant with admin access

**1. Clone & install**
```bash
git clone https://github.com/aminabk99/AzureAutoFix
cd AzureAutoFix
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**2. Train the models**
```bash
python model/train_local.py      # LogBERT fix-category classifier
python model/train_sequence.py   # DeepLog LSTM anomaly detector
```

**3. Configure Azure**
```bash
cp .env.example .env
# Fill in AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID
```

Register an app in Azure Portal with these Graph API permissions (grant admin consent):
`User.ReadWrite.All` · `Application.ReadWrite.All` · `Directory.ReadWrite.All` · `DelegatedPermissionGrant.ReadWrite.All`

**4. Run**
```bash
uvicorn backend.main:app --reload --port 8000
```

**5. Load Chrome extension**

Open `chrome://extensions` → enable **Developer mode** → **Load unpacked** → select the `extension/` folder.

**Try the live demo** — no setup needed: [azureautofix-production.up.railway.app/docs](https://azureautofix-production.up.railway.app/docs)

> `/analyze` and `/escalate` are open and read-only. `/fix` requires an admin `access_token` and `X-API-Key` header.

---

## Stack

| Layer | Tech |
|---|---|
| Log Parsing | Drain fixed-depth parse tree (He et al., ICWS 2017) |
| Error Classifier | LogBERT bidirectional Transformer (Guo et al., IJCNN 2021) |
| Anomaly Detector | DeepLog 2-layer LSTM (Du et al., CCS 2017) |
| Backend | FastAPI + httpx |
| Graph API | Microsoft Graph API v1.0 |
| Extension | Chrome Manifest V3 |
| Deploy | Docker + Railway |

---

<div align="center">
  <sub>Built by <a href="https://github.com/aminabk99">Amina Bilal</a> · <a href="https://linkedin.com/in/amina-bilal-926340382">LinkedIn</a></sub>
</div>
