<div align="center">

# ⚡ AzureAutoFix
### Agentic Azure AD Error Resolution — Detect, Diagnose, and Fix in Under 10 Seconds

An agentic system that detects Azure Active Directory (Entra ID) authentication errors, classifies them using a **three-paper AIOps research pipeline** (Drain + LogBERT + DeepLog), and resolves them automatically via the **Microsoft Graph API**. Exposed via a **FastAPI** backend and a **Chrome extension** for live error detection.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-LogBERT_+_DeepLog-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Azure](https://img.shields.io/badge/Azure-MS_Graph_API-0078D4?style=for-the-badge&logo=microsoftazure&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Railway_Deploy-2496ED?style=for-the-badge&logo=docker&logoColor=white)

</div>

---

**🔗 Live API:** [azureautofix-production.up.railway.app](https://azureautofix-production.up.railway.app) · [API docs (Swagger)](https://azureautofix-production.up.railway.app/docs)

---

## What It Does

When a user or IT admin hits an Azure AD error like `AADSTS50126` or `AADSTS900971`, they normally have to Google it, dig through Microsoft docs, and figure out the fix manually. AzureAutoFix does that instantly — and for admin-level errors, applies the fix automatically via the Microsoft Graph API.

**Three outcomes depending on error type:**
- **User-fixable** — tells the end user exactly what to do in plain English (e.g., wrong password → reset it)
- **Auto-fix** — backend calls the Graph API and resolves the issue without any manual steps (e.g., unlocks account, adds redirect URI)
- **Escalate** — generates a pre-written email + Teams message the user can send directly to IT

---

## The Research Pipeline

AzureAutoFix implements three published AIOps papers as a sequential log analysis pipeline. This is the same methodology used in production monitoring systems at large-scale cloud providers.

### Stage 1 — Log Parsing: Drain (He et al., ICWS 2017)

**The problem:** Raw Azure AD errors are noisy strings like `"AADSTS50057 - User account has been disabled. Trace ID: abc-123-def-456"`. Before any ML can run, you need to strip the noise (trace IDs, UUIDs, timestamps) and extract the structured signal.

**What Drain does:** Builds a fixed-depth parse tree that groups similar log messages together and replaces variable parts (trace IDs, numeric IDs) with a wildcard `*`, producing a clean template. The AADSTS code becomes the canonical "log key" — the atomic unit everything downstream operates on.

**Our implementation** (`model/parser.py`): Fixed-depth tree with `depth=3`, `similarity_threshold=0.4`, `max_children=100` (Table II parameters from the paper). Regex preprocessing removes UUIDs, ISO timestamps, IPv4 addresses, and long numeric IDs before tree traversal.

> He, P., Zhu, J., Zheng, Z., & Lyu, M.R. (2017). Drain: An Online Log Parsing Approach with Fixed Depth Tree. *IEEE ICWS 2017.* DOI: 10.1109/ICWS.2017.13

---

### Stage 2 — Fix Classification: LogBERT (Guo et al., IJCNN 2021)

**The problem:** Given a structured log key (AADSTS code), determine what category of fix it needs — and do this for error codes the system has never seen before, not just a lookup table.

**What LogBERT does:** A bidirectional Transformer encoder with a `[DIST]` token (equivalent to BERT's `[CLS]`) that aggregates the whole sequence into a single representation. Bidirectional means every token attends to every other token simultaneously, encoding richer context than a left-to-right model. The Masked Log Key Prediction (MLKP) pre-training objective teaches the model which error codes tend to co-occur before fine-tuning on fix labels.

**Our implementation** (`model/logbert_classifier.py`): Lightweight replica with `d_model=128`, `num_heads=4`, `num_layers=2`, `d_ff=256`. Sinusoidal position embeddings (Vaswani et al. 2017). Two heads: MLKP head for pre-training, fix-category classifier head for inference. Evaluated with Leave-One-Out Cross-Validation (LOOCV, Kohavi 1995) on N=15 Azure AD error codes — the correct methodology for small datasets.

**The 4 fix categories:**

| Category | Meaning | Who acts |
|---|---|---|
| `user` | End user can fix it themselves | User |
| `retry` | Transient error, try again | User |
| `admin_auto` | Graph API call can fix it automatically | Backend |
| `admin_escalate` | Needs human admin investigation | IT team |

> Guo, H., Yuan, S., & Wu, X. (2021). LogBERT: Log Anomaly Detection via BERT. *IJCNN 2021.* arXiv:2103.04475

---

### Stage 3 — Sequence Anomaly Detection: DeepLog (Du et al., CCS 2017)

**The problem:** Individual errors can look routine while the *pattern across a session* reveals an attack. Three wrong-password errors (`AADSTS50126`) followed by an account lockout (`AADSTS50053`) is the textbook credential-stuffing signature — but per-error classification would just say "wrong password" three times.

**What DeepLog does:** Trains a 2-layer LSTM on *normal* authentication sessions only, learning what the next error code should be given the last h=5 errors. At inference, it predicts the top-g=3 most likely next codes. If the actual next code isn't in those top 3, that transition is flagged as anomalous. Anomaly score = fraction of anomalous transitions in the session.

**Our implementation** (`model/sequence_detector.py`): `DeepLogLSTM` — Embedding → 2-layer LSTM (hidden=64) → FC → logits. Window size h=5, top-g=3 (adapted from DeepLog's h=10, g=9 for HDFS, scaled down for shorter Azure AD sessions). Trained on 49 normal synthetic sessions. Heuristic fallback when weights unavailable: detects credential-stuffing (≥3× `AADSTS50126` + `AADSTS50053`) and retry storms (≥4 identical errors) via rule-based H1/H2 checks.

Exposed via `/analyze_sequence` endpoint: takes a list of recent error codes from a session, returns anomaly score, flagged transitions, and escalation recommendation.

> Du, M., Li, F., Zheng, G., & Srikumar, V. (2017). DeepLog: Anomaly Detection and Diagnosis from System Logs through Deep Learning. *CCS 2017.* DOI: 10.1145/3133956.3134015

---

## Supported Errors

### Auto-fixed via Microsoft Graph API (`admin_auto`)

These 7 errors are resolved automatically — the backend makes the exact Graph API call needed, no manual steps required. Requires an admin access token.

| Error Code | Issue | Graph API Action |
|---|---|---|
| `AADSTS50053` | Account locked out | Unlock account |
| `AADSTS50057` | Account disabled | Re-enable account |
| `AADSTS50055` | Password expired | Force password reset |
| `AADSTS900971` | Redirect URI not registered | Add redirect URI to app registration |
| `AADSTS90094` | Admin consent not granted | Grant admin consent |
| `AADSTS70011` | Invalid scope requested | Update API permissions |
| `AADSTS700011` | App client secret expired | Rotate client secret |

### User-fixable (`user`)

| Error Code | Issue | Guidance given |
|---|---|---|
| `AADSTS50126` | Wrong username or password | Reset credentials |
| `AADSTS50076` | MFA required | Complete MFA setup |
| `AADSTS50079` | MFA registration required | Register MFA method |
| `AADSTS50158` | External security challenge | Complete conditional access check |
| `AADSTS50020` | User not found in tenant | Contact IT to check guest access |

### Transient / retry (`retry`)

| Error Code | Issue | Action |
|---|---|---|
| `AADSTS90033` | Temporary Microsoft-side error | Wait and retry |

### Admin investigation required (`admin_escalate`)

| Error Code | Issue | Action |
|---|---|---|
| `AADSTS65001` | App permissions misconfigured | IT admin reviews app registration |
| `AADSTS50034` | User account does not exist | IT admin provisions account |

For error codes outside these 15, the LogBERT classifier predicts a fix category from the error description text — so the system degrades gracefully on unknown codes rather than returning an error.

---

## How It Works End-to-End

```
User hits Azure AD error
        │
        ▼
Chrome Extension detects AADSTS code on page
        │
        ▼
POST /analyze
  └─ Drain parser  →  extracts log key from raw string
  └─ LogBERT      →  classifies fix_category
  └─ Returns: fix_category, explanation, action, confidence
        │
        ├─── fix_category = "user"            →  Show user what to do
        ├─── fix_category = "retry"           →  Tell user to retry
        ├─── fix_category = "admin_escalate"  →  Generate email + Teams message
        └─── fix_category = "admin_auto"      →  Show "Fix Now" button
                                                        │
                                                        ▼
                                                 POST /fix (admin token required)
                                                   └─ Graph API call
                                                   └─ Issue resolved ✅

[Optional] POST /analyze_sequence
  └─ Drain parser  →  extracts log keys from session
  └─ DeepLog LSTM  →  scores anomaly across sequence
  └─ If anomalous  →  escalate to security regardless of per-error category
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/status` | API version + model status |
| `POST` | `/analyze` | Classify a single error → fix category + action |
| `POST` | `/fix` | Apply auto-fix via Graph API (admin token required) |
| `POST` | `/escalate` | Generate IT escalation email + Teams message |
| `POST` | `/analyze_sequence` | DeepLog session-level anomaly detection |
| `GET` | `/privacy` | Privacy policy (Chrome Web Store requirement) |

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

**2. Train the models**

```bash
# Fix-category classifier (LogBERT-style, LOOCV evaluation)
python model/train_local.py

# Sequence anomaly detector (DeepLog LSTM)
python model/train_sequence.py
```

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
uvicorn backend.main:app --reload --port 8000
```

**5. Load Chrome extension**

1. Open `chrome://extensions`
2. Enable **Developer mode**
3. Click **Load unpacked** → select the `extension/` folder

**Optional: Collect real Azure AD logs for retraining**

```bash
# Requires AuditLog.Read.All application permission + admin consent
python model/collect_azure_logs.py --days 30
python model/train_sequence.py  # retrain on real session data
```

---

## Project Structure

```
AzureAutoFix/
├── model/
│   ├── parser.py              # Stage 1: Drain log parser (ICWS 2017)
│   ├── logbert_classifier.py  # Stage 2: LogBERT Transformer classifier (IJCNN 2021)
│   ├── sequence_detector.py   # Stage 3: DeepLog LSTM anomaly detector (CCS 2017)
│   ├── train_local.py         # LOOCV training for fix-category classifier
│   ├── train_sequence.py      # DeepLog LSTM training on normal sequences
│   ├── collect_azure_logs.py  # Microsoft Graph API log collector (real data)
│   ├── inference.py           # Model loader + classify() function
│   ├── azure_error_model.pt   # Trained fix-category weights
│   ├── sequence_model.pt      # Trained DeepLog weights
│   └── sequence_vocab.json    # DeepLog vocabulary
├── backend/
│   ├── main.py                # FastAPI app + routes
│   ├── analyze_sequence.py    # /analyze_sequence endpoint router
│   ├── graph_api.py           # MS Graph API client
│   ├── escalation.py          # Admin message generator
│   └── auth.py                # OAuth2 client-credentials token
├── data/
│   ├── azure_errors.json      # 15 labeled Azure AD errors (training data)
│   └── synthetic_sequences.json  # 49 normal + 8 anomalous sessions (DeepLog training)
├── extension/
│   ├── manifest.json          # Chrome Manifest V3
│   ├── content.js             # DOM watcher (error detection)
│   ├── background.js          # Service worker (API calls)
│   ├── popup.html             # Extension popup UI
│   └── popup.js               # Popup logic
├── .github/workflows/
│   └── test.yml               # CI: pytest on every push
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

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

## Security

- `/fix` requires a valid `access_token` from the caller — no silent fallback to app-level Graph credentials
- `/fix` requires an `X-API-Key` header matching `DEMO_API_KEY`
- `/analyze`, `/escalate`, and `/analyze_sequence` are read-only and open on the live deployment
- `ALLOW_APP_TOKEN_FALLBACK` must remain unset/`false` on any public deployment

---

## Future Work

- **Expanded error coverage** — grow `data/azure_errors.json` from 15 to 40-50 of the most common AADSTS codes, covering ~95% of real-world Azure AD issues. Each new entry expands the lookup table the system handles at 100% confidence
- **Fine-tuned DistilBERT** — replace the custom lightweight Transformer with a pre-trained `distilbert-base-uncased` fine-tuned on Azure error descriptions. Unlike the current model which learns from 15 labeled examples, DistilBERT understands English and can predict fix categories for error codes it has never seen — making the system genuinely generalize beyond the lookup table
- **More auto-fixable errors** — extend the Graph API client with additional write operations (conditional access policy adjustments, guest account provisioning, license assignment) to grow the `admin_auto` category beyond the current 7 errors
- **Real training data** — connect `collect_azure_logs.py` to a live Azure tenant and retrain DeepLog on real sign-in session logs for production-grade anomaly detection, replacing the current 49 synthetic sequences
- **Real-time streaming** — hook into Azure Event Hub to run DeepLog on live sign-in events as they arrive rather than analyzing sessions after the fact
- **Multi-tenant support** — extend the Graph API client to manage multiple Azure tenants from a single deployment
- **Feedback loop** — log whether auto-fixes succeeded and retrain the classifier on failure cases to improve accuracy over time

---

## Resume Bullet

> Built an agentic Azure AD error resolution system implementing three AIOps research papers (Drain ICWS 2017, LogBERT IJCNN 2021, DeepLog CCS 2017) as a production log analysis pipeline — parsing raw error strings, classifying fix categories via a bidirectional Transformer with LOOCV evaluation, detecting session-level attack patterns (credential stuffing, brute force) via LSTM sequence modeling, and resolving admin-level errors automatically via the Microsoft Graph API

---

<div align="center">
  <sub>Built by <a href="https://github.com/aminabk99">Amina Bilal</a> · <a href="https://linkedin.com/in/amina-bilal-926340382">LinkedIn</a></sub>
</div>
