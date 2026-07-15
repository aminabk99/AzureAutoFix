# AzureAutoFix

Chrome extension that catches Microsoft Entra (Azure AD) errors the moment they appear and resolves them — automatically where possible, with guided steps or a drafted IT email where not.

---

## What it does

Watches Microsoft login pages for AADSTS error codes in real time. The instant an error appears it:

1. Translates it into plain English — no Googling
2. Decides the right fix path based on what kind of problem it is
3. Acts — either walks you through fixing it yourself, calls the Graph API to fix it automatically, or writes the IT support email for you

**Three fix paths:**

- **You fix it** — wrong password, MFA not enrolled, account not yet provisioned. Exact steps shown inline.
- **Auto-fixed** — backend calls Microsoft Graph and resolves it without human involvement (unlock account, add redirect URI, rotate expired secret, etc.)
- **Escalate** — generates a pre-filled IT support email. You hit send.

---

## Error coverage

| Category | Count |
|----------|-------|
| Known errors handled with 100% confidence (lookup table) | 15 |
| Of those, auto-fixed via Graph API | 7 |
| Unknown errors classified by ML model | ∞ |

---

## The AI layer — three research papers

Most error-detection tools stop at pattern matching. AzureAutoFix layers three published techniques on top of the lookup table to handle errors it has never seen before and to detect attacks across a session.

### Drain — log parsing (ICWS 2017)
> He, P. et al. *Drain: An Online Log Parsing Approach with Fixed Depth Tree.* IEEE ICWS 2017.

Raw AADSTS error strings are noisy — they embed dynamic values (tenant IDs, redirect URIs, timestamps) that make direct comparison impossible. Drain parses each error string into a clean, stable template by building a fixed-depth prefix tree. The result is a structured log key that the downstream models can reason about consistently.

### LogBERT — error classification (IJCNN 2021)
> Guo, H. et al. *LogBERT: Log Anomaly Detection via BERT.* IJCNN 2021.

A bidirectional Transformer fine-tuned on tokenised log sequences. For errors not in the lookup table, LogBERT classifies which fix path applies — user error, auto-fixable, or escalation — based on learned representations of error semantics. Bidirectional context means it reads the full error message before deciding, not just the error code prefix.

### DeepLog — session-level attack detection (CCS 2017)
> Du, M. et al. *DeepLog: Anomaly Detection and Diagnosis from System Logs using Deep Learning.* ACM CCS 2017.

An LSTM that models the expected sequence of log events across a login session. Where Drain + LogBERT handle individual errors, DeepLog watches the pattern: repeated failures across accounts, rapid succession of different error codes, unusual sequences that match credential stuffing or brute force profiles. When the session pattern deviates from the learned normal, it raises an alert.

---

## CI pipeline — regression gating and quality metrics

The existing CI suite is extended with:

- **Regression gating** — each PR must pass all 15 known-error lookup cases before merge; any fix-path regression blocks the build
- **Classification accuracy gate** — LogBERT F1 on the held-out error set must stay above threshold; a model change that degrades classification fails CI
- **Auto-fix integration tests** — Graph API calls are mocked; the seven auto-fix paths are exercised end-to-end on every push
- **DeepLog sequence tests** — known attack sequences (credential stuffing pattern, brute force pattern) must be flagged; false-positive rate on normal sessions is tracked as a CI metric

---

## Setup

**Requirements:** Node 18+, Python 3.10+, a Microsoft Entra app registration with Graph API permissions.

```bash
git clone https://github.com/aminabk99/azureautofix
cd azureautofix
pip install -r requirements.txt        # backend + ML models
cd extension && npm install && npm run build
```

Load the built extension in Chrome: `chrome://extensions` → Developer mode → Load unpacked → `extension/dist`

Add to `.env`:

```
AZURE_CLIENT_ID=...
AZURE_TENANT_ID=...
AZURE_CLIENT_SECRET=...
```

---

## Tech

Chrome Extension (MV3) · FastAPI · Microsoft Graph API · MSAL · Drain · LogBERT (HuggingFace) · DeepLog (PyTorch) · pytest · GitHub Actions
