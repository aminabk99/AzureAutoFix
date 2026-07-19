<div align="center">

# ⚡ AzureAutoFix

### Detect, explain, and resolve Azure AD sign-in errors — in the browser, in three languages

A Chrome extension + **FastAPI** backend that catches Microsoft Entra (**Azure AD**) sign-in errors the instant they appear, explains them in plain English, and routes each to the right fix — **guided steps**, an **automatic Graph API remediation**, or a **drafted escalation email**. Errors outside the curated set are resolved by a **hybrid retrieval** layer over the full 350-code AADSTS catalog that cites the Microsoft doc it matched — and abstains when it isn't sure.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-from--scratch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![MS Graph](https://img.shields.io/badge/MS_Graph_API-0078D4?style=for-the-badge&logo=microsoftazure&logoColor=white)
![Chrome MV3](https://img.shields.io/badge/Chrome_MV3-Extension-4285F4?style=for-the-badge&logo=googlechrome&logoColor=white)
![i18n](https://img.shields.io/badge/i18n-EN_·_ES_·_FR-4CAF50?style=for-the-badge)

<img width="880" alt="AzureAutoFix web UI" src="https://github.com/user-attachments/assets/635b3501-3ead-4db2-9fa5-95e57a683e5b" />

<img src="assets/demo.gif" alt="AzureAutoFix detects an AADSTS error and resolves it live via the MS Graph API" width="760" />

</div>

---

## How It Works

Watches Microsoft login pages for AADSTS error codes in real time. The instant an error appears it:

1. **Detects** the error code (Chrome extension) or takes a pasted code / plain-English description (web app)
2. **Classifies** it through three tiers — cheapest and most certain first
3. **Routes** it to one of three fix paths based on what kind of problem it is
4. **Acts** — walks you through the fix, resolves it via the Graph API, or drafts the escalation email

**Three fix paths:**

- **You fix it** — wrong password, MFA not enrolled, account not provisioned. Exact steps shown inline.
- **Auto-fix** — given an admin's Graph access token and the target's object ID, the backend calls Microsoft Graph directly (unlock account, add redirect URI, rotate expired secret, …), triggered from the Chrome extension or the `/fix` endpoint. The web app shows the manual portal steps for the same codes.
- **Escalate** — generates a pre-filled IT support email. You hit send.

<div align="center"><img src="assets/architecture.svg" alt="Architecture: Chrome extension detects errors, FastAPI backend classifies via a three-tier pipeline and resolves via MS Graph API" width="760" /></div>

---

## Error coverage

Resolution runs in three tiers. Each tier only sees what the one above it couldn't answer.

| Tier | Mechanism | Codes | Latency (p50) | Confidence |
|------|-----------|-------|---------------|------------|
| 1 | Curated lookup table | 15 | ~2.6 ms | 1.00 |
| 2 | Hybrid retrieval over the full AADSTS catalog | 350 | ~4.9 ms | 0.45–0.92 |
| 3 | From-scratch transformer | fallback | model-bound | gated ≥ 0.55 |
| — | **Abstain** | — | — | 0.00 |

Of the 15 curated codes, 7 are auto-fixable via the Graph API (given an admin token + object ID).

**Why 350 and not 15.** The original design classified *which error* occurred, which caps coverage at whatever is hand-labelled. The retrieval layer classifies *which remediation applies* instead — there are hundreds of AADSTS codes but only ~20 things you can actually do about them, so the label space stays bounded while coverage grows 23×. `model/build_catalog.py` parses Microsoft's published error reference into `data/aadsts_catalog.json` and auto-labels each code by action.

**Abstain is a first-class outcome.** When retrieval scores below the out-of-domain floor, or the model falls under its confidence gate, the API returns `abstained: true` at confidence 0.0 rather than a plausible-looking guess. A wrong automated write against a live tenant costs far more than saying "I don't know."

### Measured accuracy

| Metric | Result | How it's measured |
|--------|--------|-------------------|
| Resolves the 15 curated codes | **100%** | Checked in CI on every push |
| Retrieves the right code (top-3) | **98.9%** | Self-retrieval across all 349 catalog descriptions |
| Retrieves the right code (top-1) | **95.1%** | Description-only query |
| **Right fix from plain-English phrasing** | **91.7%** | 12 hand-written admin queries, none verbatim in the corpus |
| Out-of-domain input correctly abstained | **4/4** | Deliberate off-topic queries |
| p50 latency, lookup path | **2.6 ms** | Live request traces |

All enforced in CI via `python -m monitoring.eval_retrieval`. The two paraphrase "misses" retrieve a different code with an *identical* remediation, so the user gets the right fix either way — which is why action accuracy, not code accuracy, is the number that matters.

**The transformer tier scores 53.3%** under leave-one-out cross-validation against a 47% majority-class baseline — one extra correct answer out of fifteen. That's a dataset limit, not a modelling one: N=15, and two of the four classes have a single example each, so LOOCV leaves nothing to train on for them. It's reported rather than hidden, because it's the reason retrieval runs first and the model sits behind a 0.55 confidence floor. Training is seeded (`SEED`, default 42) so the figure reproduces.

---

## The AI layer

Most error-detection tools stop at pattern matching. AzureAutoFix layers published techniques on top of the lookup table so it can handle errors it has never seen — and say so when it can't.

### Drain — log parsing (ICWS 2017)
> He, P. et al. *Drain: An Online Log Parsing Approach with Fixed Depth Tree.* IEEE ICWS 2017.

Raw AADSTS strings embed dynamic values (tenant IDs, redirect URIs, timestamps) that make direct comparison impossible. Drain parses each error into a stable log key via a fixed-depth parse tree, so the same error against two tenants collapses to one key. Implemented in `model/parser.py`, called from `backend/analyze_sequence.py`.

### DeepLog — session-level attack detection (CCS 2017)
> Du, M. et al. *DeepLog: Anomaly Detection and Diagnosis from System Logs using Deep Learning.* ACM CCS 2017.

A 2-layer LSTM that models the expected sequence of log keys across a login session. Where Drain and retrieval handle individual errors, DeepLog watches the *pattern* — if the observed next key falls outside the model's top-g predictions, the session is flagged. That catches credential-stuffing and brute-force shapes that look unremarkable one error at a time. Implemented in `model/sequence_detector.py` (`WINDOW_SIZE=5`, `TOP_G=3`, `NUM_LAYERS=2`), weights in `model/sequence_model.pt`.

### Hybrid retrieval — unseen error resolution

Classifying the fix path for errors outside the curated set is handled by retrieval (`model/retrieval.py`), not a neural classifier: BM25 over word tokens fused with character-3gram TF-IDF via Reciprocal Rank Fusion, reranked on exact-code and symbolic-name matches.

A deliberate trade. A fine-tuned encoder is the fashionable answer, but the corpus is 350 short, highly templated documents with very high lexical overlap. Retrieval is ~1 ms/query, adds no model weights to a free-tier container, and — critically — **cites the specific Microsoft doc it matched**, so every answer is auditable. It scores 95.1% top-1 and 91.7% action accuracy on unseen phrasing.

> `model/logbert_classifier.py` is a from-scratch bidirectional Transformer encoder written for this role. It is **not currently wired into the request path** — no trained checkpoint, not imported by the backend — and is kept as reference for a future A/B against retrieval. Listed here rather than left implied, because a repo shouldn't suggest a component is live when it isn't.

---

## Setup

**Requirements:** Python 3.11+. A Microsoft Entra app registration is needed only for the auto-fix write path (extension / `/fix`) — analysis works without one.

**1. Clone & install**
```bash
git clone https://github.com/aminabk99/AzureAutoFix
cd AzureAutoFix
pip install -r requirements.txt
```

**2. Build the catalog & run the API**
```bash
python model/build_catalog.py            # 350 AADSTS codes -> data/aadsts_catalog.json
uvicorn backend.main:app --port 8000
```
Open **http://localhost:8000/app** for the web UI, or `/docs` for the interactive API.

**3. (Optional) Load the Chrome extension** — plain Manifest V3, no build step:
`chrome://extensions` → Developer mode → Load unpacked → select `extension/`.

`torch` is optional. Tiers 1 and 2 (lookup + retrieval) are pure stdlib and answer virtually all traffic; without torch the service reports the model tier as unavailable via `/status` rather than failing to start — which is why CI runs the quality gates in ~15 s without it.

For the auto-fix write path, add an Entra app registration to `.env`:
```
AZURE_CLIENT_ID=...
AZURE_TENANT_ID=...
AZURE_CLIENT_SECRET=...
```

---

## Project Structure

```
AzureAutoFix/
├── backend/
│   ├── main.py              # FastAPI routes: /analyze /fix /escalate /metrics /i18n /app
│   ├── i18n.py              # Locale loading, Accept-Language negotiation, response localisation
│   ├── graph_api.py         # MS Graph client — the real auto-fix write calls
│   ├── escalation.py        # Escalation-email generator
│   └── analyze_sequence.py  # DeepLog sequence-anomaly endpoint
├── model/
│   ├── inference.py         # Three-tier classify(): lookup → retrieval → transformer → abstain
│   ├── retrieval.py         # BM25 + char-ngram hybrid retrieval with RRF
│   ├── build_catalog.py     # Parses the AADSTS reference into aadsts_catalog.json
│   ├── parser.py            # Drain log parser
│   ├── sequence_detector.py # DeepLog LSTM
│   ├── train_local.py       # From-scratch transformer training (seeded LOOCV)
│   └── logbert_classifier.py# Reference encoder (not wired in)
├── frontend/index.html      # Single-page UI served at /app (EN · ES · FR)
├── extension/               # Chrome MV3 extension — live detection + token-based fix
├── data/
│   ├── azure_errors.json    # 15 curated errors
│   ├── aadsts_catalog.json  # 350-code catalog (generated)
│   └── i18n/{en,es,fr}.json # UI + content locales
├── monitoring/
│   ├── middleware.py        # Non-blocking latency tracing
│   ├── writer.py            # Background trace writer
│   ├── metrics.py           # p50/p95/p99 + classification-source split
│   ├── check_i18n.py        # Locale completeness gate
│   ├── regression_gate.py   # 15-code classification gate
│   └── eval_retrieval.py    # Retrieval coverage gate
└── .github/workflows/       # Quality Gates + Tests
```

---

## API

**POST `/analyze`** — classify an error and return the fix path
```json
{ "error_input": "AADSTS50053", "lang": "en" }
```
Returns `error_code`, `fix_category`, `explanation`, `action`, `manual_steps`, `confidence`, `source` (`lookup` \| `retrieval` \| `model` \| `abstain`), `citations`, and `abstained`.

**POST `/fix`** — execute an auto-fix via MS Graph (requires the caller's `access_token` + `X-API-Key`)
**POST `/escalate`** — draft an IT support email for an error (read-only)
**GET `/metrics`** — p50/p95/p99 latency, per-endpoint + per-source breakdown, trace-writer health
**GET `/languages`** · **GET `/i18n/{lang}`** — supported languages and the UI string bundle
**GET `/status`** — per-tier availability (lookup / retrieval / model)

---

## Languages

The UI and all authored error text ship in **English, Spanish (Español) and French (Français)**. The picker sits in the header and persists to `localStorage`; with no explicit choice the backend negotiates from `Accept-Language`, so a Spanish-locale visitor gets Spanish automatically.

```bash
curl -X POST localhost:8000/analyze -H 'Content-Type: application/json' \
     -d '{"error_input":"AADSTS50053","lang":"es"}'
# -> "La cuenta está bloqueada por demasiados intentos fallidos..."
```

**What is and isn't translated.** We translate the text we wrote — the interface, the remediation actions, the 15 curated messages, the portal steps, the abstain response. We do **not** machine-translate the ~335 auto-labelled descriptions quoted from Microsoft's documentation; shipping unreviewed MT of technical auth guidance is how an admin ends up doing the wrong thing in a language nobody can proofread. When a response carries untranslated source text the API sets `explanation_translated: false` and the UI labels it.

Adding a language is one JSON file in `data/i18n/`. `monitoring/check_i18n.py` fails the build if any locale drifts — missing keys, orphans after a rename, empty values, copy-paste, undefined frontend keys, or a catalog action with no translation.

---

## Performance

The tracing middleware previously wrote each trace to disk synchronously, under a global lock, from inside an async handler — so every request serialised behind every other request's disk write. Observability was a measurable share of the latency it was reporting. It now enqueues to a bounded, non-blocking queue drained by a background thread.

Measured, 20k records across 16 threads:

| Load | Before (p95) | After (p95) | Dropped |
|------|--------------|-------------|---------|
| ~1k rec/s (realistic) | 715 µs | 126 µs | 0 |
| ~4k rec/s (heavy) | 2,335 µs | 87 µs | 0 |
| ~25k rec/s (saturation) | 11.6 ms | 2.5 ms | sheds excess |

Under saturation the queue drops records rather than backpressuring live requests, and reports the drop count via `/metrics` so the percentiles are never silently sampled. Two cold-start wins: `torch` is imported lazily (tiers 1–2 need none of it, saving 1–2 s per deploy), and the retrieval index is warmed at startup (~125 ms for 350 docs).

---

## CI pipeline

Two workflows run on every push. Neither needs torch, which keeps Quality Gates at ~15 s and proves the service degrades correctly when the model tier is unavailable.

**Quality Gates** (`regression.yml`) — five sequential gates:

| Gate | Fails the build when |
|------|----------------------|
| Catalog integrity | The catalog drops below 300 codes, the 15 curated entries drift, or an entry loses its citation |
| Classification regression | Any curated code stops resolving via exact lookup at confidence 1.0 |
| Localisation completeness | A locale has missing / orphaned / empty / copy-pasted strings, an undefined frontend key, or an untranslated action |
| Retrieval coverage | Self-retrieval top-1 < 85%, paraphrase top-3 < 75%, action accuracy < 80%, or an OOD query fails to abstain |
| Trace upload | (always runs — publishes `traces.jsonl`) |

**Tests** (`test.yml`) — installs torch, runs `pytest`: dataset integrity, the `classify()` contract, and API checks against a `TestClient`. Not gated (stated plainly to avoid overclaiming): Graph auto-fix paths aren't exercised end-to-end, and DeepLog has no coverage gate of its own.

---

## Hardest Part

**Mapping free-text Azure AD errors to structured fixes without exploding the label space.** Classifying *which error* occurred caps coverage at whatever's hand-labelled. Flipping the target to *which remediation applies* — hundreds of codes, ~20 actions — is what let coverage grow 23× while the label space stayed bounded, and it's what makes a retrieval hit on an uncurated code map straight to a Graph call or an escalation template rather than just a label.

## Most Interesting

**The `/fix` security model.** Because `/fix` performs real writes against a live Entra tenant, the public deployment requires both a caller-supplied `access_token` (no silent fallback to app-level Graph credentials) and an `X-API-Key`. `/analyze` and `/escalate` stay read-only and open, so the live demo is safe to share without exposing tenant write access. The token is the admin's own — the fix acts as *them*, never with more privilege than they already hold.

---

## Security

- `/fix` requires the caller's own `access_token` — **no** silent fallback to app-level Graph credentials. `ALLOW_APP_TOKEN_FALLBACK` must stay unset/`false` on any public deployment (local/dev only).
- `/fix` requires an `X-API-Key` matching `DEMO_API_KEY` when set.
- `/analyze` and `/escalate` are read-only — no Graph writes.
- The auto-fix acts with the signed-in admin's delegated permissions, so it can never do more than that admin already can.

---

<div align="center">
  <sub>Built by <a href="https://github.com/aminabk99">Amina Bilal</a> · <a href="https://linkedin.com/in/amina-bilal-926340382">LinkedIn</a></sub>
</div>
