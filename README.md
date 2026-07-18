# AzureAutoFix

Chrome extension that catches Microsoft Entra (Azure AD) errors the moment they appear and resolves them — automatically where possible, with guided steps or a drafted IT email where not.

---

## What it does

Watches Microsoft login pages for AADSTS error codes in real time. The instant an error appears it:

1. Translates it into plain English — no Googling
2. Decides the right fix path based on what kind of problem it is
3. Acts — either walks you through fixing it yourself, calls the Graph API to fix it automatically, or writes the IT support email for you

**Three fix paths:**

- **You fix it** : wrong password, MFA not enrolled, account not yet provisioned. Exact steps shown inline.
- **Auto-fixed** : backend calls Microsoft Graph and resolves it without human involvement (unlock account, add redirect URI, rotate expired secret, etc.)
- **Escalate** : generates a pre-filled IT support email. You hit send.

---

## Error coverage

Resolution runs in three tiers, cheapest and most certain first. Each tier only
sees what the one above it couldn't answer.

| Tier | Mechanism | Codes | Latency (p50) | Confidence |
|------|-----------|-------|---------------|------------|
| 1 | Curated lookup table | 15 | ~2.6 ms | 1.00 |
| 2 | Hybrid retrieval over the full AADSTS catalog | 350 | ~4.9 ms | 0.45–0.92 |
| 3 | From-scratch transformer | fallback | model-bound | gated ≥ 0.55 |
| — | **Abstain** | — | — | 0.00 |

Of the 15 curated codes, 7 are auto-fixed via the Graph API with no human involvement.

**Why 350 and not 15.** The original design classified *which error* occurred,
which caps coverage at whatever is hand-labelled. The retrieval layer classifies
*which remediation applies* instead — there are hundreds of AADSTS codes but only
~20 things you can actually do about them, so the label space stays bounded while
coverage grows 23×. `model/build_catalog.py` parses Microsoft's published error
reference into `data/aadsts_catalog.json` and auto-labels each code by action.

**Abstain is a first-class outcome.** When retrieval scores below the
out-of-domain floor, or the model falls under its confidence gate, the API
returns `abstained: true` with confidence 0.0 rather than a plausible-looking
guess. A wrong automated write against a live tenant costs far more than saying
"I don't know."

### Measured coverage

`python -m monitoring.eval_retrieval` — enforced in CI:

| Metric | Result |
|--------|--------|
| Self-retrieval top-1 (349 codes, description-only query) | 95.1% |
| Self-retrieval top-3 | 98.9% |
| Paraphrase top-3 (unseen admin phrasing) | 83.3% |
| **Paraphrase action accuracy** | **91.7%** |
| Out-of-domain queries correctly abstained | 4/4 |

Action accuracy is the number that matters operationally. The two paraphrase
"misses" retrieve `AADSTS90094` instead of `AADSTS65001`, and `AADSTS7000215`
instead of `AADSTS7000222` — different codes, identical remediation. The user
gets the right fix either way.

---

## The AI layer — Three research papers

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

Build the error catalog and start the API:

```bash
python model/build_catalog.py            # 350 AADSTS codes -> data/aadsts_catalog.json
uvicorn backend.main:app --port 8000
```

Then open **http://localhost:8000/app**.

### Web app and sign-in

The demo UI is a single static page (`frontend/index.html`) served by FastAPI at
`/app`. It replaced a Streamlit app, for two reasons:

- Streamlit reruns its entire script on every interaction and has no routing,
  which makes holding an OAuth round-trip awkward.
- More importantly, sign-in was never actually implemented. The old UI had a
  "paste your MS Graph access token" text box, so pressing anything resembling
  a sign-in button could only fail. There was nothing there to work.

The page now signs the admin in with **MSAL.js** (`loginPopup`), so `/fix` acts
with the signed-in administrator's delegated token rather than stored app
credentials.

> **App registration:** add `http://localhost:8000/app` (and your deployed URL)
> under **Authentication → Single-page application**. Registering it as a *Web*
> platform instead is the usual cause of `AADSTS50011` / `AADSTS7000215`, since a
> SPA authenticates without a client secret. The page decodes both of those codes
> inline if you hit them.

`GET /auth/config` serves the client and tenant IDs to the browser. Both are
public identifiers that already appear in the sign-in URL; the client secret is
never sent to the frontend.

---

## Languages

The UI and all authored error text are available in **English, Spanish (Español)
and French (Français)**. The picker sits in the page header and persists to
`localStorage`; with no explicit choice the backend negotiates from the
browser's `Accept-Language` header, so a Spanish-locale visitor gets Spanish
without touching it.

```bash
curl -X POST localhost:8000/analyze -H 'Content-Type: application/json' \
     -d '{"error_input":"AADSTS50053","lang":"es"}'
# -> "La cuenta está bloqueada por demasiados intentos fallidos..."

curl localhost:8000/languages     # picker list
curl localhost:8000/i18n/fr       # UI string bundle
```

**What is and isn't translated.** We translate the text we wrote: the interface,
the 30 remediation actions, the 15 curated error messages, and the abstain
response. We do **not** machine-translate the ~335 auto-labelled descriptions
parsed from Microsoft's documentation — that is quoted source material, and
shipping unreviewed machine translations of technical auth guidance is how an
admin ends up doing the wrong thing in a language nobody on the team can
proofread.

When a response contains untranslated source text the API sets
`explanation_translated: false` and the UI labels it, rather than silently
mixing languages and leaving the user to work out which half they're reading.
Note that this keys off *authorship, not tier*: a retrieval hit that lands on
one of the 15 curated codes is our own text, so it does get translated.

Adding a language is one JSON file in `data/i18n/` and nothing else.
`monitoring/check_i18n.py` fails the build if any locale drifts — missing keys,
orphaned keys after a rename, empty values, untranslated copy-paste, frontend
keys with no definition, or a catalog action with no translation. That last
check caught six missing strings the first time it ran.

| Language | Code | Strings |
|----------|------|---------|
| English | `en` | 97 |
| Español | `es` | 97 |
| Français | `fr` | 97 |

---

## Performance

The tracing middleware previously wrote each trace record to disk synchronously,
under a global lock, from inside an async handler — so every request serialised
behind every other request's disk write. Observability was a measurable share of
the latency it was reporting. It now enqueues to a bounded, non-blocking queue
drained by a background thread.

Measured, 20k records across 16 threads:

| Load | Before (p95) | After (p95) | Dropped |
|------|--------------|-------------|---------|
| ~1k rec/s (realistic) | 715 µs | 126 µs | 0 |
| ~4k rec/s (heavy) | 2,335 µs | 87 µs | 0 |
| ~25k rec/s (saturation) | 11.6 ms | 2.5 ms | sheds excess |

Under saturation the queue deliberately drops records rather than applying
backpressure to live requests, and reports the drop count via `/metrics` so the
percentiles are never silently computed from a sampled subset.

Two other cold-start wins:

- **`torch` is imported lazily.** Tiers 1 and 2 need none of it, so a deploy no
  longer pays 1–2 s of import cost before serving. A missing or untrained
  checkpoint now degrades the model tier instead of failing the boot.
- **The retrieval index is warmed at startup** (~125 ms for 350 docs), so the
  first non-curated error after a cold start doesn't pay for building it.

---

## Tech

Chrome Extension (MV3) · FastAPI · Microsoft Graph API · MSAL.js · BM25 + char-ngram hybrid retrieval · Drain · LogBERT · DeepLog (PyTorch) · pytest · GitHub Actions
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      