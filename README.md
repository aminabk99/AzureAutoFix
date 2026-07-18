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

## Accuracy

| What it does | Result | How it's measured |
|---|---|---|
| Resolves the 15 curated codes | **100%** | Every code checked in CI on every push |
| Finds the right fix from a plain-English description | **91.7%** | 12 hand-written admin-phrasing queries, none appearing verbatim in the corpus |
| Retrieves the right error code (top-3) | **98.9%** | Self-retrieval across all 349 catalog descriptions |
| Recognises non-Azure input and abstains | **4/4** | Deliberate out-of-domain queries |
| p50 latency, lookup path | **2.6 ms** | Measured from live request traces |

**The transformer tier scores 53.3%** under leave-one-out cross-validation
against a 47% majority-class baseline — roughly one extra correct answer out of
fifteen. That is a dataset limit, not a modelling one: N=15, and two of the four
classes have a single example each, so LOOCV leaves nothing to train on for
them. Reported here rather than omitted, because it is the reason the
architecture looks the way it does.

Retrieval outperforms the model at the same task by a wide margin, so it runs
first and the model sits behind a 0.55 confidence floor as a last resort. When
all three tiers fall short the API returns `abstained: true` at confidence 0.0
instead of guessing — a wrong automated write against a live tenant costs far
more than an unanswered question.

Training is seeded (`SEED`, default 42) so these figures reproduce across runs.

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

## The AI layer

Most error-detection tools stop at pattern matching. AzureAutoFix layers
published techniques on top of the lookup table so it can handle errors it has
never seen before — and, just as importantly, say so when it can't.

### Drain — log parsing (ICWS 2017)
> He, P. et al. *Drain: An Online Log Parsing Approach with Fixed Depth Tree.* IEEE ICWS 2017.

Raw AADSTS strings embed dynamic values (tenant IDs, redirect URIs, timestamps)
that make direct comparison impossible. Drain parses each error into a stable
log key via a fixed-depth parse tree, so `AADSTS50053` raised against two
different tenants collapses to the same key.

Implemented in `model/parser.py`, called from `backend/analyze_sequence.py`.

### DeepLog — session-level attack detection (CCS 2017)
> Du, M. et al. *DeepLog: Anomaly Detection and Diagnosis from System Logs using Deep Learning.* ACM CCS 2017.

A 2-layer LSTM that models the expected sequence of log keys across a login
session. Where Drain and retrieval handle individual errors, DeepLog watches the
*pattern*: if the observed next key falls outside the model's top-g predictions,
the session is flagged. That catches credential stuffing and brute-force shapes
that look unremarkable one error at a time.

Implemented in `model/sequence_detector.py` (`WINDOW_SIZE=5`, `TOP_G=3`,
`NUM_LAYERS=2`, matching the paper), trained weights in `model/sequence_model.pt`,
called from `backend/analyze_sequence.py`.

### Hybrid retrieval — unseen error resolution

Classifying which fix path applies to an error outside the curated set is
handled by the retrieval tier (`model/retrieval.py`), not by a neural
classifier. BM25 over word tokens fused with character-3gram TF-IDF via
Reciprocal Rank Fusion, reranked on exact code and symbolic-name matches.

This was a deliberate trade. A fine-tuned encoder is the more fashionable
answer, but the corpus is 350 short, highly templated documents where lexical
overlap is very high, and the retrieval approach is ~1ms per query, adds no
model weights to a free-tier container, and — critically — **cites the specific
Microsoft doc it matched**, so every answer is auditable. It scores 95.1% top-1
on self-retrieval and 91.7% action accuracy on unseen phrasing (see
*Error coverage* above).

`model/logbert_classifier.py` contains a from-scratch bidirectional Transformer
encoder written for this role. It is **not currently wired into the request
path** — no trained checkpoint, not imported by the backend — and is kept as
reference for a future A/B against the retrieval tier. It is listed here rather
than quietly left in the tree, because a repository should not imply a component
is live when it isn't.

---

## CI pipeline

Two workflows run on every push. Neither requires torch — tiers 1 and 2 are
pure stdlib, which keeps `Quality Gates` at ~15s and also proves the service
degrades correctly when the model tier is unavailable.

**Quality Gates** (`.github/workflows/regression.yml`) — five sequential gates:

| Gate | Fails the build when |
|------|----------------------|
| Catalog integrity | The AADSTS catalog drops below 300 codes, the 15 curated entries drift, or any entry loses its citation |
| Classification regression | Any of the 15 curated codes stops resolving via the exact-lookup path at confidence 1.0 |
| Localisation completeness | A locale has missing, orphaned, empty or copy-pasted strings; the frontend references an undefined key; or a catalog action has no translation |
| Retrieval coverage | Self-retrieval top-1 < 85%, paraphrase top-3 < 75%, action accuracy < 80%, or any out-of-domain query fails to abstain |
| Trace upload | (always runs — publishes `traces.jsonl` as a build artifact) |

**Tests** (`.github/workflows/test.yml`) — installs torch and runs `pytest
test_graph.py`: dataset integrity, the `classify()` contract across fix paths,
and API-level checks against a `TestClient`.

Two notes on what is *not* gated, to avoid overstating the setup: Graph API
auto-fix paths are not exercised end-to-end in CI (they are covered by the
`classify()` contract tests, not by mocked Graph calls), and the DeepLog
sequence detector has no coverage gate of its own.

---

## Setup

**Requirements:** Python 3.10+. A Microsoft Entra app registration is only
needed for sign-in and auto-fix — analysis works without one.

```bash
git clone https://github.com/aminabk99/AzureAutoFix
cd AzureAutoFix
pip install -r requirements.txt
```

The Chrome extension is plain Manifest V3 with no build step — load
`extension/` directly: `chrome://extensions` → Developer mode → Load unpacked.

`torch` is optional. Tiers 1 and 2 (lookup and retrieval) are pure stdlib and
answer virtually all traffic; without torch the service simply reports the
model tier as unavailable via `/status` rather than failing to start.

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

Chrome Extension (MV3) · FastAPI · Microsoft Graph API · MSAL.js · BM25 + char-ngram hybrid retrieval · Drain · DeepLog (PyTorch) · pytest · GitHub Actions
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      