"""
AzureAutoFix — Hybrid retrieval over the AADSTS catalog.

Replaces "if the code isn't one of our 15, guess a label" with "find the most
similar documented errors, and cite them."

Design
------
Two independent retrievers, fused:

  1. BM25 over word tokens -- strong on exact terminology ("redirect uri",
     "admin consent"). This is what actually carries most queries, because
     Azure error text is highly formulaic.

  2. Character 3-gram TF-IDF cosine -- robust to the things BM25 is brittle
     about: typos, truncated pastes, and morphological variants
     ("consent"/"consented"/"consenting"), plus partial error codes.

Scores are combined with Reciprocal Rank Fusion rather than a weighted sum of
raw scores. RRF only uses rank position, so it doesn't require the two scoring
scales to be comparable or normalised -- which they aren't.

Why not sentence-transformers
-----------------------------
A dense embedding model would likely beat this on paraphrased queries. It also
adds ~90MB of weights plus the torch/transformers import cost to a container
that currently cold-starts on a free tier, for a corpus of 350 short, highly
templated documents where lexical overlap is very high. The retriever is
deliberately behind an interface (`Retriever.search`) so a dense backend can be
dropped in and A/B'd against this one -- see `EMBEDDING_BACKEND`.

Everything here is stdlib. No numpy, no torch, no network at import time.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = ROOT / "data" / "aadsts_catalog.json"

# Swap point for a dense retriever later. "lexical" is the only backend today.
EMBEDDING_BACKEND = "lexical"

_TOKEN = re.compile(r"[a-z0-9]+")
_CODE = re.compile(r"AADSTS\d+", re.I)

# Domain stopwords -- these appear in most AADSTS descriptions and carry
# almost no discriminative signal, so they'd otherwise dominate short queries.
_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "to", "of",
    "in", "on", "for", "and", "or", "not", "this", "that", "it", "with",
    "from", "by", "at", "as", "you", "your", "user", "error", "request",
    "please", "can", "cannot", "cant", "doesnt", "does", "has", "have",
}


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


def _ngrams(text: str, n: int = 3) -> Counter:
    s = re.sub(r"\s+", " ", text.lower().strip())
    return Counter(s[i:i + n] for i in range(max(0, len(s) - n + 1)))


# ── BM25 ─────────────────────────────────────────────────────────────────────

class BM25:
    """Standard Okapi BM25. k1/b at the usual defaults for short documents."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = docs
        self.N = len(docs)
        self.avgdl = sum(len(d) for d in docs) / max(self.N, 1)
        self.tf = [Counter(d) for d in docs]

        df: Counter = Counter()
        for d in docs:
            df.update(set(d))
        # +0.5 smoothing keeps IDF positive for terms present in every doc.
        self.idf = {
            t: math.log(1 + (self.N - n + 0.5) / (n + 0.5))
            for t, n in df.items()
        }
        self.postings: dict[str, list[int]] = defaultdict(list)
        for i, d in enumerate(docs):
            for t in set(d):
                self.postings[t].append(i)

    def search(self, query: str) -> dict[int, float]:
        q = _tokens(query)
        scores: dict[int, float] = defaultdict(float)
        for term in q:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i in self.postings[term]:
                f = self.tf[i][term]
                dl = len(self.docs[i])
                denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[i] += idf * (f * (self.k1 + 1)) / denom
        return scores


# ── Character n-gram TF-IDF ──────────────────────────────────────────────────

class CharTfidf:
    """Cosine similarity over L2-normalised character 3-gram TF-IDF vectors."""

    def __init__(self, texts: list[str], n: int = 3):
        self.n = n
        self.grams = [_ngrams(t, n) for t in texts]
        N = len(texts)
        df: Counter = Counter()
        for g in self.grams:
            df.update(g.keys())
        self.idf = {g: math.log((N + 1) / (c + 1)) + 1.0 for g, c in df.items()}
        self.vectors = [self._vec(g) for g in self.grams]
        self.postings: dict[str, list[int]] = defaultdict(list)
        for i, v in enumerate(self.vectors):
            for g in v:
                self.postings[g].append(i)

    def _vec(self, grams: Counter) -> dict[str, float]:
        v = {g: (1 + math.log(c)) * self.idf.get(g, 1.0) for g, c in grams.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {g: x / norm for g, x in v.items()}

    def search(self, query: str) -> dict[int, float]:
        qv = self._vec(_ngrams(query, self.n))
        scores: dict[int, float] = defaultdict(float)
        for g, qw in qv.items():
            for i in self.postings.get(g, ()):
                dw = self.vectors[i].get(g)
                if dw:
                    scores[i] += qw * dw
        return scores


# ── Fusion + rerank ──────────────────────────────────────────────────────────

def _rrf(rankings: Iterable[dict[int, float]], k: int = 60) -> dict[int, float]:
    """
    Reciprocal Rank Fusion. Uses only rank position, so the two retrievers'
    incomparable score scales don't need normalising.
    """
    fused: dict[int, float] = defaultdict(float)
    for scores in rankings:
        for rank, (doc_id, _) in enumerate(
            sorted(scores.items(), key=lambda kv: -kv[1])
        ):
            fused[doc_id] += 1.0 / (k + rank + 1)
    return fused


class Retriever:
    def __init__(self, catalog_path: Path = CATALOG_PATH):
        if not catalog_path.exists():
            raise FileNotFoundError(
                f"{catalog_path} not found. Run: python model/build_catalog.py"
            )
        self.catalog: list[dict] = json.loads(catalog_path.read_text(encoding="utf-8"))
        self.by_code = {d["error_code"].upper(): d for d in self.catalog}

        # The indexed text deliberately repeats the symbolic name -- it's the
        # highest-signal field when present ("ConsentRequired", "UserDisabled").
        texts = [
            f'{d["error_code"]} {d.get("symbolic_name","")} {d.get("symbolic_name","")} '
            f'{d.get("description","")} {d.get("action","")}'
            for d in self.catalog
        ]
        self.bm25 = BM25([_tokens(t) for t in texts])
        self.char = CharTfidf(texts)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Return the top_k catalog entries with fused scores attached."""
        fused = _rrf([self.bm25.search(query), self.char.search(query)])

        # Rerank: an explicit AADSTS code in the query is near-decisive, and
        # a literal symbolic-name mention is a strong secondary signal.
        codes = {c.upper() for c in _CODE.findall(query)}
        low = query.lower()
        for i, entry in enumerate(self.catalog):
            if entry["error_code"].upper() in codes:
                fused[i] = fused.get(i, 0.0) + 1.0
            sym = entry.get("symbolic_name", "")
            if sym and len(sym) > 5 and sym.lower() in low:
                fused[i] = fused.get(i, 0.0) + 0.25

        ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]
        return [{**self.catalog[i], "_score": round(s, 5)} for i, s in ranked]

    # ── public entry point ───────────────────────────────────────────────

    # Out-of-domain gate. Measured separation on the 350-doc catalog:
    # in-domain queries score BM25 5-15 with ~100% of their tokens in the
    # corpus vocabulary; off-topic ones ("banana bread recipe") score 0.0
    # with 0% known tokens. Both conditions must hold, so a query that merely
    # shares a common word doesn't sneak through.
    _MIN_BM25 = 3.0
    _MIN_CHAR = 0.15
    _MIN_KNOWN_RATIO = 0.5

    def resolve(self, query: str, top_k: int = 5) -> tuple[dict | None, str]:
        """
        Like classify(), but also reports *why* it failed:

          "hit"           -- resolved
          "out_of_domain" -- the query isn't an Azure AD error at all
          "no_match"      -- plausibly in-domain, but nothing ranked

        The caller needs this distinction. On "out_of_domain" there is no point
        consulting the transformer: it was trained on Azure error text and has
        no notion of an off-topic input, so it will confidently emit some class
        for "banana bread recipe". On "no_match" the model is worth asking.
        """
        result = self.classify(query, top_k=top_k)
        if result is not None:
            return result, "hit"
        return None, "out_of_domain" if not self._in_domain(query) else "no_match"

    def _in_domain(self, query: str) -> bool:
        bm = self.bm25.search(query)
        ch = self.char.search(query)
        toks = _tokens(query)
        if not toks:
            return False
        known = sum(1 for t in toks if t in self.bm25.idf) / len(toks)
        return ((max(bm.values(), default=0.0) >= self._MIN_BM25
                 or max(ch.values(), default=0.0) >= self._MIN_CHAR)
                and known >= self._MIN_KNOWN_RATIO)

    def classify(self, query: str, top_k: int = 5) -> dict | None:
        """
        Resolve a query to a remediation. Returns None when the query is
        out-of-domain or nothing matched confidently, so the caller can fall
        back rather than emit a confident-looking guess.
        """
        # Exact code hit: no retrieval needed, and confidence is not a guess.
        for code in _CODE.findall(query):
            entry = self.by_code.get(code.upper())
            if entry:
                # _score is synthetic here -- the exact-code path never ran a
                # similarity search, so give it the ceiling value.
                return self._result(entry, 1.0, [{**entry, "_score": 1.0}], exact=True)

        bm25_scores = self.bm25.search(query)
        char_scores = self.char.search(query)

        bm25_max = max(bm25_scores.values()) if bm25_scores else 0.0
        char_max = max(char_scores.values()) if char_scores else 0.0

        toks = _tokens(query)
        known_ratio = (
            sum(1 for t in toks if t in self.bm25.idf) / len(toks) if toks else 0.0
        )

        in_domain = (
            (bm25_max >= self._MIN_BM25 or char_max >= self._MIN_CHAR)
            and known_ratio >= self._MIN_KNOWN_RATIO
        )
        if not in_domain:
            return None

        fused = _rrf([bm25_scores, char_scores])
        codes = {c.upper() for c in _CODE.findall(query)}
        low = query.lower()
        for i, entry in enumerate(self.catalog):
            if entry["error_code"].upper() in codes:
                fused[i] = fused.get(i, 0.0) + 1.0
            sym = entry.get("symbolic_name", "")
            if sym and len(sym) > 5 and sym.lower() in low:
                fused[i] = fused.get(i, 0.0) + 0.25

        ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]
        if not ranked:
            return None
        hits = [{**self.catalog[i], "_score": round(sc, 5)} for i, sc in ranked]

        # Confidence blends two independent signals:
        #   * absolute lexical strength (bm25_max, saturating) -- "is this
        #     query actually well covered by the corpus?"
        #   * rank margin between hit 1 and hit 2 -- "is the answer unambiguous?"
        # Capped below 1.0: only an exact code match earns full confidence.
        s1 = hits[0]["_score"]
        s2 = hits[1]["_score"] if len(hits) > 1 else 0.0
        margin = (s1 - s2) / s1 if s1 > 0 else 0.0
        strength = min(1.0, bm25_max / 14.0)
        confidence = round(min(0.92, 0.30 + 0.45 * strength + 0.25 * margin), 3)

        return self._result(hits[0], confidence, hits, exact=False)

    @staticmethod
    def _result(entry: dict, confidence: float, hits: list[dict], exact: bool) -> dict:
        action = entry.get("action", "manual_review")
        abstained = action == "manual_review"

        return {
            "error_code":    entry["error_code"],
            "fix_category":  entry.get("fix_category", "admin_escalate"),
            "user_or_admin": entry.get("user_or_admin", "admin"),
            "explanation":   entry.get("description", ""),
            "reasoning": (
                f'Exact match in the Microsoft AADSTS catalog.'
                if exact else
                f'Closest documented match (retrieval score {hits[0]["_score"]:.3f}).'
            ) + (
                " No automated remediation is mapped for this code, so this is "
                "reported for manual review rather than guessed."
                if abstained else ""
            ),
            "action":        action,
            "action_detail": entry.get("action_detail")
                             or _ACTION_DETAIL.get(action, entry.get("description", "")),
            "user_message":  entry.get("user_message") or entry.get("description", ""),
            "confidence":    confidence,
            "source":        "retrieval",
            "abstained":     abstained,
            "citations": [
                {
                    "title": h.get("citation", {}).get("title", h["error_code"]),
                    "url":   h.get("citation", {}).get("url", ""),
                    "score": h["_score"],
                }
                for h in hits[:3]
            ],
        }


_ACTION_DETAIL = {
    "retry_with_backoff":            "Transient failure. Retry with exponential backoff before escalating.",
    "reset_credentials":             "Have the user re-enter or reset their password.",
    "complete_mfa":                  "The user needs to complete multi-factor authentication.",
    "trigger_interactive_login":     "Force an interactive sign-in instead of a silent token request.",
    "redirect_correct_tenant":       "Sign in with the correct work or school account.",
    "complete_consent_prompt":       "The user must accept the consent or terms prompt.",
    "restart_signout_flow":          "Restart the sign-out flow from the application.",
    "add_redirect_uri":              "Add the missing redirect URI to the app registration.",
    "grant_admin_consent":           "An administrator must grant consent for the requested permissions.",
    "enable_or_unlock_account":      "Re-enable or unlock the user account in the directory.",
    "rotate_client_secret":          "Generate a new client secret and update the application.",
    "update_api_permissions":        "Correct the requested API permissions or scope.",
    "fix_app_registration":          "Review the app registration: identifier URI, audience, and reply URLs.",
    "review_token_configuration":    "Inspect the token/assertion configuration and claim mappings.",
    "review_conditional_access":     "Review the Conditional Access policies applying to this sign-in.",
    "review_device_registration":    "Check the device registration or compliance state.",
    "provision_or_invite_user":      "Provision or invite the user into the tenant.",
    "review_certificate_or_federation": "Check the signing certificate or federation configuration.",
    "fix_request_format":            "The request is malformed. Check parameters and endpoint.",
    "contact_tenant_admin":          "This requires a tenant administrator.",
    "open_microsoft_support_ticket": "Service-side issue. Open a Microsoft support ticket.",
    "manual_review":                 "No automated remediation is mapped. Review the documented description below.",
}


# Module-level singleton -- the index is built once at import/startup, never
# per request. Building it is ~10ms for 350 docs but doing it per request
# would put it directly in the p50.
_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever
