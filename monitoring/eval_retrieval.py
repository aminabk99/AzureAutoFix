#!/usr/bin/env python3
"""
AzureAutoFix — Retrieval coverage evaluation.

The classification regression gate proves the 15 curated codes still resolve.
This measures the thing the retrieval layer was actually added for: can we
resolve an error we never hand-labelled, given only its description?

Protocol
--------
For every code in the catalog, query with the *description text only* (the
error code is stripped out, so the retriever can't cheat via exact match) and
check whether the correct code is returned.

This is a self-retrieval test, which is an optimistic upper bound -- the query
text is drawn from the indexed document itself. It measures index quality and
discriminative power, not paraphrase robustness. The paraphrase set below is
the harder, more honest check: hand-written queries in the phrasing a real
admin would use, none of which appear verbatim in the corpus.

Usage:
    python -m monitoring.eval_retrieval
    python -m monitoring.eval_retrieval --k 3 --min-top1 0.75
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.retrieval import get_retriever  # noqa: E402

CATALOG = Path(__file__).resolve().parent.parent / "data" / "aadsts_catalog.json"

# Hand-written queries in real admin phrasing. None of these are verbatim
# corpus text, so this is the honest generalisation check.
PARAPHRASES: list[tuple[str, str]] = [
    ("my account got locked after too many tries",      "AADSTS50053"),
    ("the reply url in the request doesn't match",      "AADSTS50011"),
    ("app needs an admin to approve permissions",       "AADSTS65001"),
    ("password is expired and must be changed",         "AADSTS50055"),
    ("this user account has been disabled",             "AADSTS50057"),
    ("we need multi factor authentication",             "AADSTS50076"),
    ("the client secret is no longer valid",            "AADSTS7000222"),
    ("no reply address was provided for the app",       "AADSTS900971"),
    ("conditional access policy is blocking sign in",   "AADSTS53003"),
    ("username or password is wrong",                   "AADSTS50126"),
    ("the refresh token has expired",                   "AADSTS700082"),
    ("silent sign in didn't work, need interaction",    "AADSTS50058"),
]

OUT_OF_DOMAIN = [
    "banana bread recipe",
    "how do I bake sourdough",
    "what is the capital of France",
    "please book me a flight to Lisbon",
]


def strip_code(text: str) -> str:
    return re.sub(r"AADSTS\d+", " ", text, flags=re.I).strip()


def evaluate(k: int = 3) -> dict:
    r = get_retriever()
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))

    # ── 1. self-retrieval over the whole catalog ─────────────────────────
    top1 = topk = 0
    misses: list[tuple[str, str]] = []
    for entry in catalog:
        code = entry["error_code"]
        query = strip_code(entry.get("description", ""))
        if len(query) < 12:
            continue
        hits = r.search(query, top_k=k)
        got = [h["error_code"] for h in hits]
        if got and got[0] == code:
            top1 += 1
        elif code in got:
            topk += 1
        else:
            misses.append((code, query[:60]))
    total = top1 + topk + len(misses)

    # ── 2. paraphrase generalisation ─────────────────────────────────────
    p_top1 = p_topk = 0
    p_miss: list[tuple[str, str, str]] = []
    for query, expected in PARAPHRASES:
        hits = r.search(query, top_k=k)
        got = [h["error_code"] for h in hits]
        if got and got[0] == expected:
            p_top1 += 1
        elif expected in got:
            p_topk += 1
        else:
            p_miss.append((query, expected, got[0] if got else "-"))

    # ── 3. action-level accuracy ─────────────────────────────────────────
    # The operationally meaningful metric. Retrieving AADSTS90094 when the
    # "right" answer was AADSTS65001 is scored as a miss above, but both map
    # to grant_admin_consent -- the user gets the correct remediation either
    # way. Since the whole design classifies *actions* rather than error
    # identities, this is the number that reflects real behaviour.
    by_code = {d["error_code"]: d for d in catalog}
    a_hit = 0
    for query, expected in PARAPHRASES:
        hits = r.search(query, top_k=1)
        if not hits:
            continue
        want = by_code.get(expected, {}).get("action")
        if want and hits[0].get("action") == want:
            a_hit += 1

    # ── 4. out-of-domain rejection ───────────────────────────────────────
    rejected = sum(1 for q in OUT_OF_DOMAIN if r.classify(q) is None)

    return {
        "catalog_size": len(catalog),
        "self_total": total,
        "self_top1": top1 / total if total else 0.0,
        "self_topk": (top1 + topk) / total if total else 0.0,
        "self_misses": misses[:10],
        "para_total": len(PARAPHRASES),
        "para_top1": p_top1 / len(PARAPHRASES),
        "para_topk": (p_top1 + p_topk) / len(PARAPHRASES),
        "para_misses": p_miss,
        "para_action_acc": a_hit / len(PARAPHRASES),
        "ood_rejected": rejected,
        "ood_total": len(OUT_OF_DOMAIN),
        "k": k,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--min-top1", type=float, default=0.70,
                    help="CI floor for self-retrieval top-1")
    ap.add_argument("--min-para", type=float, default=0.50,
                    help="CI floor for paraphrase top-k")
    ap.add_argument("--min-action", type=float, default=0.75,
                    help="CI floor for paraphrase action-level accuracy")
    args = ap.parse_args()

    m = evaluate(args.k)
    k = m["k"]

    print(f"\nRetrieval coverage — {m['catalog_size']} AADSTS codes indexed\n")
    print(f"  self-retrieval  (n={m['self_total']})")
    print(f"    top-1         {m['self_top1']:6.1%}")
    print(f"    top-{k}         {m['self_topk']:6.1%}")
    print(f"\n  paraphrase      (n={m['para_total']}, unseen phrasing)")
    print(f"    top-1         {m['para_top1']:6.1%}")
    print(f"    top-{k}         {m['para_topk']:6.1%}")
    print(f"    action        {m['para_action_acc']:6.1%}   <- correct remediation, "
          f"regardless of which code matched")
    print(f"\n  out-of-domain   {m['ood_rejected']}/{m['ood_total']} correctly abstained")

    if m["para_misses"]:
        print("\n  paraphrase misses:")
        for q, exp, got in m["para_misses"]:
            print(f"    {q[:44]:<46} want {exp:<14} got {got}")

    ok = (m["self_top1"] >= args.min_top1
          and m["para_topk"] >= args.min_para
          and m["para_action_acc"] >= args.min_action
          and m["ood_rejected"] == m["ood_total"])

    print("\n" + ("PASS" if ok else "FAIL")
          + f" — floors: self top-1 >= {args.min_top1:.0%}, "
            f"paraphrase top-{k} >= {args.min_para:.0%}, "
            f"action >= {args.min_action:.0%}, all OOD rejected\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
