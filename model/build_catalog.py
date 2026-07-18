"""
AzureAutoFix — Build the full AADSTS error catalog.

Turns Microsoft's published AADSTS error-code reference into a structured
catalog the retrieval layer can search.

Why this exists
---------------
The original system knew exactly 15 error codes, hand-written into
data/azure_errors.json. Anything outside that set fell through to a tiny
transformer trained on those same 15 rows -- so "unknown error" support was
really "guess one of 15 labels."

Microsoft documents ~350 AADSTS codes. This script parses all of them and
auto-labels each one with a *remediation action* rather than an error identity.
That flip is the important part: there are hundreds of error codes but only a
handful of things you can actually do about them, so the label space stays
small and bounded while coverage grows ~23x.

The 15 hand-curated entries in data/azure_errors.json always win on conflict --
they carry human-written explanations and Graph API dispatch wiring that the
auto-labeller can't infer.

Usage:
    python model/build_catalog.py
    python model/build_catalog.py --source data/aadsts_reference.md
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = ROOT / "data" / "aadsts_reference.md"
CURATED = ROOT / "data" / "azure_errors.json"
ALIASES = ROOT / "data" / "action_aliases.json"
OUT = ROOT / "data" / "aadsts_catalog.json"

DOC_URL = ("https://learn.microsoft.com/en-us/entra/identity-platform/"
           "reference-error-codes")

# ── Remediation taxonomy ─────────────────────────────────────────────────────
# Ordered: first rule that matches wins, so put the specific ones first.
#
# Categories mirror what backend/main.py already dispatches on:
#   admin_auto     -- resolvable via a Graph API write we can make
#   admin_escalate -- needs an admin, but not something we should automate
#   user           -- the signed-in user can fix it themselves
#   retry          -- transient; back off and try again
RULES: list[tuple[str, str, str]] = [
    # (fix_category, action, regex)

    # -- transient / service-side ------------------------------------------
    ("retry", "retry_with_backoff",
     r"\btransient\b|\btemporar|\btry again\b|service (is )?unavailable|"
     r"throttl|rate limit|timed? ?out|internal server error|"
     r"issue with the sign-?in service|try the request again"),

    ("admin_escalate", "open_microsoft_support_ticket",
     r"open a support ticket|contact (microsoft )?support|"
     r"unexpected error|report this error|if this (error )?persists"),

    # -- credentials / MFA / session ---------------------------------------
    ("user", "reset_credentials",
     r"invalid username or password|incorrect password|password (is |has )?expired|"
     r"credential.{0,20}(invalid|incorrect)|wrong password|weak password|"
     r"password.{0,20}(must be|needs to be) (changed|reset)"),

    ("user", "complete_mfa",
     r"multi-?factor|\bMFA\b|strong authentication|second factor|"
     r"proof ?up|authenticator app|verification code|\bOTP\b"),

    ("user", "trigger_interactive_login",
     r"silent sign-?in|interaction.?required|prompt.{0,15}(required|login)|"
     r"session (has )?expired|refresh token.{0,25}expired|re-?authenticat|"
     r"session select|sign in again|\bsso\b.{0,20}fail"),

    ("user", "redirect_correct_tenant",
     r"wrong (account|tenant)|personal (microsoft )?account|"
     r"work or school account|account type|valid login domain"),

    ("user", "complete_consent_prompt",
     r"legal age|age group consent|terms of use|privacy statement|"
     r"user (must )?accept"),

    # -- auto-fixable via Graph --------------------------------------------
    ("admin_auto", "add_redirect_uri",
     r"reply (url|address)|redirect ?uri"),

    ("admin_auto", "grant_admin_consent",
     r"admin consent|administrator (has not )?consent|requires admin|"
     r"consent (is )?required|not consented"),

    ("admin_auto", "enable_or_unlock_account",
     r"account is (disabled|locked)|account (has been )?locked|"
     r"disabled in the directory|lockout|user (is |account )?disabled"),

    ("admin_auto", "rotate_client_secret",
     r"client secret|invalid client secret|secret (is |has )?expired|"
     r"credential.{0,20}expired.{0,20}app|key (has )?expired"),

    ("admin_auto", "update_api_permissions",
     r"\bscope\b|permission.{0,20}(invalid|missing|not )|invalid resource|"
     r"insufficient privileges"),

    # -- app registration / manifest ---------------------------------------
    ("admin_escalate", "fix_app_registration",
     r"application with identifier.{0,40}(not found|was not found)|"
     r"app(lication)? (is )?not (found|configured|in the tenant)|"
     r"service principal.{0,25}not found|"
     r"resource (is )?disabled|resource URL|audience (uri|validation)|"
     r"identifier ?uri|app ?id ?uri|manifest|"
     r"no token audiences|reply address.{0,20}mismatch|"
     r"not configured (as|for)|\bSID\b requirement"),

    ("admin_escalate", "review_token_configuration",
     r"assertion (is )?invalid|token issuer|issuer (doesn'?t|does not) match|"
     r"invalid (id_?token|assertion|grant|jwt)|claim.{0,25}(missing|invalid)|"
     r"signature (validation|verification)|token.{0,20}malformed|"
     r"authorization code.{0,25}(invalid|expired|used)"),

    # -- policy / device / network -----------------------------------------
    ("admin_escalate", "review_conditional_access",
     r"conditional access|access policy|compliant device|managed device|"
     r"device (is )?not (registered|joined|compliant)|"
     r"blocked by (your )?(organization|policy)|restricted proxy|"
     r"outbound (access )?policy|inbound (access )?policy|"
     r"cross-?tenant access|location.{0,20}(blocked|not allowed)|"
     r"\bIP\b address.{0,25}(blocked|not allowed)"),

    ("admin_escalate", "review_device_registration",
     r"device (registration|join|id)|\bWPJ\b|workplace join|"
     r"device (object|record).{0,20}not found"),

    # -- directory / user provisioning -------------------------------------
    ("admin_escalate", "provision_or_invite_user",
     r"does not exist in tenant|not found in.{0,25}(directory|tenant)|"
     r"external user|\bB2B\b|guest user|user account.{0,30}(missing|not exist)|"
     r"hasn'?t been (explicitly )?added to the tenant|"
     r"can'?t provision the user|user (is )?not (assigned|authorized)|"
     r"unauthorized to call this endpoint"),

    # -- federation / certs -------------------------------------------------
    ("admin_escalate", "review_certificate_or_federation",
     r"certificat|federat|\bSAML\b|token signing|\bWS-?Fed\b|"
     r"\bADFS\b|identity provider|\bRSA\b key|encryption key"),

    # -- endpoint / protocol misuse ----------------------------------------
    ("admin_escalate", "fix_request_format",
     r"not supported (over|for|by)|unsupported (grant|response|request)|"
     r"invalid (request|parameter|endpoint|domain name|uri)|"
     r"missing (a )?(required )?(parameter|header)|"
     r"malformed|endpoint (only )?accepts|wrong endpoint|"
     r"contains invalid characters"),

    # -- sign-out / session lifecycle ---------------------------------------
    ("user", "restart_signout_flow",
     r"sign ?out|logout|log ?off"),

    # -- token internals ----------------------------------------------------
    ("admin_escalate", "review_token_configuration",
     r"\bJWT\b|claims transformer|transforming the claims|"
     r"name identifier|pairwise identifier|\bsalt\b|"
     r"subject mismatch|issuer claim|flow token|nonce"),

    # -- credentials (broader) ----------------------------------------------
    ("user", "reset_credentials",
     r"password (doesn'?t|does not) exist|null password|"
     r"credential validation|social IDP login"),

    ("admin_escalate", "review_device_registration",
     r"device authentication (is )?required"),

    ("admin_escalate", "fix_request_format",
     r"unknown or invalid instance|tenant-?identifying information|"
     r"request was invalid|invalid instance"),

    # Generic catch-all: the docs explicitly say to involve an admin. Weakest
    # signal, so it sits last -- anything above outranks it.
    ("admin_escalate", "contact_tenant_admin",
     r"contact (the |your )?(tenant |global )?admin"),
]

FALLBACK = ("admin_escalate", "manual_review")


def classify_doc(text: str) -> tuple[str, str]:
    """Map a documented error description onto a remediation action."""
    low = text.lower()
    for category, action, pattern in RULES:
        if re.search(pattern, low, flags=re.I):
            return category, action
    return FALLBACK


# ── Parsing ──────────────────────────────────────────────────────────────────

ROW = re.compile(r"^\|\s*(AADSTS\d+)\s*\|\s*(.+?)\s*\|\s*$")


def clean(text: str) -> str:
    """Strip the markdown/HTML noise Microsoft's table carries."""
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\\_", "_").replace("\\*", "*")
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)   # md links -> label
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_name(desc: str) -> tuple[str, str]:
    """
    Microsoft writes most rows as 'SymbolicName - Human description.'
    Pull the symbolic name out; it's a strong retrieval signal on its own.
    """
    m = re.match(r"^([A-Z][A-Za-z0-9_]{3,60})\s+-\s+(.*)$", desc)
    if m:
        return m.group(1), m.group(2).strip()
    return "", desc


def parse(source: Path) -> list[dict]:
    entries: dict[str, dict] = {}
    for line in source.read_text(encoding="utf-8").splitlines():
        m = ROW.match(line.strip())
        if not m:
            continue
        code, raw = m.group(1), clean(m.group(2))
        if not raw or len(raw) < 8:
            continue
        symbolic, desc = split_name(raw)
        category, action = classify_doc(raw)
        entries.setdefault(code, {
            "error_code": code,
            "symbolic_name": symbolic,
            "description": desc,
            "fix_category": category,
            "action": action,
            "user_or_admin": "admin" if category.startswith("admin") else "user",
            "source": "microsoft_docs",
            "citation": {
                "title": f"Microsoft Entra error reference — {code}",
                "url": f"{DOC_URL}#{code.lower()}",
            },
        })
    return list(entries.values())


def _alias_map() -> dict[str, str]:
    """Synonym -> canonical remediation name. See data/action_aliases.json."""
    if not ALIASES.exists():
        return {}
    return json.loads(ALIASES.read_text(encoding="utf-8")).get("aliases", {})


def merge_curated(catalog: list[dict]) -> list[dict]:
    """Hand-written entries override the auto-labelled ones."""
    if not CURATED.exists():
        return catalog
    curated = {d["error_code"]: d for d in json.loads(CURATED.read_text())}
    by_code = {d["error_code"]: d for d in catalog}

    for code, entry in curated.items():
        merged = by_code.get(code, {})
        merged.update({
            "error_code":    code,
            "description":   entry.get("cause", merged.get("description", "")),
            "fix_category":  entry["fix_category"],
            "action":        entry["action"],
            "user_or_admin": entry["user_or_admin"],
            "reasoning":     entry.get("reasoning", ""),
            "action_detail": entry.get("action_detail", ""),
            "user_message":  entry.get("user_message", ""),
            "source":        "curated",
            "citation": merged.get("citation", {
                "title": f"Microsoft Entra error reference — {code}",
                "url": f"{DOC_URL}#{code.lower()}",
            }),
        })
        merged.setdefault("symbolic_name", "")
        by_code[code] = merged

    # Collapse synonym action names onto one canonical vocabulary. Without
    # this the curated entries carry labels that exist nowhere else in the
    # catalog, so a classifier trained on the rule-labelled rows physically
    # cannot predict them.
    aliases = _alias_map()
    if aliases:
        for entry in by_code.values():
            entry["action"] = aliases.get(entry.get("action"), entry.get("action"))

    return sorted(by_code.values(), key=lambda d: int(d["error_code"][6:]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    if not args.source.exists():
        raise SystemExit(
            f"{args.source} not found. Save the Microsoft AADSTS reference page "
            f"there first ({DOC_URL})."
        )

    catalog = merge_curated(parse(args.source))
    args.out.write_text(json.dumps(catalog, indent=2), encoding="utf-8")

    from collections import Counter
    cats = Counter(d["fix_category"] for d in catalog)
    acts = Counter(d["action"] for d in catalog)
    curated_n = sum(1 for d in catalog if d["source"] == "curated")

    print(f"Wrote {args.out.relative_to(ROOT)} — {len(catalog)} error codes "
          f"({curated_n} curated, {len(catalog)-curated_n} auto-labelled)\n")
    print("fix_category:")
    for k, v in cats.most_common():
        print(f"  {k:<16} {v:>4}")
    print("\ntop actions:")
    for k, v in acts.most_common(8):
        print(f"  {k:<32} {v:>4}")


if __name__ == "__main__":
    main()
