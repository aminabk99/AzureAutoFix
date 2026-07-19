"""
AzureAutoFix — Localisation.

Scope, stated plainly: we translate the text *we* wrote, and we don't pretend
to translate the text we didn't.

  Translated   UI strings, the 22 remediation action descriptions, the 15
               curated error messages, and the abstain response.
  Not          The ~335 auto-labelled AADSTS descriptions parsed from
               Microsoft's documentation. Those are quoted source material.
               Machine-translating technical auth text and presenting it as
               authoritative is how you end up telling an admin to do the
               wrong thing in a language nobody on the team can proofread.

When a response contains untranslated source text, it is flagged with
`explanation_translated: false` so the UI can label it honestly rather than
leaving the user to guess which half they're reading.

Adding a language is a JSON file in data/i18n/ plus nothing else -- the CI
parity gate (monitoring/check_i18n.py) will fail the build if it's incomplete.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

I18N_DIR = Path(__file__).resolve().parent.parent / "data" / "i18n"
DEFAULT_LANG = "en"

_locales: dict[str, dict] | None = None


def _load() -> dict[str, dict]:
    """Load every locale once at first use."""
    global _locales
    if _locales is None:
        _locales = {}
        for path in sorted(I18N_DIR.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                _locales[data.get("_meta", {}).get("code", path.stem)] = data
            except Exception as exc:
                print(f"[i18n] skipping {path.name}: {type(exc).__name__}: {exc}")
    return _locales


def available() -> list[dict]:
    """Language list for the UI picker."""
    return [
        {
            "code": code,
            "name": d.get("_meta", {}).get("name", code),
            "native_name": d.get("_meta", {}).get("native_name", code),
            "dir": d.get("_meta", {}).get("dir", "ltr"),
        }
        for code, d in sorted(_load().items())
    ]


def resolve(lang: str | None, accept_language: str | None = None) -> str:
    """
    Pick a supported language.

    Explicit `?lang=` wins. Otherwise fall back to the browser's
    Accept-Language header, honouring its q-weights, so a Spanish-locale
    browser gets Spanish without having to ask. Unsupported values degrade to
    English rather than erroring -- a bad locale should never fail a request.
    """
    locales = _load()

    if lang:
        code = lang.strip().lower().replace("_", "-").split("-")[0]
        if code in locales:
            return code

    if accept_language:
        # "fr-CA,fr;q=0.9,en;q=0.8" -> [("fr", 1.0), ("fr", 0.9), ("en", 0.8)]
        parsed: list[tuple[str, float]] = []
        for part in accept_language.split(","):
            bits = part.strip().split(";")
            tag = bits[0].strip().lower().split("-")[0]
            q = 1.0
            for extra in bits[1:]:
                m = re.match(r"\s*q=([\d.]+)", extra)
                if m:
                    try:
                        q = float(m.group(1))
                    except ValueError:
                        pass
            if tag:
                parsed.append((tag, q))
        for tag, _ in sorted(parsed, key=lambda kv: -kv[1]):
            if tag in locales:
                return tag

    return DEFAULT_LANG


def strings(lang: str) -> dict:
    """UI string bundle for the frontend."""
    locales = _load()
    d = locales.get(lang) or locales.get(DEFAULT_LANG) or {}
    base = locales.get(DEFAULT_LANG, {}).get("ui", {})
    return {**base, **d.get("ui", {})}


def localize(result: dict, lang: str) -> dict:
    """
    Overlay translations onto a classify() result.

    Never mutates the input, and never blanks a field: any string without a
    translation keeps its English value. A partially translated locale
    degrades to mixed-language output, not to empty output.
    """
    locales = _load()
    code_now = result.get("error_code", "")

    if lang == DEFAULT_LANG or lang not in locales:
        out = dict(result)
        out.setdefault("lang", DEFAULT_LANG)
        out.setdefault("explanation_translated", True)
        en = locales.get(DEFAULT_LANG, {})
        out["manual_steps"] = en.get("manual_steps", {}).get(code_now, [])
        return out

    loc = locales[lang]
    en = locales.get(DEFAULT_LANG, {})
    out = dict(result)
    code = result.get("error_code", "")
    source = result.get("source", "")
    action = result.get("action", "")

    # action_detail: keyed by remediation action, so it covers every tier.
    if action:
        translated = loc.get("action", {}).get(action)
        if translated:
            out["action_detail"] = translated

    # user_message + explanation: only the 15 curated codes are authored by us.
    if code and code in loc.get("user_message", {}):
        out["user_message"] = loc["user_message"][code]

    if source == "abstain":
        ab = loc.get("abstain", {})
        out["explanation"] = ab.get("explanation", out.get("explanation", ""))
        out["action_detail"] = ab.get("action_detail", out.get("action_detail", ""))
        out["user_message"] = ab.get("user_message", out.get("user_message", ""))

    # Whether `explanation` is text we authored (and therefore translated) or
    # source material quoted from Microsoft's docs (which we leave in English).
    #
    # Tier is not the right test on its own: a retrieval hit can land on one of
    # the 15 curated codes, in which case the description in the catalog is our
    # own curated `cause` text, not Microsoft's. So key off whether we actually
    # authored copy for this code.
    is_curated = bool(code) and code in loc.get("user_message", {})
    explanation_translated = is_curated or source == "abstain"

    if is_curated:
        # Curated `explanation` isn't translated as a separate field -- the
        # user-facing message is the sentence people actually read, so surface
        # that rather than leaving English prose under a translated heading.
        out["explanation"] = loc["user_message"][code]
    elif source == "retrieval":
        # Untranslated Microsoft description. Don't echo it into the
        # user-facing slot when we have a translated action description.
        if not out.get("user_message") or out["user_message"] == result.get("explanation"):
            out["user_message"] = out.get("action_detail", out.get("user_message", ""))

    # Manual portal steps, translated where available (English fallback).
    steps = loc.get("manual_steps", {}).get(code)
    if not steps:
        steps = en.get("manual_steps", {}).get(code, [])
    out["manual_steps"] = steps

    out["lang"] = lang
    out["explanation_translated"] = explanation_translated
    return out
