#!/usr/bin/env python3
"""
AzureAutoFix — Localisation completeness gate.

A half-translated UI is worse than an untranslated one: the user can't tell
which strings are stale and which are current. This fails the build if any
locale drifts from the English reference.

Checks:
  1. Every locale has exactly the same key set as en.json -- no missing keys,
     no orphans left behind after an English string is renamed.
  2. No empty or whitespace-only values.
  3. No value is byte-identical to English in a non-English locale, unless it
     is on an allowlist of legitimately-identical strings (product names,
     protocol terms). Catches placeholder copy-paste.
  4. Every t() key the frontend references exists in the locale files.
  5. Every remediation action emitted by the catalog has an `action` string,
     so no user can hit an untranslated remediation.

Usage:
    python -m monitoring.check_i18n
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
I18N = ROOT / "data" / "i18n"
FRONTEND = ROOT / "frontend" / "index.html"
CATALOG = ROOT / "data" / "aadsts_catalog.json"
REFERENCE = "en"

# Strings that are legitimately identical across languages.
IDENTICAL_OK = {
    "ui.stat_p50", "ui.stat_p95",
    "ui.source_label",          # "Source" / "Fuente" / "Source" -- fr matches en
    "ui.field_redirect_uri",    # "Redirect URI" is the Azure portal's own label
    "ui.nav_support",           # "GitHub" -- brand name
    "ui.nav_settings",          # "README" -- technical term
    "ui.nav_detection",         # "Extension" -- same word in EN/FR
}


def flatten(d: dict, prefix: str = "") -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in d.items():
        if k == "_meta":
            continue
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def main() -> int:
    locales = {p.stem: json.loads(p.read_text(encoding="utf-8"))
               for p in sorted(I18N.glob("*.json"))}
    if REFERENCE not in locales:
        print(f"FAIL: {REFERENCE}.json missing")
        return 1

    ref = flatten(locales[REFERENCE])
    errors: list[str] = []

    print(f"Locales: {', '.join(sorted(locales))}  ({len(ref)} strings each)\n")

    for code, data in sorted(locales.items()):
        flat = flatten(data)
        missing = sorted(set(ref) - set(flat))
        orphan = sorted(set(flat) - set(ref))
        empty = sorted(k for k, v in flat.items() if not str(v).strip())
        identical = sorted(
            k for k, v in flat.items()
            if code != REFERENCE and k in ref and v == ref[k] and k not in IDENTICAL_OK
        )

        for label, items in (("missing", missing), ("orphaned", orphan),
                             ("empty", empty), ("untranslated", identical)):
            if items:
                errors.append(f"{code}: {len(items)} {label} -> {items[:6]}")

        meta = data.get("_meta", {})
        for field in ("code", "name", "native_name"):
            if not meta.get(field):
                errors.append(f"{code}: _meta.{field} missing")

        status = "OK" if not (missing or orphan or empty or identical) else "FAIL"
        print(f"  {code:<4} {meta.get('native_name','?'):<12} {len(flat):>3} strings  {status}")

    # 4. frontend key coverage
    if FRONTEND.exists():
        html = FRONTEND.read_text(encoding="utf-8")
        used = set(re.findall(r'(?<![a-zA-Z])t\(\s*"([a-z0-9_]+)"', html))
        used |= set(re.findall(r'set\(\s*"[a-z-]+",\s*"([a-z0-9_]+)"', html))
        used |= set(re.findall(r'"(cat_[a-z_]+)"', html))
        ui_keys = {k.split(".", 1)[1] for k in ref if k.startswith("ui.")}
        unknown = sorted(used - ui_keys)
        if unknown:
            errors.append(f"frontend references undefined keys: {unknown}")
        print(f"\n  frontend: {len(used)} keys referenced, "
              f"{'all defined' if not unknown else f'{len(unknown)} UNDEFINED'}")

    # 5. every catalog action has a translation
    if CATALOG.exists():
        actions = {d.get("action") for d in json.loads(CATALOG.read_text(encoding="utf-8"))}
        actions.discard(None)
        action_keys = {k.split(".", 1)[1] for k in ref if k.startswith("action.")}
        missing_actions = sorted(actions - action_keys)
        if missing_actions:
            errors.append(f"catalog actions with no translation: {missing_actions}")
        print(f"  catalog:  {len(actions)} distinct actions, "
              f"{'all translated' if not missing_actions else f'{len(missing_actions)} MISSING'}")

    if errors:
        print("\nFAIL")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("\nPASS — all locales complete and consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
