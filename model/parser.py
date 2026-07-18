"""
Azure AD Log Parser — Drain-inspired (He et al., ICWS 2017)
============================================================
Implements a simplified fixed-depth parse tree to extract structured
log keys (AADSTS codes and event templates) from raw Azure AD error strings.

The three-paper AIOps pipeline this belongs to:
  [1] Drain (ICWS 2017)  — THIS FILE: parse raw strings → log keys
  [2] DeepLog (CCS 2017) — model/sequence_detector.py: detect anomalous sequences
  [3] Fix-category classification — model/retrieval.py (hybrid retrieval).
      model/logbert_classifier.py is reference only and is not wired in.

Drain design principles applied here:
  - Step 1: Preprocess by domain knowledge (regex for AADSTS codes, UUIDs, timestamps)
  - Step 2: Search by log message length (token count partitioning)
  - Step 3: Search by preceding tokens (first non-variable token as branch key)
  - Step 4: Search by token similarity (simSeq = matching tokens / length)
  - Step 5: Update parse tree (merge differing positions to wildcard '*')

Parameters (Table II from Drain paper, adapted for Azure AD short messages):
  depth=3, similarity_threshold=0.4, max_children=100

Reference: He, P., Zhu, J., Zheng, Z., & Lyu, M.R. (2017). Drain: An Online
Log Parsing Approach with Fixed Depth Tree. In IEEE ICWS 2017.
DOI: 10.1109/ICWS.2017.13
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Drain Step 1: Domain-knowledge preprocessing patterns
# ---------------------------------------------------------------------------

# Primary log key extractor — AADSTS error codes are the Azure AD "log key"
# in the same sense that HDFS block operation names are log keys in the paper.
AADSTS_RE = re.compile(r"AADSTS\d+", re.IGNORECASE)

# Variable-pattern normalizers (equivalent to Drain's block-ID / IP removal)
_VARIABLE_PATTERNS = [
    re.compile(r"\b[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\b"),  # UUID / Trace ID
    re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?\b"),     # ISO timestamp
    re.compile(r"\b\d{10,}\b"),                                                   # Long numeric IDs
    re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),                                  # IPv4 addresses
]

# Tokens that contain digits are treated as variables (Drain Step 3 rule)
_DIGIT_RE = re.compile(r"\d")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LogGroup:
    """
    Drain leaf-level log group.

    log_event: mutable token list; positions with variable values become '*'
               (Drain Step 5 update rule).
    log_ids:   IDs of all messages matched to this group.
    aadsts_code: the AADSTS error code extracted from the first message,
                 used as the canonical log key for this group.
    """
    log_event: list[str]
    log_ids: list[int] = field(default_factory=list)
    aadsts_code: Optional[str] = None


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class DrainParser:
    """
    Drain fixed-depth parse tree adapted for Azure AD log messages.

    Tree layout (depth=3):
      Layer 0  — root (dict)
      Layer 1  — keyed by token count (log message length)
      Layer 2  — keyed by first non-variable token (or '*')
      Layer 3  — leaf: list[LogGroup]

    This matches Figure 2 of the Drain paper exactly, with depth=3.
    """

    def __init__(
        self,
        depth: int = 3,
        similarity_threshold: float = 0.4,
        max_children: int = 100,
    ):
        self.depth = depth
        self.st = similarity_threshold
        self.max_children = max_children
        # root: {length: {first_token: [LogGroup]}}
        self._tree: dict[int, dict[str, list[LogGroup]]] = {}
        self._next_id = 0

    # ------------------------------------------------------------------
    # Step 1: Preprocess
    # ------------------------------------------------------------------

    def _preprocess(self, raw: str) -> tuple[list[str], Optional[str]]:
        """
        Extract log key and normalize variable tokens.

        Returns:
            tokens     — list of string tokens with variables replaced by '*'
            aadsts_code — the AADSTS code found (None if absent)
        """
        # Extract AADSTS code before normalizing (it would get replaced otherwise)
        m = AADSTS_RE.search(raw)
        aadsts_code = m.group(0).upper() if m else None

        # Replace variable patterns with '*'
        normalized = raw
        for pat in _VARIABLE_PATTERNS:
            normalized = pat.sub("*", normalized)

        # Tokenize on whitespace and common separators
        tokens = re.split(r"[\s\-:,;()\[\]\"']+", normalized.strip())
        tokens = [t for t in tokens if t]
        return tokens, aadsts_code

    # ------------------------------------------------------------------
    # Steps 2-3: Navigate tree
    # ------------------------------------------------------------------

    def _first_key(self, tokens: list[str]) -> str:
        """
        Drain Step 3: first token that does not contain a digit.
        Tokens starting with digits are mapped to '*' to prevent branch explosion.
        """
        for t in tokens:
            if not _DIGIT_RE.search(t):
                return t
        return "*"

    def _get_candidates(self, tokens: list[str]) -> list[LogGroup]:
        """Return the leaf-node group list for these tokens (Steps 2-3)."""
        length = len(tokens)
        if length not in self._tree:
            return []
        by_first = self._tree[length]
        key = self._first_key(tokens)
        # Fall back to wildcard bucket if exact key absent
        return by_first.get(key, by_first.get("*", []))

    def _insert_to_tree(self, tokens: list[str], group: LogGroup) -> None:
        """Insert a new log group into the correct leaf bucket."""
        length = len(tokens)
        if length not in self._tree:
            self._tree[length] = {}
        by_first = self._tree[length]
        key = self._first_key(tokens)
        if key not in by_first:
            if len(by_first) >= self.max_children:
                key = "*"
            if key not in by_first:
                by_first[key] = []
        by_first[key].append(group)

    # ------------------------------------------------------------------
    # Step 4: Token similarity (Equation 1 from the paper)
    # ------------------------------------------------------------------

    @staticmethod
    def _sim_seq(tokens: list[str], log_event: list[str]) -> float:
        """
        simSeq = Σ equ(seq1(i), seq2(i)) / n    (Drain Eq. 1)

        Wildcards ('*') in the log_event match any token.
        Returns 0.0 if lengths differ (length partitioning ensures they match).
        """
        if len(tokens) != len(log_event):
            return 0.0
        n = len(tokens)
        if n == 0:
            return 1.0
        matches = sum(
            1 for t, e in zip(tokens, log_event)
            if e == "*" or t == e
        )
        return matches / n

    # ------------------------------------------------------------------
    # Step 5: Update parse tree
    # ------------------------------------------------------------------

    def _update_log_event(self, group: LogGroup, tokens: list[str]) -> None:
        """
        Merge new message tokens into existing log event.
        Positions where tokens differ become wildcards (Drain Step 5).
        """
        for i, (t, e) in enumerate(zip(tokens, group.log_event)):
            if e != "*" and t != e:
                group.log_event[i] = "*"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, raw: str) -> dict:
        """
        Parse a raw Azure AD log string into a structured log event.

        Returns a dict with:
            log_key    — AADSTS code, or 'UNKNOWN'
            log_event  — template string (constants + '*' for variables)
            raw        — original input
        """
        tokens, aadsts_code = self._preprocess(raw)

        if not tokens:
            return {"log_key": "UNKNOWN", "log_event": "", "raw": raw}

        # Step 4: Find best matching log group
        candidates = self._get_candidates(tokens)
        best_group: Optional[LogGroup] = None
        best_sim = -1.0

        for grp in candidates:
            sim = self._sim_seq(tokens, grp.log_event)
            if sim > best_sim:
                best_sim = sim
                best_group = grp

        # Step 5: Update or create
        msg_id = self._next_id
        self._next_id += 1

        if best_group is not None and best_sim >= self.st:
            best_group.log_ids.append(msg_id)
            self._update_log_event(best_group, tokens)
            group = best_group
        else:
            group = LogGroup(
                log_event=list(tokens),
                log_ids=[msg_id],
                aadsts_code=aadsts_code,
            )
            self._insert_to_tree(tokens, group)

        return {
            "log_key": aadsts_code or "UNKNOWN",
            "log_event": " ".join(group.log_event),
            "raw": raw,
        }

    @property
    def num_groups(self) -> int:
        """Total number of log groups currently in the parse tree."""
        return sum(
            len(grps)
            for by_first in self._tree.values()
            for grps in by_first.values()
        )


# ---------------------------------------------------------------------------
# Module-level singleton (shared parser instance for streaming use)
# ---------------------------------------------------------------------------

_default_parser = DrainParser()


def parse_log(raw: str) -> dict:
    """
    Parse a raw Azure AD log string using the shared Drain parser instance.

    Example:
        >>> parse_log("Error AADSTS50057 - User account has been disabled. Trace ID: abc-123")
        {'log_key': 'AADSTS50057', 'log_event': 'Error AADSTS50057 * ...', 'raw': '...'}
    """
    return _default_parser.parse(raw)


def extract_log_key(raw: str) -> str:
    """
    Fast-path extractor: returns just the AADSTS code from a raw string.

    Used by the sequence detector (model/sequence_detector.py) to build
    log-key sequences without full parse-tree traversal.
    """
    m = AADSTS_RE.search(raw)
    return m.group(0).upper() if m else "UNKNOWN"


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _samples = [
        "Error AADSTS50057 - User account has been disabled. Trace ID: abc-123-def",
        "AADSTS50126: Invalid username or password",
        "Sign-in failed: AADSTS90094 - Admin consent required for application",
        "AADSTS90033: A transient error has occurred. Please retry.",
        "AADSTS50020: User account from identity provider does not exist in tenant",
        # Duplicate-style — should merge into same template
        "Error AADSTS50057 - User account has been disabled. Trace ID: xyz-999-abc",
    ]

    parser = DrainParser(depth=3, similarity_threshold=0.4)
    for s in _samples:
        r = parser.parse(s)
        print(f"  Key:      {r['log_key']}")
        print(f"  Template: {r['log_event']}")
        print()
    print(f"Total log groups in parse tree: {parser.num_groups}")
