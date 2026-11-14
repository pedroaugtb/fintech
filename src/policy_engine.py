#!/usr/bin/env python3
"""
Policy-as-Code study - Policy-as-Code runtime engine (conditions C2 and C3)

Executes policies/policies_v1.yaml over generated memos. Every claim gets a
PASS / BLOCK / ESCALATE decision per policy, and every decision is written to
an audit log that is sufficient to replay it.

C2 and C3 are post-generation controls (paper proposal, Section 4.3), so they
are applied to existing C1 outputs rather than regenerated:
    C2  BLOCK and ESCALATE both withhold the claim (automatic abstention)
    C3  BLOCK withholds; ESCALATE routes the claim to human review

Outputs:
    data/generated/C2/<model>/*.json
    data/generated/C3/<model>/*.json
    data/generated/<C2|C3>/<model>/policy_audit.jsonl

Usage:
    python src/policy_engine.py                     # all models, source C1
    python src/policy_engine.py --source-condition C0
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
GENERATED_DIR = DATA_DIR / "generated"
CHUNKS_DIR = DATA_DIR / "chunks"
POLICY_FILE = PROJECT_ROOT / "policies" / "policies_v1.yaml"

FY = 2025

POLICY_IDS = (
    "P1_GROUNDEDNESS",
    "P2_NUMERIC_CONSISTENCY",
    "P3_UNCERTAINTY",
    "P4_DECISION_BOUNDARY",
)


# ============================================================
# POLICY SET
# ============================================================

def load_policy_set(path: Path = POLICY_FILE) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    policy_set = yaml.safe_load(raw)

    policy_set["sha256"] = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    policy_set["by_id"] = {
        policy["policy_id"]: policy
        for policy in policy_set["policies"]
    }

    for policy in policy_set["policies"]:
        if policy["policy_id"] == "P4_DECISION_BOUNDARY":
            policy["compiled"] = [
                re.compile(pattern, re.IGNORECASE)
                for pattern in policy["rule"]["patterns"]
            ]

    return policy_set


# ============================================================
# TEXT SIGNALS
# ============================================================

STOPWORDS = set("""
a an the and or but if of in on to by at for with from into over under about
as is are was were be been being has have had its it their our we they this
that these those which who whom than then also such other can could may might
will would shall should not no per each both all any more most less least very
fy fiscal year years company company's corporation inc
""".split())

# Tokens that look numeric but are not quantities.
NON_QUANTITY_RE = re.compile(
    r"\b[A-Z]{1,5}_FY\d{4}_[0-9A-Za-z_]+\b"          # evidence ids
    r"|\b(?:10|8)-[KQ](?:/A)?\b"                      # form names
    r"|\b(?:Item|Note|Section|Rule|Form|Tier|Level|Phase|Part)\s+\d+[A-Za-z]?\b"
    r"|\bCOVID-19\b|\bS&P\s+500\b|\bASU\s+\d{4}-\d{2}\b"
    r"|\b(?:January|February|March|April|May|June|July|August|September"
    r"|October|November|December)\s+\d{1,2}\b",
    re.IGNORECASE,
)

NUMBER_RE = re.compile(
    r"(?P<cur>\$\s?)?"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"\s*(?P<unit>%|percent\b|percentage\s+points?\b|basis\s+points\b|bps\b"
    r"|trillion\b|billion\b|million\b|thousand\b|[BMK]\b)?",
    re.IGNORECASE,
)

YEAR_RE = re.compile(r"(?<![\d$.,])(?:19[89]\d|20[0-3]\d)(?![\d,%])")

SCALE = {
    "trillion": 1e12,
    "billion": 1e9, "b": 1e9,
    "million": 1e6, "m": 1e6,
    "thousand": 1e3, "k": 1e3,
}


def normalize_token(token: str) -> str:
    if len(token) > 4 and token.endswith("s"):
        return token[:-1]
    return token


def content_tokens(text: str, extra_stop: set[str] = frozenset()) -> list[str]:
    return [
        normalize_token(token)
        for token in re.findall(r"[a-z][a-z'\-]+", text.lower())
        if len(token) > 2
        and token not in STOPWORDS
        and token not in extra_stop
    ]


def extract_years(text: str) -> set[int]:
    return {int(match) for match in YEAR_RE.findall(text)}


def extract_numbers(text: str) -> list[dict[str, Any]]:
    """
    Quantities in text. Years, form names, dates and ids are excluded; bare
    small integers (counts like "3 segments") are ignored.
    """

    cleaned = NON_QUANTITY_RE.sub(" ", text)
    numbers = []

    for match in NUMBER_RE.finditer(cleaned):
        raw = match.group("num")
        unit = (match.group("unit") or "").lower().strip()
        currency = bool(match.group("cur"))
        value = float(raw.replace(",", ""))

        is_percent = unit in ("%", "percent") or unit.startswith("percentage") \
            or unit in ("basis points", "bps")

        if unit in ("basis points", "bps"):
            value = value / 100

        is_year = (
            not currency and not unit and "," not in raw and "." not in raw
            and 1980 <= value <= 2039
        )

        if is_year:
            continue

        if not currency and not unit and value < 32 and "." not in raw:
            continue

        decimals = len(raw.split(".")[1]) if "." in raw else 0
        scale = 1.0 if is_percent else SCALE.get(unit, 1.0)

        numbers.append({
            "raw": match.group(0).strip(),
            "value": value * scale,
            "half_unit": 0.5 * 10 ** (-decimals) * scale,
            "percent": is_percent,
            "scaled": unit in SCALE,
        })

    return numbers


def number_supported(
    claim_number: dict[str, Any],
    evidence_numbers: list[dict[str, Any]],
    relative_tolerance: float,
) -> bool:
    target = claim_number["value"]
    tolerance = max(
        relative_tolerance * abs(target),
        claim_number["half_unit"],
    )

    for evidence in evidence_numbers:

        if claim_number["percent"] != evidence["percent"]:
            continue

        # Tables in 10-Ks are often "in millions" without a unit next to
        # the figure, so unscaled evidence numbers may be rescaled.
        candidates = [evidence["value"]]

        if not evidence["percent"] and not evidence["scaled"]:
            candidates += [evidence["value"] * s for s in (1e3, 1e6, 1e9)]

        if any(abs(candidate - target) <= tolerance for candidate in candidates):
            return True

    return False


# ============================================================
# POLICY CHECKS
# ============================================================

def check_claim(
    claim: dict[str, Any],
    evidence_text: dict[str, str],
    available_ids: set[str],
    policy_set: dict[str, Any],
    company_tokens: set[str] = frozenset(),
) -> dict[str, dict[str, Any]]:
    """
    Run every policy on one claim. Returns policy_id -> result with keys
    passed (bool), action (on fail), score and details. Thresholds are read
    from the policy set, so sweeps only need to edit the loaded dict.
    """

    policies = policy_set["by_id"]
    text = str(claim.get("text", ""))
    cited = [str(e) for e in claim.get("evidence_ids", []) or []]
    cited_in_context = [e for e in cited if e in available_ids]
    cited_text = "\n".join(evidence_text.get(e, "") for e in cited_in_context)

    results: dict[str, dict[str, Any]] = {}

    # --- P1 groundedness -------------------------------------------------
    p1 = policies["P1_GROUNDEDNESS"]
    tokens = content_tokens(text, company_tokens)
    evidence_tokens = set(content_tokens(cited_text))
    support = (
        sum(token in evidence_tokens for token in tokens) / len(tokens)
        if tokens else 1.0
    )
    fabricated = [e for e in cited if e not in available_ids]

    p1_passed = (
        (bool(cited) or not p1["rule"]["require_citation"])
        and (not fabricated or not p1["rule"]["require_cited_ids_in_context"])
        and support >= p1["rule"]["min_lexical_support"]
    )

    results["P1_GROUNDEDNESS"] = {
        "passed": p1_passed,
        "action": p1["action_on_fail"],
        "score": round(support, 4),
        "threshold": p1["rule"]["min_lexical_support"],
        "details": {
            "n_cited": len(cited),
            "fabricated_ids": fabricated,
        },
    }

    # --- P2 numeric consistency -----------------------------------------
    p2 = policies["P2_NUMERIC_CONSISTENCY"]
    claim_numbers = extract_numbers(text)
    evidence_numbers = extract_numbers(cited_text)
    unmatched = [
        number["raw"]
        for number in claim_numbers
        if not number_supported(
            number,
            evidence_numbers,
            p2["rule"]["relative_tolerance"],
        )
    ]

    results["P2_NUMERIC_CONSISTENCY"] = {
        "passed": not unmatched,
        "action": p2["action_on_fail"],
        "score": (
            round(1 - len(unmatched) / len(claim_numbers), 4)
            if claim_numbers else None
        ),
        "threshold": p2["rule"]["relative_tolerance"],
        "details": {
            "in_scope": bool(claim_numbers),
            "unmatched_numbers": unmatched,
        },
    }

    # --- P3 uncertainty / conflicts -------------------------------------
    p3 = policies["P3_UNCERTAINTY"]
    self_flagged = claim.get("sufficient_evidence") is False
    evidence_years = extract_years(cited_text)
    unmatched_years = sorted(
        year
        for year in extract_years(text) - {FY}
        if year not in evidence_years
    )

    p3_passed = not (
        (self_flagged and p3["rule"]["escalate_if_self_flagged"])
        or (unmatched_years and p3["rule"]["require_claim_years_in_evidence"])
    )

    results["P3_UNCERTAINTY"] = {
        "passed": p3_passed,
        "action": p3["action_on_fail"],
        "score": None,
        "threshold": None,
        "details": {
            "self_flagged": self_flagged,
            "unmatched_years": unmatched_years,
        },
    }

    # --- P4 decision boundary -------------------------------------------
    p4 = policies["P4_DECISION_BOUNDARY"]
    matched = next(
        (pattern.pattern for pattern in p4["compiled"] if pattern.search(text)),
        None,
    )

    results["P4_DECISION_BOUNDARY"] = {
        "passed": matched is None,
        "action": p4["action_on_fail"],
        "score": None,
        "threshold": None,
        "details": {"matched_pattern": matched},
    }

    return results


def aggregate(
    results: dict[str, dict[str, Any]],
    enabled: tuple[str, ...] = POLICY_IDS,
) -> str:
    """BLOCK beats ESCALATE beats PASS, over the enabled policies only."""

    actions = {
        results[policy_id]["action"]
        for policy_id in enabled
        if not results[policy_id]["passed"]
    }

    if "BLOCK" in actions:
        return "BLOCK"
    if "ESCALATE" in actions:
        return "ESCALATE"
    return "PASS"


def memo_boundary_violation(memo: str, policy_set: dict[str, Any]) -> str | None:
    p4 = policy_set["by_id"]["P4_DECISION_BOUNDARY"]
    return next(
        (pattern.pattern for pattern in p4["compiled"] if pattern.search(memo)),
        None,
    )


# ============================================================
# DATA
# ============================================================

def load_evidence_text() -> dict[str, str]:
    evidence: dict[str, str] = {}

    for path in CHUNKS_DIR.glob(f"*_FY{FY}.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                evidence[item["evidence_id"]] = item["text"]

    return evidence


def load_company_tokens() -> dict[str, set[str]]:
    """ticker -> tokens of the company name (not evidence of anything)."""

    tokens: dict[str, set[str]] = {}

    with (DATA_DIR / "dataset_manifest.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tokens[row["ticker"].upper()] = set(
                content_tokens(row["company"])
            ) | {row["ticker"].lower()}

    return tokens


# ============================================================
# RUNTIME APPLICATION (C2 / C3)
# ============================================================

def apply_to_memo(
    doc: dict[str, Any],
    evidence_text: dict[str, str],
    policy_set: dict[str, Any],
    company_tokens: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Returns (per-claim decisions, memo-level summary)."""

    result = doc.get("result") or {}
    available = set(doc.get("evidence_ids_available", []))
    decisions = []

    for index, claim in enumerate(result.get("claims", [])):
        checks = check_claim(
            claim,
            evidence_text,
            available,
            policy_set,
            company_tokens,
        )
        decisions.append({
            "claim_index": index,
            "claim": claim,
            "checks": checks,
            "decision": aggregate(checks),
        })

    memo_violation = memo_boundary_violation(
        str(result.get("memo", "")),
        policy_set,
    )

    aggregation = policy_set["memo_aggregation"]

    if memo_violation and aggregation["block_memo_on_decision_boundary"]:
        memo_decision = "BLOCK"
    elif (
        aggregation["escalate_memo_if_any_claim_escalated"]
        and any(d["decision"] == "ESCALATE" for d in decisions)
    ):
        memo_decision = "ESCALATE"
    else:
        memo_decision = "PASS"

    return decisions, {
        "memo_decision": memo_decision,
        "memo_boundary_pattern": memo_violation,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply the runtime policies to generated memos."
    )
    parser.add_argument("--source-condition", default="C1")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--policy-file", type=Path, default=POLICY_FILE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    policy_set = load_policy_set(args.policy_file)
    evidence_text = load_evidence_text()
    company_tokens = load_company_tokens()

    source_root = GENERATED_DIR / args.source_condition
    models = args.models or sorted(
        path.name for path in source_root.iterdir() if path.is_dir()
    )

    versions = {
        policy["policy_id"]: policy["version"]
        for policy in policy_set["policies"]
    }

    for model in models:

        output_dirs = {
            condition: GENERATED_DIR / condition / model
            for condition in ("C2", "C3")
        }

        for directory in output_dirs.values():
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "policy_audit.jsonl").unlink(missing_ok=True)

        counts = {"PASS": 0, "BLOCK": 0, "ESCALATE": 0}

        for path in sorted((source_root / model).glob(f"*_FY{FY}.json")):

            doc = json.loads(path.read_text(encoding="utf-8"))

            if not doc.get("result"):
                continue

            start = time.perf_counter()
            decisions, memo = apply_to_memo(
                doc,
                evidence_text,
                policy_set,
                company_tokens.get(doc["ticker"], set()),
            )
            engine_ms = (time.perf_counter() - start) * 1000

            timestamp = datetime.now(timezone.utc).isoformat()

            for condition, directory in output_dirs.items():

                released, withheld, escalated = [], [], []

                for decision in decisions:
                    if decision["decision"] == "PASS":
                        released.append(decision["claim"])
                    elif decision["decision"] == "ESCALATE" and condition == "C3":
                        escalated.append(decision["claim"])
                    else:
                        withheld.append(decision["claim"])

                out = {
                    **{k: v for k, v in doc.items() if k not in ("result", "raw_output")},
                    "condition": condition,
                    "source_condition": args.source_condition,
                    "policy_set": policy_set["policy_set"],
                    "policy_set_version": policy_set["version"],
                    "policy_set_sha256": policy_set["sha256"],
                    "memo_decision": memo["memo_decision"],
                    "result": {
                        "memo": (
                            None if memo["memo_decision"] == "BLOCK"
                            else doc["result"].get("memo")
                        ),
                        "claims": released,
                    },
                    "withheld_claims": withheld,
                    "escalated_for_review": escalated,
                }

                (directory / path.name).write_text(
                    json.dumps(out, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                with (directory / "policy_audit.jsonl").open("a", encoding="utf-8") as audit:
                    for decision in decisions:
                        claim = decision["claim"]
                        for policy_id, check in decision["checks"].items():
                            audit.write(json.dumps({
                                "timestamp": timestamp,
                                "condition": condition,
                                "source_condition": args.source_condition,
                                "model": model,
                                "ticker": doc["ticker"],
                                "claim_index": decision["claim_index"],
                                "claim_id": claim.get("claim_id"),
                                "evidence_ids": claim.get("evidence_ids", []),
                                "policy_id": policy_id,
                                "policy_version": versions[policy_id],
                                "policy_set_sha256": policy_set["sha256"],
                                "passed": check["passed"],
                                "action_on_fail": check["action"],
                                "score": check["score"],
                                "threshold": check["threshold"],
                                "details": check["details"],
                                "claim_decision": decision["decision"],
                                "memo_decision": memo["memo_decision"],
                                "engine_ms_memo": round(engine_ms, 3),
                            }, ensure_ascii=False) + "\n")

            for decision in decisions:
                counts[decision["decision"]] += 1

        total = sum(counts.values()) or 1
        print(
            f"{model:22s} claims={total:5d} "
            + " ".join(f"{k}={v} ({v / total:.1%})" for k, v in counts.items())
        )

    print(f"\npolicy set {policy_set['version']} sha256={policy_set['sha256'][:12]}")


if __name__ == "__main__":
    main()
