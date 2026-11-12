"""
RAIOps4Fin - claim-level evaluation (paper proposal, Section 5)

Loads generated memos, runs the runtime policies once per claim, attaches
error labels, and computes the proposal's metrics per scenario with a
company-level cluster bootstrap.

Scenarios (a claim is "released" when it reaches the user):
    C0, C1  everything generated is released
    C2      runtime policies; BLOCK and ESCALATE are withheld
    C3      runtime policies; ESCALATE goes to a human reviewer, modeled as
            an oracle that releases correct claims and drops erroneous ones

Label sources, in priority order (see load_labels):
    panel   majority of the LLM-judge panel (data/annotations/judges/<judge>/),
            judges from families other than the generators, every claim
    judge   data/annotations/judge_labels.csv  (single Qwen3.5-27B judge)
    judge:<name>  one judge of data/annotations/judges/
    human   data/annotations/human_labels.csv  (stratified sample, weighted)
    proxy   automatic checks against ALL retrieved evidence and XBRL.
            Proxy labels overlap with the policy signals by construction, so
            they validate the pipeline, not the paper's claims.
"""

from __future__ import annotations

import itertools
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import policy_engine as pe


PROJECT_ROOT = pe.PROJECT_ROOT
DATA_DIR = pe.DATA_DIR
GENERATED_DIR = pe.GENERATED_DIR
ANNOTATION_DIR = DATA_DIR / "annotations"
XBRL_DIR = DATA_DIR / "xbrl" / "normalized"

SEVERITY_WEIGHT = {"S0": 0, "S1": 1, "S2": 3, "S3": 5}
SEVERE = ("S2", "S3")
SEVERITIES = ("S0", "S1", "S2", "S3")

JUDGES_DIR = ANNOTATION_DIR / "judges"

# Consensus panel: no Qwen/Llama/Mistral family, so no judge grades its own
# generator and three votes never tie. Other judges are reference raters.
PANEL = ("phi4", "gpt_oss_20b", "gemma4_26b_a4b")
JUDGE_NAME = {
    "phi4": "Phi-4 (14B)",
    "gpt_oss_20b": "gpt-oss-20b",
    "gemma4_26b_a4b": "Gemma-4-26B-A4B",
    "qwen35_27b": "Qwen3.5-27B",
    "codex_sample": "Codex/GPT (sample)",
    "human_sample": "Human expert (sample)",
    "panel_consensus": "Panel majority",
}

# Proxy-label thresholds (evaluation side, deliberately looser than P1 and
# computed against all retrieved evidence rather than the cited chunks).
PROXY_MIN_SUPPORT_ALL = 0.35
PROXY_STRONG_RISK_RE = re.compile(
    r"\b(significant|substantial|severe|material)\s+(risk|weakness|doubt|uncertaint)"
    r"|\b(default|insolven|going\s+concern|distress|downgrade)",
    re.IGNORECASE,
)

SHORT = {
    "P1_GROUNDEDNESS": "P1",
    "P2_NUMERIC_CONSISTENCY": "P2",
    "P3_UNCERTAINTY": "P3",
    "P4_DECISION_BOUNDARY": "P4",
}


# ============================================================
# LOADING
# ============================================================

def load_memos(conditions=("C0", "C1")) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (memos, claims) for the generation conditions."""

    memo_rows, claim_rows = [], []

    for condition in conditions:
        for model_dir in sorted((GENERATED_DIR / condition).glob("*")):

            audit_path = model_dir / "audit_log.jsonl"
            audits = {}

            if audit_path.exists():
                for line in audit_path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        record = json.loads(line)
                        audits[record["ticker"]] = record

            for path in sorted(model_dir.glob(f"*_FY{pe.FY}.json")):
                doc = json.loads(path.read_text(encoding="utf-8"))
                result = doc.get("result") or {}
                audit = audits.get(doc["ticker"], {})

                memo_rows.append({
                    "model": model_dir.name,
                    "condition": condition,
                    "ticker": doc["ticker"],
                    "parse_ok": bool(result),
                    "memo": result.get("memo"),
                    "n_claims": len(result.get("claims", [])),
                    "prompt_tokens": audit.get("prompt_tokens"),
                    "output_tokens": audit.get("output_tokens"),
                    "gen_latency_s": audit.get("latency_seconds_amortized"),
                    "evidence_ids_available": doc.get("evidence_ids_available", []),
                })

                for index, claim in enumerate(result.get("claims", [])):
                    claim_rows.append({
                        "model": model_dir.name,
                        "condition": condition,
                        "ticker": doc["ticker"],
                        "claim_index": index,
                        "claim_id": claim.get("claim_id"),
                        "text": claim.get("text", ""),
                        "topic": claim.get("topic"),
                        "evidence_ids": list(claim.get("evidence_ids") or []),
                        "sufficient_evidence": claim.get("sufficient_evidence"),
                    })

    return pd.DataFrame(memo_rows), pd.DataFrame(claim_rows)


def load_xbrl_values() -> dict[str, list[float]]:
    """ticker -> every XBRL value reported for FY2024/FY2025 (for the oracle)."""

    values: dict[str, list[float]] = {}

    for path in XBRL_DIR.glob("*_FY202[45].json"):
        doc = json.loads(path.read_text(encoding="utf-8"))
        bucket = values.setdefault(doc["ticker"], [])

        for metric in doc["metrics"].values():
            if metric and metric.get("value") is not None:
                bucket.append(float(metric["value"]))

        if doc.get("total_debt"):
            bucket.append(float(doc["total_debt"]["value"]))

    return values


# ============================================================
# POLICY SIGNALS (computed once per claim)
# ============================================================

def add_policy_signals(
    memos: pd.DataFrame,
    claims: pd.DataFrame,
    policy_set: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Adds one column per policy signal, so decisions for any policy subset or
    P1 threshold can be derived without re-running the checks.
    """

    evidence_text = pe.load_evidence_text()
    company_tokens = pe.load_company_tokens()
    available = {
        (row.model, row.condition, row.ticker): set(row.evidence_ids_available)
        for row in memos.itertuples()
    }

    records = []
    engine_ms: dict[tuple, float] = {}

    for row in claims.itertuples():
        key = (row.model, row.condition, row.ticker)
        start = pe.time.perf_counter()
        checks = pe.check_claim(
            {
                "text": row.text,
                "evidence_ids": row.evidence_ids,
                "sufficient_evidence": row.sufficient_evidence,
            },
            evidence_text,
            available[key],
            policy_set,
            company_tokens.get(row.ticker, set()),
        )
        engine_ms[key] = engine_ms.get(key, 0.0) + (pe.time.perf_counter() - start) * 1000

        p1 = checks["P1_GROUNDEDNESS"]
        records.append({
            "P1_pass": p1["passed"],
            "P1_support": p1["score"],
            "P1_cited": p1["details"]["n_cited"] > 0,
            "P1_fabricated": bool(p1["details"]["fabricated_ids"]),
            "P2_pass": checks["P2_NUMERIC_CONSISTENCY"]["passed"],
            "numeric": checks["P2_NUMERIC_CONSISTENCY"]["details"]["in_scope"],
            "P3_pass": checks["P3_UNCERTAINTY"]["passed"],
            "self_flagged": checks["P3_UNCERTAINTY"]["details"]["self_flagged"],
            "P4_pass": checks["P4_DECISION_BOUNDARY"]["passed"],
        })

    claims = pd.concat([claims.reset_index(drop=True), pd.DataFrame(records)], axis=1)

    memos = memos.copy()
    memos["engine_ms"] = [
        engine_ms.get((r.model, r.condition, r.ticker), 0.0)
        for r in memos.itertuples()
    ]
    memos["memo_db_violation"] = [
        pe.memo_boundary_violation(str(m or ""), policy_set) is not None
        for m in memos["memo"]
    ]

    return memos, claims


def decide(
    claims: pd.DataFrame,
    enabled=pe.POLICY_IDS,
    p1_threshold: float | None = None,
) -> pd.Series:
    """PASS / BLOCK / ESCALATE per claim for a policy subset (vectorized)."""

    p1_fail = ~claims["P1_pass"]

    if p1_threshold is not None:
        p1_fail = (
            ~claims["P1_cited"]
            | claims["P1_fabricated"]
            | (claims["P1_support"] < p1_threshold)
        )

    block = pd.Series(False, index=claims.index)
    escalate = pd.Series(False, index=claims.index)

    if "P1_GROUNDEDNESS" in enabled:
        block |= p1_fail
    if "P4_DECISION_BOUNDARY" in enabled:
        block |= ~claims["P4_pass"]
    if "P2_NUMERIC_CONSISTENCY" in enabled:
        escalate |= ~claims["P2_pass"]
    if "P3_UNCERTAINTY" in enabled:
        escalate |= ~claims["P3_pass"]

    return pd.Series(
        np.where(block, "BLOCK", np.where(escalate, "ESCALATE", "PASS")),
        index=claims.index,
    )


# ============================================================
# LABELS
# ============================================================

def proxy_labels(
    memos: pd.DataFrame,
    claims: pd.DataFrame,
    policy_set: dict[str, Any],
) -> pd.DataFrame:
    """
    Automatic labels from all retrieved evidence + XBRL. Placeholder until
    judge/human labels exist; see module docstring for the caveat.
    """

    evidence_text = pe.load_evidence_text()
    company_tokens = pe.load_company_tokens()
    xbrl = load_xbrl_values()
    context = {
        (r.model, r.condition, r.ticker): "\n".join(
            evidence_text.get(e, "") for e in r.evidence_ids_available
        )
        for r in memos.itertuples()
    }
    context_tokens = {k: set(pe.content_tokens(v)) for k, v in context.items()}
    context_numbers = {k: pe.extract_numbers(v) for k, v in context.items()}
    context_years = {k: pe.extract_years(v) for k, v in context.items()}
    tolerance = policy_set["by_id"]["P2_NUMERIC_CONSISTENCY"]["rule"]["relative_tolerance"]

    rows = []

    for row in claims.itertuples():
        key = (row.model, row.condition, row.ticker)
        tokens = pe.content_tokens(row.text, company_tokens.get(row.ticker, set()))
        support_all = (
            sum(t in context_tokens[key] for t in tokens) / len(tokens)
            if tokens else 1.0
        )

        xbrl_numbers = [
            {"value": v, "percent": False, "scaled": True, "half_unit": 0}
            for v in xbrl.get(row.ticker, [])
        ]
        bad_number = any(
            not pe.number_supported(n, context_numbers[key], tolerance)
            and not pe.number_supported(n, xbrl_numbers, max(tolerance, 0.01))
            for n in pe.extract_numbers(row.text)
        )
        bad_year = bool(pe.extract_years(row.text) - {pe.FY} - context_years[key])

        if not row.P4_pass:
            error_type, severity = "decision_boundary", "S3"
        elif not row.P1_cited or row.P1_fabricated or support_all < PROXY_MIN_SUPPORT_ALL:
            error_type = "unsupported"
            severity = "S3" if PROXY_STRONG_RISK_RE.search(row.text) else "S2"
        elif bad_number:
            error_type, severity = "numeric", "S2"
        elif bad_year:
            error_type, severity = "temporal", "S2"
        else:
            error_type, severity = "none", "S0"

        rows.append({"error_type": error_type, "severity": severity})

    labels = pd.DataFrame(rows, index=claims.index)
    labels["weight"] = 1.0
    return labels


def load_labels(
    memos: pd.DataFrame,
    claims: pd.DataFrame,
    policy_set: dict[str, Any],
    source: str = "auto",
) -> tuple[pd.DataFrame, str]:
    """
    Returns (labels aligned to claims.index with error_type/severity/weight,
    source name). Human labels cover a sample: unlabeled claims get weight 0
    and labeled ones their inverse sampling probability.
    """

    keys = ["model", "condition", "ticker", "claim_index"]
    judge = ANNOTATION_DIR / "judge_labels.csv"
    human = ANNOTATION_DIR / "human_labels.csv"

    if source == "auto":
        source = (
            "panel" if len(available_judges(PANEL)) >= 2
            else "judge" if judge.exists()
            else "human" if human.exists()
            else "proxy"
        )

    if source == "proxy":
        return proxy_labels(memos, claims, policy_set), "proxy"

    if source == "panel":
        return panel_labels(claims), "panel"

    if source.startswith("judge:"):
        path = JUDGES_DIR / source.split(":", 1)[1] / "labels.csv"
    else:
        path = judge if source == "judge" else human
    labeled = pd.read_csv(path)
    merged = claims[keys].merge(labeled, on=keys, how="left")
    merged.index = claims.index

    if "weight" not in merged:
        merged["weight"] = 1.0

    merged["weight"] = merged["weight"].where(merged["severity"].notna(), 0.0)
    merged["severity"] = merged["severity"].fillna("S0")
    merged["error_type"] = merged["error_type"].fillna("none")

    return merged[["error_type", "severity", "weight"]], source


# ============================================================
# JUDGE PANEL AND AGREEMENT
# ============================================================

def available_judges(names=None) -> list[str]:
    names = names or sorted(p.name for p in JUDGES_DIR.glob("*") if p.is_dir())
    return [n for n in names if (JUDGES_DIR / n / "labels.csv").exists()]


def judge_severity_table(claims: pd.DataFrame, judges=None) -> pd.DataFrame:
    """claims.index x judge -> severity (S0..S3), NaN where a judge has no label."""

    keys = ["model", "condition", "ticker", "claim_index"]
    table = pd.DataFrame(index=claims.index)

    for name in available_judges(judges):
        labels = pd.read_csv(JUDGES_DIR / name / "labels.csv")[keys + ["severity", "error_type"]]
        merged = claims[keys].merge(labels, on=keys, how="left")
        merged.index = claims.index
        table[name] = merged["severity"]
        table[f"{name}__type"] = merged["error_type"]

    return table


def panel_labels(claims: pd.DataFrame, judges=PANEL) -> pd.DataFrame:
    """
    Consensus = median severity over the panel (ordinal S0<S1<S2<S3). With
    three judges a claim is severe iff at least two call it S2+; with two
    judges the lower value is taken, i.e. both must agree.
    """

    names = available_judges(judges)
    table = judge_severity_table(claims, names)
    ranks = table[names].apply(lambda col: col.map({s: i for i, s in enumerate(SEVERITIES)}))
    ordered = np.sort(ranks.to_numpy(dtype=float), axis=1)       # NaN sorts last
    n_valid = ranks.notna().sum(axis=1).to_numpy()
    lower_median = np.array([
        row[(n - 1) // 2] if n else np.nan for row, n in zip(ordered, n_valid)
    ])

    severity = pd.Series(
        [SEVERITIES[int(r)] if not np.isnan(r) else "S0" for r in lower_median],
        index=claims.index,
    )

    # Error type: most common type among the judges that agree with the
    # consensus severity class (error vs. no error).
    types = table[[f"{n}__type" for n in names]]
    def consensus_type(i):
        if severity[i] == "S0":
            return "none"
        votes = [t for t in types.loc[i] if isinstance(t, str) and t != "none"]
        return pd.Series(votes).mode().iloc[0] if votes else "unsupported"

    labels = pd.DataFrame({
        "severity": severity,
        "error_type": [consensus_type(i) for i in claims.index],
        "weight": np.where(n_valid > 0, 1.0, 0.0),
    }, index=claims.index)
    labels["n_judges"] = n_valid
    return labels


def cohen_kappa(a, b, weights: str | None = None) -> float:
    """
    Cohen's kappa for two raters. Binary when a/b are booleans; ordinal
    (weights="linear" or "quadratic") when they are S0..S3 labels.
    """

    a, b = pd.Series(a).reset_index(drop=True), pd.Series(b).reset_index(drop=True)
    keep = a.notna() & b.notna()
    a, b = a[keep], b[keep]

    if weights is None:
        a, b = a.astype(bool), b.astype(bool)
        po = (a == b).mean()
        pe_ = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
        return float((po - pe_) / (1 - pe_)) if pe_ < 1 else float("nan")

    k = len(SEVERITIES)
    ia = a.map({s: i for i, s in enumerate(SEVERITIES)}).to_numpy()
    ib = b.map({s: i for i, s in enumerate(SEVERITIES)}).to_numpy()
    observed = np.zeros((k, k))
    np.add.at(observed, (ia, ib), 1)
    observed /= observed.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    grid = np.abs(np.subtract.outer(np.arange(k), np.arange(k))) / (k - 1)
    w = grid if weights == "linear" else grid ** 2
    return float(1 - (w * observed).sum() / (w * expected).sum())


def fleiss_kappa(ratings: pd.DataFrame) -> float:
    """Fleiss' kappa for boolean ratings (rows = items, columns = raters)."""

    r = ratings.dropna().astype(bool)
    n = r.shape[1]
    positive = r.sum(axis=1).to_numpy()
    counts = np.stack([n - positive, positive], axis=1)
    p_item = ((counts ** 2).sum(axis=1) - n) / (n * (n - 1))
    p_cat = counts.sum(axis=0) / counts.sum()
    pe_ = (p_cat ** 2).sum()
    return float((p_item.mean() - pe_) / (1 - pe_))


def kappa_ci(a, b, n_boot: int = 1000, seed: int = 2026, weights=None) -> tuple[float, float]:
    """Percentile CI for Cohen's kappa by resampling items."""

    a, b = pd.Series(a).reset_index(drop=True), pd.Series(b).reset_index(drop=True)
    keep = (a.notna() & b.notna()).to_numpy()
    a, b = a[keep].reset_index(drop=True), b[keep].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    values = [
        cohen_kappa(a.iloc[idx], b.iloc[idx], weights)
        for idx in (rng.integers(0, len(a), len(a)) for _ in range(n_boot))
    ]
    lo, hi = np.nanpercentile(values, [2.5, 97.5])
    return float(lo), float(hi)


# ============================================================
# METRICS
# ============================================================

def scenario_frame(
    claims: pd.DataFrame,
    labels: pd.DataFrame,
    memos: pd.DataFrame,
    scenario: str,
    source_condition: str = "C1",
    enabled=pe.POLICY_IDS,
    p1_threshold: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-claim release flags for one scenario, plus its memos."""

    generation = scenario if scenario in ("C0", "C1") else source_condition
    mask = claims["condition"] == generation
    frame = claims.loc[mask].join(labels.loc[mask])

    # Material errors (S2/S3) define "wrong": S1 imprecision is not worth
    # blocking, so withholding an S1 claim counts as a false block.
    frame["severe"] = frame["severity"].isin(SEVERE)
    frame["is_error"] = frame["severe"]
    frame["sev_weight"] = frame["severity"].map(SEVERITY_WEIGHT)

    if scenario in ("C0", "C1"):
        frame["action"] = "PASS"
        frame["released"] = True
    else:
        frame["action"] = decide(frame, enabled, p1_threshold)
        frame["released"] = frame["action"] == "PASS"
        if scenario == "C3":
            frame["released"] |= (frame["action"] == "ESCALATE") & ~frame["is_error"]

    memo_frame = memos.loc[memos["condition"] == generation].copy()
    return frame, memo_frame


def per_ticker_sums(frame: pd.DataFrame, memo_frame: pd.DataFrame) -> pd.DataFrame:
    """Numerators/denominators per (model, ticker) for fast cluster bootstrap."""

    w = frame["weight"]
    rel = frame["released"]
    parts = pd.DataFrame({
        "model": frame["model"],
        "ticker": frame["ticker"],
        "claims": w,
        "released": w * rel,
        "severe_all": w * frame["severe"],
        "severe_rel": w * (frame["severe"] & rel),
        "sevw_all": w * frame["sev_weight"],
        "sevw_rel": w * frame["sev_weight"] * rel,
        "unsup_rel": w * ((frame["error_type"] == "unsupported") & rel),
        "numeric_rel": w * (frame["numeric"] & rel),
        "numerr_rel": w * ((frame["error_type"] == "numeric") & rel),
        "cited_rel": w * (frame["P1_cited"] & ~frame["P1_fabricated"] & rel),
        "db_rel": w * ((frame["error_type"] == "decision_boundary") & rel),
        "correct_all": w * ~frame["is_error"],
        "correct_withheld": w * (~frame["is_error"] & ~rel),
        "withheld": w * (frame["action"] != "PASS"),
        "escalated": w * (frame["action"] == "ESCALATE"),
    })
    sums = parts.groupby(["model", "ticker"]).sum()

    memo_sums = memo_frame.groupby(["model", "ticker"]).agg(
        memos=("ticker", "size"),
        engine_ms=("engine_ms", "sum"),
        gen_latency_s=("gen_latency_s", "sum"),
    )

    return sums.join(memo_sums, how="outer").fillna(0.0)


def ratios(s: pd.DataFrame | pd.Series) -> dict[str, Any]:
    """Proposal metrics from summed numerators (works on rows or bootstrap arrays)."""

    def div(a, b):
        return np.divide(a, b, out=np.full_like(np.asarray(a, dtype=float), np.nan),
                         where=np.asarray(b) > 0)

    return {
        "severe_per_memo": div(s["severe_rel"], s["memos"]),
        "severe_rate_released": div(s["severe_rel"], s["released"]),
        "sepr": div(s["severe_rel"], s["severe_all"]),
        "rwl": div(s["sevw_rel"], s["sevw_all"]),
        "unsupported_rate": div(s["unsup_rel"], s["released"]),
        "numeric_error_rate": div(s["numerr_rel"], s["numeric_rel"]),
        "evidence_coverage": div(s["cited_rel"], s["released"]),
        "db_violation_per_memo": div(s["db_rel"], s["memos"]),
        "false_block_rate": div(s["correct_withheld"], s["correct_all"]),
        "withheld_share": div(s["withheld"], s["claims"]),
        "escalation_rate": div(s["escalated"], s["claims"]),
        "claims_released_per_memo": div(s["released"], s["memos"]),
        "engine_ms_per_memo": div(s["engine_ms"], s["memos"]),
        "gen_s_per_memo": div(s["gen_latency_s"], s["memos"]),
    }


def bootstrap(
    sums_by_scenario: dict[str, pd.DataFrame],
    n_boot: int = 2000,
    seed: int = 2026,
) -> pd.DataFrame:
    """
    Company-level cluster bootstrap. All scenarios share the same resampled
    companies, so differences between scenarios are paired.
    Returns long frame: scenario, model, metric, point, lo, hi.
    """

    rng = np.random.default_rng(seed)
    rows = []

    tickers = sorted(set().union(*[
        set(df.index.get_level_values("ticker")) for df in sums_by_scenario.values()
    ]))
    counts = rng.multinomial(len(tickers), [1 / len(tickers)] * len(tickers), size=n_boot)
    count_frame = pd.DataFrame(counts.T, index=tickers)

    for scenario, sums in sums_by_scenario.items():
        models = sorted(set(sums.index.get_level_values("model")))
        for model in [*models, "all"]:
            block = sums if model == "all" else sums.xs(model, level="model", drop_level=False)
            per_ticker = block.groupby(level="ticker").sum()
            weights = count_frame.reindex(per_ticker.index).fillna(0).to_numpy()
            boot = {col: weights.T @ per_ticker[col].to_numpy() for col in per_ticker}
            point = ratios(per_ticker.sum())
            spread = ratios(boot)

            for metric, value in point.items():
                lo, hi = np.nanpercentile(spread[metric], [2.5, 97.5])
                rows.append({
                    "scenario": scenario, "model": model, "metric": metric,
                    "point": float(value), "lo": lo, "hi": hi,
                    "boot": spread[metric],
                })

    return pd.DataFrame(rows)


def paired_difference(boot: pd.DataFrame, metric: str, a: str, b: str, model: str = "all"):
    """Point and 95% CI of metric(a) - metric(b) using the shared resamples."""

    def pick(scenario):
        mask = (
            (boot["scenario"] == scenario)
            & (boot["model"] == model)
            & (boot["metric"] == metric)
        )
        return boot[mask].iloc[0]

    ra, rb = pick(a), pick(b)
    diff = ra["boot"] - rb["boot"]
    lo, hi = np.nanpercentile(diff, [2.5, 97.5])
    return ra["point"] - rb["point"], lo, hi


# ============================================================
# ABLATIONS AND SWEEPS
# ============================================================

def policy_subsets():
    for size in range(len(pe.POLICY_IDS) + 1):
        yield from itertools.combinations(pe.POLICY_IDS, size)


def subset_label(subset) -> str:
    return "+".join(SHORT[p] for p in subset) if subset else "none"


# ============================================================
# AUDIT REPLAY (RQ4)
# ============================================================

def replay_audit(model: str, policy_set: dict[str, Any], condition: str = "C2") -> dict[str, Any]:
    """
    Re-executes the logged policy version on the logged inputs and checks
    that every per-policy verdict and claim decision is reproduced.
    """

    audit_path = GENERATED_DIR / condition / model / "policy_audit.jsonl"
    evidence_text = pe.load_evidence_text()
    company_tokens = pe.load_company_tokens()
    cache: dict[tuple, Any] = {}
    total = matched = complete = 0
    required = {"policy_id", "policy_version", "policy_set_sha256", "claim_id",
                "evidence_ids", "passed", "claim_decision", "timestamp"}

    for line in audit_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        total += 1
        complete += required.issubset(k for k, v in record.items() if v is not None)

        source = GENERATED_DIR / record["source_condition"] / model / f"{record['ticker']}_FY{pe.FY}.json"
        key = (record["ticker"], record["claim_index"])

        if key not in cache:
            doc = json.loads(source.read_text(encoding="utf-8"))
            claim = doc["result"]["claims"][record["claim_index"]]
            checks = pe.check_claim(
                claim, evidence_text, set(doc["evidence_ids_available"]),
                policy_set, company_tokens.get(record["ticker"], set()),
            )
            cache[key] = (checks, pe.aggregate(checks))

        checks, decision = cache[key]
        matched += (
            checks[record["policy_id"]]["passed"] == record["passed"]
            and decision == record["claim_decision"]
            and record["policy_set_sha256"] == policy_set["sha256"]
        )

    return {"records": total, "replayed_identically": matched, "complete": complete}


# ============================================================
# ANNOTATION SAMPLE
# ============================================================

def export_annotation_sample(
    claims: pd.DataFrame,
    n_total: int = 240,
    seed: int = 2026,
    path: Path = ANNOTATION_DIR / "annotation_sample_v1.csv",
) -> pd.DataFrame:
    """
    Stratified sample (model x condition x full-policy decision) with inverse
    sampling weights, for human labels and for validating an LLM judge.
    """

    evidence_text = pe.load_evidence_text()
    frame = claims.copy()
    frame["auto_decision"] = decide(frame)
    frame["stratum"] = frame["model"] + "|" + frame["condition"] + "|" + frame["auto_decision"]

    sizes = frame["stratum"].value_counts()
    allocation = np.maximum(5, np.round(n_total * sizes / sizes.sum())).astype(int)

    parts = []
    for stratum, n in allocation.items():
        members = frame[frame["stratum"] == stratum]
        take = members.sample(min(n, len(members)), random_state=seed)
        take = take.assign(weight=len(members) / len(take))
        parts.append(take)

    sample = pd.concat(parts).sample(frac=1, random_state=seed)
    sample["cited_evidence"] = sample["evidence_ids"].map(
        lambda ids: "\n---\n".join(f"[{e}] {evidence_text.get(e, '(not in corpus)')}" for e in ids)
    )

    out = sample[["model", "condition", "ticker", "claim_index", "stratum", "weight",
                  "topic", "text", "cited_evidence"]].copy()
    for column in ("supported", "numeric_error", "recommendation", "error_type", "severity", "notes"):
        out[column] = ""

    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out
