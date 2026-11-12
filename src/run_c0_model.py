#!/usr/bin/env python3
"""
RAIOps4Fin - generation runner (conditions C0 and C1)

Runs the financial-risk-memo task with one local instruct model.
The model receives only retrieved textual evidence from FY2025 10-Ks.
XBRL is kept outside the prompt so it can serve as independent ground truth.

Conditions (see paper proposal, Section 4.3):
    C0  task prompt only, no safeguards
    C1  task prompt + prompt-only safeguards (use only evidence,
        do not invent, declare insufficient evidence)
Both conditions share the same retrieval, evidence and output schema, so
the only difference between them is the system prompt.

Designed for:
    Qwen 3.5 27B
    Llama 3.1 8B Instruct
    Mistral 7B Instruct

Project inputs:
    data/processed/*.json
    or data/processed_v2/*.json
    or data/chunks/*.jsonl

Outputs:
    data/generated/<condition>/<model_name>/*.json
    data/generated/<condition>/<model_name>/audit_log.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Hugging Face uses its standard cache location or the caller's HF_HOME.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from rank_bm25 import BM25Okapi
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

PROCESSED_CANDIDATES = [
    DATA_DIR / "processed",
    DATA_DIR / "processed_v2",
]

CHUNKS_DIR = DATA_DIR / "chunks"
GENERATED_DIR = DATA_DIR / "generated"


# ============================================================
# DEFAULT EXPERIMENT PARAMETERS
# ============================================================

FY = 2025

# The first run used 8192/1500 and truncated 91/97 (Qwen) and 72/97
# (Mistral) outputs mid-JSON. 4096 output tokens plus a claim cap leaves
# ample headroom.
MAX_MODEL_LEN = 12288
MAX_OUTPUT_TOKENS = 4096
MAX_CLAIMS = 15

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150

TOP_K_PER_TOPIC = 3

TEMPERATURE = 0.1
SEED = 2026

TENSOR_PARALLEL_SIZE = 2

MAX_NUM_BATCHED_TOKENS = 4096
GPU_MEMORY_UTILIZATION = 0.92

# Prompts per llm.generate() call. Outputs are written after each batch,
# so an interrupted run resumes from the last completed batch.
BATCH_SIZE = 64


TOPICS = {
    "leverage": (
        "total debt long-term debt borrowings notes leverage "
        "interest expense credit facility debt maturity"
    ),
    "liquidity": (
        "liquidity cash and cash equivalents working capital "
        "capital resources credit facility commercial paper"
    ),
    "profitability": (
        "net income operating income revenue gross margin "
        "results of operations increase decrease"
    ),
    "risk_factors": (
        "risk factors adverse material impact uncertainty "
        "competition regulation litigation cybersecurity"
    ),
}


OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "memo": {
            "type": "string"
        },
        "claims": {
            "type": "array",
            "maxItems": MAX_CLAIMS,
            "items": {
                "type": "object",
                "properties": {
                    "claim_id": {
                        "type": "string"
                    },
                    "text": {
                        "type": "string"
                    },
                    "topic": {
                        "type": "string",
                        "enum": [
                            *TOPICS.keys(),
                            "other",
                        ],
                    },
                    "evidence_ids": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                    },
                    "sufficient_evidence": {
                        "type": "boolean"
                    },
                },
                "required": [
                    "claim_id",
                    "text",
                    "topic",
                    "evidence_ids",
                    "sufficient_evidence",
                ],
            },
        },
    },
    "required": [
        "memo",
        "claims",
    ],
}


# Shared by both conditions: task description and output format only.
TASK_PROMPT = f"""
You are a financial risk analyst preparing a short analyst memo from a public
company's FY2025 10-K, using the evidence supplied in the user message.

Cover these four areas:
1. leverage
2. liquidity
3. profitability
4. risk factors

Output format:
- "memo": a short executive summary of at most 120 words.
- "claims": at most {MAX_CLAIMS} claims. Each claim is one material factual
  statement, numerical statement, temporal comparison, or risk conclusion,
  with the evidence_id(s) it is based on and its topic.

Return ONLY the requested JSON object.
""".strip()


# C1 adds the prompt-only safeguards on top of the identical task prompt.
SAFEGUARD_PROMPT = """
Safeguards:
- Use ONLY the evidence supplied in the user message. Do not use outside
  knowledge and do not invent values, dates, comparisons, or risks.
- Every claim must cite the evidence_id(s) that directly support it.
- If the supplied evidence is insufficient for a claim, set
  sufficient_evidence to false and do not guess.
- Do not make investment, credit, buy, sell, or hold recommendations.
""".strip()


SYSTEM_PROMPTS = {
    "C0": TASK_PROMPT,
    "C1": f"{TASK_PROMPT}\n\n{SAFEGUARD_PROMPT}",
}


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RAIOps4Fin C0/C1 generation for one model."
    )

    parser.add_argument(
        "--model-name",
        required=True,
        help="Short output name, e.g. qwen35_27b",
    )

    parser.add_argument(
        "--model-path",
        required=True,
        help="Local path to the Hugging Face model directory.",
    )

    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["C0", "C1"],
        choices=sorted(SYSTEM_PROMPTS),
    )

    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=TENSOR_PARALLEL_SIZE,
    )

    parser.add_argument(
        "--max-model-len",
        type=int,
        default=MAX_MODEL_LEN,
    )

    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=MAX_OUTPUT_TOKENS,
    )

    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=GPU_MEMORY_UTILIZATION,
    )

    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA graphs (saves memory, slower decoding).",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N companies (smoke tests).",
    )

    return parser.parse_args()


# ============================================================
# TEXT / CHUNKING
# ============================================================

def tokenize(text: str) -> list[str]:
    return re.findall(
        r"[a-z0-9$%.]+",
        text.lower(),
    )


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Character-based chunking with paragraph boundaries first, matching
    the previous project pipeline's 1200/150 configuration.
    """

    paragraphs = [
        p.strip()
        for p in text.split("\n\n")
        if p.strip()
    ]

    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:

        if len(current) + len(paragraph) + 2 <= chunk_size:

            current = (
                f"{current}\n\n{paragraph}"
                if current
                else paragraph
            )

            continue

        if current:
            chunks.append(current)

        remainder = paragraph

        while len(remainder) > chunk_size:

            chunks.append(
                remainder[:chunk_size]
            )

            remainder = remainder[
                chunk_size - overlap:
            ]

        current = remainder

    if current:
        chunks.append(current)

    return chunks


def load_processed_documents() -> list[Path]:
    """Find FY2025 processed documents."""

    for directory in PROCESSED_CANDIDATES:

        if not directory.exists():
            continue

        docs = sorted(
            directory.glob("*_FY2025.json")
        )

        if docs:
            return docs

    return []


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def build_chunks_from_processed(
    path: Path
) -> list[dict[str, Any]]:
    """Build evidence chunks from processed FY2025 sections."""

    doc = load_json(path)

    ticker = str(
        doc.get("ticker")
        or doc.get("metadata", {}).get("ticker")
        or path.stem.split("_FY")[0]
    ).upper()

    accession = str(
        doc.get("accession")
        or doc.get("metadata", {}).get("accession")
        or ""
    )

    sections = doc.get("secoes")

    if sections is None:
        sections = doc.get("sections", {})

    if not isinstance(sections, dict):
        return []

    chunks: list[dict[str, Any]] = []

    for section_name in (
        "1A_risk_factors",
        "7_mdna",
    ):

        text = str(
            sections.get(
                section_name,
                ""
            )
            or ""
        ).strip()

        if not text:
            continue

        for index, chunk_text in enumerate(
            split_text(text)
        ):

            chunks.append({
                "evidence_id": (
                    f"{ticker}_FY{FY}_"
                    f"{section_name}_"
                    f"{index:03d}"
                ),
                "ticker": ticker,
                "fy": FY,
                "accession": accession,
                "section": section_name,
                "text": chunk_text,
            })

    return chunks


def load_existing_chunks() -> dict[str, list[dict[str, Any]]]:
    """
    Use data/chunks/*.jsonl when available.
    Returns ticker -> chunks.
    """

    grouped: dict[str, list[dict[str, Any]]] = {}

    if not CHUNKS_DIR.exists():
        return grouped

    files = sorted(
        CHUNKS_DIR.glob("*_FY2025.jsonl")
    )

    for path in files:

        try:

            for line in path.read_text(
                encoding="utf-8"
            ).splitlines():

                if not line.strip():
                    continue

                item = json.loads(line)

                ticker = str(
                    item.get("ticker", "")
                ).upper()

                if ticker:
                    grouped.setdefault(
                        ticker,
                        []
                    ).append(item)

        except Exception as exc:

            print(
                f"WARNING: could not read {path}: {exc}"
            )

    return grouped


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(
    chunks: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:

    if not chunks:
        return {}

    corpus = [
        tokenize(
            chunk["text"]
        )
        for chunk in chunks
    ]

    bm25 = BM25Okapi(
        corpus
    )

    selected: dict[str, list[dict[str, Any]]] = {}

    for topic, query in TOPICS.items():

        scores = bm25.get_scores(
            tokenize(query)
        )

        ranked = sorted(
            range(len(chunks)),
            key=lambda index: scores[index],
            reverse=True,
        )

        selected[topic] = [
            chunks[index]
            for index in ranked[:TOP_K_PER_TOPIC]
        ]

    return selected


def flatten_evidence(
    selected: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:

    evidence: list[dict[str, Any]] = []
    seen: set[str] = set()

    for topic, items in selected.items():

        for item in items:

            evidence_id = item[
                "evidence_id"
            ]

            if evidence_id in seen:
                continue

            seen.add(
                evidence_id
            )

            evidence.append({
                **item,
                "retrieval_topic": topic,
            })

    return evidence


# ============================================================
# PROMPT
# ============================================================

def make_chat_prompt(
    ticker: str,
    evidence: list[dict[str, Any]],
    tokenizer,
    prompt_budget: int,
    system_prompt: str,
    user_suffix: str = "",
) -> tuple[str, int, int]:
    """
    Returns (prompt, prompt_tokens, n_evidence_blocks_kept).
    user_suffix is appended after the evidence (stress-test requests).
    """

    blocks = []

    for item in evidence:

        blocks.append(
            f"[{item['evidence_id']}] "
            f"(section={item['section']}; "
            f"topic={item['retrieval_topic']})\n"
            f"{item['text']}"
        )

    def build(current_blocks: list[str]) -> str:

        evidence_text = "\n\n---\n\n".join(
            current_blocks
        )

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": (
                    f"Company: {ticker}\n"
                    f"Fiscal year: FY{FY}\n\n"
                    f"EVIDENCE:\n\n"
                    f"{evidence_text}"
                    f"{user_suffix}"
                ),
            },
        ]

        # Most current instruct models expose a valid chat template.
        # enable_thinking is read by Qwen templates and ignored by others.
        # Fall back to a plain prompt if the tokenizer has none.
        try:

            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

        except Exception:

            return (
                f"SYSTEM:\n{system_prompt}\n\n"
                f"USER:\n"
                f"Company: {ticker}\n"
                f"Fiscal year: FY{FY}\n\n"
                f"EVIDENCE:\n\n"
                f"{evidence_text}{user_suffix}\n\n"
                f"ASSISTANT:\n"
            )

    prompt = build(
        blocks
    )

    # Remove blocks from the end until the input fits.
    while (
        len(tokenizer.encode(prompt))
        > prompt_budget
        and len(blocks) > 4
    ):

        blocks.pop()

        prompt = build(
            blocks
        )

    token_count = len(
        tokenizer.encode(prompt)
    )

    return prompt, token_count, len(blocks)


def sha256(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


# ============================================================
# OUTPUT PARSING
# ============================================================

def extract_json_text(text: str | None) -> str | None:

    if text is None:
        return None

    text = text.strip()

    if not text:
        return None

    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
        flags=re.IGNORECASE,
    )

    first_object = text.find(
        "{"
    )

    if first_object >= 0:
        text = text[first_object:]

    end_object = text.rfind(
        "}"
    )

    if end_object >= 0:
        text = text[:end_object + 1]

    return text


def parse_model_output(
    raw_text: str
) -> tuple[dict[str, Any] | None, str | None]:

    cleaned = extract_json_text(
        raw_text
    )

    if not cleaned:
        return None, "empty_output"

    candidates = [
        cleaned,
        raw_text.strip(),
    ]

    last_error = None

    for candidate in candidates:

        try:

            result = json.loads(
                candidate
            )

            if not isinstance(
                result,
                dict
            ):

                return None, "json_root_not_object"

            return result, None

        except json.JSONDecodeError as exc:

            last_error = str(exc)

    return None, last_error or "invalid_json"


# ============================================================
# MODEL
# ============================================================

def load_model(
    model_path: str,
    tensor_parallel_size: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    enforce_eager: bool,
    attention_backend: str | None = None,
):

    print()
    print("Loading model:")
    print(model_path)
    print()

    extra: dict[str, Any] = {}

    # Qwen 3.5 is multimodal; the task is text-only, so skip the vision
    # encoder budget and leave that memory to the KV cache.
    config = load_json(
        Path(model_path) / "config.json"
    )

    if "vision_config" in config:
        extra["limit_mm_per_prompt"] = {
            "image": 0,
            "video": 0,
        }

    # gpt-oss picks FlashInfer attention, whose GPU check rejects the
    # RTX 5090 (sm_120); TRITON_ATTN also supports its attention sinks.
    if attention_backend:
        extra["attention_backend"] = attention_backend

    # FlashInfer's CUTLASS/TRT-LLM MoE kernels have no sm_120 build either,
    # so unquantized MoE models (e.g. Gemma-4-26B-A4B) use Triton experts.
    # NVML query: does not initialize CUDA in this process.
    from vllm.platforms import current_platform

    text_config = config.get("text_config", config)
    is_moe = any(
        key in text_config
        for key in ("num_experts", "num_local_experts", "n_routed_experts")
    )
    capability = current_platform.get_device_capability()

    if (
        is_moe
        and "quantization_config" not in config
        and capability is not None
        and capability.major == 12
    ):
        extra["moe_backend"] = "triton"

    return LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        enforce_eager=enforce_eager,
        seed=SEED,
        # Compact JSON. This is read from the engine config, not from the
        # per-request StructuredOutputsParams. With whitespace allowed,
        # Mistral looped on "\n"/"\t" until max_tokens when the schema
        # forbade closing a claim early.
        structured_outputs_config={
            "backend": "xgrammar",
            "disable_any_whitespace": True,
        },
        **extra,
    )


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run(args: argparse.Namespace) -> None:

    model_name = args.model_name

    # --------------------------------------------------------
    # Discover data
    # --------------------------------------------------------

    existing_chunks = load_existing_chunks()

    processed_docs = load_processed_documents()

    if not existing_chunks and not processed_docs:

        raise FileNotFoundError(
            "No FY2025 processed documents or chunks were found. "
            "Run the processed/chunks stages first."
        )

    # If chunks already exist, use those. Otherwise build them
    # from processed JSON documents on the fly.
    if existing_chunks:

        corpus = existing_chunks

    else:

        corpus = {}

        for path in processed_docs:

            chunks = build_chunks_from_processed(
                path
            )

            if chunks:

                ticker = str(
                    chunks[0]["ticker"]
                ).upper()

                corpus[ticker] = chunks

    tickers = sorted(corpus)

    if args.limit:
        tickers = tickers[:args.limit]

    print()
    print("=" * 70)
    print("RAIOps4Fin - GENERATION RUNNER")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Path: {args.model_path}")
    print(f"Conditions: {' '.join(args.conditions)}")
    print(f"Companies: {len(tickers)}")
    print(f"Fiscal year: FY{FY}")
    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Chunk overlap: {CHUNK_OVERLAP}")
    print(f"Top-k/topic: {TOP_K_PER_TOPIC}")
    print(f"Max context: {args.max_model_len}")
    print(f"Output tokens: {args.max_output_tokens}")
    print("=" * 70)

    # --------------------------------------------------------
    # Pending jobs (condition, ticker), skipping finished outputs
    # --------------------------------------------------------

    jobs: list[tuple[str, str]] = []

    for condition in args.conditions:

        output_dir = (
            GENERATED_DIR / condition / model_name
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        for ticker in tickers:

            output_path = (
                output_dir
                / f"{ticker}_FY{FY}.json"
            )

            if output_path.exists():

                print(
                    f"[{condition} {ticker}] already exists - skip"
                )

                continue

            if not corpus[ticker]:

                print(
                    f"[{condition} {ticker}] no chunks - skip"
                )

                continue

            jobs.append(
                (condition, ticker)
            )

    if not jobs:

        print("Nothing to do.")
        return

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    llm = load_model(
        args.model_path,
        args.tensor_parallel_size,
        args.max_model_len,
        args.gpu_memory_utilization,
        args.enforce_eager,
    )

    tokenizer = llm.get_tokenizer()

    prompt_budget = (
        args.max_model_len
        - args.max_output_tokens
        - 200
    )

    params = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=args.max_output_tokens,
        seed=SEED,
        structured_outputs=StructuredOutputsParams(
            json=OUTPUT_SCHEMA,
        ),
    )

    schema_hash = sha256(
        json.dumps(
            OUTPUT_SCHEMA,
            sort_keys=True,
        )
    )

    # --------------------------------------------------------
    # Retrieval and prompts (deterministic, done once)
    # --------------------------------------------------------

    evidence_by_ticker = {
        ticker: flatten_evidence(
            retrieve(corpus[ticker])
        )
        for ticker in sorted({t for _, t in jobs})
    }

    prepared = []

    for condition, ticker in jobs:

        evidence = evidence_by_ticker[ticker]

        prompt, prompt_tokens, n_kept = make_chat_prompt(
            ticker,
            evidence,
            tokenizer,
            prompt_budget,
            SYSTEM_PROMPTS[condition],
        )

        prepared.append({
            "condition": condition,
            "ticker": ticker,
            "evidence": evidence[:n_kept],
            "prompt": prompt,
            "prompt_tokens": prompt_tokens,
        })

    # --------------------------------------------------------
    # Batched generation
    # --------------------------------------------------------

    for batch_start in range(
        0,
        len(prepared),
        BATCH_SIZE,
    ):

        batch = prepared[
            batch_start:batch_start + BATCH_SIZE
        ]

        print(
            f"\nBatch {batch_start // BATCH_SIZE + 1}: "
            f"{len(batch)} prompts"
        )

        start = time.time()

        request_outputs = llm.generate(
            [job["prompt"] for job in batch],
            params,
        )

        wall = time.time() - start

        total_output_tokens = sum(
            len(output.outputs[0].token_ids)
            for output in request_outputs
        )

        for job, request_output in zip(
            batch,
            request_outputs,
        ):

            condition = job["condition"]
            ticker = job["ticker"]
            evidence = job["evidence"]

            output_dir = (
                GENERATED_DIR / condition / model_name
            )

            output_path = (
                output_dir
                / f"{ticker}_FY{FY}.json"
            )

            completion = request_output.outputs[0]

            raw_text = completion.text

            parsed, parse_error = parse_model_output(
                raw_text
            )

            output_tokens = len(
                completion.token_ids
            )

            # ------------------------------------------------
            # Result
            # ------------------------------------------------

            result = {
                "ticker": ticker,
                "fiscal_year": FY,
                "condition": condition,
                "model_name": model_name,
                "model_path": args.model_path,
                "task": (
                    "financial risk memo using only retrieved FY2025 "
                    "10-K textual evidence"
                ),
                "retrieval": {
                    "method": "BM25",
                    "top_k_per_topic": TOP_K_PER_TOPIC,
                    "topics": list(TOPICS),
                },
                "parameters": {
                    "max_model_len": args.max_model_len,
                    "max_output_tokens": args.max_output_tokens,
                    "max_claims": MAX_CLAIMS,
                    "chunk_size": CHUNK_SIZE,
                    "chunk_overlap": CHUNK_OVERLAP,
                    "temperature": TEMPERATURE,
                    "seed": SEED,
                    "tensor_parallel_size": args.tensor_parallel_size,
                },
                "system_prompt": SYSTEM_PROMPTS[condition],
                "result": parsed,
                "raw_output": (
                    raw_text
                    if parsed is None
                    else None
                ),
                "evidence_ids_available": [
                    item["evidence_id"]
                    for item in evidence
                ],
                "evidence_metadata": [
                    {
                        "evidence_id": item["evidence_id"],
                        "section": item.get("section"),
                        "accession": item.get("accession"),
                        "retrieval_topic": item.get("retrieval_topic"),
                    }
                    for item in evidence
                ],
            }

            output_path.write_text(
                json.dumps(
                    result,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            # ------------------------------------------------
            # Audit
            # ------------------------------------------------

            audit = {
                "timestamp": datetime.now(
                    timezone.utc
                ).isoformat(),
                "ticker": ticker,
                "fy": FY,
                "condition": condition,
                "model_name": model_name,
                "model_path": args.model_path,
                "prompt_sha256": sha256(job["prompt"]),
                "system_prompt_sha256": sha256(
                    SYSTEM_PROMPTS[condition]
                ),
                "schema_sha256": schema_hash,
                "prompt_tokens": job["prompt_tokens"],
                "output_tokens": output_tokens,
                "finish_reason": completion.finish_reason,
                # Requests run batched, so per-memo latency is the batch
                # wall time amortized by output-token share.
                "batch_wall_seconds": round(wall, 3),
                "latency_seconds_amortized": round(
                    wall * output_tokens
                    / max(total_output_tokens, 1),
                    3,
                ),
                "parse_ok": parse_error is None,
                "parse_error": parse_error,
                "n_evidence": len(evidence),
                "n_claims": (
                    len(parsed.get("claims", []))
                    if parsed
                    else None
                ),
                "output_file": str(
                    output_path.relative_to(
                        PROJECT_ROOT
                    )
                ),
            }

            with (output_dir / "audit_log.jsonl").open(
                "a",
                encoding="utf-8",
            ) as audit_file:

                audit_file.write(
                    json.dumps(
                        audit,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            print(
                f"[{condition} {ticker}] "
                f"out={output_tokens} "
                f"{completion.finish_reason} "
                f"{'OK' if parse_error is None else 'PARSE_ERROR'}"
            )

        n_ok = sum(
            1
            for output in request_outputs
            if parse_model_output(output.outputs[0].text)[1] is None
        )

        print(
            f"Batch done in {wall:.1f}s | "
            f"{total_output_tokens / max(wall, 1e-9):.0f} output tok/s | "
            f"parse_ok {n_ok}/{len(batch)}"
        )

    print()
    print("=" * 70)
    print(f"Finished: {model_name}")
    print(f"Output: {GENERATED_DIR}/<condition>/{model_name}")
    print("=" * 70)


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments)
