import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CHUNKS_DIR = DATA_DIR / "chunks"
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 1200
OVERLAP = 150

# Matches the legacy C0 context design: Risk Factors + MD&A.
CHUNK_SECTIONS = (
    "1A_risk_factors",
    "7_mdna",
)


def read_manifest() -> list[dict]:
    with (DATA_DIR / "filing_manifest.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as f:
        return list(csv.DictReader(f))


def hard_split(text: str, size: int, overlap: int) -> list[tuple[str, int, int]]:
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + size, len(text))

        # Prefer ending on whitespace when possible.
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start + int(size * 0.60):
                end = boundary

        chunk = text[start:end].strip()
        if chunk:
            actual_start = start
            actual_end = end
            chunks.append((chunk, actual_start, actual_end))

        if end >= len(text):
            break

        next_start = max(end - overlap, start + 1)
        start = next_start

    return chunks


def paragraph_chunks(text: str) -> list[tuple[str, int, int]]:
    """Pack paragraphs into 1,200-char chunks; hard-split oversized paragraphs."""
    paragraphs = []
    cursor = 0

    for part in text.split("\n\n"):
        clean = part.strip()
        if not clean:
            cursor += len(part) + 2
            continue

        idx = text.find(clean, cursor)
        if idx == -1:
            idx = cursor
        end = idx + len(clean)
        paragraphs.append((clean, idx, end))
        cursor = end

    chunks = []
    current_text = ""
    current_start = None
    current_end = None

    for paragraph, start, end in paragraphs:
        if len(paragraph) > CHUNK_SIZE:
            if current_text:
                chunks.append((current_text.strip(), current_start, current_end))
                current_text = ""
                current_start = None

            chunks.extend(hard_split(paragraph, CHUNK_SIZE, OVERLAP))
            current_end = end
            continue

        candidate = paragraph if not current_text else current_text + "\n\n" + paragraph

        if len(candidate) <= CHUNK_SIZE:
            if current_start is None:
                current_start = start
            current_end = end
            current_text = candidate
        else:
            chunks.append((current_text.strip(), current_start, current_end))
            current_text = paragraph
            current_start = start
            current_end = end

    if current_text:
        chunks.append((current_text.strip(), current_start, current_end))

    return chunks


def process_one(row: dict) -> list[dict]:
    ticker = row["ticker"].upper()
    fy = int(row["fy"])
    accession = row["accession"]
    source_file = row.get("local_file") or row.get("local_primary_document", "")
    source_url = row.get("document_url") or row.get("filing_url", "")

    path = PROCESSED_DIR / f"{ticker}_FY{fy}.json"
    if not path.exists():
        raise FileNotFoundError(path)

    doc = json.loads(path.read_text(encoding="utf-8"))
    records = []
    global_index = 0

    for section in CHUNK_SECTIONS:
        section_text = doc["sections"].get(section, "")
        if not section_text:
            continue

        parts = paragraph_chunks(section_text)

        for section_index, (chunk_text, start, end) in enumerate(parts):
            evidence_id = f"{ticker}_FY{fy}_{section}_{section_index:03d}"

            records.append({
                "evidence_id": evidence_id,
                "ticker": ticker,
                "company": row.get("company", ""),
                "sector": row.get("sector", ""),
                "fy": fy,
                "accession": accession,
                "filing_date": row.get("filing_date", ""),
                "report_date": row.get("report_date", ""),
                "section": section,
                "chunk_index": global_index,
                "section_chunk_index": section_index,
                "char_start": start,
                "char_end": end,
                "text": chunk_text,
                "source_file": source_file,
                "source_url": source_url,
            })
            global_index += 1

    if not records:
        raise ValueError("No chunks produced from extracted 10-K sections")

    out_path = CHUNKS_DIR / f"{ticker}_FY{fy}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return records


def main() -> None:
    all_rows = read_manifest()
    rows = [row for row in all_rows if row.get("status") == "ok"]
    skipped = len(all_rows) - len(rows)
    all_records = []
    summary = []
    failures = []

    for row in rows:
        try:
            records = process_one(row)
            all_records.extend(records)
            summary.append({
                "ticker": row["ticker"],
                "fy": int(row["fy"]),
                "n_chunks": len(records),
            })
            print(
                f"{row['ticker']} FY{row['fy']}: "
                f"{len(records)} chunks"
            )
        except Exception as exc:
            print(f"ERROR {row.get('ticker')} FY{row.get('fy')}: {exc}")
            output_path = CHUNKS_DIR / f"{row.get('ticker', '').upper()}_FY{row.get('fy')}.jsonl"
            if output_path.exists():
                output_path.unlink()
            failures.append({
                "ticker": row.get("ticker"),
                "fy": row.get("fy"),
                "error": str(exc),
            })

    combined = CHUNKS_DIR / "all_chunks.jsonl"
    with combined.open("w", encoding="utf-8") as f:
        for item in all_records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    (CHUNKS_DIR / "_resumo.json").write_text(
        json.dumps({
            "chunk_size": CHUNK_SIZE,
            "overlap": OVERLAP,
            "sections": list(CHUNK_SECTIONS),
            "skipped_non_ok_filings": skipped,
            "documents": summary,
            "failures": failures,
            "total_chunks": len(all_records),
        }, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\nCHUNKS COMPLETE")
    print(f"Documents: {len(summary)}/{len(rows)}")
    print(f"Skipped non-ok filings: {skipped}")
    print(f"Total chunks: {len(all_records)}")
    print(f"Failures: {len(failures)}")
    print(f"Combined file: {combined}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
