import csv
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FILING_MANIFEST = DATA_DIR / "filing_manifest.csv"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

MIN_SECTION_CHARACTERS = 500


# Only these sections enter the retrieval context by default.
# Item 8 is kept for audit/reference and later numeric validation.
SECTIONS = {
    "1A_risk_factors": (
        r"(?im)^\s*(?:(?:item\s*)?1a\b[\s.\-:–—]*risk\s*factors|risk\s*factors(?:[\s.\-:–—]*(?:item\s*)?1a\b)?)",
        r"(?im)^\s*(?:(?:item\s*)?1b\b|(?:item\s*)?1c\b|cybersecurity)"
    ),
    "7_mdna": (
        r"(?im)^\s*(?:(?:item\s*)?7\b[\s.\-:–—]*management|management[’']s\s+discussion)",
        r"(?im)^\s*(?:(?:item\s*)?7a\b|(?:item\s*)?8\b)"
    ),
    "8_financial_statements": (
        r"(?im)^\s*(?:(?:item\s*)?8\b[\s.\-:–—]*financial\s*statements|financial\s*statements\s+and\s+supplementary\s+data)",
        r"(?im)^\s*(?:item\s*)?9(?:a|b)?\b"
    ),
}


def read_manifest() -> list[dict]:
    with FILING_MANIFEST.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def html_content_to_text(raw: str) -> str:
    parser = "lxml"
    try:
        soup = BeautifulSoup(raw, parser)
    except Exception:
        soup = BeautifulSoup(raw, "html.parser")

    for tag in soup(["script", "style", "noscript", "svg", "iframe"]):
        tag.decompose()

    text = soup.get_text("\n")
    text = text.replace("\xa0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Normalize broken Item headings such as:
    # Item 7.\nManagement's Discussion...
    text = re.sub(
        r"(?im)^(\s*item\s*\d+[a-c]?\.?)[ \t]*\n+[ \t]*",
        r"\1 ",
        text
    )

    # Collapse horizontal whitespace but retain paragraph boundaries.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

    return text.strip()


def html_to_text(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    return html_content_to_text(raw)


def extract_submission_document(path: Path, accepted_types: set[str]) -> str:
    submission = path.read_text(encoding="utf-8", errors="ignore")

    for document in re.finditer(
        r"<DOCUMENT>(.*?)</DOCUMENT>",
        submission,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        block = document.group(1)
        form = re.search(r"(?im)^\s*<TYPE>\s*([^\s<]+)", block)
        if not form or form.group(1).upper() not in accepted_types:
            continue

        body = re.search(r"(?is)<TEXT>(.*?)</TEXT>", block)
        if body:
            return body.group(1)

    raise ValueError(f"No matching document found in SEC submission: {path}")


def extract_10k_document(path: Path) -> str:
    return extract_submission_document(path, {"10-K", "10-K/A"})


def extract_annual_report_document(path: Path) -> str:
    return extract_submission_document(path, {"EX-13", "EX-13.1"})


def extract_longest_section(text: str, start_pattern: str, end_pattern: str) -> str:
    """Select the longest candidate section to avoid using the table of contents."""
    best = ""

    starts = list(re.finditer(start_pattern, text))
    for start in starts:
        end = re.search(end_pattern, text[start.end():])
        if not end:
            continue

        section = text[start.start(): start.end() + end.start()]
        if len(section) > len(best):
            best = section

    return best.strip()


def process_one(row: dict) -> dict:
    ticker = row["ticker"].upper()
    fy = int(row["fy"])

    source_file = row.get("local_file") or row.get("local_primary_document", "")
    if not source_file:
        raise ValueError("Manifest row has no local_file")

    source = PROJECT_ROOT / source_file
    if not source.exists():
        raise FileNotFoundError(source)

    if source.suffix.lower() == ".txt":
        text = source.read_text(encoding="utf-8", errors="ignore")
    else:
        text = html_to_text(source)

    def extract_sections(document_text: str) -> dict[str, str]:
        extracted = {
            name: extract_longest_section(document_text, start_pattern, end_pattern)
            for name, (start_pattern, end_pattern) in SECTIONS.items()
        }
        return {
            name: value if len(value) >= MIN_SECTION_CHARACTERS else ""
            for name, value in extracted.items()
        }

    sections = extract_sections(text)
    submission_file = row.get("local_submission", "")
    submission = PROJECT_ROOT / submission_file if submission_file else None

    if not sections["1A_risk_factors"] and not sections["7_mdna"] and submission and submission.exists():
        try:
            text = html_content_to_text(extract_10k_document(submission))
            sections = extract_sections(text)
            source_file = submission_file
        except ValueError:
            pass

    if not sections["1A_risk_factors"] and not sections["7_mdna"] and submission and submission.exists():
        try:
            text = html_content_to_text(extract_annual_report_document(submission))
            sections = extract_sections(text)
            source_file = submission_file
        except ValueError:
            pass

    if not sections["1A_risk_factors"] and not sections["7_mdna"]:
        raise ValueError("No substantive Risk Factors or MD&A section was extracted")

    source_url = row.get("document_url") or row.get("filing_url", "")
    if submission_file and source_file == submission_file:
        source_url = row.get("submission_url", source_url)

    out = {
        "metadata": {
            "ticker": ticker,
            "company": row.get("company", ""),
            "sector": row.get("sector", ""),
            "fy": fy,
            "cik": row.get("cik", ""),
            "accession": row.get("accession", ""),
            "filing_date": row.get("filing_date", ""),
            "report_date": row.get("report_date", ""),
            "filing_url": source_url,
            "source_file": source_file,
        },
        "statistics": {
            "full_text_characters": len(text),
            "full_text_words": len(text.split()),
            "section_characters": {
                name: len(value) for name, value in sections.items()
            },
            "section_words": {
                name: len(value.split()) for name, value in sections.items()
            },
        },
        "sections": sections,
    }

    out_path = PROCESSED_DIR / f"{ticker}_FY{fy}.json"
    out_path.write_text(
        json.dumps(out, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    return out


def main() -> None:
    all_rows = read_manifest()
    rows = [row for row in all_rows if row.get("status") == "ok"]
    skipped = len(all_rows) - len(rows)
    successes = 0
    failures = []

    for row in rows:
        try:
            result = process_one(row)
            chars = result["statistics"]["section_characters"]
            print(
                f"{row['ticker']} FY{row['fy']} | "
                f"1A={chars['1A_risk_factors']:,} | "
                f"7={chars['7_mdna']:,} | "
                f"8={chars['8_financial_statements']:,}"
            )
            successes += 1
        except Exception as exc:
            print(f"ERROR {row.get('ticker')} FY{row.get('fy')}: {exc}")
            output_path = PROCESSED_DIR / f"{row.get('ticker', '').upper()}_FY{row.get('fy')}.json"
            if output_path.exists():
                output_path.unlink()
            failures.append({
                "ticker": row.get("ticker"),
                "fy": row.get("fy"),
                "error": str(exc),
            })

    summary = {
        "expected": len(all_rows),
        "processed": successes,
        "skipped": skipped,
        "failed": len(failures),
        "failures": failures,
    }

    (PROCESSED_DIR / "_resumo.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\nPROCESSED COMPLETE")
    print(f"Processed: {successes}/{len(rows)}")
    print(f"Skipped non-ok filings: {skipped}")
    print(f"Failed: {len(failures)}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
