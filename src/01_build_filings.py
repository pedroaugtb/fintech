from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

INPUT_MANIFEST = DATA_DIR / "dataset_manifest.csv"

OUTPUT_DIR = DATA_DIR / "filings" / "sec-edgar-filings"

OUTPUT_MANIFEST = DATA_DIR / "filing_manifest.csv"


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_FISCAL_YEARS = [2023, 2024, 2025]

REQUEST_DELAY = 0.25

TIMEOUT = 60

MAX_RETRIES = 4


# ============================================================
# SEC USER AGENT
# ============================================================

# Set this before running:
#
# export SEC_USER_AGENT="Your Name your.email@example.com"

SEC_USER_AGENT = os.environ.get(
    "SEC_USER_AGENT",
    "Research YOUR_EMAIL@example.com"
)


# ============================================================
# SEC URLS
# ============================================================

COMPANY_TICKERS_URL = (
    "https://www.sec.gov/files/company_tickers.json"
)

SUBMISSIONS_URL = (
    "https://data.sec.gov/submissions/CIK{}.json"
)

SUBMISSIONS_BASE_URL = (
    "https://data.sec.gov/submissions/"
)

ARCHIVES_BASE_URL = (
    "https://www.sec.gov/Archives/edgar/data"
)


# ============================================================
# DIRECTORIES
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# HTTP SESSION
# ============================================================

session = requests.Session()

session.headers.update({
    "User-Agent": SEC_USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
    "Accept": "*/*"
})


# ============================================================
# SEC REQUEST
# ============================================================

def sec_get(
    url: str,
    timeout: int = TIMEOUT
) -> requests.Response:
    """
    GET request with retries for temporary SEC/network errors.
    """

    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            response = session.get(
                url,
                timeout=timeout
            )

            # Temporary errors
            if response.status_code in {
                429,
                500,
                502,
                503,
                504
            }:

                wait = 2 ** (attempt - 1)

                print(
                    f"    HTTP {response.status_code}; "
                    f"retrying in {wait}s..."
                )

                time.sleep(
                    wait
                )

                continue

            response.raise_for_status()

            return response

        except requests.RequestException as exc:

            last_error = exc

            if attempt < MAX_RETRIES:

                wait = 2 ** (attempt - 1)

                print(
                    f"    Request error: {exc}"
                )

                print(
                    f"    Retrying in {wait}s..."
                )

                time.sleep(
                    wait
                )

            else:

                raise

    raise last_error


# ============================================================
# SAFE ARRAY ACCESS
# ============================================================

def safe_get(
    values,
    index,
    default=""
):
    """
    SEC submissions are columnar arrays. Some optional
    fields can be missing, so never assume equal lengths.
    """

    if not isinstance(
        values,
        list
    ):
        return default

    if index < 0:
        return default

    if index >= len(values):
        return default

    value = values[index]

    if value is None:
        return default

    return str(value)


# ============================================================
# LOAD SEC TICKER DATABASE
# ============================================================

def load_ticker_database():

    print(
        "\nLoading SEC ticker database..."
    )

    response = sec_get(
        COMPANY_TICKERS_URL
    )

    data = response.json()

    ticker_to_cik = {}

    for item in data.values():

        ticker = str(
            item.get(
                "ticker",
                ""
            )
        ).strip().upper()

        cik_raw = item.get(
            "cik_str"
        )

        if not ticker or cik_raw is None:
            continue

        cik = (
            str(cik_raw)
            .split(".")[0]
            .zfill(10)
        )

        ticker_to_cik[ticker] = cik

    print(
        f"Ticker records: {len(ticker_to_cik)}"
    )

    return ticker_to_cik


# ============================================================
# LOAD SUBMISSIONS
# ============================================================

def load_submissions(
    ticker,
    cik
):

    cik = str(cik).zfill(10)

    url = SUBMISSIONS_URL.format(
        cik
    )

    print(
        f"Requesting submissions for "
        f"{ticker}: {url}"
    )

    response = sec_get(
        url
    )

    return response.json()


# ============================================================
# CONVERT COLUMNAR SUBMISSIONS TO RECORDS
# ============================================================

def parse_submission_records(
    data
):
    """
    Handles recent SEC submissions.

    Converts:

        form[]
        filingDate[]
        reportDate[]
        accessionNumber[]
        ...

    into:

        [
            {
                ...
            },
            ...
        ]
    """

    filings = data.get(
        "filings",
        {}
    )

    recent = filings.get(
        "recent",
        {}
    )

    if not isinstance(
        recent,
        dict
    ):
        return []

    forms = recent.get(
        "form",
        []
    )

    records = []

    for i in range(
        len(forms)
    ):

        record = {

            "form": safe_get(
                recent.get(
                    "form",
                    []
                ),
                i
            ),

            "filing_date": safe_get(
                recent.get(
                    "filingDate",
                    []
                ),
                i
            ),

            "report_date": safe_get(
                recent.get(
                    "reportDate",
                    []
                ),
                i
            ),

            "accession": safe_get(
                recent.get(
                    "accessionNumber",
                    []
                ),
                i
            ),

            "primary_document": safe_get(
                recent.get(
                    "primaryDocument",
                    []
                ),
                i
            ),

            "fy": safe_get(
                recent.get(
                    "fy",
                    []
                ),
                i
            ),

            "fp": safe_get(
                recent.get(
                    "fp",
                    []
                ),
                i
            )
        }

        records.append(
            record
        )

    return records


# ============================================================
# LOAD ARCHIVED SUBMISSIONS
# ============================================================

def load_archived_records(
    submissions
):
    """
    SEC provides extra submission JSON files through:

        filings.files

    These contain older filing history.
    """

    filings = submissions.get(
        "filings",
        {}
    )

    files = filings.get(
        "files",
        []
    )

    if not isinstance(
        files,
        list
    ):
        return []

    all_records = []

    for file_info in files:

        filename = file_info.get(
            "name"
        )

        if not filename:
            continue

        url = (
            SUBMISSIONS_BASE_URL
            + filename
        )

        print(
            f"    Reading historical submissions: "
            f"{filename}"
        )

        try:

            response = sec_get(
                url
            )

            data = response.json()

            forms = data.get(
                "form",
                []
            )

            for i in range(
                len(forms)
            ):

                all_records.append({

                    "form": safe_get(
                        data.get(
                            "form",
                            []
                        ),
                        i
                    ),

                    "filing_date": safe_get(
                        data.get(
                            "filingDate",
                            []
                        ),
                        i
                    ),

                    "report_date": safe_get(
                        data.get(
                            "reportDate",
                            []
                        ),
                        i
                    ),

                    "accession": safe_get(
                        data.get(
                            "accessionNumber",
                            []
                        ),
                        i
                    ),

                    "primary_document": safe_get(
                        data.get(
                            "primaryDocument",
                            []
                        ),
                        i
                    ),

                    "fy": safe_get(
                        data.get(
                            "fy",
                            []
                        ),
                        i
                    ),

                    "fp": safe_get(
                        data.get(
                            "fp",
                            []
                        ),
                        i
                    )
                })

        except Exception as exc:

            print(
                f"    WARNING: historical file "
                f"{filename} failed: {exc}"
            )

        time.sleep(
            REQUEST_DELAY
        )

    return all_records


# ============================================================
# PARSE YEAR
# ============================================================

def parse_year(
    value
):
    try:

        return int(
            float(
                str(value).strip()
            )
        )

    except (
        ValueError,
        TypeError
    ):

        return None


# ============================================================
# REPORT DATE YEAR
# ============================================================

def report_date_year(
    report_date
):

    report_date = str(
        report_date
    ).strip()

    if not re.match(
        r"^\d{4}-\d{2}-\d{2}$",
        report_date
    ):

        return None

    try:

        return int(
            report_date[:4]
        )

    except ValueError:

        return None


# ============================================================
# FISCAL YEAR INFERENCE
# ============================================================

def infer_fiscal_year(
    record
):
    """
    Primary:
        SEC fy

    Fallback:
        reportDate

    For January/February reporting dates, use the previous
    year, matching your legacy corpus convention.
    """

    sec_fy = parse_year(
        record.get(
            "fy"
        )
    )

    if sec_fy is not None:

        return sec_fy

    date_text = str(
        record.get(
            "report_date",
            ""
        )
    )

    if not date_text:
        return None

    try:

        year = int(
            date_text[:4]
        )

        month = int(
            date_text[5:7]
        )

        if month <= 2:

            return year - 1

        return year

    except (
        ValueError,
        IndexError
    ):

        return None


# ============================================================
# FIND CANDIDATES
# ============================================================

def find_candidates(
    records,
    target_fy
):

    candidates = []

    for record in records:

        form = str(
            record.get(
                "form",
                ""
            )
        ).strip().upper()

        # Only annual reports
        if form not in {
            "10-K",
            "10-K/A"
        }:

            continue

        accession = str(
            record.get(
                "accession",
                ""
            )
        ).strip()

        if not accession:

            continue

        inferred_fy = infer_fiscal_year(
            record
        )

        report_year = report_date_year(
            record.get(
                "report_date",
                ""
            )
        )

        sec_fy = parse_year(
            record.get(
                "fy"
            )
        )

        # -----------------------------------------------
        # Match
        # -----------------------------------------------

        fy_match = (
            inferred_fy == target_fy
        )

        report_match = (
            report_year == target_fy
        )

        if not fy_match and not report_match:
            continue

        # -----------------------------------------------
        # Score
        # -----------------------------------------------

        score = 0

        if sec_fy == target_fy:
            score += 100

        if report_match:
            score += 50

        if form == "10-K":
            score += 20

        if str(
            record.get(
                "fp",
                ""
            )
        ).upper() == "FY":

            score += 10

        candidate = dict(
            record
        )

        candidate["_score"] = score

        candidates.append(
            candidate
        )

    candidates.sort(
        key=lambda x: (
            x["_score"],
            x.get(
                "filing_date",
                ""
            )
        ),
        reverse=True
    )

    return candidates


# ============================================================
# FIND PRIMARY DOCUMENT FROM FILING INDEX
# ============================================================

def find_primary_document_from_index(
    cik,
    accession
):
    """
    Uses SEC filing index.json.

    This is more reliable than assuming that primaryDocument
    in submissions.json is present/correct.
    """

    cik_archive = str(
        int(
            str(cik).zfill(10)
        )
    )

    accession_clean = (
        str(accession)
        .replace(
            "-",
            ""
        )
    )

    index_url = (
        f"{ARCHIVES_BASE_URL}/"
        f"{cik_archive}/"
        f"{accession_clean}/"
        f"index.json"
    )

    print(
        f"    Filing index:"
        f" {index_url}"
    )

    try:

        response = sec_get(
            index_url
        )

        data = response.json()

    except Exception as exc:

        print(
            f"    WARNING: index.json failed: "
            f"{exc}"
        )

        return None

    # SEC index structure:
    #
    # directory -> item
    #
    # Be slightly flexible because structure can vary.

    directory = data.get(
        "directory",
        {}
    )

    items = directory.get(
        "item",
        []
    )

    if not isinstance(
        items,
        list
    ):

        items = []

    # --------------------------------------------------------
    # Preferred: document explicitly marked 10-K
    # --------------------------------------------------------

    for item in items:

        item_type = str(
            item.get(
                "type",
                ""
            )
        ).upper()

        name = str(
            item.get(
                "name",
                ""
            )
        )

        if item_type in {
            "10-K",
            "10-K/A"
        }:

            if "index" not in name.lower() and name.lower().endswith(
                (
                    ".htm",
                    ".html"
                )
            ):

                return name

    # --------------------------------------------------------
    # Fallback: HTML/HTM documents that aren't exhibits
    # --------------------------------------------------------

    excluded_terms = {
        "xsd",
        "xml",
        "json",
        "txt",
        "jpg",
        "jpeg",
        "png",
        "gif",
        "css",
        "js"
    }

    html_candidates = []

    for item in items:

        name = str(
            item.get(
                "name",
                ""
            )
        )

        lower = name.lower()

        if "index" in lower:
            continue

        if not lower.endswith(
            (
                ".htm",
                ".html"
            )
        ):
            continue

        extension = (
            lower.split(".")[-1]
        )

        if extension in excluded_terms:
            continue

        html_candidates.append(
            name
        )

    if html_candidates:

        # Primary filing document is usually one of the
        # first HTML documents. Prefer filenames containing
        # the filing/accession identifier.
        accession_stem = accession.replace(
            "-",
            ""
        ).lower()

        for name in html_candidates:

            if accession_stem in name.lower():

                return name

        return html_candidates[0]

    return None


# ============================================================
# BUILD ARCHIVE URL
# ============================================================

def build_archive_url(
    cik,
    accession,
    filename
):

    cik_archive = str(
        int(
            str(cik).zfill(10)
        )
    )

    accession_clean = (
        str(accession)
        .replace(
            "-",
            ""
        )
    )

    return (
        f"{ARCHIVES_BASE_URL}/"
        f"{cik_archive}/"
        f"{accession_clean}/"
        f"{filename}"
    )


# ============================================================
# DOWNLOAD BINARY FILE
# ============================================================

def download_file(
    url,
    destination
):

    response = sec_get(
        url
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    destination.write_bytes(
        response.content
    )

    return len(
        response.content
    )


# ============================================================
# PROCESS COMPANY
# ============================================================

def process_company(
    ticker,
    company,
    sector,
    cik,
    submissions,
    all_records,
    manifest_records
):

    for target_fy in TARGET_FISCAL_YEARS:

        print()
        print(
            "=" * 70
        )

        print(
            f"{ticker} | FY{target_fy}"
        )

        print(
            "=" * 70
        )

        candidates = find_candidates(
            all_records,
            target_fy
        )

        if not candidates:

            print(
                f"  ERROR {ticker} FY{target_fy}: "
                f"No 10-K candidate found."
            )

            manifest_records.append({

                "ticker": ticker,

                "company": company,

                "sector": sector,

                "cik": cik,

                "fy": target_fy,

                "status": "missing"

            })

            continue

        # ----------------------------------------------------
        # Try candidates until one works
        # ----------------------------------------------------

        successful = False

        for candidate in candidates:

            accession = candidate[
                "accession"
            ]

            form = candidate[
                "form"
            ]

            print(
                f"  Candidate:"
                f" {form} "
                f"{accession}"
            )

            print(
                f"  filingDate:"
                f" {candidate.get('filing_date', '')}"
            )

            print(
                f"  reportDate:"
                f" {candidate.get('report_date', '')}"
            )

            print(
                f"  SEC fy:"
                f" {candidate.get('fy', '')}"
            )

            # ------------------------------------------------
            # Find primary document
            # ------------------------------------------------

            primary = (
                find_primary_document_from_index(
                    cik,
                    accession
                )
            )

            if not primary:

                # Sometimes the primary document from the
                # submission metadata is still useful.
                submission_primary = str(
                    candidate.get(
                        "primary_document",
                        ""
                    )
                ).strip()

                if "index" not in submission_primary.lower():
                    primary = submission_primary

            if not primary:

                print(
                    "    No primary document found. "
                    "Trying next candidate."
                )

                continue

            print(
                f"    Primary document:"
                f" {primary}"
            )

            # ------------------------------------------------
            # Build directory
            # ------------------------------------------------

            filing_dir = (
                OUTPUT_DIR
                / ticker
                / "10-K"
                / accession
            )

            filing_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            # ------------------------------------------------
            # Primary document destination
            # ------------------------------------------------

            extension = Path(
                primary
            ).suffix.lower()

            if extension not in {
                ".htm",
                ".html"
            }:

                extension = ".htm"

            primary_destination = (
                filing_dir
                / f"primary-document{extension}"
            )

            # ------------------------------------------------
            # Download primary document
            # ------------------------------------------------

            document_url = build_archive_url(
                cik,
                accession,
                primary
            )

            try:

                print(
                    f"    Downloading:"
                )

                print(
                    f"    {document_url}"
                )

                size = download_file(
                    document_url,
                    primary_destination
                )

                print(
                    f"    Saved:"
                    f" {size:,} bytes"
                )

            except Exception as exc:

                print(
                    f"    Primary download failed:"
                    f" {exc}"
                )

                continue

            # ------------------------------------------------
            # Full submission
            # ------------------------------------------------

            submission_destination = (
                filing_dir
                / "full-submission.txt"
            )

            submission_url = (
                f"{ARCHIVES_BASE_URL}/"
                f"{str(int(str(cik).zfill(10)))}/"
                f"{accession.replace('-', '')}/"
                f"{accession}.txt"
            )

            try:

                size = download_file(
                    submission_url,
                    submission_destination
                )

                print(
                    f"    Full submission:"
                    f" {size:,} bytes"
                )

            except Exception as exc:

                print(
                    f"    WARNING: full submission "
                    f"failed: {exc}"
                )

            # ------------------------------------------------
            # Metadata
            # ------------------------------------------------

            metadata = {

                "ticker": ticker,

                "company": company,

                "sector": sector,

                "cik": cik,

                "fy_requested": target_fy,

                "fy_sec": candidate.get(
                    "fy",
                    ""
                ),

                "fp_sec": candidate.get(
                    "fp",
                    ""
                ),

                "report_date": candidate.get(
                    "report_date",
                    ""
                ),

                "filing_date": candidate.get(
                    "filing_date",
                    ""
                ),

                "form": form,

                "accession": accession,

                "primary_document": primary,

                "document_url": document_url,

                "submission_url": submission_url,

                "local_primary_document": str(
                    primary_destination.relative_to(
                        PROJECT_ROOT
                    )
                ),

                "local_submission": str(
                    submission_destination.relative_to(
                        PROJECT_ROOT
                    )
                ),

                "downloaded_at": (
                    datetime.now(
                        timezone.utc
                    ).isoformat()
                )
            }

            metadata_path = (
                filing_dir
                / "filing_metadata.json"
            )

            metadata_path.write_text(
                json.dumps(
                    metadata,
                    ensure_ascii=False,
                    indent=2
                ),
                encoding="utf-8"
            )

            # ------------------------------------------------
            # Manifest record
            # ------------------------------------------------

            manifest_records.append({

                "ticker": ticker,

                "company": company,

                "sector": sector,

                "cik": cik,

                "fy": target_fy,

                "form": form,

                "filing_date": candidate.get(
                    "filing_date",
                    ""
                ),

                "report_date": candidate.get(
                    "report_date",
                    ""
                ),

                "accession": accession,

                "primary_document": primary,

                "document_url": document_url,

                "submission_url": submission_url,

                "local_file": str(
                    primary_destination.relative_to(
                        PROJECT_ROOT
                    )
                ),

                "local_submission": str(
                    submission_destination.relative_to(
                        PROJECT_ROOT
                    )
                ),

                "status": "ok"

            })

            successful = True

            print(
                f"\n  ✓ {ticker} FY{target_fy} COMPLETE"
            )

            break

        # ----------------------------------------------------
        # No candidate successfully downloaded
        # ----------------------------------------------------

        if not successful:

            print(
                f"\n  ✗ {ticker} FY{target_fy} FAILED"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "Policy-as-Code runtime controls"
    )

    print(
        "01_BUILD_FILINGS"
    )

    print(
        "=" * 70
    )

    print(
        f"\nProject:"
        f"\n{PROJECT_ROOT}"
    )

    print(
        f"\nInput:"
        f"\n{INPUT_MANIFEST}"
    )

    print(
        f"\nOutput:"
        f"\n{OUTPUT_DIR}"
    )

    print(
        f"\nTarget FYs:"
        f"\n{TARGET_FISCAL_YEARS}"
    )

    # --------------------------------------------------------
    # User Agent
    # --------------------------------------------------------

    if "YOUR_EMAIL" in SEC_USER_AGENT:

        print()
        print(
            "WARNING: SEC_USER_AGENT still contains "
            "YOUR_EMAIL."
        )

        print(
            'Run, for example:'
        )

        print(
            'export SEC_USER_AGENT="Your Name '
            'your@email.com"'
        )

    # --------------------------------------------------------
    # Input manifest
    # --------------------------------------------------------

    if not INPUT_MANIFEST.exists():

        raise FileNotFoundError(
            f"Input manifest not found: "
            f"{INPUT_MANIFEST}"
        )

    manifest = pd.read_csv(
        INPUT_MANIFEST,
        dtype=str
    ).fillna("")

    companies = []

    for _, row in manifest.iterrows():

        ticker = str(
            row.get(
                "ticker",
                ""
            )
        ).strip().upper()

        if not ticker:
            continue

        companies.append({

            "ticker": ticker,

            "company": str(
                row.get(
                    "company",
                    ""
                )
            ).strip(),

            "sector": str(
                row.get(
                    "sector",
                    ""
                )
            ).strip()
        })

    print(
        f"\nCompanies:"
        f" {len(companies)}"
    )

    print(
        f"Expected filing records:"
        f" {len(companies) * 3}"
    )

    # --------------------------------------------------------
    # Ticker database
    # --------------------------------------------------------

    ticker_to_cik = (
        load_ticker_database()
    )

    # --------------------------------------------------------
    # All manifest results
    # --------------------------------------------------------

    manifest_records = []

    # ========================================================
    # PROCESS
    # ========================================================

    for company_info in companies:

        ticker = company_info[
            "ticker"
        ]

        company = company_info[
            "company"
        ]

        sector = company_info[
            "sector"
        ]

        print()
        print(
            "#" * 70
        )

        print(
            f"COMPANY: {ticker}"
        )

        print(
            "#" * 70
        )

        # ----------------------------------------------------
        # CIK
        # ----------------------------------------------------

        cik = ticker_to_cik.get(
            ticker
        )

        if not cik:

            print(
                f"  ERROR: CIK not found for {ticker}"
            )

            for fy in TARGET_FISCAL_YEARS:

                manifest_records.append({

                    "ticker": ticker,

                    "company": company,

                    "sector": sector,

                    "cik": "",

                    "fy": fy,

                    "status": "cik_not_found"

                })

            continue

        print(
            f"  CIK: {cik}"
        )

        # ----------------------------------------------------
        # Submissions
        # ----------------------------------------------------

        try:

            submissions = load_submissions(
                ticker,
                cik
            )

        except Exception as exc:

            print(
                f"  ERROR retrieving submissions:"
                f" {exc}"
            )

            for fy in TARGET_FISCAL_YEARS:

                manifest_records.append({

                    "ticker": ticker,

                    "company": company,

                    "sector": sector,

                    "cik": cik,

                    "fy": fy,

                    "status": "submission_error",

                    "error": str(exc)

                })

            continue

        time.sleep(
            REQUEST_DELAY
        )

        # ----------------------------------------------------
        # Recent records
        # ----------------------------------------------------

        recent_records = (
            parse_submission_records(
                submissions
            )
        )

        print(
            f"  Recent filing records:"
            f" {len(recent_records)}"
        )

        # ----------------------------------------------------
        # Historical records
        # ----------------------------------------------------

        historical_records = (
            load_archived_records(
                submissions
            )
        )

        print(
            f"  Historical records:"
            f" {len(historical_records)}"
        )

        # ----------------------------------------------------
        # Combine
        # ----------------------------------------------------

        all_records = (
            recent_records
            + historical_records
        )

        # ----------------------------------------------------
        # Remove duplicate accessions
        # ----------------------------------------------------

        unique = {}

        for record in all_records:

            accession = str(
                record.get(
                    "accession",
                    ""
                )
            )

            if accession:

                unique[
                    accession
                ] = record

        all_records = list(
            unique.values()
        )

        print(
            f"  Unique filings:"
            f" {len(all_records)}"
        )

        # ----------------------------------------------------
        # Diagnostic: show all 10-Ks
        # ----------------------------------------------------

        ten_ks = [
            record
            for record in all_records
            if str(
                record.get(
                    "form",
                    ""
                )
            ).upper()
            in {
                "10-K",
                "10-K/A"
            }
        ]

        print(
            f"  10-K records discovered:"
            f" {len(ten_ks)}"
        )

        for record in sorted(
            ten_ks,
            key=lambda x: x.get(
                "filing_date",
                ""
            ),
            reverse=True
        )[:10]:

            inferred = infer_fiscal_year(
                record
            )

            print(
                f"    {record.get('form')} | "
                f"FY={record.get('fy')} | "
                f"report={record.get('report_date')} | "
                f"inferred={inferred} | "
                f"{record.get('accession')}"
            )

        # ----------------------------------------------------
        # Process company
        # ----------------------------------------------------

        process_company(
            ticker=ticker,
            company=company,
            sector=sector,
            cik=cik,
            submissions=submissions,
            all_records=all_records,
            manifest_records=manifest_records
        )

        # Save progress after each company
        pd.DataFrame(
            manifest_records
        ).to_csv(
            OUTPUT_MANIFEST,
            index=False
        )

        time.sleep(
            REQUEST_DELAY
        )

    # ========================================================
    # FINAL MANIFEST
    # ========================================================

    dataframe = pd.DataFrame(
        manifest_records
    )

    if not dataframe.empty:

        dataframe = dataframe.sort_values(
            [
                "ticker",
                "fy"
            ]
        )

    dataframe.to_csv(
        OUTPUT_MANIFEST,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    ok = dataframe[
        dataframe["status"] == "ok"
    ]

    missing = dataframe[
        dataframe["status"] == "missing"
    ]

    errors = dataframe[
        ~dataframe["status"].isin(
            [
                "ok",
                "missing"
            ]
        )
    ]

    print()
    print(
        "=" * 70
    )

    print(
        "FINAL SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"Expected:"
        f" {len(companies) * 3}"
    )

    print(
        f"Successful:"
        f" {len(ok)}"
    )

    print(
        f"Missing:"
        f" {len(missing)}"
    )

    print(
        f"Errors:"
        f" {len(errors)}"
    )

    print()
    print(
        "Manifest:"
    )

    print(
        OUTPUT_MANIFEST
    )

    # --------------------------------------------------------
    # Missing records
    # --------------------------------------------------------

    if not missing.empty:

        print()
        print(
            "MISSING:"
        )

        for _, row in missing.iterrows():

            print(
                f"  {row['ticker']} FY{row['fy']}"
            )

    print()
    print(
        "Done."
    )


if __name__ == "__main__":
    main()