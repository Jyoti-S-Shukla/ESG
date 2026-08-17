"""
Extracts every page into a column-aware grid.

CORRECTED after testing against the real PDF: per-page gutter auto-detection
(src.pdf_table_extractor.detect_column_gutters) is unreliable on this
document -- it returns ZERO gutters on the Table 2 title page (page 19,
sparse content) and drifts by 10+ points on others, because true column
boundaries here are often not perfectly empty across every row (wrapped
text frequently runs close to, or into, the visual gutter).

What actually works, verified across 5 structurally different Table 2 pages
(the title page, a normal page, and pages deep into different Principles):
this document uses FIXED, consistent column x-positions throughout each
table section. So we hardcode verified bounds per section instead of
re-detecting them per page.

    Table 1 (pages 6-18):  effectively ONE content column -- the BRSR ID
                            and its GRI citation run together as one string
                            (e.g. "P5-E8 GRI 2: General Disclosures 2021
                            Disclosures 2-23..."). This is not a parsing
                            failure -- the source document really is laid
                            out this way here. scripts/03 already handles
                            it correctly via regex + flattening.

    Table 2 (pages 19+):   FOUR columns: Sl.No | BRSR text | GRI text |
                            Remarks, at fixed x-positions [46, 102, 302,
                            556, 797].

If you're running this against a different edition/version of the PDF and
these bounds don't line up, re-run the diagnostic snippet in
tests/diagnose_columns.py (below) against a few sample pages and adjust
TABLE2_BOUNDS.

Output: data/interim/pages_raw.json
    [
      {"page_number": int, "table": "table1"|"table2"|"other",
       "grid": [[cell, ...], ...], "text": "..."},
      ...
    ]

Usage:
    python scripts/02_extract_pdf_tables.py
"""

import json
import pathlib
import sys

import pdfplumber

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.pdf_table_extractor import words_to_grid

BASE = pathlib.Path(__file__).resolve().parents[1]
PDF_PATH = BASE / "data" / "raw" / "sebi_brsb_gri_linkage_doc.pdf"
OUT_PATH = BASE / "data" / "interim" / "pages_raw.json"

# CONFIRM these page ranges against your copy's own table of contents --
# different PDF export/versions can shift page numbers by a page or two.
TABLE1_PAGE_RANGE = (6, 18)
TABLE2_PAGE_START = 19

TABLE1_BOUNDS = [35, 808]                  # single content column, verified
TABLE2_BOUNDS = [46, 102, 302, 556, 797]   # Sl.No | BRSR | GRI | Remarks, verified


def classify_page(page_number: int) -> str:
    if TABLE1_PAGE_RANGE[0] <= page_number <= TABLE1_PAGE_RANGE[1]:
        return "table1"
    if page_number >= TABLE2_PAGE_START:
        return "table2"
    return "other"


def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"{PDF_PATH} not found. Place the source PDF there first."
        )

    pages_out = []
    with pdfplumber.open(PDF_PATH) as pdf:
        print(f"Opened PDF with {len(pdf.pages)} pages")
        for i, page in enumerate(pdf.pages, start=1):
            table = classify_page(i)
            bounds = TABLE1_BOUNDS if table == "table1" else TABLE2_BOUNDS
            grid = words_to_grid(page, bounds) if table != "other" else []
            text = page.extract_text() or ""
            pages_out.append({
                "page_number": i,
                "table": table,
                "grid": grid,
                "text": text,
            })
            if i % 20 == 0 or i == len(pdf.pages):
                print(f"  ...processed page {i}/{len(pdf.pages)}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(pages_out, indent=2, ensure_ascii=False))
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
