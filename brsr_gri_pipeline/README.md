# BRSR–GRI Gold Dataset Pipeline

Builds a structured, machine-actionable gold-label dataset from the official
GRI/BSI/BSE "Linking the GRI Standards and the SEBI BRSR Framework" (2022)
document. This gold set is the foundation for everything downstream
(ontology population, matcher calibration, evaluation).

## Directory layout

```
brsr_gri_pipeline/
  data/
    raw/          # source PDF goes here (see scripts/01)
    interim/      # intermediate extraction artifacts (raw table dumps)
    processed/    # final gold_pairs.csv and QA reports
  scripts/        # run in numeric order
  src/            # shared reference data + regex patterns
  tests/          # sanity checks against known values from the doc
```

## Execution order

1. `scripts/01_fetch_source_doc.py` — run **locally**, not in a sandboxed
   environment without internet, since it needs to reach
   globalreporting.org. Downloads the PDF into `data/raw/`.
2. `scripts/02_extract_pdf_tables.py` — extracts raw table structure from
   the PDF using pdfplumber (table-aware, not plain-text extraction —
   important, see note in the script).
3. `scripts/03_parse_summary_table.py` — parses Table 1 (Summary Table,
   pages 6–18) into `(brsr_id, gri_refs, match_type)` triples.
4. `scripts/04_parse_comprehensive_table.py` — parses Table 2
   (Comprehensive Table, pages 19+) into fine-grained records including
   the expert "Remarks" text (your granularity-gap gold signal).
5. `scripts/05_merge_and_validate_gold.py` — cross-checks Table 1 vs.
   Table 2, flags disagreements/parse failures for manual review, and
   writes the final `data/processed/gold_pairs.csv`.

## Mandatory manual QA step

After step 5, open `data/processed/qa_flags.csv` and manually review every
flagged row before treating `gold_pairs.csv` as ground truth. PDF table
extraction on a real-world report is never 100% clean — budget an hour for
this, it's the single highest-leverage hour in the whole project since
every downstream metric depends on this file being correct.

## Install

```
pip install -r requirements.txt --break-system-packages
```
