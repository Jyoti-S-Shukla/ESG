"""
Sanity checks for the final gold dataset.

These facts were read directly from the source PDF and are treated as
known-correct anchors. The test checks the schema produced by script 05.

Run after scripts/05:
    python tests/test_known_values.py

Important:
    Script 05 no longer writes a single `gri_codes_json` column.
    It preserves Table 1 and Table 2 extraction separately:

        gri_codes_table1
        gri_codes_table2
        gri_codes_shared
        gri_codes_union
        gri_codes_consensus
        gri_code_agreement
"""


import csv
import json
import pathlib
import re
import sys


BASE = pathlib.Path(__file__).resolve().parents[1]
GOLD_PATH = BASE / "data" / "processed" / "gold_pairs.csv"


# ---------------------------------------------------------------------
# Known facts read directly from the source PDF
# ---------------------------------------------------------------------
#
# (brsr_id,
#  expected_match_type,
#  expected_gri_standard,
#  expected_source,
#  expected_code_source)
#
# expected_code_source tells the test which table should contain the
# expected GRI standard.
#
KNOWN_FACTS = [
    (
        "A6",
        "direct",
        "2",
        'Table 2: "E-mail ... GRI 2: General Disclosures 2021 '
        'Disclosure 2-3 ... d. specify the contact point for questions '
        'about the report..."',
        "table2",
    ),
    (
        "P6-E1",
        "direct",
        "302",
        'Table 1: "P6-E1 GRI 302: Energy 2016 Disclosure 302-1-a; '
        '302-1-b; 302-1-c-I; 302-1-e"',
        "table1",
    ),
    (
        "P2-E1",
        "no_direct_linkage",
        None,
        'Table 1: "Essential Indicators P2-E1 No direct linkage"',
        None,
    ),
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def parse_json_list(value, column_name, brsr_id):
    """
    Parse a JSON list from one of the Script 05 GRI-code columns.

    Script 05 writes these columns as JSON arrays, e.g.:

        ["302-1-A", "302-1-B"]

    Return an empty list for blank values.
    """
    if value is None or not str(value).strip():
        return []

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"[{brsr_id}] invalid JSON in {column_name}: "
            f"{value!r} ({exc})"
        )

    if not isinstance(parsed, list):
        raise AssertionError(
            f"[{brsr_id}] {column_name} is not a JSON list: "
            f"{parsed!r}"
        )

    return parsed


def standard_from_code(code):
    """
    Extract the GRI standard number from a normalized disclosure code.

    Examples:
        302-1-A       -> 302
        302-1-C-I     -> 302
        405-1-A-I     -> 405
        3-3-C         -> 3
        2-7           -> 2
    """
    if not isinstance(code, str):
        return None

    m = re.match(r"^(\d+)-", code.strip())
    return m.group(1) if m else None


def standards_from_codes(codes):
    """Return the set of GRI standard numbers represented by codes."""
    return {
        standard_from_code(code)
        for code in codes
        if standard_from_code(code) is not None
    }


def check_schema(rows):
    """
    Verify that Script 05 produced the schema this test expects.
    """
    if not rows:
        print("ERROR: gold_pairs.csv contains no rows.")
        return False

    required_columns = {
        "brsr_id",
        "match_type",
        "gri_codes_table1",
        "gri_codes_table2",
        "gri_codes_shared",
        "gri_codes_union",
        "gri_codes_consensus",
        "gri_code_agreement",
    }

    actual_columns = set(rows[0].keys())
    missing = required_columns - actual_columns

    if missing:
        print("FAIL: gold_pairs.csv is missing required columns:")
        for col in sorted(missing):
            print(f"  - {col}")
        print("\nActual columns:")
        print(sorted(actual_columns))
        return False

    # Explicitly catch the old schema so the error is easy to understand.
    if "gri_codes_json" in actual_columns:
        print(
            "WARNING: gold_pairs.csv still contains the old "
            "'gri_codes_json' column."
        )

    return True


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    if not GOLD_PATH.exists():
        print(
            f"ERROR: {GOLD_PATH} not found.\n"
            "Run scripts 01-05 first."
        )
        sys.exit(1)

    with open(GOLD_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows_list = list(reader)

    if not check_schema(rows_list):
        sys.exit(1)

    rows = {r["brsr_id"]: r for r in rows_list}

    failures = 0

    print("=" * 70)
    print("KNOWN-VALUE SANITY CHECK")
    print("=" * 70)
    print(f"Gold dataset: {GOLD_PATH}")
    print(f"Rows: {len(rows_list)}")
    print()

    for (
        brsr_id,
        expected_type,
        expected_standard,
        quote,
        code_source,
    ) in KNOWN_FACTS:

        print("-" * 70)
        print(f"CHECK: {brsr_id}")

        row = rows.get(brsr_id)

        # -------------------------------------------------------------
        # 1. ID existence
        # -------------------------------------------------------------
        if row is None:
            print(f"FAIL [{brsr_id}]: not found in gold_pairs.csv")
            print(f"      source: {quote}")
            failures += 1
            continue

        # -------------------------------------------------------------
        # 2. Match type
        # -------------------------------------------------------------
        actual_type = row["match_type"]

        if actual_type != expected_type:
            print(
                f"FAIL [{brsr_id}]: match_type = {actual_type!r}, "
                f"expected {expected_type!r}"
            )
            print(f"      source: {quote}")
            failures += 1
            continue

        # -------------------------------------------------------------
        # 3. GRI standard check
        # -------------------------------------------------------------
        if expected_standard is not None:

            if code_source == "table1":
                column = "gri_codes_table1"
            elif code_source == "table2":
                column = "gri_codes_table2"
            else:
                column = "gri_codes_union"

            codes = parse_json_list(
                row[column],
                column,
                brsr_id,
            )

            standards_found = standards_from_codes(codes)

            if expected_standard not in standards_found:
                print(
                    f"FAIL [{brsr_id}]: expected GRI standard "
                    f"{expected_standard} not found."
                )
                print(f"      source column: {column}")
                print(f"      codes: {codes}")
                print(f"      standards found: {sorted(standards_found)}")
                print(f"      source: {quote}")
                failures += 1
                continue

            print(f"  match_type: PASS ({actual_type})")
            print(
                f"  GRI standard: PASS "
                f"({expected_standard} found in {column})"
            )
            print(f"  codes: {codes}")

        else:
            print(f"  match_type: PASS ({actual_type})")

        print(f"PASS [{brsr_id}]")

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    passed = len(KNOWN_FACTS) - failures

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)
    print(
        f"{passed}/{len(KNOWN_FACTS)} known-fact checks passed"
    )

    if failures:
        print()
        print(
            "FAILURES DETECTED."
        )
        print(
            "Do NOT proceed to Script 06 yet. "
            "A known PDF anchor is inconsistent with the generated gold data."
        )
        sys.exit(1)

    print()
    print(
        "All known-value checks passed."
    )
    print(
        "The Script 05 output is consistent with these PDF anchors."
    )
    print(
        "You can proceed to Script 06."
    )


if __name__ == "__main__":
    main()