"""
Parse the official GRI-BRSR Table 1 (Summary Table).

Purpose
-------
Extract one record per BRSR indicator from Table 1 and recover the GRI
disclosure/sub-point codes embedded in the free-text GRI column.

Important design decision
-------------------------
The PDF text extraction is not sufficiently regular to rely on one large
regex matching:

    GRI <number>: <name> <year> Disclosure <codes>

Instead, this script uses two stages:

1. Reconstruct Table 1 records from the PDF grid:
       BRSR ID -> GRI text / continuation rows

2. Extract GRI disclosure codes independently from the reconstructed GRI
   text using a permissive disclosure-code recognizer.

Examples recovered:

    GRI 405 ... Disclosure 405-1-a-I; 405-1-b-i
        -> 405-1-A-I
        -> 405-1-B-I

    GRI 403 ... Disclosure 403-1-a, 403-1-b
        -> 403-1-A
        -> 403-1-B

    GRI 305 ... Disclosure 305-4
        -> 305-4

    GRI 305 ... Disclosure 305-5
        -> 305-5

The output intentionally contains BOTH:
    gri_codes_json
    gri_requirement_codes_json

to avoid schema drift with downstream scripts.

Output
------
data/interim/table1_summary.csv
"""

import csv
import json
import pathlib
import re
import sys


# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

BASE = pathlib.Path(__file__).resolve().parents[1]

IN_PATH = BASE / "data" / "interim" / "pages_raw.json"
OUT_PATH = BASE / "data" / "interim" / "table1_summary.csv"


# ---------------------------------------------------------------------------
# TABLE 1 PAGE RANGE
# ---------------------------------------------------------------------------

PAGE_START = 6
PAGE_END = 18


# ---------------------------------------------------------------------------
# BRSR IDENTIFIERS
# ---------------------------------------------------------------------------

# Examples:
#   A19
#   A17a
#   B3
#   P3-E10
#   P3-E10a
#   P6-E1
#
# We deliberately allow the sub-part letter as part of the identifier.
#
BRSR_ID_RE = re.compile(
    r"^("
    r"[AB]\d{1,2}[a-c]?"
    r"|"
    r"P[1-9]-[EL]\d{1,2}[a-d]?"
    r")"
    r"\b",
    re.IGNORECASE,
)


# In the actual PDF, some multi-part rows appear as:
#
#     P3-E10 a GRI 403...
#     P3-E10 b GRI 403...
#
# Therefore after identifying P3-E10, inspect the immediate remainder for
# a standalone sub-part letter.
#
SUBPART_RE = re.compile(
    r"^[ \t]*([a-d])(?=\s+(?:GRI\b|No\s+direct\b|Can\s+be\s+covered\b))",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# HEADER DETECTION
# ---------------------------------------------------------------------------

HEADER_LINE_RE = re.compile(
    r"^(?:"
    r"Section\s+[ABC]\s*:"
    r"|PRINCIPLE\s+\d+"
    r"|Essential\s+Indicators"
    r"|Leadership\s+Indicators"
    r"|SEBI\s*-\s*BRSR\s+Framework"
    r")",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# MATCH TYPE
# ---------------------------------------------------------------------------

NO_LINKAGE_RE = re.compile(
    r"\bno\s+direct\s+linkage\b",
    re.IGNORECASE,
)

CAN_BE_COVERED_RE = re.compile(
    r"\bcan\s+be\s+covered\s+by\b",
    re.IGNORECASE,
)


def classify_match_type(text):
    """
    Classify the official Table 1 mapping statement.

    Order matters:
        no direct linkage
        can be covered by
        otherwise direct
    """

    if not text or not text.strip():
        return "no_direct_linkage"

    if NO_LINKAGE_RE.search(text):
        return "no_direct_linkage"

    if CAN_BE_COVERED_RE.search(text):
        return "can_be_covered_by"

    return "direct"


# ---------------------------------------------------------------------------
# GRI CODE EXTRACTION
# ---------------------------------------------------------------------------

# A GRI disclosure identifier has the general structure:
#
#     2-1
#     2-1-a
#     2-1-a-i
#     302-1
#     302-1-a
#     302-1-a-i
#     403-10-b-ii
#
# We do NOT try to encode the entire GRI ontology into the regex.
#
# The important invariant is:
#
#     <GRI standard number>-<disclosure number>
#     optionally followed by alphabetic / roman subparts.
#
# The first component is normally 1-3 digits.
#
GRI_CODE_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(\d{1,3}"
    r"-"
    r"\d{1,3}"
    r"(?:"
    r"-[A-Za-z]+"
    r")*"
    r")"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)


# More restrictive pattern used to identify the actual disclosure section.
#
# Examples:
#
#   Disclosure 405-1-a-I; 405-1-b-i
#   Disclosures 403-2-b; 403-2-c
#   Disclosure 305-4
#
DISCLOSURE_ANCHOR_RE = re.compile(
    r"\bDisclosures?\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# NORMALIZATION
# ---------------------------------------------------------------------------

def normalize_gri_code(code):
    """
    Normalize a GRI disclosure code.

    Examples
    --------
    405-1-a-I  -> 405-1-A-I
    403-10-b-ii -> 403-10-B-II
    302-1 -> 302-1
    """

    if not code:
        return None

    code = str(code).strip()
    code = code.rstrip(".,;:")

    # Normalize whitespace around hyphens.
    code = re.sub(r"\s*-\s*", "-", code)

    # GRI codes are conventionally represented uppercase here so that
    # comparison between Table 1 / Table 2 / ESRS is deterministic.
    code = code.upper()

    return code


def is_probable_gri_disclosure(code):
    """
    Defensive validation.

    This prevents ordinary numeric fragments from being treated as GRI
    disclosure codes.

    Examples accepted:
        2-1
        2-23-A
        302-1
        403-10-B-II

    Examples rejected:
        2016
        2021
        1-2-3-4-5-6-7
    """

    if not code:
        return False

    parts = code.split("-")

    if len(parts) < 2:
        return False

    # Standard number.
    if not parts[0].isdigit():
        return False

    # Disclosure number.
    if not parts[1].isdigit():
        return False

    standard = int(parts[0])
    disclosure = int(parts[1])

    # GRI standard numbers in these documents are generally 1-3 digits.
    if standard < 1 or standard > 999:
        return False

    if disclosure < 1 or disclosure > 999:
        return False

    # Optional sub-parts should be alphabetic tokens.
    for part in parts[2:]:
        if not re.fullmatch(r"[A-Z]+", part, re.IGNORECASE):
            return False

    return True


def extract_gri_codes(text):
    """
    Robustly extract GRI disclosure codes from a reconstructed GRI cell.

    This function intentionally does NOT depend on the GRI standard-name
    regex. It searches the text for disclosure-shaped identifiers.

    Returns
    -------
    list[str]
        Sorted unique normalized GRI disclosure codes.
    """

    if not text:
        return []

    candidates = GRI_CODE_RE.findall(text)

    normalized = []

    for candidate in candidates:
        code = normalize_gri_code(candidate)

        if is_probable_gri_disclosure(code):
            normalized.append(code)

    # Deduplicate while preserving deterministic ordering.
    return sorted(set(normalized))


def extract_gri_structured_codes(text):
    """
    Produce the richer structure used by earlier versions of the pipeline.

    Example
    -------
    Input:
        GRI 405 ... Disclosure 405-1-a-I; 405-1-b-i

    Output:
        [
            {
                "standard": "405",
                "disclosures": [
                    "405-1-A-I",
                    "405-1-B-I"
                ]
            }
        ]

    This is useful for compatibility with script 05.
    """

    codes = extract_gri_codes(text)

    grouped = {}

    for code in codes:
        standard = code.split("-", 1)[0]

        grouped.setdefault(standard, []).append(code)

    result = []

    for standard in sorted(grouped):
        result.append({
            "standard": standard,
            "disclosures": sorted(grouped[standard]),
        })

    return result


# ---------------------------------------------------------------------------
# ROW HANDLING
# ---------------------------------------------------------------------------

def flatten_row(row):
    """
    Join non-empty cells into one text string.

    Table 1 does not have a reliable dedicated BRSR-ID column, so ID
    detection is performed against the flattened row.
    """

    return " ".join(
        str(c).strip()
        for c in row
        if c is not None and str(c).strip()
    )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():

    if not IN_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{IN_PATH}\n"
            "Run script 02 first."
        )

    pages = json.loads(
        IN_PATH.read_text(encoding="utf-8")
    )

    records = []

    current = None

    def close_current():

        nonlocal current

        if current is None:
            return

        full_text = " ".join(
            p.strip()
            for p in current["text_parts"]
            if p and p.strip()
        ).strip()

        match_type = classify_match_type(full_text)

        # ------------------------------------------------------------------
        # ROBUST EXTRACTION
        # ------------------------------------------------------------------

        flat_codes = extract_gri_codes(full_text)

        structured_codes = extract_gri_structured_codes(full_text)

        # A useful semantic field for downstream code.
        if flat_codes:
            mapping_semantics = "gri_disclosure_mapping"
        elif match_type == "no_direct_linkage":
            mapping_semantics = "no_gri_mapping"
        elif match_type == "can_be_covered_by":
            mapping_semantics = "partial_gri_mapping"
        else:
            mapping_semantics = "unparsed_gri_mapping"

        # ------------------------------------------------------------------
        # RECORD
        # ------------------------------------------------------------------

        records.append({

            "brsr_id":
                current["brsr_id"],

            "match_type":
                match_type,

            "mapping_semantics":
                mapping_semantics,

            "gri_text_raw":
                full_text,

            # Rich structured form.
            #
            # This is intentionally retained because script 05 can use it
            # without reparsing free text.
            "gri_codes_json":
                json.dumps(
                    structured_codes,
                    ensure_ascii=False,
                ),

            # Flat form.
            #
            # This makes debugging and cross-framework comparison much
            # easier.
            "gri_requirement_codes_json":
                json.dumps(
                    flat_codes,
                    ensure_ascii=False,
                ),

            "source_page":
                current["source_page"],
        })

        current = None

    # ----------------------------------------------------------------------
    # WALK PDF TABLE
    # ----------------------------------------------------------------------

    for page in pages:

        page_number = page.get("page_number")

        if page_number is None:
            continue

        if not (
            PAGE_START <= page_number <= PAGE_END
        ):
            continue

        for row in page.get("grid", []):

            text = flatten_row(row)

            if not text:
                continue

            # Skip known headers.
            if HEADER_LINE_RE.match(text):
                continue

            # --------------------------------------------------------------
            # New BRSR record
            # --------------------------------------------------------------

            m = BRSR_ID_RE.match(text)

            if m:

                # Finish previous record.
                close_current()

                base_id = m.group(1)

                remainder = text[m.end():].strip()

                # ----------------------------------------------------------
                # Multi-part indicator
                #
                # Example:
                #
                # P3-E10 a GRI 403...
                #
                # becomes:
                #
                # P3-E10a
                # ----------------------------------------------------------

                subpart_match = SUBPART_RE.match(
                    remainder
                )

                if subpart_match:

                    subpart = (
                        subpart_match.group(1)
                        .lower()
                    )

                    brsr_id = (
                        f"{base_id}{subpart}"
                    )

                    remainder = (
                        remainder[
                            subpart_match.end():
                        ].strip()
                    )

                else:

                    brsr_id = base_id

                current = {
                    "brsr_id":
                        brsr_id,

                    "text_parts":
                        [remainder]
                        if remainder
                        else [],

                    "source_page":
                        page_number,
                }

                continue

            # --------------------------------------------------------------
            # Continuation row
            # --------------------------------------------------------------

            if current is not None:

                current["text_parts"].append(text)

            # Otherwise this is stray content before the first BRSR ID.
            # Ignore it deliberately.

    # Flush final record.
    close_current()

    # ----------------------------------------------------------------------
    # WRITE CSV
    # ----------------------------------------------------------------------

    OUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "brsr_id",
        "match_type",
        "mapping_semantics",
        "gri_text_raw",
        "gri_codes_json",
        "gri_requirement_codes_json",
        "source_page",
    ]

    with open(
        OUT_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(records)

    # ----------------------------------------------------------------------
    # DIAGNOSTICS
    # ----------------------------------------------------------------------

    by_type = {}

    for r in records:

        mt = r["match_type"]

        by_type[mt] = (
            by_type.get(mt, 0) + 1
        )

    records_with_codes = sum(
        bool(
            json.loads(
                r["gri_requirement_codes_json"]
            )
        )
        for r in records
    )

    print()
    print("=" * 70)
    print("TABLE 1 SUMMARY PARSER")
    print("=" * 70)

    print(
        f"Parsed {len(records)} Table 1 records"
    )

    print(
        f"Output: {OUT_PATH}"
    )

    print()
    print("MATCH TYPE BREAKDOWN")

    for k, v in sorted(by_type.items()):
        print(
            f"  {k:<25} {v}"
        )

    print()
    print("GRI CODE EXTRACTION")

    print(
        f"  Records containing GRI codes: "
        f"{records_with_codes}/{len(records)}"
    )

    print()
    print("ANCHOR CHECKS")

    anchors = [
        "A19",
        "P3-E10a",
        "P3-E10b",
        "P3-E10c",
        "P3-E10d",
        "P6-E1",
        "P6-E8",
        "P6-E9",
    ]

    by_id = {
        r["brsr_id"]: r
        for r in records
    }

    for bid in anchors:

        r = by_id.get(bid)

        print()
        print("-" * 70)
        print(bid)

        if r is None:
            print("NOT FOUND")
            continue

        print(
            "match_type:",
            r["match_type"]
        )

        print(
            "raw:",
            r["gri_text_raw"]
        )

        print(
            "codes:",
            r["gri_requirement_codes_json"]
        )

        print(
            "structured:",
            r["gri_codes_json"]
        )

    print()
    print("=" * 70)
    print(
        "IMPORTANT: inspect the anchor codes above before running script 05."
    )
    print("=" * 70)


if __name__ == "__main__":
    main()