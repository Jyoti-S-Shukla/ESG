"""
04_parse_comprehensive_table.py

Parse Table 2 (Comprehensive Table) from pages_raw.json.

IMPORTANT DESIGN DECISIONS
--------------------------
Table 2 does NOT reliably contain explicit BRSR IDs. Therefore IDs are
reconstructed from:

    Section
    Principle
    Indicator type (Essential / Leadership)
    Sl.No

Do NOT use Table 1 as an ID whitelist. Table 1 and Table 2 have different
granularities, and some Table 1 IDs may legitimately have no exact Table 2
counterpart.

The parser preserves the physical four-column structure:

    Sl.No | BRSR text | GRI text | Remarks

Continuation rows are appended to the corresponding column only.

Table 2 can contain a parent indicator where Table 1 has multiple subparts.
For example, Table 1 may contain:

    P3-E10a
    P3-E10b
    P3-E10c
    P3-E10d

while Table 2 may contain one reconstructed:

    P3-E10

That is NOT a parsing error. The later merge/normalisation stage must handle
parent/sub-indicator relationships.

Outputs
-------
data/interim/table2_comprehensive.csv
data/interim/table2_unparsed.json
"""

import csv
import json
import pathlib
import re
import sys
from collections import Counter, defaultdict

# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.gri_patterns import classify_match_type, extract_gri_codes
from src.brsr_ids import ParserState


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

BASE = pathlib.Path(__file__).resolve().parents[1]

IN_PATH = BASE / "data" / "interim" / "pages_raw.json"

OUT_CSV = BASE / "data" / "interim" / "table2_comprehensive.csv"

OUT_UNPARSED = (
    BASE / "data" / "interim" / "table2_unparsed.json"
)


# ---------------------------------------------------------------------
# Header / structural patterns
# ---------------------------------------------------------------------

SECTION_A_RE = re.compile(
    r"Section\s+A\s*:\s*General\s+Disclosures",
    re.IGNORECASE,
)

SECTION_B_RE = re.compile(
    r"Section\s+B\s*:\s*Management\s+and\s+Process",
    re.IGNORECASE,
)

SECTION_C_RE = re.compile(
    r"Section\s+C\s*:\s*Principle",
    re.IGNORECASE,
)

PRINCIPLE_RE = re.compile(
    r"\bPRINCIPLE\s+(\d+)\b",
    re.IGNORECASE,
)

ESSENTIAL_RE = re.compile(
    r"^\s*Essential\s+Indicators\b",
    re.IGNORECASE,
)

LEADERSHIP_RE = re.compile(
    r"^\s*Leadership\s+Indicators\b",
    re.IGNORECASE,
)

TABLE_TITLE_RE = re.compile(
    r"Table\s+2\s*:\s*Comprehensive\s+Table",
    re.IGNORECASE,
)

COLUMN_HEADER_RE = re.compile(
    r"^(Sl\.?\s*No|BRSR\s+Code|BRSR\s+Indicator|"
    r"GRI\s+Standards\s+and\s+Disclosures|Remarks)$",
    re.IGNORECASE,
)

# FIX (Phase 0, header-leak): the repeating running header splits into
# cells like ["", "SEBI - BRSR Framework", "GRI Standards and Disclosures",
# "Remarks"] on every new page. COLUMN_HEADER_RE above only matches when
# the WHOLE joined row equals one exact phrase, which this row never does
# (it's three different phrases concatenated) -- so it fell through to the
# continuation branch and got silently appended to whatever record was
# still open. This checks each cell independently instead.
HEADER_CELL_RE = re.compile(
    r"^(Sl\.?\s*No|SEBI\s*-\s*BRSR\s+Framework|BRSR\s+Code|"
    r"BRSR\s+Indicator|GRI\s+Standards\s+and\s+Disclosures|Remarks)$",
    re.IGNORECASE,
)

# Defensive second layer: strip these exact fragments if one still ends up
# glued onto real content on the same physical line (belt-and-suspenders,
# in case a header fragment shares a row with genuine data in some edge
# case the cell-level check above doesn't catch).
HEADER_FRAGMENT_STRIP_RE = re.compile(
    r"\s*(SEBI\s*-\s*BRSR\s+Framework|GRI\s+Standards\s+and\s+Disclosures|"
    r"Remarks)\s*$",
    re.IGNORECASE,
)

# Flag (not fix) remarks that mention a BRSR ID from a DIFFERENT principle
# than the record currently being assembled -- a strong signal that two
# records' content bled together across a missed record boundary. This
# makes contamination visible/countable instead of silently trusting it.
BRSR_ID_MENTION_RE = re.compile(r"\(?P(\d)[\s-]?[EL]\d{1,2}[a-d]?\)?", re.IGNORECASE)

# Arabic Sl.No = actual data row.
SL_NO_RE = re.compile(r"^\d{1,3}$")

# Roman numerals are organisational headings inside Section A/B.
ROMAN_SUBHEADING_RE = re.compile(
    r"^[IVXLC]{1,6}$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def clean_cell(value):
    """Normalise a grid cell without destroying meaningful text."""
    if value is None:
        return ""
    return str(value).strip()


def normalise_whitespace(text):
    """Collapse repeated whitespace while preserving content."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def row_to_text(row):
    """Flatten a row ONLY for header detection / diagnostics."""
    return normalise_whitespace(
        " ".join(x for x in row if x)
    )


def safe_json(value):
    """
    Parse a JSON field safely.

    Returns [] for empty/malformed values rather than crashing the pipeline.
    """
    if not value:
        return []

    try:
        obj = json.loads(value)
        return obj if isinstance(obj, list) else []
    except Exception:
        return []


def infer_mapping_semantics(match_type, gri_text, remarks):
    """
    Derive a coarse semantic label for Table 2.

    This is intentionally conservative.

    direct:
        Explicit GRI linkage / disclosure.

    no_direct_linkage:
        Explicit statement that no direct linkage exists.

    partial:
        Only used if Table 2 explicitly contains partial/coverage language.

    none:
        Could not infer anything beyond the classifier.
    """

    text = normalise_whitespace(
        f"{gri_text} {remarks}"
    ).lower()

    if match_type == "no_direct_linkage":
        return "no_direct_linkage"

    partial_patterns = [
        "partially covered",
        "partial coverage",
        "partially covers",
        "partially covered by",
        "partially addressed",
        "not fully covered",
        "does not fully cover",
        "only partially",
    ]

    if any(p in text for p in partial_patterns):
        return "partial"

    if match_type == "direct":
        return "direct"

    return "none"


# ---------------------------------------------------------------------
# Parser state
# ---------------------------------------------------------------------

def update_state_from_row_text(
    state: ParserState,
    row_text: str,
):
    """
    Update parser state from section/principle/indicator headers.

    Returns True if the row is structural/header content and should NOT
    be treated as a data row.

    Crucial point:
    Repeated running headers on subsequent PDF pages must NOT reset the
    Sl.No counter.
    """

    text = normalise_whitespace(row_text)

    if not text:
        return False

    # -------------------------------------------------------------
    # Table title / column headers
    # -------------------------------------------------------------

    if TABLE_TITLE_RE.search(text):
        return True

    if COLUMN_HEADER_RE.match(text):
        return True

    # -------------------------------------------------------------
    # Section
    # -------------------------------------------------------------

    if SECTION_A_RE.search(text):
        if state.section != "A":
            state.section = "A"
            state.principle = None
            state.indicator_type = None
            state.reset_counter()
        return True

    if SECTION_B_RE.search(text):
        if state.section != "B":
            state.section = "B"
            state.principle = None
            state.indicator_type = None
            state.reset_counter()
        return True

    if SECTION_C_RE.search(text):
        if state.section != "C":
            state.section = "C"
            state.principle = None
            state.indicator_type = None
            state.reset_counter()
        return True

    # -------------------------------------------------------------
    # Principle
    # -------------------------------------------------------------

    m = PRINCIPLE_RE.search(text)

    if m:
        new_principle = int(m.group(1))

        if state.principle != new_principle:
            state.principle = new_principle
            state.indicator_type = None

        return True

    # -------------------------------------------------------------
    # Indicator type
    # -------------------------------------------------------------

    if ESSENTIAL_RE.match(text):
        if state.indicator_type != "E":
            state.indicator_type = "E"
            state.reset_counter()
        return True

    if LEADERSHIP_RE.match(text):
        if state.indicator_type != "L":
            state.indicator_type = "L"
            state.reset_counter()
        return True

    return False


# ---------------------------------------------------------------------
# State diagnostics
# ---------------------------------------------------------------------

def state_snapshot(state):
    return {
        "section": getattr(state, "section", None),
        "principle": getattr(state, "principle", None),
        "indicator_type": getattr(state, "indicator_type", None),
    }


# ---------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------

def main():

    pages = json.loads(
        IN_PATH.read_text(encoding="utf-8")
    )

    state = ParserState()

    records = []
    unparsed = []

    current = None

    # -------------------------------------------------------------
    # Close current record
    # -------------------------------------------------------------

    def close_current():

        nonlocal current

        if current is None:
            return

        brsr_text = normalise_whitespace(
            " ".join(
                p for p in current["brsr_parts"]
                if p
            )
        )

        gri_text = normalise_whitespace(
            " ".join(
                p for p in current["gri_parts"]
                if p
            )
        )

        remarks = normalise_whitespace(
            " ".join(
                p for p in current["remarks_parts"]
                if p
            )
        )

        # ---------------------------------------------------------
        # FIX (Phase 0, header-leak): defensive second-layer strip.
        # The cell-level skip above should catch nearly all header
        # leakage at the source; this is the belt-and-suspenders pass
        # in case a fragment still made it into the accumulated text.
        # ---------------------------------------------------------

        brsr_text = HEADER_FRAGMENT_STRIP_RE.sub("", brsr_text).strip()
        gri_text = HEADER_FRAGMENT_STRIP_RE.sub("", gri_text).strip()
        remarks = HEADER_FRAGMENT_STRIP_RE.sub("", remarks).strip()

        # ---------------------------------------------------------
        # FIX (Phase 0, contamination flag): if remarks mentions a BRSR
        # ID from a DIFFERENT principle than this record, that's a
        # strong signal content from a neighbouring record bled in
        # across a missed boundary (this is what happened to P6-E1's
        # remarks, which contained a P3-L5 discussion). We do NOT try
        # to auto-split -- too risky to guess right -- we just make it
        # visible in table2_unparsed.json instead of silently trusting it.
        # ---------------------------------------------------------

        for m in BRSR_ID_MENTION_RE.finditer(remarks):
            mentioned_principle = int(m.group(1))
            if current["principle"] is not None and mentioned_principle != current["principle"]:
                unparsed.append({
                    "page": current["page"],
                    "brsr_id": current["brsr_id"],
                    "reason": (
                        f"remarks mentions {m.group(0)} (Principle "
                        f"{mentioned_principle}) but this record belongs to "
                        f"Principle {current['principle']} -- likely "
                        f"cross-record contamination, not auto-fixed"
                    ),
                    "remarks_excerpt": remarks[:200],
                })

        # ---------------------------------------------------------
        # Extract GRI codes
        # ---------------------------------------------------------

        try:
            gri_codes = extract_gri_codes(gri_text)
        except Exception as exc:
            gri_codes = []

            unparsed.append({
                "page": current["page"],
                "brsr_id": current["brsr_id"],
                "reason": (
                    f"GRI extraction failed: {repr(exc)}"
                ),
                "gri_text_raw": gri_text,
            })

        # ---------------------------------------------------------
        # Classify mapping
        # ---------------------------------------------------------

        try:
            match_type = classify_match_type(gri_text)
        except Exception as exc:
            match_type = "unknown"

            unparsed.append({
                "page": current["page"],
                "brsr_id": current["brsr_id"],
                "reason": (
                    f"match classification failed: {repr(exc)}"
                ),
                "gri_text_raw": gri_text,
            })

        mapping_semantics = infer_mapping_semantics(
            match_type=match_type,
            gri_text=gri_text,
            remarks=remarks,
        )

        records.append({
            "brsr_id": current["brsr_id"],
            "section": current["section"],
            "principle": current["principle"],
            "indicator_type": current["indicator_type"],
            "brsr_text": brsr_text,
            "gri_text_raw": gri_text,
            "gri_codes_json": json.dumps(
                gri_codes,
                ensure_ascii=False,
            ),
            "match_type": match_type,
            "mapping_semantics": mapping_semantics,
            "remarks": remarks,
            "source_page": current["page"],
        })

        current = None

    # -----------------------------------------------------------------
    # Iterate pages
    # -----------------------------------------------------------------

    for page in pages:

        # Only Table 2 pages.
        if page.get("table") != "table2":
            continue

        page_number = page.get("page_number")

        grid = page.get("grid", [])

        for raw_row in grid:

            row = [
                clean_cell(c)
                for c in raw_row
            ]

            # Guarantee four columns.
            if len(row) < 4:
                row.extend(
                    [""] * (4 - len(row))
                )

            # If extraction generated more than four cells, preserve
            # the first four physical columns. This is consistent with
            # script 02's fixed Table-2 column boundaries.
            row = row[:4]

            sl_no_cell = row[0]
            brsr_cell = row[1]
            gri_cell = row[2]
            remarks_cell = row[3]

            row_text = row_to_text(row)

            if not row_text:
                continue

            # ---------------------------------------------------------
            # Header / state rows
            # ---------------------------------------------------------

            if update_state_from_row_text(
                state,
                row_text,
            ):
                continue

            # ---------------------------------------------------------
            # Roman organisational subheading
            # ---------------------------------------------------------

            if ROMAN_SUBHEADING_RE.match(
                sl_no_cell
            ):
                continue

            # ---------------------------------------------------------
            # FIX (Phase 0, header-leak): running header row, cell-level
            # check. Skip entirely -- do NOT let it reach the
            # continuation branch below, which is what was silently
            # appending "SEBI - BRSR Framework" / "GRI Standards and
            # Disclosures" / "Remarks" onto open records.
            # ---------------------------------------------------------

            if (
                HEADER_CELL_RE.match(sl_no_cell)
                or HEADER_CELL_RE.match(brsr_cell)
                or HEADER_CELL_RE.match(gri_cell)
                or HEADER_CELL_RE.match(remarks_cell)
            ):
                continue

            # ---------------------------------------------------------
            # Actual data row
            # ---------------------------------------------------------

            if SL_NO_RE.match(sl_no_cell):

                # FIX (Phase 0, ID-drift root cause): the source PDF
                # re-prints the SAME Sl.No at the top of the next page
                # whenever one item's content wraps across a page break
                # (verified: Sl.No 3, 6, 8, 11 all repeat this way in the
                # Principle 6 Essential Indicators span, inflating 12 real
                # items to 16 reconstructed and fragmenting their content).
                # If this Sl.No matches the currently open record's own
                # Sl.No, it's a continuation of the SAME item, not a new
                # one -- do not close/reopen, do not advance the counter.
                if current is not None and current.get("raw_sl_no") == sl_no_cell:
                    if brsr_cell:
                        current["brsr_parts"].append(brsr_cell)
                    if gri_cell:
                        current["gri_parts"].append(gri_cell)
                    if remarks_cell:
                        current["remarks_parts"].append(remarks_cell)
                    continue

                # Finish previous record.
                close_current()

                # -----------------------------------------------------
                # Reconstruct BRSR ID.
                # -----------------------------------------------------

                try:
                    brsr_id = state.next_id()
                except Exception as exc:

                    unparsed.append({
                        "page": page_number,
                        "row": row,
                        "state": state_snapshot(state),
                        "reason": (
                            "Could not reconstruct BRSR ID: "
                            f"{repr(exc)}"
                        ),
                    })

                    current = None
                    continue

                current = {
                    "brsr_id": brsr_id,
                    "raw_sl_no": sl_no_cell,
                    "section": state.section,
                    "principle": state.principle,
                    "indicator_type": state.indicator_type,
                    "brsr_parts": (
                        [brsr_cell]
                        if brsr_cell
                        else []
                    ),
                    "gri_parts": (
                        [gri_cell]
                        if gri_cell
                        else []
                    ),
                    "remarks_parts": (
                        [remarks_cell]
                        if remarks_cell
                        else []
                    ),
                    "page": page_number,
                }

                continue

            # ---------------------------------------------------------
            # Continuation row
            # ---------------------------------------------------------

            if current is None:

                # Ignore completely blank structural rows.
                if not (
                    brsr_cell
                    or gri_cell
                    or remarks_cell
                ):
                    continue

                unparsed.append({
                    "page": page_number,
                    "row": row,
                    "state": state_snapshot(state),
                    "reason": (
                        "Non-empty continuation row encountered "
                        "without an open Table-2 record."
                    ),
                })

                continue

            # IMPORTANT:
            # Never flatten continuation content across columns.
            if brsr_cell:
                current["brsr_parts"].append(
                    brsr_cell
                )

            if gri_cell:
                current["gri_parts"].append(
                    gri_cell
                )

            if remarks_cell:
                current["remarks_parts"].append(
                    remarks_cell
                )

    # -----------------------------------------------------------------
    # Flush final record
    # -----------------------------------------------------------------

    close_current()

    # -----------------------------------------------------------------
    # Write outputs
    # -----------------------------------------------------------------

    OUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "brsr_id",
        "section",
        "principle",
        "indicator_type",
        "brsr_text",
        "gri_text_raw",
        "gri_codes_json",
        "match_type",
        "mapping_semantics",
        "remarks",
        "source_page",
    ]

    with open(
        OUT_CSV,
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

    OUT_UNPARSED.write_text(
        json.dumps(
            unparsed,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # -----------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------

    match_counts = Counter(
        r["match_type"]
        for r in records
    )

    semantic_counts = Counter(
        r["mapping_semantics"]
        for r in records
    )

    id_counts = Counter(
        r["brsr_id"]
        for r in records
    )

    duplicates = {
        k: v
        for k, v in id_counts.items()
        if v > 1
    }

    ids_with_codes = sum(
        bool(safe_json(r["gri_codes_json"]))
        for r in records
    )

    explicit_id_count = 0
    # Table 2 reconstruction is intentional; therefore this diagnostic
    # should NOT be interpreted as a requirement for explicit IDs.
    # We retain it only to make the design obvious.
    explicit_id_count = 0

    print("=" * 70)
    print("TABLE 2 COMPREHENSIVE TABLE PARSER")
    print("=" * 70)

    print(
        f"Parsed {len(records)} Table 2 records"
    )

    print(
        f"Unique BRSR IDs: {len(id_counts)}"
    )

    print(
        f"Records containing GRI codes: "
        f"{ids_with_codes}/{len(records)}"
    )

    print("\nMATCH TYPE BREAKDOWN")

    for key, value in sorted(
        match_counts.items()
    ):
        print(
            f"  {key:<25} {value}"
        )

    print("\nMAPPING SEMANTICS BREAKDOWN")

    for key, value in sorted(
        semantic_counts.items()
    ):
        print(
            f"  {key:<25} {value}"
        )

    print(
        f"\nRows requiring review: "
        f"{len(unparsed)}"
    )

    print(
        f"Duplicate BRSR IDs: "
        f"{len(duplicates)}"
    )

    if duplicates:
        print("\nDUPLICATE IDS:")
        for k, v in sorted(
            duplicates.items()
        ):
            print(
                f"  {k}: {v} records"
            )

    print("\nOutput CSV:")
    print(f"  {OUT_CSV}")

    print("\nUnparsed / suspicious rows:")
    print(f"  {OUT_UNPARSED}")

    # -----------------------------------------------------------------
    # Anchor checks
    # -----------------------------------------------------------------

    by_id = defaultdict(list)

    for r in records:
        by_id[r["brsr_id"]].append(r)

    anchors = [
        "A19",
        "P3-E10",
        "P3-E10a",
        "P3-E10b",
        "P3-E10c",
        "P3-E10d",
        "P6-E1",
        "P6-E8",
        "P6-E9",
    ]

    print("\n" + "=" * 70)
    print("ANCHOR CHECKS")
    print("=" * 70)

    for bid in anchors:

        print("\n" + "-" * 70)
        print(bid)

        rows = by_id.get(bid, [])

        if not rows:
            print("NOT FOUND")
            continue

        for r in rows:

            print(
                "brsr_text:"
            )
            print(
                repr(r["brsr_text"])
            )

            print(
                "gri_text_raw:"
            )
            print(
                repr(r["gri_text_raw"])
            )

            print(
                "gri_codes_json:"
            )
            print(
                r["gri_codes_json"]
            )

            print(
                "match_type:"
            )
            print(
                r["match_type"]
            )

            print(
                "mapping_semantics:"
            )
            print(
                r["mapping_semantics"]
            )

            print(
                "remarks:"
            )
            print(
                repr(r["remarks"])
            )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()