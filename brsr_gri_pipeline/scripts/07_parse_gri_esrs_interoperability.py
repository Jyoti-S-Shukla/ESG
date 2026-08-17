"""
Parse the official GRI-ESRS Interoperability Index (GRI + EFRAG, Nov 2024).

Source:
    data/raw/esrs-gri-interoperability-index-november-2024.pdf

The official index uses a five-column table:

    GRI STANDARDS
    GRI DISCLOSURES AND REQUIREMENTS
    ESRS DISCLOSURE REQUIREMENTS
    NOTES
    EXPLANATION

Important source semantics:
- The index maps GRI disclosures / required datapoints to ESRS disclosure
  requirements at a granular level.
- If the GRI disclosure is not fully covered by ESRS, the covered GRI
  requirements are indicated in parentheses in the GRI column.
- Notes classify differences as granularity/data type, scope, or definition.
- Some GRI disclosures are not covered by the ESRS sustainability matters
  list, while some are covered through MDR-P/MDR-A/MDR-T and/or
  entity-specific metrics.

This script preserves the raw source text and creates structured fields
without pretending that all mappings are exact equivalences.

Output:
    data/interim/gri_esrs_interoperability.csv
"""

import csv
import json
import pathlib
import re
import sys

import pdfplumber


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

BASE = pathlib.Path(__file__).resolve().parents[1]

PDF_PATH = (
    BASE
    / "data"
    / "raw"
    / "esrs-gri-interoperability-index-november-2024.pdf"
)

OUT_PATH = (
    BASE
    / "data"
    / "interim"
    / "gri_esrs_interoperability.csv"
)


# ---------------------------------------------------------------------
# PDF table configuration
# ---------------------------------------------------------------------

# The official November 2024 PDF has:
#   pages 1-3  -> front matter / disclaimer
#   pages 4-24 -> interoperability table
#   page 25    -> notes legend
TABLE_START_PAGE = 4
TABLE_END_PAGE = 24


EXPECTED_HEADERS = [
    "GRI STANDARDS",
    "GRI DISCLOSURES AND REQUIREMENTS",
    "ESRS DISCLOSURE REQUIREMENTS",
    "NOTES",
    "EXPLANATION",
]


# ---------------------------------------------------------------------
# Regex
# ---------------------------------------------------------------------

# Examples:
#   2-1
#   2-23
#   101-2
#   403-10
DISCLOSURE_ID_RE = re.compile(
    r"^\s*(\d{1,3}-\d+)\b",
    re.IGNORECASE,
)

# Parenthesized covered requirements, e.g.
#
#   (2-3-a and 2-3-b)
#   (403-9-a-i, a-iii, b-i, b-iii, c-iii, d, e)
#
PARENTHETICAL_RE = re.compile(
    r"\(([^()]*(?:\([^()]*\)[^()]*)?)\)"
)

# Full GRI requirement token:
#   403-9-a-i
#   2-23-a-iv
#   101-2-c-i
FULL_GRI_REQ_RE = re.compile(
    r"\b\d{1,3}-\d+(?:-[a-z0-9]+)+\b",
    re.IGNORECASE,
)

# Relative requirement token:
#   a
#   b
#   a-i
#   c-iii
#   a-iv
#
# Used only inside the parenthesized qualification.
RELATIVE_REQ_RE = re.compile(
    r"(?<![\w-])"
    r"(?:[a-z](?:-[a-z0-9]+)*|[ivx]+)"
    r"(?![\w-])",
    re.IGNORECASE,
)

# Official index language indicating that a topic is not in the ESRS
# sustainability-matters list.
NOT_COVERED_RE = re.compile(
    r"this topic is not covered by the list of sustainability matters",
    re.IGNORECASE,
)

# Broader ESRS mapping language used by the official index.
BROADER_TOPIC_RE = re.compile(
    r"(?:"
    r"sustainability matter"
    r"|covered by MDR-P"
    r"|covered by MDR-A"
    r"|covered by MDR-T"
    r"|entity-specific metric"
    r")",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def clean_text(value):
    """Normalize PDF-extracted cell text while preserving content."""
    if value is None:
        return ""

    text = str(value)

    # PDF line breaks inside cells become spaces.
    text = text.replace("\r", " ")
    text = text.replace("\n", " ")

    # Collapse repeated whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def is_header_row(row):
    """Detect repeated table-header rows on subsequent PDF pages."""
    if not row:
        return False

    cells = [clean_text(x).upper() for x in row]

    joined = " | ".join(cells)

    return (
        "GRI STANDARDS" in joined
        and "ESRS DISCLOSURE" in joined
    )


def is_notes_legend_row(row):
    """
    Page 25 contains the Notes legend rather than data rows.
    """
    if not row:
        return False

    text = " ".join(clean_text(x) for x in row if x)

    return text.lower().startswith("notes legend")


def extract_disclosure_id(gri_disclosure_text):
    """
    Extract the disclosure-level identifier.

    Examples:
        '2-23 Policy commitments' -> '2-23'
        '403-9 Work-related injuries' -> '403-9'
        '101-2 Management of biodiversity impacts' -> '101-2'
    """
    m = DISCLOSURE_ID_RE.match(gri_disclosure_text)

    if not m:
        return None

    return m.group(1).upper()


def extract_parenthetical_text(gri_disclosure_text):
    """
    Extract the qualification inside parentheses.

    Example:
        403-9 Work-related injuries
        (403-9-a-i, a-iii, b-i, b-iii, c-iii, d, e)

    returns:
        '403-9-a-i, a-iii, b-i, b-iii, c-iii, d, e'
    """
    matches = PARENTHETICAL_RE.findall(gri_disclosure_text)

    if not matches:
        return None

    # Normally there is one qualification. Preserve all if there are
    # multiple parenthesized expressions.
    return " ; ".join(m.strip() for m in matches if m.strip())


def expand_parenthetical_requirements(
    parenthetical_text,
    disclosure_id,
):
    """
    Convert the official parenthetical notation into explicit
    GRI requirement identifiers.

    Examples:

        disclosure_id = 403-9
        '(403-9-a-i, a-iii, b-i, b-iii, c-iii, d, e)'

    becomes approximately:

        [
            '403-9-a-i',
            '403-9-a-iii',
            '403-9-b-i',
            '403-9-b-iii',
            '403-9-c-iii',
            '403-9-d',
            '403-9-e'
        ]

    The source notation sometimes omits the parent disclosure prefix
    after the first occurrence. We expand those relative tokens while
    retaining the raw source expression separately.
    """

    if not parenthetical_text or not disclosure_id:
        return []

    results = []

    # Process each parenthetical expression independently.
    expressions = [
        x.strip()
        for x in parenthetical_text.split(";")
        if x.strip()
    ]

    current_prefix = disclosure_id

    for expression in expressions:

        # -------------------------------------------------------------
        # First recover fully qualified identifiers.
        # -------------------------------------------------------------

        full_codes = FULL_GRI_REQ_RE.findall(expression)

        for code in full_codes:
            code = code.upper()

            if code not in results:
                results.append(code)

            # The most recent full code gives us the prefix for relative
            # tokens such as "a-iii".
            parts = code.split("-")

            if len(parts) >= 3:
                current_prefix = "-".join(parts[:2])

        # -------------------------------------------------------------
        # Remove fully qualified identifiers so that the remaining
        # relative tokens can be interpreted.
        # -------------------------------------------------------------

        remaining = FULL_GRI_REQ_RE.sub(" ", expression)

        # Ignore common connective words and bracket qualifiers.
        remaining = re.sub(
            r"\b(?:AND|TO|FOR|PUBLIC-INTEREST|ENTITIES?|ONLY|LISTED|"
            r"UNDERTAKINGS?)\b",
            " ",
            remaining,
            flags=re.IGNORECASE,
        )

        # Remove descriptive qualifier text such as:
        # "[for public-interest entities only]"
        remaining = re.sub(
            r"\[[^\]]*\]",
            " ",
            remaining,
        )

        # -------------------------------------------------------------
        # Relative tokens.
        #
        # We handle:
        #   a
        #   b
        #   a-i
        #   c-iii
        #
        # We deliberately do not try to infer complicated prose.
        # The raw parenthetical text is retained for auditability.
        # -------------------------------------------------------------

        tokens = RELATIVE_REQ_RE.findall(remaining)

        for token in tokens:
            token = token.strip().upper()

            if not token:
                continue

            # Ignore roman numerals that are likely fragments of prose
            # unless they are attached to a letter.
            if re.fullmatch(r"[IVX]+", token):
                continue

            code = f"{current_prefix}-{token}"

            if code not in results:
                results.append(code)

    # -----------------------------------------------------------------
    # A second, more reliable pass for the common official patterns.
    # This handles cases such as:
    #
    #   403-9-a-i, a-iii, b-i, b-iii, c-iii, d, e
    #
    # where the relative items all inherit 403-9.
    # -----------------------------------------------------------------

    if parenthetical_text:

        flat = parenthetical_text.replace(";", ",")

        # Find every full disclosure requirement and then inspect
        # subsequent relative tokens.
        full_matches = list(FULL_GRI_REQ_RE.finditer(flat))

        for match in full_matches:
            full_code = match.group(0).upper()

            if full_code not in results:
                results.append(full_code)

            prefix = "-".join(full_code.split("-")[:2])

            # Region until next full code or end.
            start = match.end()

            next_start = len(flat)
            for nxt in full_matches:
                if nxt.start() > start:
                    next_start = nxt.start()
                    break

            region = flat[start:next_start]

            relative_tokens = re.findall(
                r"(?<![\w-])"
                r"(?:[a-z](?:-[a-z0-9]+)*|[a-z])"
                r"(?![\w-])",
                region,
                flags=re.IGNORECASE,
            )

            for token in relative_tokens:
                token = token.strip().upper()

                if token in {"AND", "TO", "FOR"}:
                    continue

                if re.fullmatch(r"[A-Z]", token):
                    candidate = f"{prefix}-{token}"
                elif "-" in token and re.match(r"^[A-Z]-", token):
                    candidate = f"{prefix}-{token}"
                else:
                    # Standalone roman numeral such as "iii" is
                    # ambiguous without the immediately preceding
                    # letter. Do not fabricate a code here.
                    continue

                if candidate not in results:
                    results.append(candidate)

    return results


def classify_mapping(
    gri_disclosure_text,
    esrs_requirement_text,
    notes_text,
    explanation_text,
):
    """
    Conservative classification of the official GRI-ESRS mapping.

    Categories:

        direct
            Explicit ESRS disclosure requirement is given and the
            GRI disclosure is not restricted to selected sub-requirements.

        partial_requirement
            Parenthesized GRI requirements indicate that only selected
            requirements within the GRI disclosure are covered.

        broader_topic
            Coverage is expressed through sustainability matters,
            MDR-P/MDR-A/MDR-T and/or entity-specific metrics.

        not_covered
            Official text explicitly states that the topic is not
            covered by the ESRS sustainability-matter list.

        no_explicit_mapping
            The GRI disclosure is present in the index but the ESRS
            disclosure-requirement field is blank and the source does
            not explicitly say that the topic is not covered.

    Important:
        no_explicit_mapping is NOT a negative semantic label.
    """

    gri_text = clean_text(gri_disclosure_text)
    esrs_text = clean_text(esrs_requirement_text)
    notes = clean_text(notes_text)
    explanation = clean_text(explanation_text)

    combined = " ".join(
        x for x in [esrs_text, notes, explanation] if x
    )

    # -------------------------------------------------------------
    # Explicitly not covered by ESRS sustainability matters
    # -------------------------------------------------------------

    if NOT_COVERED_RE.search(combined):
        return "not_covered"

    # -------------------------------------------------------------
    # Explicit partial coverage
    # -------------------------------------------------------------

    if extract_parenthetical_text(gri_text):
        return "partial_requirement"

    # -------------------------------------------------------------
    # Broader topic-level / MDR coverage
    # -------------------------------------------------------------

    if BROADER_TOPIC_RE.search(combined):
        return "broader_topic"

    # -------------------------------------------------------------
    # Explicit ESRS mapping
    # -------------------------------------------------------------

    if esrs_text:
        return "direct"

    # -------------------------------------------------------------
    # GRI disclosure exists, but no explicit ESRS mapping is stated
    # -------------------------------------------------------------

    return "no_explicit_mapping"


def extract_esrs_references(esrs_text):
    """
    Extract ESRS standard/reference strings where they are explicitly
    present.

    This is deliberately conservative. The complete raw ESRS text is
    always retained in `esrs_requirement_raw`.

    Examples:
        ESRS S1 S1-3 §32 (b) and §33
        ESRS E1 E1-5 §37; §38
        ESRS G1 G1-4 §24 (a)
    """

    if not esrs_text:
        return []

    refs = []

    # ESRS topical / cross-cutting standard labels.
    pattern = re.compile(
        r"\bESRS\s+(?:1|2|E[1-5]|S[1-4]|G1)\b"
        r"(?:\s+[A-Z0-9]+(?:-[A-Z0-9]+)?)?",
        re.IGNORECASE,
    )

    for m in pattern.findall(esrs_text):
        x = clean_text(m)

        if x not in refs:
            refs.append(x)

    # Some cells contain references without repeating "ESRS", e.g.
    # "S1-3 §32..." after an initial ESRS S1.
    short_refs = re.findall(
        r"\b(?:E[1-5]|S[1-4]|G1|GOV-\d+|SBM-\d+|"
        r"IRO-\d+|MDR-[PATM])\b",
        esrs_text,
        flags=re.IGNORECASE,
    )

    for x in short_refs:
        x = x.upper()

        if x not in refs:
            refs.append(x)

    return refs


# ---------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------

def main():

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"{PDF_PATH} not found.\n"
            "Place the official November 2024 GRI-ESRS "
            "Interoperability Index PDF under data/raw/."
        )

    records = []

    current_gri_standard = None
    current_gri_standard_raw = None

    pages_processed = 0
    rows_seen = 0
    rows_skipped = 0

    with pdfplumber.open(PDF_PATH) as pdf:

        for page_number in range(
            TABLE_START_PAGE,
            TABLE_END_PAGE + 1,
        ):

            page = pdf.pages[page_number - 1]

            tables = page.extract_tables()

            if not tables:
                print(
                    f"WARNING: no table detected on PDF page "
                    f"{page_number}"
                )
                continue

            # The source has one principal five-column table per page.
            table = tables[0]

            pages_processed += 1

            for row_number, row in enumerate(table, start=1):

                rows_seen += 1

                if not row:
                    rows_skipped += 1
                    continue

                cells = [
                    clean_text(cell)
                    for cell in row
                ]

                # Ensure exactly five logical columns.
                while len(cells) < 5:
                    cells.append("")

                if len(cells) > 5:
                    cells = cells[:5]

                (
                    gri_standard_raw,
                    gri_disclosure_raw,
                    esrs_requirement_raw,
                    notes_raw,
                    explanation_raw,
                ) = cells

                # Skip repeated table header.
                if is_header_row(row):
                    rows_skipped += 1
                    continue

                if is_notes_legend_row(row):
                    rows_skipped += 1
                    continue

                # -----------------------------------------------------
                # Forward-fill merged GRI Standards cells.
                # -----------------------------------------------------

                if gri_standard_raw:
                    current_gri_standard_raw = gri_standard_raw
                    current_gri_standard = gri_standard_raw

                # If no GRI standard has yet been observed, this is
                # probably a malformed/extraneous row.
                if not current_gri_standard:
                    rows_skipped += 1
                    continue

                # A valid data row must have a GRI disclosure.
                if not gri_disclosure_raw:
                    rows_skipped += 1
                    continue

                disclosure_id = extract_disclosure_id(
                    gri_disclosure_raw
                )

                # The table may contain a continuation or unusual row
                # without a conventional disclosure identifier.
                # Preserve it rather than silently deleting it.
                if disclosure_id is None:
                    rows_skipped += 1
                    continue

                parenthetical_raw = extract_parenthetical_text(
                    gri_disclosure_raw
                )

                covered_requirements = (
                    expand_parenthetical_requirements(
                        parenthetical_raw,
                        disclosure_id,
                    )
                    if parenthetical_raw
                    else []
                )

                mapping_type = classify_mapping(
                    gri_disclosure_raw,
                    esrs_requirement_raw,
                    notes_raw,
                    explanation_raw,
                )

                esrs_refs = extract_esrs_references(
                    esrs_requirement_raw
                )

                records.append(
                    {
                        "source_page": page_number,
                        "source_row": row_number,

                        "gri_standard": current_gri_standard,
                        "gri_standard_raw": current_gri_standard_raw,

                        "gri_disclosure_id": disclosure_id,

                        "gri_disclosure_raw": gri_disclosure_raw,

                        "gri_covered_requirements_raw": (
                            parenthetical_raw or ""
                        ),

                        "gri_covered_requirements_json": json.dumps(
                            covered_requirements,
                            ensure_ascii=False,
                        ),

                        "esrs_requirement_raw": (
                            esrs_requirement_raw
                        ),

                        "esrs_references_json": json.dumps(
                            esrs_refs,
                            ensure_ascii=False,
                        ),

                        "notes_raw": notes_raw,

                        "explanation_raw": explanation_raw,

                        "mapping_type": mapping_type,
                    }
                )

    # -----------------------------------------------------------------
    # Write output
    # -----------------------------------------------------------------

    OUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "source_page",
        "source_row",

        "gri_standard",
        "gri_standard_raw",

        "gri_disclosure_id",
        "gri_disclosure_raw",

        "gri_covered_requirements_raw",
        "gri_covered_requirements_json",

        "esrs_requirement_raw",
        "esrs_references_json",

        "notes_raw",
        "explanation_raw",

        "mapping_type",
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

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    print(
        f"Parsed {len(records)} GRI-ESRS interoperability rows "
        f"-> {OUT_PATH}"
    )

    print(
        f"  PDF pages processed: {pages_processed}"
    )

    print(
        f"  Table rows seen: {rows_seen}"
    )

    print(
        f"  Rows skipped: {rows_skipped}"
    )

    print("\nMapping type breakdown:")

    counts = {}

    for record in records:
        key = record["mapping_type"]
        counts[key] = counts.get(key, 0) + 1

    for key, value in sorted(counts.items()):
        print(
            f"  {key}: {value}"
        )

    print("\nGRI standards represented:")

    standards = sorted(
        {
            r["gri_standard"]
            for r in records
            if r["gri_standard"]
        }
    )

    for standard in standards:
        print(
            f"  {standard}"
        )

    print(
        "\nIMPORTANT: inspect partial mappings and broader_topic "
        "mappings against the source PDF before treating them as "
        "training positives."
    )


if __name__ == "__main__":
    main()