"""
Robust parsing utilities for the official SEBI BRSR -> GRI linkage PDF.

Design principle:
    DO NOT try to parse an entire GRI citation with one regex.

The official document contains several syntactic variants, e.g.:

    GRI 405: Diversity and Equal Opportunity 2016
    Disclosure 405-1-a-I; 405-1-b-i

    GRI 403: Occupational Health and Safety 2018
    Disclosure 403-1-a, 403-1-b

    GRI 305: Emissions 2016
    Disclosure 305-4 GHG emissions intensity a. ...

Therefore extraction is performed in two independent stages:

    1. identify GRI standard mentions
    2. identify disclosure identifiers

The raw source text is always preserved.
"""

import re


# ---------------------------------------------------------------------
# Match types
# ---------------------------------------------------------------------

MATCH_TYPE_DIRECT = "direct"
MATCH_TYPE_PARTIAL = "can_be_covered_by"
MATCH_TYPE_NONE = "no_direct_linkage"


NO_LINKAGE_RE = re.compile(
    r"\bno\s+direct\s+linkage\b",
    re.IGNORECASE,
)

CAN_BE_COVERED_RE = re.compile(
    r"\bcan\s+be\s+covered\s+by\b",
    re.IGNORECASE,
)


def classify_match_type(cell_text: str) -> str:
    """
    Classify the linkage text.

    Priority:
        no direct linkage
        can be covered by
        otherwise direct
    """
    if not cell_text or not cell_text.strip():
        return MATCH_TYPE_NONE

    if NO_LINKAGE_RE.search(cell_text):
        return MATCH_TYPE_NONE

    if CAN_BE_COVERED_RE.search(cell_text):
        return MATCH_TYPE_PARTIAL

    return MATCH_TYPE_DIRECT


# ---------------------------------------------------------------------
# GRI standard extraction
# ---------------------------------------------------------------------

GRI_STANDARD_RE = re.compile(
    r"""
    \bGRI\s+
    (?P<number>\d{1,3})
    \s*:\s*
    (?P<name>.*?)
    \s+
    (?P<year>20\d{2})
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)


# ---------------------------------------------------------------------
# Disclosure-code extraction
# ---------------------------------------------------------------------

"""
A GRI disclosure identifier generally looks like:

    2-23
    2-23-a
    2-23-a-iv
    302-1
    302-1-a
    302-1-c-I
    403-10-a
    403-10-b-ii
    3-3-d-i-iii
    305-4-a

We intentionally do NOT require the word "Disclosure" immediately
before the identifier.

This is important because the official PDF contains variants such as:

    Disclosure 305-4 GHG emissions intensity a. ...

and wrapped/merged text can alter the immediate context.
"""

DISCLOSURE_CODE_RE = re.compile(
    r"""
    (?<![\w.-])
    (?P<code>
        \d{1,3}
        -
        \d+
        (?:
            -
            [A-Za-z]+
        )*
    )
    (?![\w.-])
    """,
    re.VERBOSE,
)


def normalize_code(code: str) -> str:
    """
    Normalize a GRI disclosure identifier.

    Examples:
        405-1-a-I      -> 405-1-A-I
        403-10-b-ii    -> 403-10-B-II
        305-1 a        -> handled separately by tokenization
    """
    if not code:
        return ""

    parts = code.strip().split("-")

    normalized = [parts[0], parts[1]]

    for p in parts[2:]:
        normalized.append(p.upper())

    return "-".join(normalized)


def extract_disclosure_codes(text: str):
    """
    Extract all disclosure identifiers from arbitrary GRI linkage text.

    This is deliberately independent from GRI standard-name parsing.

    Returns:
        list[str]

    Example:
        'GRI 403 ... Disclosure 403-1-a, 403-1-b'
        ->
        ['403-1-A', '403-1-B']
    """
    if not text:
        return []

    codes = []

    for match in DISCLOSURE_CODE_RE.finditer(text):
        code = normalize_code(match.group("code"))

        if code and code not in codes:
            codes.append(code)

    return codes


def extract_gri_standards(text: str):
    """
    Extract GRI standard metadata.

    Returns:
        [
            {
                "standard": "405",
                "standard_name": "Diversity and Equal Opportunity",
                "year": "2016"
            }
        ]
    """
    if not text:
        return []

    results = []

    for match in GRI_STANDARD_RE.finditer(text):
        item = {
            "standard": match.group("number"),
            "standard_name": match.group("name").strip(" ,;"),
            "year": match.group("year"),
        }

        if item not in results:
            results.append(item)

    return results


def extract_gri_codes(text: str):
    """
    Main backward-compatible extraction function.

    Returns the same broad structure expected by scripts 03/04:

        [
            {
                "standard": "403",
                "standard_name": "Occupational Health and Safety",
                "year": "2018",
                "disclosures": [
                    "403-1-A",
                    "403-1-B"
                ]
            }
        ]

    Important:
        If standard parsing fails but disclosure parsing succeeds,
        the disclosures are STILL returned.

    This prevents a malformed standard-name match from destroying
    otherwise valid supervision signals.
    """

    if not text:
        return []

    standards = extract_gri_standards(text)
    all_codes = extract_disclosure_codes(text)

    if not all_codes:
        return []

    results = []

    # -------------------------------------------------------------
    # If we can identify GRI standards, associate codes by their
    # numeric prefix.
    # -------------------------------------------------------------

    for standard in standards:
        prefix = standard["standard"] + "-"

        codes_for_standard = [
            c for c in all_codes
            if c.startswith(prefix)
        ]

        if codes_for_standard:
            results.append({
                "standard": standard["standard"],
                "standard_name": standard["standard_name"],
                "year": standard["year"],
                "disclosures": codes_for_standard,
            })

    # -------------------------------------------------------------
    # Codes whose parent standard was not parsed.
    #
    # Do NOT discard them.
    # -------------------------------------------------------------

    assigned = set()

    for item in results:
        assigned.update(item["disclosures"])

    unassigned = [
        c for c in all_codes
        if c not in assigned
    ]

    for code in unassigned:
        standard_number = code.split("-")[0]

        results.append({
            "standard": standard_number,
            "standard_name": "",
            "year": "",
            "disclosures": [code],
        })

    return results


def extract_gri_code_set(text: str):
    """
    Convenience function for QA/merge scripts.

    Returns:
        set of normalized disclosure IDs.
    """
    return set(extract_disclosure_codes(text))