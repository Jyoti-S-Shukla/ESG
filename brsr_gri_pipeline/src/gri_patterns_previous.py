"""
Regex patterns for pulling structured signal out of the linkage document's
free-text cells. Keep these centralized so 03/04/05 stay in sync.
"""

import re

# Matches a GRI standard declaration followed by one or more
# disclosure identifiers.
#
# Examples:
#   GRI 302: Energy 2016 Disclosure 302-1-a; 302-1-b
#   GRI 2: General Disclosures 2021 Disclosures 2-3-a, 2-3-b
#
# IMPORTANT:
# The disclosure part is deliberately tokenized as identifiers rather
# than allowing arbitrary letters/whitespace. This prevents wrapped
# disclosure titles from being captured as GRI codes.

GRI_STANDARD_RE = re.compile(
    r"GRI\s+(\d{1,3}):\s*"
    r"([A-Za-z0-9 ,&\-/]+?)\s+"
    r"(\d{4})\s*"
    r"(?:Topic management disclosures\s*)?"
    r"Disclosures?\s+"
    r"((?:\d{1,3}-\d+(?:-[A-Za-z0-9]+)*"
    r"(?:\s*[,;]\s*|\s*$|(?=\s+GRI\s))"
    r")+)",
    re.IGNORECASE,
)

# Table 1's composite BRSR IDs, e.g. "A17a", "P6-E1", "P3-L4"
BRSR_ID_RE = re.compile(r"^([AB]\d{1,2}[a-c]?|P[1-9]-[EL]\d{1,2}[a-d]?)$")

NO_LINKAGE_RE = re.compile(r"no\s+direct\s+linkage", re.IGNORECASE)
CAN_BE_COVERED_RE = re.compile(r"can\s+be\s+covered\s+by\s*-?\s*", re.IGNORECASE)

MATCH_TYPE_DIRECT = "direct"
MATCH_TYPE_PARTIAL = "can_be_covered_by"
MATCH_TYPE_NONE = "no_direct_linkage"


def classify_match_type(cell_text: str) -> str:
    """Classify a GRI-column cell from Table 1/2 into a match type."""
    if not cell_text or not cell_text.strip():
        return MATCH_TYPE_NONE
    if NO_LINKAGE_RE.search(cell_text):
        return MATCH_TYPE_NONE
    if CAN_BE_COVERED_RE.search(cell_text):
        return MATCH_TYPE_PARTIAL
    return MATCH_TYPE_DIRECT


def extract_gri_codes(cell_text: str):
    """
    Pull out individual GRI disclosure codes.

    Example:
        GRI 302: Energy 2016 Disclosure 302-1-a; 302-1-b

    returns:
        [
            {
                "standard": "302",
                "standard_name": "Energy",
                "year": "2016",
                "disclosures": ["302-1-a", "302-1-b"]
            }
        ]

    The parser intentionally extracts identifier-shaped tokens only.
    Text belonging to the disclosure title is not treated as part of
    the GRI code.
    """

    results = []

    if not cell_text:
        return results

    for m in GRI_STANDARD_RE.finditer(cell_text):

        standard_no, standard_name, year, disclosure_blob = m.groups()

        # Extract only valid identifier-shaped tokens.
        codes = re.findall(
            r"\b\d{1,3}-\d+(?:-[A-Za-z0-9]+)*\b",
            disclosure_blob,
        )

        # Remove duplicates while preserving order.
        codes = list(dict.fromkeys(codes))

        if not codes:
            continue

        results.append({
            "standard": standard_no,
            "standard_name": standard_name.strip(),
            "year": year,
            "disclosures": codes,
        })

    return results
