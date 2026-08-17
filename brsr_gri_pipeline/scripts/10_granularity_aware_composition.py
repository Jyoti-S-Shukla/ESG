"""
SCRIPT 10 — FINAL TRANSITIVE DATASET CONSTRUCTION

Constructs the final BRSR -> GRI -> ESRS graph with explicit
precision/recall separation.

Outputs:

    data/processed/transitive_candidates.csv
        High-recall candidate graph.

    data/processed/transitive_gold.csv
        High-precision graph suitable for training/evaluation.

    data/processed/transitive_review.csv
        Ambiguous / high-expansion mappings requiring review.

Important:
    These three datasets must NOT be conflated.

    candidates = retrieval space
    gold       = supervised positives
    review     = uncertain evidence

BRSR -> GRI:
    Table 1 is primary evidence.
    Table 2 is used only when Table 1 has no code.

GRI -> ESRS:
    The official EFRAG/GRI datapoint workbook is primary.
    The interoperability index is used as supporting evidence.
"""

import csv
import json
import pathlib
import re
from collections import Counter, defaultdict


# ================================================================
# PATHS
# ================================================================

BASE = pathlib.Path(__file__).resolve().parents[1]

BRSR_PATH = (
    BASE / "data" / "processed" / "gold_pairs.csv"
)

GRI_ESRS_PATH = (
    BASE / "data" / "interim"
    / "gri_esrs_datapoint_mapping.csv"
)

INTEROP_PATH = (
    BASE / "data" / "interim"
    / "gri_esrs_interoperability.csv"
)

CANDIDATE_OUT = (
    BASE / "data" / "processed"
    / "transitive_candidates.csv"
)

GOLD_OUT = (
    BASE / "data" / "processed"
    / "transitive_gold.csv"
)

REVIEW_OUT = (
    BASE / "data" / "processed"
    / "transitive_review.csv"
)


# ================================================================
# CONFIGURATION
# ================================================================

# Maximum expansion allowed for a high-confidence gold edge.
GOLD_MAX_EXPANSION = 3

# Maximum expansion allowed for medium-confidence edges.
MEDIUM_MAX_EXPANSION = 12


# ================================================================
# HELPERS
# ================================================================

def clean(x):
    if x is None:
        return ""
    return str(x).strip()


def parse_json_list(value):
    if not value:
        return []

    try:
        obj = json.loads(value)
    except Exception:
        return []

    if not isinstance(obj, list):
        return []

    return [
        clean(x).upper()
        for x in obj
        if clean(x)
    ]


def normalize_gri_code(code):
    """
    Normalize GRI identifiers.

    Examples:
        302-1-A     -> 302-1-A
        302-1-a     -> 302-1-A
        GRI 302-1   -> 302-1
    """

    code = clean(code).upper()

    code = re.sub(
        r"^GRI\s+",
        "",
        code,
        flags=re.IGNORECASE,
    )

    code = code.replace(" ", "")

    return code


def disclosure_id(code):
    """
    Return the GRI disclosure-level identifier.

        302-1-A     -> 302-1
        405-1-A-I   -> 405-1
        3-3-D-I-II  -> 3-3
    """

    code = normalize_gri_code(code)

    parts = code.split("-")

    if len(parts) < 2:
        return code

    return "-".join(parts[:2])


def code_specificity(code):
    """
    Number of components beyond the disclosure.

        302-1       -> 0
        302-1-A     -> 1
        302-1-A-I   -> 2
    """

    code = normalize_gri_code(code)

    return max(0, len(code.split("-")) - 2)


# ================================================================
# LOAD BRSR -> GRI
# ================================================================

def load_brsr_gri():

    rows = []

    with open(
        BRSR_PATH,
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        required = {
            "brsr_id",
            "match_type",
            "gri_codes_table1",
            "gri_codes_table2",
        }

        missing = required - set(reader.fieldnames or [])

        if missing:
            raise ValueError(
                f"gold_pairs.csv missing columns: {sorted(missing)}"
            )

        for row in reader:

            brsr_id = clean(row["brsr_id"])

            if not brsr_id:
                continue

            match_type = clean(
                row["match_type"]
            ).lower()

            # ----------------------------------------------------
            # Table 1 is the primary source.
            # ----------------------------------------------------

            table1 = [
                normalize_gri_code(x)
                for x in parse_json_list(
                    row["gri_codes_table1"]
                )
            ]

            # ----------------------------------------------------
            # Table 2 is fallback only.
            # ----------------------------------------------------

            table2 = [
                normalize_gri_code(x)
                for x in parse_json_list(
                    row["gri_codes_table2"]
                )
            ]

            if table1:
                codes = table1
                evidence_source = "table1"
            else:
                codes = table2
                evidence_source = "table2_fallback"

            # No direct linkage means no positive BRSR -> GRI edge.
            if match_type == "no_direct_linkage":
                codes = []

            for code in sorted(set(codes)):

                rows.append(
                    {
                        "brsr_id": brsr_id,
                        "gri_code": code,
                        "gri_disclosure": disclosure_id(code),
                        "brsr_match_type": match_type,
                        "brsr_mapping_semantics": clean(
                            row.get("mapping_semantics", "")
                        ),
                        "brsr_evidence_source": evidence_source,
                    }
                )

    return rows


# ================================================================
# LOAD GRI -> ESRS
# ================================================================

def load_gri_esrs():

    exact = defaultdict(list)
    disclosure = defaultdict(list)

    with open(
        GRI_ESRS_PATH,
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            if clean(
                row.get("has_gri_mapping", "")
            ).lower() != "true":
                continue

            standard = clean(
                row.get("gri_standard", "")
            )

            gri_disclosure = clean(
                row.get("gri_disclosure", "")
            )

            gri_number = clean(
                row.get("gri_number", "")
            )

            dp_id = clean(
                row.get("esrs_datapoint_id", "")
            )

            if not dp_id:
                continue

            # ----------------------------------------------------
            # Convert "GRI 302" + "302-1" + "A"
            # into requirement-level code when possible.
            # ----------------------------------------------------

            if gri_disclosure and gri_number:

                base = normalize_gri_code(
                    gri_disclosure
                )

                number = normalize_gri_code(
                    gri_number
                )

                if re.fullmatch(
                    r"\d{1,3}-\d+",
                    base,
                ):

                    exact_code = (
                        f"{base}-{number}"
                    )

                    exact[
                        exact_code
                    ].append(row)

                    disclosure[
                        base
                    ].append(row)

            # ----------------------------------------------------
            # Also index disclosure from the disclosure column.
            # ----------------------------------------------------

            if gri_disclosure:

                d = normalize_gri_code(
                    gri_disclosure
                )

                if re.fullmatch(
                    r"\d{1,3}-\d+",
                    d,
                ):

                    disclosure[d].append(row)

    return exact, disclosure


# ================================================================
# LOAD INTEROPERABILITY SUPPORT
# ================================================================

def load_interoperability():

    support = defaultdict(set)

    with open(
        INTEROP_PATH,
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            d = normalize_gri_code(
                row.get("gri_disclosure_id", "")
            )

            mapping_type = clean(
                row.get("mapping_type", "")
            )

            if d:
                support[d].add(mapping_type)

    return support


# ================================================================
# CLASSIFY EDGE
# ================================================================

def classify_edge(
    brsr_row,
    gri_code,
    datapoints,
    interop_support,
):
    """
    Determine evidence tier.

    The important principle is that direct interoperability
    does NOT automatically make every expanded datapoint gold.
    """

    expansion = len(datapoints)

    d = disclosure_id(gri_code)

    support = interop_support.get(
        d,
        set(),
    )

    specificity = code_specificity(
        gri_code
    )

    match_type = brsr_row[
        "brsr_match_type"
    ]

    # ------------------------------------------------------------
    # Strong evidence
    #
    # Requirement-level BRSR mapping + small expansion.
    # ------------------------------------------------------------

    if (
        specificity >= 1
        and match_type == "direct"
        and expansion <= GOLD_MAX_EXPANSION
        and "direct" in support
    ):
        return "gold"

    # ------------------------------------------------------------
    # Medium evidence
    # ------------------------------------------------------------

    if (
        match_type == "direct"
        and expansion <= MEDIUM_MAX_EXPANSION
        and (
            "direct" in support
            or "partial_requirement" in support
        )
    ):
        return "medium"

    # ------------------------------------------------------------
    # High recall only.
    #
    # Still useful for candidate retrieval, but should not
    # be used as a supervised positive.
    # ------------------------------------------------------------

    if (
        expansion <= MEDIUM_MAX_EXPANSION
        and support
    ):
        return "high_recall"

    # ------------------------------------------------------------
    # Everything else is review.
    # ------------------------------------------------------------

    return "review"


# ================================================================
# MAIN CONSTRUCTION
# ================================================================

def main():

    print("=" * 72)
    print("SCRIPT 10 — FINAL TRANSITIVE DATASET CONSTRUCTION")
    print("=" * 72)

    brsr_rows = load_brsr_gri()

    exact_index, disclosure_index = (
        load_gri_esrs()
    )

    interop = load_interoperability()

    print()
    print("Loaded:")
    print(
        f"  BRSR-GRI edges:       {len(brsr_rows)}"
    )
    print(
        f"  GRI exact keys:       {len(exact_index)}"
    )
    print(
        f"  GRI disclosure keys:  {len(disclosure_index)}"
    )
    print(
        f"  Interoperability keys:{len(interop)}"
    )

    candidates = []

    # ============================================================
    # BUILD GRAPH
    # ============================================================

    for brsr in brsr_rows:

        code = brsr["gri_code"]

        d = brsr["gri_disclosure"]

        # --------------------------------------------------------
        # Requirement-level match first.
        # --------------------------------------------------------

        datapoints = exact_index.get(
            code,
            []
        )

        match_level = "requirement"

        # --------------------------------------------------------
        # Disclosure fallback.
        # --------------------------------------------------------

        if not datapoints:

            datapoints = disclosure_index.get(
                d,
                []
            )

            match_level = "disclosure"

        # Remove duplicate datapoints.
        unique = {}

        for dp in datapoints:

            dp_id = clean(
                dp.get("esrs_datapoint_id", "")
            )

            if dp_id:
                unique[dp_id] = dp

        datapoints = list(
            unique.values()
        )

        expansion = len(datapoints)

        tier = classify_edge(
            brsr,
            code,
            datapoints,
            interop,
        )

        support = sorted(
            interop.get(
                d,
                set()
            )
        )

        for dp in datapoints:

            candidates.append(
                {
                    "brsr_id":
                        brsr["brsr_id"],

                    "gri_code":
                        code,

                    "gri_disclosure":
                        d,

                    "esrs_datapoint_id":
                        dp["esrs_datapoint_id"],

                    "esrs_sheet":
                        dp.get("esrs_sheet", ""),

                    "esrs_topic_code":
                        dp.get(
                            "esrs_topic_code",
                            ""
                        ),

                    "esrs_dr":
                        dp.get(
                            "esrs_dr",
                            ""
                        ),

                    "esrs_name":
                        dp.get(
                            "esrs_name",
                            ""
                        ),

                    "esrs_data_type":
                        dp.get(
                            "esrs_data_type",
                            ""
                        ),

                    "brsr_match_type":
                        brsr[
                            "brsr_match_type"
                        ],

                    "brsr_mapping_semantics":
                        brsr[
                            "brsr_mapping_semantics"
                        ],

                    "brsr_evidence_source":
                        brsr[
                            "brsr_evidence_source"
                        ],

                    "gri_esrs_match_level":
                        match_level,

                    "expansion_count":
                        expansion,

                    "expansion_class":
                        (
                            "low"
                            if expansion <= 3
                            else
                            "moderate"
                            if expansion <= 12
                            else
                            "high"
                            if expansion <= 30
                            else
                            "very_high"
                        ),

                    "interop_support":
                        ";".join(
                            support
                        ),

                    "evidence_tier":
                        tier,
                }
            )

    # ============================================================
    # DEDUPLICATE
    # ============================================================

    unique = {}

    for row in candidates:

        key = (
            row["brsr_id"],
            row["gri_code"],
            row["esrs_datapoint_id"],
        )

        unique[key] = row

    candidates = list(
        unique.values()
    )

    # ============================================================
    # SPLIT DATASETS
    # ============================================================

    gold = [
        r for r in candidates
        if r["evidence_tier"] == "gold"
    ]

    medium = [
        r for r in candidates
        if r["evidence_tier"] == "medium"
    ]

    high_recall = [
        r for r in candidates
        if r["evidence_tier"] == "high_recall"
    ]

    review = [
        r for r in candidates
        if r["evidence_tier"] == "review"
    ]

    # ============================================================
    # WRITE
    # ============================================================

    fieldnames = [
        "brsr_id",
        "gri_code",
        "gri_disclosure",
        "esrs_datapoint_id",
        "esrs_sheet",
        "esrs_topic_code",
        "esrs_dr",
        "esrs_name",
        "esrs_data_type",
        "brsr_match_type",
        "brsr_mapping_semantics",
        "brsr_evidence_source",
        "gri_esrs_match_level",
        "expansion_count",
        "expansion_class",
        "interop_support",
        "evidence_tier",
    ]

    for path, rows in [
        (CANDIDATE_OUT, candidates),
        (GOLD_OUT, gold),
        (
            REVIEW_OUT,
            review + medium + high_recall,
        ),
    ]:

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            path,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(rows)

    # ============================================================
    # SUMMARY
    # ============================================================

    print()
    print("=" * 72)
    print("FINAL DATASET SUMMARY")
    print("=" * 72)

    print(
        f"Candidate edges:          {len(candidates)}"
    )

    print(
        f"Gold edges:               {len(gold)}"
    )

    print(
        f"Medium-confidence edges:  {len(medium)}"
    )

    print(
        f"High-recall edges:        {len(high_recall)}"
    )

    print(
        f"Review edges:             {len(review)}"
    )

    print()
    print("Evidence tiers:")

    counts = Counter(
        r["evidence_tier"]
        for r in candidates
    )

    for k, v in sorted(
        counts.items()
    ):
        print(
            f"  {k:20s} {v}"
        )

    print()
    print("Expansion classes:")

    counts = Counter(
        r["expansion_class"]
        for r in candidates
    )

    for k, v in sorted(
        counts.items()
    ):
        print(
            f"  {k:20s} {v}"
        )

    # ============================================================
    # REACHABILITY
    # ============================================================

    brsr_all = {
        r["brsr_id"]
        for r in brsr_rows
    }

    brsr_reached = {
        r["brsr_id"]
        for r in candidates
    }

    esrs_reached = {
        r["esrs_datapoint_id"]
        for r in candidates
    }

    print()
    print("Reachability:")
    print(
        f"  BRSR IDs:              {len(brsr_all)}"
    )
    print(
        f"  BRSR IDs with ESRS:    {len(brsr_reached)}"
    )
    print(
        f"  BRSR IDs without path: "
        f"{len(brsr_all - brsr_reached)}"
    )
    print(
        f"  ESRS datapoints reached:{len(esrs_reached)}"
    )

    print()
    print("Outputs:")
    print(
        f"  candidates: {CANDIDATE_OUT}"
    )
    print(
        f"  gold:       {GOLD_OUT}"
    )
    print(
        f"  review:     {REVIEW_OUT}"
    )

    print()
    print(
        "IMPORTANT:"
    )
    print(
        "  Use transitive_gold.csv for supervised training/evaluation."
    )
    print(
        "  Use transitive_candidates.csv for high-recall retrieval."
    )
    print(
        "  Do NOT treat the candidate graph as gold labels."
    )


if __name__ == "__main__":
    main()