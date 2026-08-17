"""
Script 09 — Validate the BRSR -> GRI -> ESRS transitive graph.

Purpose
-------
Validate the composed graph produced by Script 08 before it is used for
retrieval, knowledge distillation, evaluation, or model training.

This script deliberately does NOT assume that every transitive edge is a
gold semantic equivalence.

The graph contains:

    BRSR -> GRI
    GRI  -> ESRS datapoint

and therefore candidate:

    BRSR -> GRI -> ESRS

edges.

The validator checks:

1. Structural integrity
2. Provenance integrity
3. Duplicate edges
4. Expansion behaviour
5. Missing / dangling mappings
6. Interoperability support
7. BRSR mapping semantics
8. High-expansion disclosures
9. Anchor cases
10. Candidate confidence tiers

Outputs
-------
data/processed/
    graph_validation_summary.csv
    graph_edge_validation.csv
    graph_review_queue.csv
    graph_expansion_summary.csv

The validated edge file should be treated as a candidate graph, not as
manually validated semantic gold.

Run:
    python scripts/09_validate_transitive_graph.py
"""

import csv
import json
import pathlib
from collections import Counter, defaultdict


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

BASE = pathlib.Path(__file__).resolve().parents[1]

AUDIT_PATH = (
    BASE
    / "data"
    / "processed"
    / "transitive_mapping_audit.csv"
)

TRANSITIVE_PATH = (
    BASE
    / "data"
    / "processed"
    / "transitive_mapping.csv"
)

BRSR_GRI_PATH = (
    BASE
    / "data"
    / "processed"
    / "gold_pairs.csv"
)

GRI_ESRS_PATH = (
    BASE
    / "data"
    / "interim"
    / "gri_esrs_datapoint_mapping.csv"
)

INTEROP_PATH = (
    BASE
    / "data"
    / "interim"
    / "gri_esrs_interoperability.csv"
)

OUT_DIR = BASE / "data" / "processed"

SUMMARY_PATH = OUT_DIR / "graph_validation_summary.csv"
EDGE_VALIDATION_PATH = OUT_DIR / "graph_edge_validation.csv"
REVIEW_PATH = OUT_DIR / "graph_review_queue.csv"
EXPANSION_PATH = OUT_DIR / "graph_expansion_summary.csv"


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def normalize(value):
    if value is None:
        return ""

    return str(value).strip()


def normalize_gri_code(code):
    """
    Normalize GRI identifiers for comparison.

    Examples:
        3-3
        3-3-C
        302-1-A
    """

    code = normalize(code).upper()

    if not code:
        return ""

    return code.replace(" ", "")


def parse_json(value):
    if not value:
        return []

    try:
        return json.loads(value)
    except Exception:
        return []


def safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


# ---------------------------------------------------------------------
# Confidence classification
# ---------------------------------------------------------------------

def classify_candidate(
    expansion_class,
    interop_support,
    brsr_match_type,
    brsr_mapping_semantics,
    esrs_datapoint_count,
):
    """
    Assign a conservative candidate tier.

    IMPORTANT:
        These are candidate-confidence tiers, not semantic truth labels.
    """

    expansion_class = normalize(expansion_class)
    interop_support = normalize(interop_support)
    brsr_match_type = normalize(brsr_match_type)
    brsr_mapping_semantics = normalize(brsr_mapping_semantics)

    count = safe_int(esrs_datapoint_count)

    # -------------------------------------------------------------
    # Highest confidence
    # -------------------------------------------------------------

    if (
        expansion_class == "low_expansion"
        and interop_support == "direct"
        and brsr_match_type == "direct"
        and brsr_mapping_semantics == "direct"
        and count <= 3
    ):
        return "high_confidence_candidate"

    # -------------------------------------------------------------
    # Good candidate but requires some care
    # -------------------------------------------------------------

    if (
        expansion_class in {
            "low_expansion",
            "moderate_expansion",
        }
        and interop_support in {
            "direct",
            "partial_requirement",
        }
        and brsr_match_type == "direct"
    ):
        return "medium_confidence_candidate"

    # -------------------------------------------------------------
    # Partial / broader mappings
    # -------------------------------------------------------------

    if interop_support in {
        "partial_requirement",
        "broader_topic",
    }:
        return "review_required"

    # -------------------------------------------------------------
    # Very large expansion is inherently risky
    # -------------------------------------------------------------

    if count >= 20:
        return "high_recall_only"

    # -------------------------------------------------------------
    # Default
    # -------------------------------------------------------------

    return "review_required"


# ---------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------

def main():

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SCRIPT 09 — TRANSITIVE GRAPH VALIDATION")
    print("=" * 70)

    # -------------------------------------------------------------
    # Load files
    # -------------------------------------------------------------

    audit_rows = load_csv(AUDIT_PATH)
    brsr_rows = load_csv(BRSR_GRI_PATH)
    gri_esrs_rows = load_csv(GRI_ESRS_PATH)
    interop_rows = load_csv(INTEROP_PATH)

    print()
    print("Loaded:")
    print(f"  BRSR-GRI rows:              {len(brsr_rows)}")
    print(f"  GRI-ESRS datapoint rows:    {len(gri_esrs_rows)}")
    print(f"  Interoperability rows:      {len(interop_rows)}")
    print(f"  Transitive audit rows:       {len(audit_rows)}")

    # -------------------------------------------------------------
    # Build BRSR -> GRI index
    # -------------------------------------------------------------

    brsr_to_gri = defaultdict(set)

    brsr_ids = set()

    for row in brsr_rows:

        brsr_id = normalize(row.get("brsr_id"))

        if not brsr_id:
            continue

        brsr_ids.add(brsr_id)

        # Script 05 stores the structured GRI codes here.
        codes = parse_json(row.get("gri_codes_json"))

        for item in codes:

            standard = normalize(item.get("standard"))
            disclosures = item.get("disclosures", [])

            for disclosure in disclosures:

                disclosure = normalize_gri_code(disclosure)

                if disclosure:
                    brsr_to_gri[brsr_id].add(disclosure)

    # -------------------------------------------------------------
    # Build GRI -> ESRS datapoint index
    # -------------------------------------------------------------

    gri_to_esrs = defaultdict(set)

    esrs_datapoints = set()

    for row in gri_esrs_rows:

        dp = normalize(row.get("esrs_datapoint_id"))

        if not dp:
            continue

        esrs_datapoints.add(dp)

        standard = normalize(row.get("gri_standard"))
        disclosure = normalize(row.get("gri_disclosure"))
        number = normalize(row.get("gri_number"))

        if not standard or not disclosure:
            continue

        # Remove "GRI " prefix.
        standard_code = (
            standard.upper()
            .replace("GRI ", "")
            .strip()
        )

        disclosure = disclosure.strip()

        if disclosure == "":
            continue

        # Workbook structure is:
        #
        # GRI 302 | 302-1 | a
        #
        # so reconstruct the disclosure identifier.
        if "-" in disclosure:
            gri_code = disclosure
        else:
            gri_code = f"{standard_code}-{disclosure}"

        gri_code = normalize_gri_code(gri_code)

        if gri_code:
            gri_to_esrs[gri_code].add(dp)

    # -------------------------------------------------------------
    # Build interoperability index
    # -------------------------------------------------------------

    interop = defaultdict(set)

    for row in interop_rows:

        gri_code = normalize_gri_code(
            row.get("gri_disclosure_id")
        )

        mapping_type = normalize(
            row.get("mapping_type")
        )

        if gri_code:
            interop[gri_code].add(mapping_type)

    # -------------------------------------------------------------
    # Structural diagnostics
    # -------------------------------------------------------------

    print()
    print("=" * 70)
    print("STRUCTURAL VALIDATION")
    print("=" * 70)

    unique_audit_edges = set()

    duplicate_edges = []
    dangling_brsr = []
    dangling_gri = []

    expansion_counter = Counter()
    confidence_counter = Counter()
    support_counter = Counter()

    review_rows = []
    validated_rows = []

    for row in audit_rows:

        brsr_id = normalize(row.get("brsr_id"))
        gri_code = normalize_gri_code(row.get("gri_code"))

        if not brsr_id or not gri_code:
            continue

        edge_key = (brsr_id, gri_code)

        if edge_key in unique_audit_edges:
            duplicate_edges.append(edge_key)

        unique_audit_edges.add(edge_key)

        # ---------------------------------------------------------
        # Check BRSR node
        # ---------------------------------------------------------

        brsr_exists = brsr_id in brsr_ids

        if not brsr_exists:
            dangling_brsr.append(edge_key)

        # ---------------------------------------------------------
        # Check GRI -> ESRS path
        # ---------------------------------------------------------

        actual_esrs = gri_to_esrs.get(gri_code, set())

        expected_count = safe_int(
            row.get("esrs_datapoint_count")
        )

        actual_count = len(actual_esrs)

        gri_exists = gri_code in gri_to_esrs

        if not gri_exists:
            dangling_gri.append(edge_key)

        # ---------------------------------------------------------
        # Expansion consistency
        # ---------------------------------------------------------

        expansion_class = normalize(
            row.get("expansion_class")
        )

        expansion_counter[expansion_class] += 1

        count_mismatch = (
            expected_count != actual_count
        )

        # ---------------------------------------------------------
        # Interoperability support
        # ---------------------------------------------------------

        interop_support = normalize(
            row.get("interop_support")
        )

        if not interop_support:
            if gri_code in interop:
                types = interop[gri_code]

                if "direct" in types:
                    interop_support = "direct"

                elif "partial_requirement" in types:
                    interop_support = "partial_requirement"

                elif "broader_topic" in types:
                    interop_support = "broader_topic"

                else:
                    interop_support = "no_interop_match"

            else:
                interop_support = "no_interop_match"

        support_counter[interop_support] += 1

        # ---------------------------------------------------------
        # BRSR semantics
        # ---------------------------------------------------------

        brsr_match_type = normalize(
            row.get("brsr_match_type")
        )

        brsr_mapping_semantics = normalize(
            row.get("brsr_mapping_semantics")
        )

        # ---------------------------------------------------------
        # Candidate confidence
        # ---------------------------------------------------------

        confidence = classify_candidate(
            expansion_class,
            interop_support,
            brsr_match_type,
            brsr_mapping_semantics,
            actual_count,
        )

        confidence_counter[confidence] += 1

        # ---------------------------------------------------------
        # Determine review reasons
        # ---------------------------------------------------------

        reasons = []

        if not brsr_exists:
            reasons.append("missing_brsr_node")

        if not gri_exists:
            reasons.append("missing_gri_esrs_path")

        if count_mismatch:
            reasons.append("esrs_count_mismatch")

        if interop_support in {
            "partial_requirement",
            "broader_topic",
            "no_interop_match",
        }:
            reasons.append(
                f"interop_{interop_support}"
            )

        if actual_count >= 20:
            reasons.append("very_large_expansion")

        elif actual_count >= 10:
            reasons.append("large_expansion")

        if brsr_match_type != "direct":
            reasons.append(
                f"brsr_match_{brsr_match_type}"
            )

        if brsr_mapping_semantics not in {
            "",
            "direct",
        }:
            reasons.append(
                f"brsr_semantics_{brsr_mapping_semantics}"
            )

        # ---------------------------------------------------------
        # Status
        # ---------------------------------------------------------

        if not reasons:
            status = "validated_candidate"

        elif confidence == "high_recall_only":
            status = "high_recall_candidate"

        else:
            status = "review_required"

        output = dict(row)

        output.update(
            {
                "actual_esrs_datapoint_count": actual_count,
                "count_consistent": str(
                    not count_mismatch
                ),
                "brsr_node_exists": str(brsr_exists),
                "gri_esrs_path_exists": str(gri_exists),
                "interop_support_resolved": interop_support,
                "candidate_confidence": confidence,
                "validation_status": status,
                "review_reasons": ";".join(reasons),
            }
        )

        validated_rows.append(output)

        if reasons:
            review_rows.append(output)

    # -------------------------------------------------------------
    # Duplicate edges
    # -------------------------------------------------------------

    print()
    print(f"Unique BRSR-GRI edges:       {len(unique_audit_edges)}")
    print(f"Duplicate audit edges:       {len(duplicate_edges)}")
    print(f"Dangling BRSR edges:         {len(dangling_brsr)}")
    print(f"Dangling GRI-ESRS edges:     {len(dangling_gri)}")

    # -------------------------------------------------------------
    # Count BRSR reachability
    # -------------------------------------------------------------

    reachable_brsr = set()

    for row in validated_rows:

        if (
            row["gri_esrs_path_exists"] == "True"
            and safe_int(
                row["actual_esrs_datapoint_count"]
            ) > 0
        ):
            reachable_brsr.add(
                normalize(row["brsr_id"])
            )

    unreachable_brsr = brsr_ids - reachable_brsr

    # -------------------------------------------------------------
    # Count unique ESRS datapoints reached
    # -------------------------------------------------------------

    reached_esrs = set()

    for row in validated_rows:

        if row["gri_esrs_path_exists"] != "True":
            continue

        gri_code = normalize_gri_code(
            row["gri_code"]
        )

        reached_esrs.update(
            gri_to_esrs.get(gri_code, set())
        )

    # -------------------------------------------------------------
    # Print expansion summary
    # -------------------------------------------------------------

    print()
    print("=" * 70)
    print("EXPANSION VALIDATION")
    print("=" * 70)

    for key, value in sorted(
        expansion_counter.items()
    ):
        print(
            f"  {key:25s} {value:6d}"
        )

    print()
    print("Candidate confidence:")

    for key, value in sorted(
        confidence_counter.items()
    ):
        print(
            f"  {key:30s} {value:6d}"
        )

    print()
    print("Interoperability support:")

    for key, value in sorted(
        support_counter.items()
    ):
        print(
            f"  {key:30s} {value:6d}"
        )

    # -------------------------------------------------------------
    # GRI-level expansion analysis
    # -------------------------------------------------------------

    gri_expansion = []

    all_gri_codes = sorted(
        {
            normalize_gri_code(
                row.get("gri_code")
            )
            for row in audit_rows
            if normalize_gri_code(
                row.get("gri_code")
            )
        }
    )

    for gri_code in all_gri_codes:

        esrs_set = gri_to_esrs.get(
            gri_code,
            set(),
        )

        brsr_set = {
            normalize(row.get("brsr_id"))
            for row in audit_rows
            if normalize_gri_code(
                row.get("gri_code")
            ) == gri_code
        }

        support = interop.get(
            gri_code,
            {"no_interop_match"},
        )

        if "direct" in support:
            dominant_support = "direct"

        elif "partial_requirement" in support:
            dominant_support = "partial_requirement"

        elif "broader_topic" in support:
            dominant_support = "broader_topic"

        else:
            dominant_support = "no_interop_match"

        if len(esrs_set) >= 20:
            expansion_risk = "very_high"

        elif len(esrs_set) >= 10:
            expansion_risk = "high"

        elif len(esrs_set) >= 4:
            expansion_risk = "moderate"

        else:
            expansion_risk = "low"

        gri_expansion.append(
            {
                "gri_code": gri_code,
                "brsr_count": len(brsr_set),
                "esrs_datapoint_count": len(esrs_set),
                "expansion_risk": expansion_risk,
                "interop_support": dominant_support,
            }
        )

    gri_expansion.sort(
        key=lambda x: (
            -x["esrs_datapoint_count"],
            x["gri_code"],
        )
    )

    print()
    print("Top GRI expansion hubs:")

    for row in gri_expansion[:20]:

        print(
            f"  {row['gri_code']:12s} -> "
            f"{row['esrs_datapoint_count']:4d} ESRS | "
            f"{row['expansion_risk']:10s} | "
            f"{row['interop_support']}"
        )

    # -------------------------------------------------------------
    # Anchor diagnostics
    # -------------------------------------------------------------

    print()
    print("=" * 70)
    print("ANCHOR DIAGNOSTICS")
    print("=" * 70)

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

    for anchor in anchors:

        rows = [
            r
            for r in validated_rows
            if normalize(r.get("brsr_id")) == anchor
        ]

        print()
        print(f"{anchor}: {len(rows)} BRSR-GRI edges")

        if not rows:
            print("  NO TRANSITIVE EDGE")
            continue

        for row in rows:

            print(
                f"  {row['gri_code']:18s} -> "
                f"{row['actual_esrs_datapoint_count']:4d} ESRS | "
                f"{row['candidate_confidence']}"
            )

    # -------------------------------------------------------------
    # Overall interpretation
    # -------------------------------------------------------------

    validated_count = sum(
        1
        for r in validated_rows
        if r["validation_status"]
        == "validated_candidate"
    )

    high_recall_count = sum(
        1
        for r in validated_rows
        if r["validation_status"]
        == "high_recall_candidate"
    )

    review_count = sum(
        1
        for r in validated_rows
        if r["validation_status"]
        == "review_required"
    )

    print()
    print("=" * 70)
    print("FINAL GRAPH VALIDATION")
    print("=" * 70)

    print(
        f"Total candidate BRSR-GRI edges: "
        f"{len(validated_rows)}"
    )

    print(
        f"Validated candidates:            "
        f"{validated_count}"
    )

    print(
        f"High-recall candidates:           "
        f"{high_recall_count}"
    )

    print(
        f"Review-required edges:            "
        f"{review_count}"
    )

    print(
        f"Unique BRSR nodes:                "
        f"{len(brsr_ids)}"
    )

    print(
        f"BRSR nodes reaching ESRS:        "
        f"{len(reachable_brsr)}"
    )

    print(
        f"BRSR nodes without ESRS path:    "
        f"{len(unreachable_brsr)}"
    )

    print(
        f"Unique ESRS datapoints reached:   "
        f"{len(reached_esrs)}"
    )

    # -------------------------------------------------------------
    # Write edge validation
    # -------------------------------------------------------------

    if validated_rows:

        fields = list(
            validated_rows[0].keys()
        )

        with open(
            EDGE_VALIDATION_PATH,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fields,
            )

            writer.writeheader()
            writer.writerows(validated_rows)

    # -------------------------------------------------------------
    # Write review queue
    # -------------------------------------------------------------

    if review_rows:

        fields = list(
            review_rows[0].keys()
        )

        with open(
            REVIEW_PATH,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fields,
            )

            writer.writeheader()
            writer.writerows(review_rows)

    # -------------------------------------------------------------
    # Write expansion summary
    # -------------------------------------------------------------

    with open(
        EXPANSION_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        fields = [
            "gri_code",
            "brsr_count",
            "esrs_datapoint_count",
            "expansion_risk",
            "interop_support",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(gri_expansion)

    # -------------------------------------------------------------
    # Write compact summary
    # -------------------------------------------------------------

    summary = [
        {
            "metric": "brsr_nodes",
            "value": len(brsr_ids),
        },
        {
            "metric": "brsr_nodes_reaching_esrs",
            "value": len(reachable_brsr),
        },
        {
            "metric": "brsr_nodes_without_esrs",
            "value": len(unreachable_brsr),
        },
        {
            "metric": "unique_brsr_gri_edges",
            "value": len(unique_audit_edges),
        },
        {
            "metric": "duplicate_edges",
            "value": len(duplicate_edges),
        },
        {
            "metric": "dangling_brsr_edges",
            "value": len(dangling_brsr),
        },
        {
            "metric": "dangling_gri_edges",
            "value": len(dangling_gri),
        },
        {
            "metric": "unique_esrs_datapoints_reached",
            "value": len(reached_esrs),
        },
        {
            "metric": "validated_candidates",
            "value": validated_count,
        },
        {
            "metric": "high_recall_candidates",
            "value": high_recall_count,
        },
        {
            "metric": "review_required",
            "value": review_count,
        },
    ]

    with open(
        SUMMARY_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=["metric", "value"],
        )

        writer.writeheader()
        writer.writerows(summary)

    # -------------------------------------------------------------
    # Final message
    # -------------------------------------------------------------

    print()
    print("=" * 70)
    print("OUTPUTS")
    print("=" * 70)

    print(f"  {SUMMARY_PATH}")
    print(f"  {EDGE_VALIDATION_PATH}")
    print(f"  {REVIEW_PATH}")
    print(f"  {EXPANSION_PATH}")

    print()
    print(
        "Graph validation completed. "
        "Do NOT treat all transitive edges as semantic gold."
    )


if __name__ == "__main__":
    main()