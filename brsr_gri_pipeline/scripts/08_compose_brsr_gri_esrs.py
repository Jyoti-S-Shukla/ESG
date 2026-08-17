"""
Compose the transitive BRSR -> GRI -> ESRS mapping.

Hop 1:
    BRSR -> GRI
    Source:
        data/processed/gold_pairs.csv

Hop 2:
    GRI -> ESRS datapoint
    Primary source:
        data/interim/gri_esrs_datapoint_mapping.csv
        (official GRI + EFRAG Nov 2024 XLSX)

Secondary cross-check:
    data/interim/gri_esrs_interoperability.csv
    (official GRI + EFRAG Nov 2024 interoperability index)

IMPORTANT SEMANTIC RULE
-----------------------
Do NOT treat a GRI disclosure as an ESRS disclosure equivalent.

For example:

    BRSR -> GRI 302-1
    GRI 302-1 -> ESRS E1 datapoint(s)

The output preserves:

    BRSR item
    exact GRI code from BRSR mapping
    GRI disclosure-level parent
    ESRS datapoint
    ESRS data type
    mapping evidence

This allows downstream work to distinguish:
    - exact/sub-point BRSR->GRI evidence
    - disclosure-level GRI->ESRS evidence
    - datapoint-level ESRS mapping

Output:
    data/processed/brsr_gri_esrs_mapping.csv
    data/processed/brsr_gri_esrs_unmapped.csv
"""

import ast
import csv
import json
import pathlib
import re
from collections import Counter, defaultdict


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

BASE = pathlib.Path(__file__).resolve().parents[1]

GOLD_PATH = (
    BASE / "data" / "processed" / "gold_pairs.csv"
)

DATAPOINT_PATH = (
    BASE / "data" / "interim"
    / "gri_esrs_datapoint_mapping.csv"
)

INTEROP_PATH = (
    BASE / "data" / "interim"
    / "gri_esrs_interoperability.csv"
)

OUT_PATH = (
    BASE / "data" / "processed"
    / "brsr_gri_esrs_mapping.csv"
)

UNMAPPED_PATH = (
    BASE / "data" / "processed"
    / "brsr_gri_esrs_unmapped.csv"
)


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def clean(value):
    if value is None:
        return ""
    return str(value).strip()


def normalize_gri_standard(value):
    """
    Convert:

        GRI 302: Energy 2016
        GRI 302
        GRI 302: Energy

    into:

        GRI 302
    """

    text = clean(value)

    m = re.search(
        r"\bGRI\s+(\d+)\b",
        text,
        flags=re.IGNORECASE,
    )

    if not m:
        return text.upper()

    return f"GRI {m.group(1)}"


def normalize_disclosure(value):
    """
    Normalize disclosure identifiers.

    Examples:
        302-1 -> 302-1
        302-1-a -> 302-1-a
        3-3 -> 3-3
    """

    text = clean(value).upper()

    if not text:
        return ""

    text = text.replace(" ", "")

    return text


def disclosure_parent_from_code(code):
    """
    Convert a granular GRI requirement into its parent disclosure.

    Examples:

        302-1-A
        -> 302-1

        405-1-B-I
        -> 405-1

        3-3-C
        -> 3-3

        306-2-A
        -> 306-2
    """

    code = normalize_disclosure(code)

    m = re.match(
        r"^(\d{1,3}-\d+)",
        code,
    )

    return m.group(1) if m else ""


def parse_json(value, default=None):
    if default is None:
        default = []

    value = clean(value)

    if not value:
        return default

    try:
        return json.loads(value)
    except Exception:
        pass

    # Some older outputs may contain Python-style structures.
    try:
        return ast.literal_eval(value)
    except Exception:
        return default


def unique_preserve(items):
    seen = set()
    result = []

    for item in items:
        item = clean(item)

        if not item:
            continue

        if item not in seen:
            seen.add(item)
            result.append(item)

    return result


# ---------------------------------------------------------------------
# Load BRSR -> GRI
# ---------------------------------------------------------------------

def load_brsr_gri():
    if not GOLD_PATH.exists():
        raise FileNotFoundError(
            f"Missing {GOLD_PATH}. Run scripts 01-05 first."
        )

    with open(
        GOLD_PATH,
        newline="",
        encoding="utf-8",
    ) as f:
        rows = list(csv.DictReader(f))

    print(
        f"Loaded BRSR-GRI gold rows: {len(rows)}"
    )

    return rows


# ---------------------------------------------------------------------
# Extract GRI codes from gold_pairs
# ---------------------------------------------------------------------

def extract_gri_codes(row):
    """
    Read the structured GRI codes produced by Scripts 03/04/05.

    Supports both:
        gri_codes_table1
        gri_codes_table2

    and the older generic:
        gri_codes_json
    """

    candidates = []

    for column in (
        "gri_codes_table1",
        "gri_codes_table2",
        "gri_codes_json",
    ):
        value = row.get(column)

        if not value:
            continue

        parsed = parse_json(value)

        if isinstance(parsed, list):
            candidates.extend(parsed)

    return unique_preserve(candidates)


def normalize_code_list(value):
    parsed = parse_json(value)

    if not isinstance(parsed, list):
        return []

    return unique_preserve(parsed)


# ---------------------------------------------------------------------
# Load GRI -> ESRS datapoint mapping
# ---------------------------------------------------------------------

def load_datapoint_mapping():
    if not DATAPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DATAPOINT_PATH}. "
            "Run script 06 first."
        )

    with open(
        DATAPOINT_PATH,
        newline="",
        encoding="utf-8",
    ) as f:
        rows = list(csv.DictReader(f))

    print(
        f"Loaded GRI-ESRS datapoint rows: {len(rows)}"
    )

    return rows


def build_datapoint_index(rows):
    """
    Index by:

        (GRI standard, GRI disclosure)

    Example:

        ("GRI 302", "302-1")

    -> all ESRS datapoints mapped to GRI 302-1
    """

    index = defaultdict(list)

    for row in rows:

        standard = normalize_gri_standard(
            row.get("gri_standard")
        )

        disclosure = normalize_disclosure(
            row.get("gri_disclosure")
        )

        if not standard or not disclosure:
            continue

        key = (
            standard,
            disclosure,
        )

        index[key].append(row)

    return index


# ---------------------------------------------------------------------
# Load interoperability index
# ---------------------------------------------------------------------

def load_interoperability():
    """
    Secondary cross-check only.

    Script 07 is not used as the primary source for ESRS
    datapoint construction.
    """

    if not INTEROP_PATH.exists():
        print(
            "\nWARNING: interoperability file not found."
            "\nContinuing without secondary cross-check."
        )
        return []

    with open(
        INTEROP_PATH,
        newline="",
        encoding="utf-8",
    ) as f:
        rows = list(csv.DictReader(f))

    print(
        f"Loaded interoperability rows: {len(rows)}"
    )

    return rows


def build_interop_index(rows):
    """
    Index interoperability rows by:

        (GRI standard, GRI disclosure)
    """

    index = defaultdict(list)

    for row in rows:

        standard = normalize_gri_standard(
            row.get("gri_standard")
        )

        disclosure = normalize_disclosure(
            row.get("gri_disclosure_id")
        )

        if not standard or not disclosure:
            continue

        index[
            (standard, disclosure)
        ].append(row)

    return index


# ---------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------

def compose(
    brsr_rows,
    datapoint_index,
    interop_index,
):
    final_rows = []
    unmapped_rows = []

    stats = Counter()

    for brsr in brsr_rows:

        brsr_id = clean(
            brsr.get("brsr_id")
        )

        match_type = clean(
            brsr.get("match_type")
        )

        mapping_semantics = clean(
            brsr.get("mapping_semantics")
        )

        # -------------------------------------------------------------
        # Recover GRI codes from both Table 1 and Table 2.
        # -------------------------------------------------------------

        gri_codes = extract_gri_codes(brsr)

        if not gri_codes:

            stats["brsr_without_gri"] += 1

            unmapped_rows.append({
                "brsr_id": brsr_id,
                "brsr_match_type": match_type,
                "reason": "no_gri_code",
                "gri_code": "",
                "gri_standard": "",
                "gri_disclosure": "",
                "detail": "",
            })

            continue

        # -------------------------------------------------------------
        # Each BRSR->GRI code becomes an individual candidate edge.
        # -------------------------------------------------------------

        for gri_code in gri_codes:

            gri_code = normalize_disclosure(
                gri_code
            )

            gri_disclosure = disclosure_parent_from_code(
                gri_code
            )

            if not gri_disclosure:

                stats["invalid_gri_code"] += 1

                unmapped_rows.append({
                    "brsr_id": brsr_id,
                    "brsr_match_type": match_type,
                    "reason": "invalid_gri_code",
                    "gri_code": gri_code,
                    "gri_standard": "",
                    "gri_disclosure": "",
                    "detail": "",
                })

                continue

            # ---------------------------------------------------------
            # Infer GRI standard from the code itself.
            #
            # For codes such as 302-1-A:
            #   standard = GRI 302
            # ---------------------------------------------------------

            standard_number = gri_code.split("-")[0]

            gri_standard = f"GRI {standard_number}"

            key = (
                gri_standard,
                gri_disclosure,
            )

            datapoints = datapoint_index.get(
                key,
                [],
            )

            interop_rows = interop_index.get(
                key,
                [],
            )

            # ---------------------------------------------------------
            # No ESRS datapoint mapping.
            # ---------------------------------------------------------

            if not datapoints:

                stats["gri_without_esrs_datapoint"] += 1

                interop_types = unique_preserve(
                    r.get("mapping_type")
                    for r in interop_rows
                )

                unmapped_rows.append({
                    "brsr_id": brsr_id,
                    "brsr_match_type": match_type,
                    "reason": "no_esrs_datapoint_mapping",
                    "gri_code": gri_code,
                    "gri_standard": gri_standard,
                    "gri_disclosure": gri_disclosure,
                    "detail": "; ".join(interop_types),
                })

                continue

            # ---------------------------------------------------------
            # One GRI disclosure can map to MANY ESRS datapoints.
            #
            # This is intentional.
            # ---------------------------------------------------------

            for dp in datapoints:

                stats["transitive_edges"] += 1

                interop_types = unique_preserve(
                    r.get("mapping_type")
                    for r in interop_rows
                )

                interop_esrs = unique_preserve(
                    r.get("esrs_requirement_raw")
                    for r in interop_rows
                )

                final_rows.append({

                    # -------------------------------------------------
                    # BRSR layer
                    # -------------------------------------------------

                    "brsr_id": brsr_id,

                    "brsr_text": clean(
                        brsr.get("brsr_text")
                    ),

                    "brsr_match_type": match_type,

                    "brsr_mapping_semantics": (
                        mapping_semantics
                    ),

                    # -------------------------------------------------
                    # GRI layer
                    # -------------------------------------------------

                    "gri_code_from_brsr": gri_code,

                    "gri_standard": gri_standard,

                    "gri_disclosure": gri_disclosure,

                    "gri_granularity": (
                        "requirement"
                        if gri_code != gri_disclosure
                        else "disclosure"
                    ),

                    # -------------------------------------------------
                    # ESRS layer
                    # -------------------------------------------------

                    "esrs_datapoint_id": clean(
                        dp.get("esrs_datapoint_id")
                    ),

                    "esrs_sheet": clean(
                        dp.get("esrs_sheet")
                    ),

                    "esrs_topic_code": clean(
                        dp.get("esrs_topic_code")
                    ),

                    "esrs_dr": clean(
                        dp.get("esrs_dr")
                    ),

                    "esrs_paragraph": clean(
                        dp.get("esrs_paragraph")
                    ),

                    "esrs_name": clean(
                        dp.get("esrs_name")
                    ),

                    "esrs_data_type": clean(
                        dp.get("esrs_data_type")
                    ),

                    # -------------------------------------------------
                    # Evidence / provenance
                    # -------------------------------------------------

                    "gri_esrs_source": (
                        "GRI-EFRAG-Nov2024-datapoint-mapping"
                    ),

                    "interop_mapping_type": (
                        "; ".join(interop_types)
                    ),

                    "interop_esrs_requirement": (
                        " || ".join(interop_esrs)
                    ),

                    "mapping_chain": (
                        "BRSR -> GRI -> ESRS"
                    ),

                    "mapping_confidence": (
                        "transitive_official"
                    ),
                })

    return final_rows, unmapped_rows, stats


# ---------------------------------------------------------------------
# Deduplicate
# ---------------------------------------------------------------------

def deduplicate(rows):
    """
    Remove exact duplicate transitive edges.

    Distinct ESRS datapoints are intentionally retained.
    """

    seen = set()
    output = []

    for row in rows:

        key = (
            row["brsr_id"],
            row["gri_code_from_brsr"],
            row["esrs_datapoint_id"],
        )

        if key in seen:
            continue

        seen.add(key)
        output.append(row)

    return output


# ---------------------------------------------------------------------
# Write CSV
# ---------------------------------------------------------------------

def write_csv(path, rows):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        print(
            f"No rows to write: {path}"
        )
        return

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print("=" * 70)
    print("SCRIPT 08 — BRSR -> GRI -> ESRS COMPOSITION")
    print("=" * 70)

    brsr_rows = load_brsr_gri()

    datapoint_rows = load_datapoint_mapping()

    datapoint_index = build_datapoint_index(
        datapoint_rows
    )

    interop_rows = load_interoperability()

    interop_index = build_interop_index(
        interop_rows
    )

    print(
        f"Indexed GRI->ESRS disclosure keys: "
        f"{len(datapoint_index)}"
    )

    final_rows, unmapped_rows, stats = compose(
        brsr_rows,
        datapoint_index,
        interop_index,
    )

    final_rows = deduplicate(
        final_rows
    )

    write_csv(
        OUT_PATH,
        final_rows,
    )

    write_csv(
        UNMAPPED_PATH,
        unmapped_rows,
    )

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    print("\n" + "=" * 70)
    print("COMPOSITION SUMMARY")
    print("=" * 70)

    print(
        f"BRSR-GRI source rows:          {len(brsr_rows)}"
    )

    print(
        f"Final BRSR-GRI-ESRS edges:     {len(final_rows)}"
    )

    print(
        f"Unmapped / review rows:        {len(unmapped_rows)}"
    )

    print("\nStatistics:")

    for key, value in sorted(stats.items()):
        print(
            f"  {key}: {value}"
        )

    # -----------------------------------------------------------------
    # Coverage
    # -----------------------------------------------------------------

    brsr_with_esrs = {
        r["brsr_id"]
        for r in final_rows
        if r["brsr_id"]
    }

    all_brsr = {
        clean(r.get("brsr_id"))
        for r in brsr_rows
        if clean(r.get("brsr_id"))
    }

    print("\nBRSR coverage:")

    print(
        f"  Unique BRSR IDs:             {len(all_brsr)}"
    )

    print(
        f"  BRSR IDs reaching ESRS:      {len(brsr_with_esrs)}"
    )

    print(
        f"  BRSR IDs without ESRS edge:  "
        f"{len(all_brsr - brsr_with_esrs)}"
    )

    # -----------------------------------------------------------------
    # ESRS coverage
    # -----------------------------------------------------------------

    esrs_ids = {
        r["esrs_datapoint_id"]
        for r in final_rows
        if r["esrs_datapoint_id"]
    }

    print(
        f"\nUnique ESRS datapoints reached: {len(esrs_ids)}"
    )

    # -----------------------------------------------------------------
    # Data types
    # -----------------------------------------------------------------

    data_types = Counter(
        r["esrs_data_type"]
        for r in final_rows
        if r["esrs_data_type"]
    )

    print("\nESRS data types reached:")

    for key, value in data_types.most_common():
        print(
            f"  {key}: {value}"
        )

    # -----------------------------------------------------------------
    # Mapping type cross-check
    # -----------------------------------------------------------------

    interop_types = Counter()

    for row in final_rows:

        value = clean(
            row["interop_mapping_type"]
        )

        if not value:
            interop_types["no_interop_match"] += 1
            continue

        for item in value.split(";"):
            item = item.strip()

            if item:
                interop_types[item] += 1

    print(
        "\nSecondary interoperability cross-check:"
    )

    for key, value in interop_types.most_common():
        print(
            f"  {key}: {value}"
        )

    # -----------------------------------------------------------------
    # Anchor checks
    # -----------------------------------------------------------------

    print("\n" + "=" * 70)
    print("ANCHOR CHECKS")
    print("=" * 70)

    for anchor in [
        "A19",
        "P3-E10",
        "P3-E10a",
        "P3-E10b",
        "P3-E10c",
        "P3-E10d",
        "P6-E1",
        "P6-E8",
        "P6-E9",
    ]:

        rows = [
            r
            for r in final_rows
            if r["brsr_id"] == anchor
        ]

        print(
            f"\n{anchor}: {len(rows)} BRSR-GRI-ESRS edges"
        )

        for r in rows[:10]:

            print(
                f"  GRI {r['gri_code_from_brsr']}"
                f" -> {r['esrs_datapoint_id']}"
                f" | {r['esrs_name'][:100]}"
            )

        if len(rows) > 10:
            print(
                f"  ... {len(rows) - 10} more"
            )

    print("\n" + "=" * 70)
    print("OUTPUT")
    print("=" * 70)

    print(
        f"Final mapping:\n  {OUT_PATH}"
    )

    print(
        f"Unmapped/review:\n  {UNMAPPED_PATH}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "The final table contains transitive mappings. "
        "It does NOT assert that BRSR, GRI and ESRS are equivalent."
    )

    print(
        "The exact BRSR->GRI requirement is retained separately "
        "from the parent GRI disclosure and ESRS datapoint."
    )


if __name__ == "__main__":
    main()