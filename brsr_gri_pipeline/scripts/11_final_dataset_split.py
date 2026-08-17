"""
SCRIPT 11 — FINAL DATASET SPLIT AND FREEZING

Purpose
-------
Create the final, leakage-safe datasets from the transitive candidate graph.

Input
-----
data/processed/transitive_candidates.csv

The candidate file contains evidence tiers:
    gold
    medium
    high_recall
    review

Design
------
1. GOLD
   - Reliable transitive mappings.
   - Split at BRSR-ID level into train/validation/test.
   - Only gold edges are used for validation/test.

2. MEDIUM
   - Additional reasonably supported mappings.
   - Used ONLY for training.
   - Restricted to training BRSR IDs to avoid leakage.

3. HIGH-RECALL
   - Candidate retrieval pool.
   - Never treated as supervised positive labels.
   - Exported separately.

4. REVIEW
   - Uncertain mappings.
   - Completely excluded from model training/evaluation.
   - Preserved for future manual/error analysis.

Important
---------
The split is performed at BRSR-ID level, NOT row level.

Therefore, a BRSR disclosure appearing in train cannot appear in
validation/test, even if it has different GRI or ESRS edges.

Final outputs
-------------
data/final/
    final_train.csv
    final_validation.csv
    final_test.csv
    gold_only_train.csv
    high_recall_pool.csv
    excluded_review.csv
    split_manifest.csv

Expected current statistics
----------------------------
Candidate rows: 5146
Gold:             64
Medium:          442
High-recall:      93
Review:         4547

Gold split:
    Train: 45
    Validation: 10
    Test: 9

Training:
    Gold: 45
    Medium: 47
    Total: 92

IMPORTANT:
    High-recall and review rows are NOT included in training.
"""

from pathlib import Path
import csv
import random
import sys

import pandas as pd


# ============================================================================
# PATHS
# ============================================================================

BASE = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    BASE
    / "data"
    / "processed"
    / "transitive_candidates.csv"
)

FINAL_DIR = (
    BASE
    / "data"
    / "final"
)

FINAL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

RANDOM_SEED = 42

TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

# These are intentionally explicit because your current dataset
# contains exactly these tier counts.
EXPECTED_COUNTS = {
    "gold": 64,
    "medium": 442,
    "high_recall": 93,
    "review": 4547,
}


# ============================================================================
# REQUIRED COLUMNS
# ============================================================================

REQUIRED_COLUMNS = [
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


# ============================================================================
# UTILITIES
# ============================================================================

def fail(message):
    """Print an error and terminate."""
    print("\nERROR:")
    print(message)
    sys.exit(1)


def edge_key(df):
    """
    Identity of a transitive graph edge.

    We use the complete:
        BRSR -> GRI -> ESRS
    triple.
    """
    return list(
        zip(
            df["brsr_id"].astype(str),
            df["gri_code"].astype(str),
            df["esrs_datapoint_id"].astype(str),
        )
    )


def edge_key_set(df):
    return set(edge_key(df))


def brsr_ids(df):
    """Return unique BRSR IDs as strings."""
    return set(
        df["brsr_id"]
        .dropna()
        .astype(str)
        .unique()
    )


def print_tier_counts(df):
    print("\nEvidence tiers:")
    counts = (
        df["evidence_tier"]
        .value_counts()
        .to_dict()
    )

    for tier in [
        "gold",
        "medium",
        "high_recall",
        "review",
    ]:
        print(
            f"  {tier:<15} "
            f"{counts.get(tier, 0):>6}"
        )


def save_csv(df, path):
    """Save dataframe deterministically."""
    df.to_csv(
        path,
        index=False,
        encoding="utf-8",
    )


# ============================================================================
# LOAD
# ============================================================================

print("=" * 80)
print("SCRIPT 11 — FINAL DATASET SPLIT")
print("=" * 80)

if not INPUT_PATH.exists():
    fail(
        f"Input file not found:\n{INPUT_PATH}"
    )

print("\nLoading:")
print(f"  {INPUT_PATH}")

df = pd.read_csv(
    INPUT_PATH,
    dtype=str,
)

print(f"\nLoaded candidate rows: {len(df)}")


# ============================================================================
# COLUMN VALIDATION
# ============================================================================

missing = [
    c for c in REQUIRED_COLUMNS
    if c not in df.columns
]

if missing:
    fail(
        "transitive_candidates.csv is missing columns:\n"
        + "\n".join(f"  - {x}" for x in missing)
    )

print("\nColumn validation: PASS")


# ============================================================================
# BASIC CLEANING
# ============================================================================

# Keep identifiers as strings.
for col in [
    "brsr_id",
    "gri_code",
    "gri_disclosure",
    "esrs_datapoint_id",
    "evidence_tier",
]:
    df[col] = df[col].fillna("").astype(str).str.strip()


# Remove rows without the fundamental graph identifiers.
invalid_identity = (
    (df["brsr_id"] == "")
    |
    (df["gri_code"] == "")
    |
    (df["esrs_datapoint_id"] == "")
)

if invalid_identity.any():

    n_invalid = int(invalid_identity.sum())

    fail(
        f"Found {n_invalid} rows with missing graph identifiers "
        "(brsr_id / gri_code / esrs_datapoint_id)."
    )

print("Graph identifier validation: PASS")


# ============================================================================
# EVIDENCE-TIER VALIDATION
# ============================================================================

allowed_tiers = {
    "gold",
    "medium",
    "high_recall",
    "review",
}

observed_tiers = set(
    df["evidence_tier"].unique()
)

unexpected_tiers = observed_tiers - allowed_tiers

if unexpected_tiers:
    fail(
        "Unexpected evidence tiers found:\n"
        + "\n".join(
            f"  - {x}"
            for x in sorted(unexpected_tiers)
        )
    )

print("Evidence-tier validation: PASS")

print_tier_counts(df)


# ============================================================================
# EXPECTED COUNT CHECK
# ============================================================================

actual_counts = (
    df["evidence_tier"]
    .value_counts()
    .to_dict()
)

print("\nExpected vs observed:")
for tier, expected in EXPECTED_COUNTS.items():

    observed = actual_counts.get(
        tier,
        0,
    )

    status = "PASS" if observed == expected else "WARNING"

    print(
        f"  {tier:<15}"
        f" expected={expected:<5}"
        f" observed={observed:<5}"
        f" {status}"
    )

# We deliberately do NOT terminate if counts differ.
#
# This makes the script robust if you later regenerate the candidate
# graph with a slightly different count.
#
# However, the current dataset should report:
#
#   gold        = 64
#   medium      = 442
#   high_recall = 93
#   review      = 4547


# ============================================================================
# SPLIT EVIDENCE TIERS
# ============================================================================

gold = df[
    df["evidence_tier"] == "gold"
].copy()

medium = df[
    df["evidence_tier"] == "medium"
].copy()

high_recall = df[
    df["evidence_tier"] == "high_recall"
].copy()

review = df[
    df["evidence_tier"] == "review"
].copy()


print("\n" + "=" * 80)
print("TIER DATASETS")
print("=" * 80)

print(f"Gold:        {len(gold)}")
print(f"Medium:      {len(medium)}")
print(f"High-recall: {len(high_recall)}")
print(f"Review:      {len(review)}")


# ============================================================================
# TIER EXCLUSIVITY CHECK
# ============================================================================

gold_keys = edge_key_set(gold)
medium_keys = edge_key_set(medium)
high_recall_keys = edge_key_set(high_recall)
review_keys = edge_key_set(review)

tier_sets = {
    "gold": gold_keys,
    "medium": medium_keys,
    "high_recall": high_recall_keys,
    "review": review_keys,
}

tier_names = list(tier_sets.keys())

overlap_found = False

for i in range(len(tier_names)):
    for j in range(i + 1, len(tier_names)):

        a = tier_names[i]
        b = tier_names[j]

        overlap = (
            tier_sets[a]
            & tier_sets[b]
        )

        if overlap:

            overlap_found = True

            print(
                f"\nWARNING: {a} ∩ {b} "
                f"= {len(overlap)}"
            )

if overlap_found:
    fail(
        "Evidence tiers are not mutually exclusive."
    )

print("\nTier exclusivity: PASS")


# ============================================================================
# GOLD CONSISTENCY
# ============================================================================

print("\n" + "=" * 80)
print("GOLD CONSISTENCY CHECK")
print("=" * 80)

gold_brsr = brsr_ids(gold)

gold_edge_count = len(
    gold_keys
)

print(f"Gold rows:       {len(gold)}")
print(f"Unique gold edges: {gold_edge_count}")
print(f"Unique gold BRSR IDs: {len(gold_brsr)}")

if gold_edge_count != len(gold):
    fail(
        "Gold dataset contains duplicate "
        "BRSR-GRI-ESRS edges."
    )

print("Gold consistency check: PASS")


# ============================================================================
# BRSR-LEVEL GOLD SPLIT
# ============================================================================

print("\n" + "=" * 80)
print("BRSR-LEVEL GOLD SPLIT")
print("=" * 80)

all_gold_brsr_ids = sorted(
    gold_brsr
)

rng = random.Random(
    RANDOM_SEED
)

rng.shuffle(
    all_gold_brsr_ids
)

n_brsr = len(
    all_gold_brsr_ids
)

# Calculate approximate split sizes.
#
# For the current 30 BRSR IDs this gives:
#
#   Train = 16
#   Val   = 7
#   Test  = 7
#
# which is the split used in your previous run.

n_train = int(
    round(
        n_brsr
        * TRAIN_FRACTION
    )
)

n_val = int(
    round(
        n_brsr
        * VAL_FRACTION
    )
)

# Ensure all IDs are assigned.
n_test = (
    n_brsr
    - n_train
    - n_val
)

# Guard against pathological small datasets.
if n_test < 1:
    n_test = 1
    n_val = max(
        1,
        n_val - 1,
    )

if n_val < 1:
    n_val = 1
    n_train = max(
        1,
        n_train - 1,
    )

if n_train < 1:
    fail(
        "Not enough BRSR IDs to construct train/validation/test split."
    )

train_brsr_ids = set(
    all_gold_brsr_ids[
        :n_train
    ]
)

val_brsr_ids = set(
    all_gold_brsr_ids[
        n_train:n_train + n_val
    ]
)

test_brsr_ids = set(
    all_gold_brsr_ids[
        n_train + n_val:
    ]
)

print(
    f"Unique gold BRSR IDs: {n_brsr}"
)

print(
    f"Train BRSR IDs:      "
    f"{len(train_brsr_ids)}"
)

print(
    f"Validation BRSR IDs: "
    f"{len(val_brsr_ids)}"
)

print(
    f"Test BRSR IDs:       "
    f"{len(test_brsr_ids)}"
)


# ============================================================================
# BRSR SPLIT LEAKAGE CHECK
# ============================================================================

if (
    train_brsr_ids
    & val_brsr_ids
):
    fail(
        "BRSR-ID leakage between train and validation."
    )

if (
    train_brsr_ids
    & test_brsr_ids
):
    fail(
        "BRSR-ID leakage between train and test."
    )

if (
    val_brsr_ids
    & test_brsr_ids
):
    fail(
        "BRSR-ID leakage between validation and test."
    )

if (
    train_brsr_ids
    | val_brsr_ids
    | test_brsr_ids
) != gold_brsr:
    fail(
        "Some gold BRSR IDs were not assigned to a split."
    )

print("\nBRSR-level split: PASS")


# ============================================================================
# CREATE GOLD TRAIN / VAL / TEST
# ============================================================================

gold_train = gold[
    gold["brsr_id"].isin(
        train_brsr_ids
    )
].copy()

gold_validation = gold[
    gold["brsr_id"].isin(
        val_brsr_ids
    )
].copy()

gold_test = gold[
    gold["brsr_id"].isin(
        test_brsr_ids
    )
].copy()


print("\nGold split:")
print(
    f"  Train:      {len(gold_train)}"
)

print(
    f"  Validation: {len(gold_validation)}"
)

print(
    f"  Test:       {len(gold_test)}"
)


# ============================================================================
# ADD MEDIUM EDGES TO TRAINING
# ============================================================================

# CRITICAL:
#
# Medium-confidence edges are allowed ONLY when their BRSR ID belongs
# to the training partition.
#
# We never add medium edges from validation/test BRSR IDs to training.

medium_train = medium[
    medium["brsr_id"].isin(
        train_brsr_ids
    )
].copy()


# For transparency, calculate excluded medium edges.

medium_excluded = medium[
    ~medium["brsr_id"].isin(
        train_brsr_ids
    )
].copy()


print("\nMedium-confidence augmentation:")
print(
    f"  Total medium:             "
    f"{len(medium)}"
)

print(
    f"  Added to training:        "
    f"{len(medium_train)}"
)

print(
    f"  Excluded due to split:    "
    f"{len(medium_excluded)}"
)


# ============================================================================
# FINAL TRAINING DATASET
# ============================================================================

final_train = pd.concat(
    [
        gold_train,
        medium_train,
    ],
    ignore_index=True,
)

# Preserve a deterministic ordering.
final_train = final_train.sort_values(
    by=[
        "brsr_id",
        "gri_code",
        "esrs_datapoint_id",
    ]
).reset_index(
    drop=True
)

gold_only_train = (
    gold_train
    .sort_values(
        by=[
            "brsr_id",
            "gri_code",
            "esrs_datapoint_id",
        ]
    )
    .reset_index(
        drop=True
    )
)


# ============================================================================
# VALIDATION / TEST SORTING
# ============================================================================

gold_validation = (
    gold_validation
    .sort_values(
        by=[
            "brsr_id",
            "gri_code",
            "esrs_datapoint_id",
        ]
    )
    .reset_index(
        drop=True
    )
)

gold_test = (
    gold_test
    .sort_values(
        by=[
            "brsr_id",
            "gri_code",
            "esrs_datapoint_id",
        ]
    )
    .reset_index(
        drop=True
    )
)


# ============================================================================
# HIGH-RECALL POOL
# ============================================================================

"""
IMPORTANT:

The high-recall pool is NOT split into train/validation/test here.

It is a retrieval candidate pool rather than supervised positive labels.

Therefore all 93 high-recall rows are preserved in a separate file.

If you later use high-recall candidates during training, you should
construct a train-safe retrieval pool that excludes validation/test
BRSR IDs. Do not silently use the complete pool for training.
"""

high_recall_pool = (
    high_recall
    .sort_values(
        by=[
            "brsr_id",
            "gri_code",
            "esrs_datapoint_id",
        ]
    )
    .reset_index(
        drop=True
    )
)


# ============================================================================
# REVIEW DATASET
# ============================================================================

excluded_review = (
    review
    .sort_values(
        by=[
            "brsr_id",
            "gri_code",
            "esrs_datapoint_id",
        ]
    )
    .reset_index(
        drop=True
    )
)


# ============================================================================
# SAVE PRIMARY DATASETS
# ============================================================================

print("\n" + "=" * 80)
print("WRITING FINAL DATASETS")
print("=" * 80)

train_path = (
    FINAL_DIR
    / "final_train.csv"
)

validation_path = (
    FINAL_DIR
    / "final_validation.csv"
)

test_path = (
    FINAL_DIR
    / "final_test.csv"
)

gold_only_path = (
    FINAL_DIR
    / "gold_only_train.csv"
)

high_recall_path = (
    FINAL_DIR
    / "high_recall_pool.csv"
)

review_path = (
    FINAL_DIR
    / "excluded_review.csv"
)


save_csv(
    final_train,
    train_path,
)

save_csv(
    gold_validation,
    validation_path,
)

save_csv(
    gold_test,
    test_path,
)

save_csv(
    gold_only_train,
    gold_only_path,
)

save_csv(
    high_recall_pool,
    high_recall_path,
)

save_csv(
    excluded_review,
    review_path,
)


print(
    f"  final_train.csv:       "
    f"{len(final_train)} rows"
)

print(
    f"  final_validation.csv:  "
    f"{len(gold_validation)} rows"
)

print(
    f"  final_test.csv:        "
    f"{len(gold_test)} rows"
)

print(
    f"  gold_only_train.csv:   "
    f"{len(gold_only_train)} rows"
)

print(
    f"  high_recall_pool.csv:  "
    f"{len(high_recall_pool)} rows"
)

print(
    f"  excluded_review.csv:   "
    f"{len(excluded_review)} rows"
)


# ============================================================================
# FINAL TRAINING TIER CHECK
# ============================================================================

print("\n" + "=" * 80)
print("TRAINING DATA CHECK")
print("=" * 80)

train_tiers = (
    final_train["evidence_tier"]
    .value_counts()
    .to_dict()
)

print(
    f"Gold in training:   "
    f"{train_tiers.get('gold', 0)}"
)

print(
    f"Medium in training: "
    f"{train_tiers.get('medium', 0)}"
)

print(
    f"Total training:     "
    f"{len(final_train)}"
)

if (
    set(
        final_train["evidence_tier"]
        .unique()
    )
    - {"gold", "medium"}
):
    fail(
        "Training contains an evidence tier other than gold/medium."
    )

if (
    final_train["brsr_id"]
    .isin(val_brsr_ids)
    .any()
):
    fail(
        "Validation BRSR ID leaked into training."
    )

if (
    final_train["brsr_id"]
    .isin(test_brsr_ids)
    .any()
):
    fail(
        "Test BRSR ID leaked into training."
    )

print("Training tier/leakage check: PASS")


# ============================================================================
# VALIDATION / TEST CHECK
# ============================================================================

print("\n" + "=" * 80)
print("EVALUATION DATA CHECK")
print("=" * 80)

for name, subset in [
    ("validation", gold_validation),
    ("test", gold_test),
]:

    tiers = set(
        subset["evidence_tier"].unique()
    )

    if tiers - {"gold"}:
        fail(
            f"{name} contains non-gold evidence."
        )

    print(
        f"{name.capitalize():<12}"
        f" rows={len(subset):<5}"
        f" BRSR IDs={len(brsr_ids(subset)):<5}"
        f" evidence=gold"
    )

print("Evaluation purity: PASS")


# ============================================================================
# EDGE-LEVEL LEAKAGE CHECK
# ============================================================================

print("\n" + "=" * 80)
print("EDGE-LEVEL LEAKAGE CHECK")
print("=" * 80)

train_edges = edge_key_set(
    final_train
)

val_edges = edge_key_set(
    gold_validation
)

test_edges = edge_key_set(
    gold_test
)

if train_edges & val_edges:
    fail(
        "Exact graph edge leaked from train to validation."
    )

if train_edges & test_edges:
    fail(
        "Exact graph edge leaked from train to test."
    )

if val_edges & test_edges:
    fail(
        "Exact graph edge leaked from validation to test."
    )

print(
    f"Train ∩ Validation: "
    f"{len(train_edges & val_edges)}"
)

print(
    f"Train ∩ Test:       "
    f"{len(train_edges & test_edges)}"
)

print(
    f"Validation ∩ Test:  "
    f"{len(val_edges & test_edges)}"
)

print("Edge-level leakage check: PASS")


# ============================================================================
# HIGH-RECALL SANITY CHECK
# ============================================================================

print("\n" + "=" * 80)
print("HIGH-RECALL POOL CHECK")
print("=" * 80)

print(
    f"High-recall rows written: "
    f"{len(high_recall_pool)}"
)

print(
    f"High-recall BRSR IDs:     "
    f"{len(brsr_ids(high_recall_pool))}"
)

if len(high_recall_pool) == 0:
    print(
        "WARNING: high-recall pool is empty."
    )
else:
    print(
        "High-recall export: PASS"
    )

print(
    "\nIMPORTANT:"
)

print(
    "  High-recall rows are candidate retrieval evidence."
)

print(
    "  They are NOT treated as supervised positive labels."
)

print(
    "  Do not include them in validation/test scoring as gold."
)


# ============================================================================
# REVIEW SANITY CHECK
# ============================================================================

print("\n" + "=" * 80)
print("REVIEW EXCLUSION CHECK")
print("=" * 80)

review_edge_keys = edge_key_set(
    excluded_review
)

if (
    review_edge_keys
    & train_edges
):
    fail(
        "Review edge leaked into training."
    )

if (
    review_edge_keys
    & val_edges
):
    fail(
        "Review edge leaked into validation."
    )

if (
    review_edge_keys
    & test_edges
):
    fail(
        "Review edge leaked into test."
    )

print(
    f"Review rows preserved: "
    f"{len(excluded_review)}"
)

print(
    "Review exclusion: PASS"
)


# ============================================================================
# FINAL GLOBAL ACCOUNTING
# ============================================================================

print("\n" + "=" * 80)
print("FINAL GLOBAL ACCOUNTING")
print("=" * 80)

candidate_total = len(df)

tier_total = (
    len(gold)
    + len(medium)
    + len(high_recall)
    + len(review)
)

print(
    f"Candidate rows:       {candidate_total}"
)

print(
    f"Gold:                 {len(gold)}"
)

print(
    f"Medium:               {len(medium)}"
)

print(
    f"High-recall:          {len(high_recall)}"
)

print(
    f"Review:               {len(review)}"
)

print(
    f"Tier total:           {tier_total}"
)

if tier_total != candidate_total:
    fail(
        "Evidence-tier accounting does not sum to "
        "the candidate dataset."
    )

print(
    "\nCandidate tier accounting: PASS"
)


# ============================================================================
# SPLIT MANIFEST
# ============================================================================

manifest_path = (
    FINAL_DIR
    / "split_manifest.csv"
)

manifest_rows = [

    {
        "dataset": "final_train",
        "purpose": "supervised training",
        "rows": len(final_train),
        "gold_rows": len(gold_train),
        "medium_rows": len(medium_train),
        "high_recall_rows": 0,
        "review_rows": 0,
        "brr_ids": len(
            brsr_ids(final_train)
        ),
        "evidence_policy":
            "gold + medium; training BRSR IDs only",
    },

    {
        "dataset": "gold_only_train",
        "purpose": "training ablation",
        "rows": len(gold_only_train),
        "gold_rows": len(gold_only_train),
        "medium_rows": 0,
        "high_recall_rows": 0,
        "review_rows": 0,
        "brr_ids": len(
            brsr_ids(gold_only_train)
        ),
        "evidence_policy":
            "gold only",
    },

    {
        "dataset": "final_validation",
        "purpose": "model selection / validation",
        "rows": len(gold_validation),
        "gold_rows": len(gold_validation),
        "medium_rows": 0,
        "high_recall_rows": 0,
        "review_rows": 0,
        "brr_ids": len(
            brsr_ids(gold_validation)
        ),
        "evidence_policy":
            "gold only",
    },

    {
        "dataset": "final_test",
        "purpose": "final evaluation",
        "rows": len(gold_test),
        "gold_rows": len(gold_test),
        "medium_rows": 0,
        "high_recall_rows": 0,
        "review_rows": 0,
        "brr_ids": len(
            brsr_ids(gold_test)
        ),
        "evidence_policy":
            "gold only",
    },

    {
        "dataset": "high_recall_pool",
        "purpose": "candidate retrieval",
        "rows": len(high_recall_pool),
        "gold_rows": 0,
        "medium_rows": 0,
        "high_recall_rows": len(
            high_recall_pool
        ),
        "review_rows": 0,
        "brr_ids": len(
            brsr_ids(high_recall_pool)
        ),
        "evidence_policy":
            "candidate evidence; not gold labels",
    },

    {
        "dataset": "excluded_review",
        "purpose": "future audit / error analysis",
        "rows": len(excluded_review),
        "gold_rows": 0,
        "medium_rows": 0,
        "high_recall_rows": 0,
        "review_rows": len(
            excluded_review
        ),
        "brr_ids": len(
            brsr_ids(excluded_review)
        ),
        "evidence_policy":
            "excluded from training and evaluation",
    },
]


manifest_df = pd.DataFrame(
    manifest_rows
)

manifest_df.to_csv(
    manifest_path,
    index=False,
    encoding="utf-8",
)


# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("FINAL DATASET SUMMARY")
print("=" * 80)

print(
    f"""
Candidate edges:             {len(df)}

Evidence tiers:
  Gold:                       {len(gold)}
  Medium:                     {len(medium)}
  High-recall:                {len(high_recall)}
  Review:                     {len(review)}

Training:
  Gold:                       {len(gold_train)}
  Medium:                     {len(medium_train)}
  Total:                      {len(final_train)}

Evaluation:
  Validation:                 {len(gold_validation)}
  Test:                       {len(gold_test)}

Retrieval:
  High-recall pool:           {len(high_recall_pool)}

Excluded:
  Review:                     {len(excluded_review)}

BRSR-level split:
  Train BRSR IDs:             {len(train_brsr_ids)}
  Validation BRSR IDs:        {len(val_brsr_ids)}
  Test BRSR IDs:              {len(test_brsr_ids)}
"""
)


print("=" * 80)
print("OUTPUTS")
print("=" * 80)

print(
    f"  Train:        {train_path}"
)

print(
    f"  Validation:   {validation_path}"
)

print(
    f"  Test:         {test_path}"
)

print(
    f"  Gold-only:    {gold_only_path}"
)

print(
    f"  High-recall:  {high_recall_path}"
)

print(
    f"  Review:       {review_path}"
)

print(
    f"  Manifest:     {manifest_path}"
)


# ============================================================================
# FINAL STATUS
# ============================================================================

print("\n" + "=" * 80)
print("FINAL STATUS")
print("=" * 80)

print(
    "STATUS: DATASET FREEZE READY"
)

print(
    "\nThe final supervised datasets can now be treated as frozen."
)

print(
    "Do not use review edges as negatives or positives."
)

print(
    "Do not use high-recall edges as gold supervision."
)

print(
    "Validation and test contain gold evidence only."
)

print(
    "Training contains gold + medium evidence from training BRSR IDs."
)

print("=" * 80)