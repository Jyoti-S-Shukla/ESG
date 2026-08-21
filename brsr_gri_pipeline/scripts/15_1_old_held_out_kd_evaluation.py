"""
SCRIPT 15 — FIXED HELD-OUT KD EVALUATION

Purpose
-------
Evaluate the BRSR -> ESRS retrieval systems on the authoritative
held-out test set.

This script does NOT perform:
    - training
    - cross-validation
    - candidate generation
    - zero-shot splitting
    - ranking-file construction

It evaluates frozen models on the fixed authoritative test set.

Evaluation protocol
-------------------
Authoritative test file:
    data/final/final_test.csv

Current held-out BRSR IDs:
    A13
    B6
    P1-E7
    P5-E2
    P6-E1

Training/evaluation isolation
-----------------------------
KD training IDs are read from:
    data/processed/kd_symbolic_candidates_train.csv
    data/processed/kd_training_data_gold.csv
    data/processed/kd_training_data_gold_medium.csv

The script verifies that no authoritative test BRSR ID occurs in
the KD training set.

Systems
-------
1. DIRECT
       BRSR -> ESRS

2. SYMBOLIC_MULTIHOP
       BRSR -> GRI -> ESRS

3. MULTIHOP_CROSS_ENCODER
       BRSR -> GRI -> ESRS -> Cross Encoder

4. MULTIHOP_KD_GOLD
       BRSR -> GRI -> ESRS -> KD student trained on gold supervision

5. MULTIHOP_KD_GOLD_MEDIUM
       BRSR -> GRI -> ESRS -> KD student trained on
       gold + medium supervision

6. DIRECT_KD_GOLD
       BRSR -> ESRS -> KD student trained on gold supervision

7. DIRECT_KD_GOLD_MEDIUM
       BRSR -> ESRS -> KD student trained on
       gold + medium supervision

Important
---------
The symbolic BRSR -> GRI -> ESRS candidate graph is treated as
official/distant supervision and is NOT modified using the test labels.

The test labels are used only for computing retrieval metrics.

Model directories
-----------------
SentenceTransformer KD models must point to their actual exported
SentenceTransformer directory, normally:

    models/kd_student_gold/final
    models/kd_student_gold_medium/final

and for expanded models:

    models/kd_student_expanded_gold/final
    models/kd_student_expanded_gold_medium/final

Cross-encoder directories remain:

    models/cross_encoder_gold
    models/cross_encoder_gold_medium
    models/cross_encoder_expanded_gold
    models/cross_encoder_expanded_gold_medium

Usage
-----
Original gold-trained models:

python scripts/15_kd_end_to_end_evaluation.py \
    --ce-model models/cross_encoder_gold \
    --kd-gold models/kd_student_gold/final \
    --kd-gold-medium models/kd_student_gold_medium/final \
    --output-dir data/results/original

Expanded models:

python scripts/15_kd_end_to_end_evaluation.py \
    --ce-model models/cross_encoder_expanded_gold \
    --kd-gold models/kd_student_expanded_gold/final \
    --kd-gold-medium models/kd_student_expanded_gold_medium/final \
    --output-dir data/results/expanded
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer, CrossEncoder


# ============================================================================
# PATHS
# ============================================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
FINAL_DIR = DATA_DIR / "final"
MODELS_DIR = ROOT / "models"

DEFAULT_CANDIDATE_FILE = (
    PROCESSED_DIR / "final_brsr_gri_esrs_candidates.csv"
)

DEFAULT_TEST_FILE = (
    FINAL_DIR / "final_test.csv"
)

DEFAULT_KD_TRAINING_FILES = [
    PROCESSED_DIR / "kd_symbolic_candidates_train.csv",
    PROCESSED_DIR / "kd_training_data_gold.csv",
    PROCESSED_DIR / "kd_training_data_gold_medium.csv",
]

DEFAULT_CE_MODEL = (
    MODELS_DIR / "cross_encoder_gold"
)

DEFAULT_KD_GOLD = (
    MODELS_DIR / "kd_student_gold" / "final"
)

DEFAULT_KD_GOLD_MEDIUM = (
    MODELS_DIR / "kd_student_gold_medium" / "final"
)

BASE_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================================
# ARGUMENTS
# ============================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Fixed held-out evaluation of "
            "BRSR -> GRI -> ESRS retrieval systems"
        )
    )

    parser.add_argument(
        "--candidate-file",
        default=str(DEFAULT_CANDIDATE_FILE),
        help=(
            "Official BRSR -> GRI -> ESRS candidate universe."
        ),
    )

    parser.add_argument(
        "--test-file",
        default=str(DEFAULT_TEST_FILE),
        help=(
            "Authoritative held-out evaluation file. "
            "Defaults to final_test.csv."
        ),
    )

    parser.add_argument(
        "--ce-model",
        default=str(DEFAULT_CE_MODEL),
        help=(
            "Frozen CrossEncoder directory."
        ),
    )

    parser.add_argument(
        "--kd-gold",
        default=str(DEFAULT_KD_GOLD),
        help=(
            "Frozen KD student trained using gold supervision. "
            "Point to the actual SentenceTransformer directory, "
            "normally .../final."
        ),
    )

    parser.add_argument(
        "--kd-gold-medium",
        default=str(DEFAULT_KD_GOLD_MEDIUM),
        help=(
            "Frozen KD student trained using gold + medium supervision. "
            "Point to the actual SentenceTransformer directory."
        ),
    )

    parser.add_argument(
        "--base-model",
        default=BASE_MODEL,
        help="Dense baseline SentenceTransformer.",
    )

    parser.add_argument(
        "--output-dir",
        default=str(PROCESSED_DIR),
        help="Directory for evaluation outputs.",
    )

    parser.add_argument(
        "--output-prefix",
        default="script15",
        help="Prefix for output files.",
    )

    parser.add_argument(
        "--rerank-top",
        type=int,
        default=50,
        help=(
            "Number of symbolic candidates passed to "
            "CrossEncoder/KD reranking."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# GENERAL HELPERS
# ============================================================================

def norm_id(x):

    if pd.isna(x):
        return ""

    return str(x).strip()


def clean_text(x):

    if pd.isna(x):
        return ""

    return str(x).strip()


def first_existing(df, names):

    for name in names:

        if name in df.columns:
            return name

    return None


def load_csv(path, name):

    path = Path(path)

    if not path.exists():

        raise FileNotFoundError(
            f"{name} not found:\n{path}"
        )

    df = pd.read_csv(path)

    print(
        f"{name}: "
        f"{len(df):,} rows | "
        f"{len(df.columns)} columns"
    )

    return df


# ============================================================================
# TEST GOLD
# ============================================================================

def build_gold_targets(test_df):

    required = [
        "brsr_id",
        "esrs_datapoint_id",
    ]

    missing = [
        c
        for c in required
        if c not in test_df.columns
    ]

    if missing:

        raise RuntimeError(
            "Authoritative test file is missing "
            f"required columns: {missing}"
        )

    targets = defaultdict(set)

    for _, row in test_df.iterrows():

        bid = norm_id(
            row["brsr_id"]
        )

        eid = norm_id(
            row["esrs_datapoint_id"]
        )

        if bid and eid:

            targets[bid].add(eid)

    return dict(targets)


# ============================================================================
# KD TRAINING IDS
# ============================================================================

def load_kd_training_ids():

    all_ids = set()

    print()
    print("=" * 80)
    print("KD TRAINING ISOLATION AUDIT")
    print("=" * 80)

    for path in DEFAULT_KD_TRAINING_FILES:

        if not path.exists():

            print(
                f"  [NOT FOUND] {path.name}"
            )

            continue

        df = pd.read_csv(path)

        if "brsr_id" not in df.columns:

            print(
                f"  [SKIP] {path.name}: "
                "no brsr_id column"
            )

            continue

        ids = set(
            df["brsr_id"]
            .dropna()
            .astype(str)
            .str.strip()
        )

        print(
            f"  {path.name}: "
            f"{len(ids)} BRSR IDs"
        )

        all_ids.update(ids)

    print(
        f"KD training BRSR IDs: "
        f"{len(all_ids)}"
    )

    if all_ids:

        print(
            "IDs:",
            ", ".join(sorted(all_ids)),
        )

    return all_ids


# ============================================================================
# BRSR TEXT
# ============================================================================

def build_brsr_text_map(
    candidate_df,
    test_df,
):

    result = {}

    text_col = first_existing(
        candidate_df,
        [
            "brsr_text",
            "brsr_description",
            "brsr_name",
        ],
    )

    if text_col:

        for _, row in candidate_df.iterrows():

            bid = norm_id(
                row["brsr_id"]
            )

            if not bid:
                continue

            text = clean_text(
                row[text_col]
            )

            if text:
                result[bid] = text

    if "brsr_text" in test_df.columns:

        for _, row in test_df.iterrows():

            bid = norm_id(
                row["brsr_id"]
            )

            text = clean_text(
                row["brsr_text"]
            )

            if bid and text:

                result[bid] = text

    return result


# ============================================================================
# ESRS TEXT
# ============================================================================

def build_esrs_text_map(candidate_df):

    if "esrs_datapoint_id" not in candidate_df.columns:

        raise RuntimeError(
            "Candidate table has no "
            "esrs_datapoint_id column."
        )

    text_col = first_existing(
        candidate_df,
        [
            "esrs_name",
            "esrs_text",
            "esrs_description",
        ],
    )

    if text_col is None:

        raise RuntimeError(
            "Could not identify ESRS text column."
        )

    result = {}

    for _, row in candidate_df.iterrows():

        eid = norm_id(
            row["esrs_datapoint_id"]
        )

        if not eid:
            continue

        text = clean_text(
            row[text_col]
        )

        if (
            eid not in result
            or len(text) > len(result[eid])
        ):

            result[eid] = text

    return result


# ============================================================================
# BRSR -> GRI
# ============================================================================

def build_brsr_to_gri(candidate_df):

    required = [
        "brsr_id",
        "gri_code",
    ]

    missing = [
        c
        for c in required
        if c not in candidate_df.columns
    ]

    if missing:

        raise RuntimeError(
            f"Candidate table missing: {missing}"
        )

    mapping = defaultdict(set)

    for _, row in candidate_df.iterrows():

        bid = norm_id(
            row["brsr_id"]
        )

        gri = norm_id(
            row["gri_code"]
        )

        if bid and gri:

            mapping[bid].add(gri)

    return dict(mapping)


# ============================================================================
# GRI -> ESRS
# ============================================================================

def build_gri_to_esrs(candidate_df):

    required = [
        "gri_code",
        "esrs_datapoint_id",
    ]

    missing = [
        c
        for c in required
        if c not in candidate_df.columns
    ]

    if missing:

        raise RuntimeError(
            f"Candidate table missing: {missing}"
        )

    mapping = defaultdict(set)

    for _, row in candidate_df.iterrows():

        gri = norm_id(
            row["gri_code"]
        )

        eid = norm_id(
            row["esrs_datapoint_id"]
        )

        if gri and eid:

            mapping[gri].add(eid)

    return dict(mapping)


# ============================================================================
# CANDIDATE GENERATION
# ============================================================================

def direct_candidates(
    brsr_id,
    candidate_df,
):

    mask = (
        candidate_df["brsr_id"]
        .astype(str)
        .str.strip()
        == str(brsr_id).strip()
    )

    subset = candidate_df[mask]

    if subset.empty:
        return []

    return sorted(
        set(
            subset["esrs_datapoint_id"]
            .dropna()
            .astype(str)
            .str.strip()
        )
    )


def symbolic_candidates(
    brsr_id,
    brsr_to_gri,
    gri_to_esrs,
):

    gri_nodes = brsr_to_gri.get(
        brsr_id,
        set(),
    )

    result = set()

    for gri in gri_nodes:

        result.update(
            gri_to_esrs.get(
                gri,
                set(),
            )
        )

    return sorted(result)


# ============================================================================
# DENSE RANKING
# ============================================================================

def dense_rank(
    query,
    candidate_ids,
    candidate_embeddings,
    candidate_index,
    model,
):

    if not candidate_ids:
        return []

    valid_ids = [
        eid
        for eid in candidate_ids
        if eid in candidate_index
    ]

    if not valid_ids:
        return []

    indices = [
        candidate_index[eid]
        for eid in valid_ids
    ]

    query_embedding = model.encode(
        [query],
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    doc_embeddings = candidate_embeddings[
        indices
    ]

    scores = (
        query_embedding
        @
        doc_embeddings.T
    ).squeeze(0)

    order = torch.argsort(
        scores,
        descending=True,
    )

    return [
        valid_ids[i]
        for i in order.tolist()
    ]


# ============================================================================
# CROSS ENCODER RANKING
# ============================================================================

def ce_rank(
    query,
    candidate_ids,
    esrs_texts,
    ce_model,
):

    if not candidate_ids:
        return []

    pairs = [
        (
            query,
            esrs_texts.get(
                eid,
                "",
            ),
        )
        for eid in candidate_ids
    ]

    scores = ce_model.predict(
        pairs,
        show_progress_bar=False,
    )

    ranked = sorted(
        zip(candidate_ids, scores),
        key=lambda x: float(x[1]),
        reverse=True,
    )

    return [
        eid
        for eid, _ in ranked
    ]


# ============================================================================
# KD RANKING
# ============================================================================

def kd_rank(
    query,
    candidate_ids,
    esrs_texts,
    kd_model,
):

    if not candidate_ids:
        return []

    valid_ids = [
        eid
        for eid in candidate_ids
        if esrs_texts.get(eid, "")
    ]

    if not valid_ids:
        return []

    query_texts = [query]

    document_texts = [
        esrs_texts[eid]
        for eid in valid_ids
    ]

    query_embedding = kd_model.encode(
        query_texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    document_embeddings = kd_model.encode(
        document_texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    scores = (
        query_embedding
        @
        document_embeddings.T
    ).squeeze(0)

    order = torch.argsort(
        scores,
        descending=True,
    )

    return [
        valid_ids[i]
        for i in order.tolist()
    ]


# ============================================================================
# METRICS
# ============================================================================

def rank_metrics(
    ranked_ids,
    gold_set,
    ks=(1, 5, 10, 20, 50),
):

    ranked_ids = [
        norm_id(x)
        for x in ranked_ids
    ]

    gold_set = {
        norm_id(x)
        for x in gold_set
    }

    metrics = {}

    if not gold_set:

        for k in ks:
            metrics[f"recall@{k}"] = 0.0

        metrics["mrr"] = 0.0

        return metrics

    for k in ks:

        top_k = set(
            ranked_ids[:k]
        )

        metrics[
            f"recall@{k}"
        ] = (
            len(
                top_k.intersection(
                    gold_set
                )
            )
            /
            len(gold_set)
        )

    mrr = 0.0

    for rank, eid in enumerate(
        ranked_ids,
        start=1,
    ):

        if eid in gold_set:

            mrr = 1.0 / rank
            break

    metrics["mrr"] = mrr

    return metrics


# ============================================================================
# ONE QUERY
# ============================================================================

def evaluate_query(
    bid,
    query,
    gold_set,
    candidate_df,
    brsr_to_gri,
    gri_to_esrs,
    esrs_texts,
    esrs_index,
    base_model,
    base_embeddings,
    ce_model,
    kd_gold,
    kd_gold_medium,
    rerank_top,
):

    # ------------------------------------------------------------------
    # DIRECT CANDIDATES
    # ------------------------------------------------------------------

    direct = direct_candidates(
        bid,
        candidate_df,
    )

    direct_ranked = dense_rank(
        query=query,
        candidate_ids=direct,
        candidate_embeddings=base_embeddings,
        candidate_index=esrs_index,
        model=base_model,
    )

    # ------------------------------------------------------------------
    # SYMBOLIC MULTIHOP CANDIDATES
    # ------------------------------------------------------------------

    symbolic = symbolic_candidates(
        bid,
        brsr_to_gri,
        gri_to_esrs,
    )

    symbolic_ranked = dense_rank(
        query=query,
        candidate_ids=symbolic,
        candidate_embeddings=base_embeddings,
        candidate_index=esrs_index,
        model=base_model,
    )

    # ------------------------------------------------------------------
    # RERANKING CANDIDATES
    # ------------------------------------------------------------------

    rerank_candidates = (
        symbolic_ranked[:rerank_top]
    )

    # ------------------------------------------------------------------
    # CROSS ENCODER
    # ------------------------------------------------------------------

    ce_ranked = ce_rank(
        query=query,
        candidate_ids=rerank_candidates,
        esrs_texts=esrs_texts,
        ce_model=ce_model,
    )

    # ------------------------------------------------------------------
    # KD GOLD
    # ------------------------------------------------------------------

    kd_gold_ranked = kd_rank(
        query=query,
        candidate_ids=rerank_candidates,
        esrs_texts=esrs_texts,
        kd_model=kd_gold,
    )

    # ------------------------------------------------------------------
    # KD GOLD + MEDIUM
    # ------------------------------------------------------------------

    kd_gold_medium_ranked = kd_rank(
        query=query,
        candidate_ids=rerank_candidates,
        esrs_texts=esrs_texts,
        kd_model=kd_gold_medium,
    )

    # ------------------------------------------------------------------
    # DIRECT KD
    # ------------------------------------------------------------------

    direct_kd_gold_ranked = kd_rank(
        query=query,
        candidate_ids=direct,
        esrs_texts=esrs_texts,
        kd_model=kd_gold,
    )

    direct_kd_gold_medium_ranked = kd_rank(
        query=query,
        candidate_ids=direct,
        esrs_texts=esrs_texts,
        kd_model=kd_gold_medium,
    )

    # ------------------------------------------------------------------
    # SYSTEMS
    # ------------------------------------------------------------------

    rankings = {

        "direct":
            direct_ranked,

        "symbolic_multihop":
            symbolic_ranked,

        "multihop_cross_encoder":
            ce_ranked,

        "multihop_kd_gold":
            kd_gold_ranked,

        "multihop_kd_gold_medium":
            kd_gold_medium_ranked,

        "direct_kd_gold":
            direct_kd_gold_ranked,

        "direct_kd_gold_medium":
            direct_kd_gold_medium_ranked,
    }

    metrics = {}

    for system, ranked in rankings.items():

        metrics[system] = rank_metrics(
            ranked,
            gold_set,
        )

    # ------------------------------------------------------------------
    # DIAGNOSTICS
    # ------------------------------------------------------------------

    diagnostics = {

        "brsr_id":
            bid,

        "gold_count":
            len(gold_set),

        "gri_nodes":
            len(
                brsr_to_gri.get(
                    bid,
                    set(),
                )
            ),

        "direct_candidates":
            len(direct),

        "symbolic_candidates":
            len(symbolic_ranked),

        "rerank_candidates":
            len(rerank_candidates),

        "bridge_gold_reached":
            len(
                set(symbolic_ranked)
                .intersection(gold_set)
            ),

        "direct_gold_reached":
            len(
                set(direct)
                .intersection(gold_set)
            ),

        "ce_gold_reached":
            len(
                set(ce_ranked)
                .intersection(gold_set)
            ),

        "kd_gold_gold_reached":
            len(
                set(kd_gold_ranked)
                .intersection(gold_set)
            ),

        "kd_gold_medium_gold_reached":
            len(
                set(kd_gold_medium_ranked)
                .intersection(gold_set)
            ),
    }

    return metrics, diagnostics, rankings


# ============================================================================
# AGGREGATION
# ============================================================================

SYSTEMS = [
    "direct",
    "symbolic_multihop",
    "multihop_cross_encoder",
    "multihop_kd_gold",
    "multihop_kd_gold_medium",
    "direct_kd_gold",
    "direct_kd_gold_medium",
]

METRIC_NAMES = [
    "recall@1",
    "recall@5",
    "recall@10",
    "recall@20",
    "recall@50",
    "mrr",
]


def aggregate_system(
    rows,
    system,
):

    sub = [
        row
        for row in rows
        if row["system"] == system
    ]

    if not sub:
        return {}

    return {
        metric: float(
            np.mean([
                row[metric]
                for row in sub
            ])
        )
        for metric in METRIC_NAMES
    }


# ============================================================================
# MAIN
# ============================================================================

def main():

    args = parse_args()

    set_seed(args.seed)

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 80)
    print(
        "SCRIPT 15 — FIXED HELD-OUT KD EVALUATION"
    )
    print("=" * 80)

    print(
        f"Device: {device}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    print()
    print(
        f"Base embedder: "
        f"{args.base_model}"
    )

    print(
        f"Cross encoder: "
        f"{args.ce_model}"
    )

    print(
        f"KD gold: "
        f"{args.kd_gold}"
    )

    print(
        f"KD gold+medium: "
        f"{args.kd_gold_medium}"
    )

    print(
        f"Candidate file: "
        f"{args.candidate_file}"
    )

    print(
        f"Authoritative test file: "
        f"{args.test_file}"
    )

    # ==================================================================
    # LOAD DATA
    # ==================================================================

    candidate_df = load_csv(
        args.candidate_file,
        "Candidate table",
    )

    test_df = load_csv(
        args.test_file,
        "Authoritative TEST gold",
    )

    gold_targets = build_gold_targets(
        test_df
    )

    brsr_texts = build_brsr_text_map(
        candidate_df,
        test_df,
    )

    esrs_texts = build_esrs_text_map(
        candidate_df
    )

    esrs_ids = sorted(
        esrs_texts.keys()
    )

    esrs_index = {
        eid: i
        for i, eid in enumerate(esrs_ids)
    }

    brsr_to_gri = build_brsr_to_gri(
        candidate_df
    )

    gri_to_esrs = build_gri_to_esrs(
        candidate_df
    )

    print()
    print(
        "=" * 80
    )
    print(
        "AUTHORITATIVE TEST SET"
    )
    print(
        "=" * 80
    )

    print(
        f"Test BRSR IDs: "
        f"{len(gold_targets)}"
    )

    print(
        "IDs:",
        ", ".join(
            sorted(gold_targets)
        ),
    )

    # ==================================================================
    # KD ISOLATION
    # ==================================================================

    kd_training_ids = (
        load_kd_training_ids()
    )

    test_ids = set(
        gold_targets.keys()
    )

    leakage = (
        test_ids
        .intersection(
            kd_training_ids
        )
    )

    print()
    print(
        "=" * 80
    )
    print(
        "TRAIN/TEST ISOLATION"
    )
    print(
        "=" * 80
    )

    if leakage:

        print(
            "[FAIL] KD training/test overlap detected:"
        )

        print(
            ", ".join(
                sorted(leakage)
            )
        )

        raise RuntimeError(
            "KD training/test leakage detected."
        )

    print(
        "[PASS] No authoritative test BRSR ID "
        "occurs in KD training data."
    )

    # ==================================================================
    # BASIC DATA AUDIT
    # ==================================================================

    missing_test_text = [
        bid
        for bid in test_ids
        if not brsr_texts.get(bid)
    ]

    if missing_test_text:

        raise RuntimeError(
            "Missing BRSR text for test IDs: "
            f"{missing_test_text}"
        )

    missing_gold = [
        bid
        for bid, targets in gold_targets.items()
        if not targets
    ]

    if missing_gold:

        raise RuntimeError(
            "Test IDs without ESRS gold targets: "
            f"{missing_gold}"
        )

    print()
    print(
        "=" * 80
    )
    print(
        "OFFICIAL SYMBOLIC SOURCES"
    )
    print(
        "=" * 80
    )

    print(
        f"BRSR text entries: "
        f"{len(brsr_texts)}"
    )

    print(
        f"GRI bridge nodes: "
        f"{len(gri_to_esrs)}"
    )

    print(
        f"ESRS datapoints in candidate universe: "
        f"{len(esrs_ids)}"
    )

    # ==================================================================
    # LOAD BASE MODEL
    # ==================================================================

    print()
    print(
        "Loading base retriever..."
    )

    base_model = SentenceTransformer(
        args.base_model,
        device=device,
    )

    print(
        "Encoding ESRS universe..."
    )

    base_embeddings = base_model.encode(
        [
            esrs_texts[eid]
            for eid in esrs_ids
        ],
        batch_size=args.batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    # ==================================================================
    # LOAD CROSS ENCODER
    # ==================================================================

    print()
    print(
        "Loading cross encoder..."
    )

    ce_model = CrossEncoder(
        args.ce_model,
        device=device,
    )

    # ==================================================================
    # LOAD KD MODELS
    # ==================================================================

    print()
    print(
        "Loading KD GOLD..."
    )

    kd_gold = SentenceTransformer(
        args.kd_gold,
        device=device,
    )

    print(
        "Loading KD GOLD+MEDIUM..."
    )

    kd_gold_medium = SentenceTransformer(
        args.kd_gold_medium,
        device=device,
    )

    # ==================================================================
    # EVALUATION
    # ==================================================================

    print()
    print(
        "=" * 80
    )
    print(
        "FIXED HELD-OUT EVALUATION"
    )
    print(
        "=" * 80
    )

    metric_rows = []
    diagnostic_rows = []
    ranking_rows = []

    for bid in sorted(test_ids):

        query = brsr_texts[bid]

        gold_set = gold_targets[bid]

        print()
        print(
            "-" * 80
        )

        print(
            f"BRSR: {bid}"
        )

        print(
            "Gold:",
            "|".join(
                sorted(gold_set)
            ),
        )

        metrics, diagnostics, rankings = (
            evaluate_query(
                bid=bid,
                query=query,
                gold_set=gold_set,
                candidate_df=candidate_df,
                brsr_to_gri=brsr_to_gri,
                gri_to_esrs=gri_to_esrs,
                esrs_texts=esrs_texts,
                esrs_index=esrs_index,
                base_model=base_model,
                base_embeddings=base_embeddings,
                ce_model=ce_model,
                kd_gold=kd_gold,
                kd_gold_medium=kd_gold_medium,
                rerank_top=args.rerank_top,
            )
        )

        print(
            f"GRI count: "
            f"{diagnostics['gri_nodes']}"
        )

        print(
            f"Symbolic candidates: "
            f"{diagnostics['symbolic_candidates']}"
        )

        print(
            f"Bridge gold reached: "
            f"{diagnostics['bridge_gold_reached']}"
        )

        for system in SYSTEMS:

            values = metrics[system]

            metric_rows.append({
                "brsr_id": bid,
                "system": system,
                **values,
            })

            print(
                f"{system:32s} "
                f"R@1={values['recall@1']:.4f} "
                f"R@5={values['recall@5']:.4f} "
                f"R@10={values['recall@10']:.4f} "
                f"MRR={values['mrr']:.4f}"
            )

            for rank, eid in enumerate(
                rankings[system],
                start=1,
            ):

                ranking_rows.append({
                    "brsr_id": bid,
                    "system": system,
                    "rank": rank,
                    "esrs_datapoint_id": eid,
                    "is_gold": (
                        eid in gold_set
                    ),
                })

        diagnostic_rows.append(
            diagnostics
        )

    # ==================================================================
    # SUMMARY
    # ==================================================================

    print()
    print(
        "=" * 80
    )
    print(
        "FINAL KD ABLATION RESULTS"
    )
    print(
        "=" * 80
    )

    summary_rows = []

    for system in SYSTEMS:

        values = aggregate_system(
            metric_rows,
            system,
        )

        summary_row = {
            "system": system,
            "evaluated_brsr_ids": len(
                test_ids
            ),
            **values,
        }

        summary_rows.append(
            summary_row
        )

        print()
        print(
            "-" * 80
        )

        print(
            system.upper()
        )

        print(
            f"evaluated_brsr_ids : "
            f"{len(test_ids)}"
        )

        for metric in METRIC_NAMES:

            print(
                f"{metric:20s}: "
                f"{values[metric]:.4f}"
            )

    # ==================================================================
    # BRIDGE / EFFICIENCY
    # ==================================================================

    diagnostics_df = pd.DataFrame(
        diagnostic_rows
    )

    if not diagnostics_df.empty:

        print()
        print(
            "=" * 80
        )
        print(
            "BRIDGE / EFFICIENCY SUMMARY"
        )
        print(
            "=" * 80
        )

        print(
            "Mean symbolic candidates/query: "
            f"{diagnostics_df['symbolic_candidates'].mean():.2f}"
        )

        print(
            "Mean bridge gold coverage/query:",
            f"{(diagnostics_df['bridge_gold_reached'] > 0).mean():.4f}",
        )

        print(
            "Mean GRI nodes/query: "
            f"{diagnostics_df['gri_nodes'].mean():.2f}"
        )

        print(
            "Mean direct candidates/query: "
            f"{diagnostics_df['direct_candidates'].mean():.2f}"
        )

    # ==================================================================
    # SAVE
    # ==================================================================

    metrics_df = pd.DataFrame(
        metric_rows
    )

    summary_df = pd.DataFrame(
        summary_rows
    )

    rankings_df = pd.DataFrame(
        ranking_rows
    )

    metrics_path = (
        output_dir
        /
        f"{args.output_prefix}_per_query.csv"
    )

    summary_path = (
        output_dir
        /
        f"{args.output_prefix}_summary.csv"
    )

    diagnostics_path = (
        output_dir
        /
        f"{args.output_prefix}_diagnostics.csv"
    )

    rankings_path = (
        output_dir
        /
        f"{args.output_prefix}_rankings.csv"
    )

    manifest_path = (
        output_dir
        /
        f"{args.output_prefix}_manifest.json"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    diagnostics_df.to_csv(
        diagnostics_path,
        index=False,
    )

    rankings_df.to_csv(
        rankings_path,
        index=False,
    )

    # ==================================================================
    # MANIFEST
    # ==================================================================

    manifest = {

        "script":
            "15",

        "experiment":
            "fixed_held_out_evaluation",

        "seed":
            args.seed,

        "candidate_file":
            str(
                Path(args.candidate_file)
            ),

        "test_file":
            str(
                Path(args.test_file)
            ),

        "base_model":
            str(args.base_model),

        "ce_model":
            str(
                Path(args.ce_model)
            ),

        "kd_gold":
            str(
                Path(args.kd_gold)
            ),

        "kd_gold_medium":
            str(
                Path(args.kd_gold_medium)
            ),

        "rerank_top":
            args.rerank_top,

        "test_brsr_ids":
            sorted(test_ids),

        "test_brsr_count":
            len(test_ids),

        "kd_training_brsr_ids":
            sorted(kd_training_ids),

        "kd_training_brsr_count":
            len(kd_training_ids),

        "train_test_overlap":
            sorted(leakage),

        "training_performed":
            False,

        "models_frozen":
            True,

        "authoritative_test_file":
            True,

        "candidate_generation_uses_test_labels":
            False,
    }

    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2,
        )

    # ==================================================================
    # FINAL
    # ==================================================================

    print()
    print(
        "=" * 80
    )
    print(
        "SCRIPT 15 COMPLETE"
    )
    print(
        "=" * 80
    )

    print()
    print(
        "Authoritative test IDs:"
    )

    print(
        "  "
        + ", ".join(
            sorted(test_ids)
        )
    )

    print()
    print(
        "Outputs:"
    )

    print(
        f"  Per-query metrics: "
        f"{metrics_path}"
    )

    print(
        f"  Summary: "
        f"{summary_path}"
    )

    print(
        f"  Diagnostics: "
        f"{diagnostics_path}"
    )

    print(
        f"  Rankings: "
        f"{rankings_path}"
    )

    print(
        f"  Manifest: "
        f"{manifest_path}"
    )

    print()
    print(
        "Protocol checks:"
    )

    print(
        "  [PASS] Authoritative final_test.csv used"
    )

    print(
        "  [PASS] Fixed held-out test IDs evaluated"
    )

    print(
        "  [PASS] No training performed"
    )

    print(
        "  [PASS] KD checkpoints frozen"
    )

    print(
        "  [PASS] KD training/test IDs disjoint"
    )

    print(
        "  [PASS] Test labels used only for evaluation"
    )


if __name__ == "__main__":
    main()s