#!/usr/bin/env python3
"""
SCRIPT 16 — LARGE-SCALE FROZEN ZERO-SHOT EVALUATION

Purpose
-------
Evaluate frozen retrieval / KD models on the authoritative held-out test set.

Protocol
--------
1. Authoritative test set:
       data/final/final_test.csv

2. Held-out BRSR IDs:
       A13, B6, P1-E7, P5-E2, P6-E1

3. KD training IDs:
       read from the KD training files and verified disjoint from test IDs.

4. No training, fitting, threshold tuning, or recalibration is performed.

5. Gold ESRS column:
       esrs_datapoint_id

6. Compared systems:
       DIRECT
       SYMBOLIC_MULTIHOP
       MULTIHOP_CROSS_ENCODER
       MULTIHOP_KD_GOLD
       MULTIHOP_KD_GOLD_MEDIUM

The script can be run independently for the original and expanded
checkpoints.

Example
-------
python scripts/16_large_scale_zero_shot_evaluation.py \
    --ce-model models/cross_encoder_gold \
    --kd-gold models/kd_student_gold/final \
    --kd-gold-medium models/kd_student_gold_medium/final \
    --ranking-file data/processed/script17_zero_shot_rankings.csv \
    --output-dir data/results/zero_shot_original

Expanded:
python scripts/16_large_scale_zero_shot_evaluation.py \
    --ce-model models/cross_encoder_expanded_gold \
    --kd-gold models/kd_student_expanded_gold/final \
    --kd-gold-medium models/kd_student_expanded_gold_medium/final \
    --ranking-file data/processed/script17_zero_shot_rankings.csv \
    --output-dir data/results/zero_shot_expanded
"""

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
FINAL_DIR = DATA_DIR / "final"
MODELS_DIR = ROOT / "models"

DEFAULT_CANDIDATE_FILE = (
    PROCESSED_DIR / "final_brsr_gri_esrs_candidates.csv"
)

DEFAULT_GOLD_FILE = FINAL_DIR / "final_test.csv"

DEFAULT_RANKING_FILE = (
    PROCESSED_DIR / "script17_zero_shot_rankings.csv"
)

DEFAULT_KD_TRAINING_FILES = [
    PROCESSED_DIR / "kd_symbolic_candidates_train.csv",
    PROCESSED_DIR / "kd_training_data_gold.csv",
    PROCESSED_DIR / "kd_training_data_gold_medium.csv",
]

DEFAULT_BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------------
# AUTHORITATIVE TEST IDs
# ---------------------------------------------------------------------

AUTHORITATIVE_TEST_IDS = {
    "A13",
    "B6",
    "P1-E7",
    "P5-E2",
    "P6-E1",
}


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def norm_id(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def split_gold(value):
    """
    Gold datapoints are pipe-separated in final_test.csv.
    """
    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value:
        return []

    return [
        x.strip()
        for x in value.split("|")
        if x.strip()
    ]


def safe_float(x, default=0.0):
    try:
        v = float(x)
        if np.isnan(v):
            return default
        return v
    except Exception:
        return default


# ---------------------------------------------------------------------
# GOLD
# ---------------------------------------------------------------------

def load_authoritative_gold(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    gold = pd.read_csv(path)

    required = {"brsr_id", "esrs_datapoint_id"}

    missing = required - set(gold.columns)

    if missing:
        raise ValueError(
            f"Authoritative gold file is missing columns: {sorted(missing)}\n"
            f"Columns: {list(gold.columns)}"
        )

    gold["brsr_id"] = gold["brsr_id"].map(norm_id)
    gold["esrs_datapoint_id"] = gold["esrs_datapoint_id"].map(norm_id)

    gold = gold[
        gold["brsr_id"].isin(AUTHORITATIVE_TEST_IDS)
    ].copy()

    if gold.empty:
        raise ValueError(
            "No authoritative test rows found for the expected "
            f"IDs: {sorted(AUTHORITATIVE_TEST_IDS)}"
        )

    gold_sets = {}

    for brsr_id, group in gold.groupby("brsr_id"):
        vals = set(
            x
            for x in group["esrs_datapoint_id"]
            if x
        )

        if vals:
            gold_sets[brsr_id] = vals

    found_ids = set(gold_sets)

    if found_ids != AUTHORITATIVE_TEST_IDS:
        raise ValueError(
            "Authoritative test ID mismatch.\n"
            f"Expected: {sorted(AUTHORITATIVE_TEST_IDS)}\n"
            f"Found:    {sorted(found_ids)}"
        )

    return gold, gold_sets


# ---------------------------------------------------------------------
# KD TRAINING ISOLATION
# ---------------------------------------------------------------------

def load_kd_training_ids(paths):
    all_ids = set()
    audit = []

    for path in paths:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"KD training file not found: {path}"
            )

        df = pd.read_csv(path)

        if "brsr_id" not in df.columns:
            raise ValueError(
                f"KD training file has no brsr_id column: {path}"
            )

        ids = {
            norm_id(x)
            for x in df["brsr_id"]
            if norm_id(x)
        }

        all_ids.update(ids)

        audit.append({
            "file": str(path),
            "n_ids": len(ids),
        })

    overlap = all_ids & AUTHORITATIVE_TEST_IDS

    if overlap:
        raise RuntimeError(
            "KD TRAIN/TEST ISOLATION FAILED.\n"
            f"Overlapping IDs: {sorted(overlap)}"
        )

    return all_ids, audit


# ---------------------------------------------------------------------
# CANDIDATE TABLE
# ---------------------------------------------------------------------

def load_candidates(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = {
        "brsr_id",
        "esrs_datapoint_id",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Candidate file missing columns: {sorted(missing)}\n"
            f"Columns: {list(df.columns)}"
        )

    df["brsr_id"] = df["brsr_id"].map(norm_id)
    df["esrs_datapoint_id"] = (
        df["esrs_datapoint_id"].map(norm_id)
    )

    return df


# ---------------------------------------------------------------------
# TEXT CONSTRUCTION
# ---------------------------------------------------------------------

def find_first_existing(row, columns):
    for col in columns:
        if col in row.index:
            value = row[col]

            if pd.notna(value):
                value = str(value).strip()

                if value:
                    return value

    return ""


def candidate_text(row):
    """
    Construct a stable ESRS representation from the candidate table.
    """

    parts = []

    datapoint = find_first_existing(
        row,
        [
            "esrs_datapoint_id",
            "esrs_id",
        ],
    )

    name = find_first_existing(
        row,
        [
            "esrs_name",
            "esrs_datapoint_name",
            "datapoint_name",
        ],
    )

    topic = find_first_existing(
        row,
        [
            "esrs_topic_code",
            "esrs_topic",
        ],
    )

    dr = find_first_existing(
        row,
        [
            "esrs_dr",
            "esrs_disclosure_requirement",
        ],
    )

    dtype = find_first_existing(
        row,
        [
            "esrs_data_type",
            "data_type",
        ],
    )

    if datapoint:
        parts.append(datapoint)

    if name:
        parts.append(name)

    if topic:
        parts.append(topic)

    if dr:
        parts.append(dr)

    if dtype:
        parts.append(dtype)

    return " | ".join(parts)


def brsr_text(row):
    parts = []

    for col in [
        "brsr_question",
        "brsr_text",
        "brsr_name",
        "brsr_description",
        "question",
        "description",
    ]:
        if col in row.index and pd.notna(row[col]):
            value = str(row[col]).strip()

            if value:
                parts.append(value)

    if not parts:
        parts.append(norm_id(row.get("brsr_id", "")))

    return " | ".join(parts)


# ---------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------

def reciprocal_rank(ranked_ids, gold):
    for rank, item in enumerate(ranked_ids, start=1):
        if item in gold:
            return 1.0 / rank

    return 0.0


def recall_at_k(ranked_ids, gold, k):
    if not gold:
        return 0.0

    return len(set(ranked_ids[:k]) & gold) / len(gold)


def evaluate_ranking(ranked_ids, gold):
    return {
        "recall@1": recall_at_k(ranked_ids, gold, 1),
        "recall@5": recall_at_k(ranked_ids, gold, 5),
        "recall@10": recall_at_k(ranked_ids, gold, 10),
        "recall@20": recall_at_k(ranked_ids, gold, 20),
        "recall@50": recall_at_k(ranked_ids, gold, 50),
        "mrr": reciprocal_rank(ranked_ids, gold),
    }


# ---------------------------------------------------------------------
# RANKING FILE
# ---------------------------------------------------------------------

def inspect_ranking_file(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    ranking = pd.read_csv(path)

    if "brsr_id" not in ranking.columns:
        raise ValueError(
            "Ranking file must contain brsr_id.\n"
            f"Columns: {list(ranking.columns)}"
        )

    ranking["brsr_id"] = ranking["brsr_id"].map(norm_id)

    if "esrs_datapoint_id" in ranking.columns:
        ranking["esrs_datapoint_id"] = (
            ranking["esrs_datapoint_id"].map(norm_id)
        )

    return ranking


def discover_score_column(df, keywords):
    """
    Find a ranking score column without assuming one exact
    script17 schema.
    """

    cols = list(df.columns)

    for keyword in keywords:
        for col in cols:
            c = col.lower()

            if keyword in c:
                if any(
                    token in c
                    for token in [
                        "score",
                        "sim",
                        "rank",
                        "prob",
                    ]
                ):
                    return col

    return None


def ranking_from_score(df, score_col):
    tmp = df.copy()

    tmp["_score"] = pd.to_numeric(
        tmp[score_col],
        errors="coerce"
    ).fillna(-np.inf)

    tmp = tmp.sort_values(
        ["_score"],
        ascending=False,
        kind="mergesort",
    )

    return tmp["esrs_datapoint_id"].tolist()


# ---------------------------------------------------------------------
# MODEL RANKING
# ---------------------------------------------------------------------

@torch.inference_mode()
def encode_candidates(model, texts, batch_size=64):
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )


def cosine_rank(query_embedding, candidate_embeddings, ids):
    scores = candidate_embeddings @ query_embedding

    order = np.argsort(-scores)

    return [
        ids[i]
        for i in order
    ], scores


def cross_encoder_rank(model, query, candidate_texts, ids):
    pairs = [
        [query, text]
        for text in candidate_texts
    ]

    scores = model.predict(
        pairs,
        batch_size=32,
        show_progress_bar=False,
    )

    scores = np.asarray(scores)

    order = np.argsort(-scores)

    return [
        ids[i]
        for i in order
    ], scores


# ---------------------------------------------------------------------
# SYMBOLIC CANDIDATES
# ---------------------------------------------------------------------

def build_symbolic_candidates(group):
    """
    The candidate table already represents the symbolic composition
    BRSR -> GRI -> ESRS.

    Deduplicate ESRS datapoints while preserving the best available
    candidate ordering from the source table.
    """

    if "symbolic_score" in group.columns:
        group = group.copy()

        group["symbolic_score"] = pd.to_numeric(
            group["symbolic_score"],
            errors="coerce"
        ).fillna(0.0)

        group = group.sort_values(
            "symbolic_score",
            ascending=False,
            kind="mergesort",
        )

    elif "mapping_score" in group.columns:
        group = group.copy()

        group["mapping_score"] = pd.to_numeric(
            group["mapping_score"],
            errors="coerce"
        ).fillna(0.0)

        group = group.sort_values(
            "mapping_score",
            ascending=False,
            kind="mergesort",
        )

    return list(
        dict.fromkeys(
            x
            for x in group["esrs_datapoint_id"]
            if x
        )
    )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--candidate-file",
        default=str(DEFAULT_CANDIDATE_FILE),
    )

    parser.add_argument(
        "--gold-file",
        default=str(DEFAULT_GOLD_FILE),
    )

    parser.add_argument(
        "--ranking-file",
        default=str(DEFAULT_RANKING_FILE),
    )

    parser.add_argument(
        "--ce-model",
        default=None,
    )

    parser.add_argument(
        "--kd-gold",
        default=None,
    )

    parser.add_argument(
        "--kd-gold-medium",
        default=None,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("SCRIPT 16 — LARGE-SCALE FROZEN ZERO-SHOT EVALUATION")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    print(f"Candidate file: {args.candidate_file}")
    print(f"Gold file:      {args.gold_file}")
    print(f"Ranking file:   {args.ranking_file}")

    # ---------------------------------------------------------------
    # GOLD
    # ---------------------------------------------------------------

    gold_df, gold_sets = load_authoritative_gold(
        args.gold_file
    )

    print("\n" + "=" * 80)
    print("AUTHORITATIVE ZERO-SHOT TEST SET")
    print("=" * 80)

    print(
        f"Test BRSR IDs: {len(gold_sets)}"
    )

    print(
        "IDs:",
        ", ".join(sorted(gold_sets)),
    )

    # ---------------------------------------------------------------
    # KD TRAINING ISOLATION
    # ---------------------------------------------------------------

    kd_training_ids, audit = load_kd_training_ids(
        DEFAULT_KD_TRAINING_FILES
    )

    print("\n" + "=" * 80)
    print("KD TRAINING ISOLATION")
    print("=" * 80)

    for item in audit:
        print(
            f"  {Path(item['file']).name}: "
            f"{item['n_ids']} BRSR IDs"
        )

    print(
        f"KD training BRSR IDs: {len(kd_training_ids)}"
    )

    print(
        "IDs:",
        ", ".join(sorted(kd_training_ids)),
    )

    overlap = (
        kd_training_ids &
        set(gold_sets)
    )

    if overlap:
        raise RuntimeError(
            f"ZERO-SHOT ISOLATION FAILED: {sorted(overlap)}"
        )

    print(
        "[PASS] KD training IDs are disjoint "
        "from authoritative zero-shot IDs."
    )

    # ---------------------------------------------------------------
    # CANDIDATES
    # ---------------------------------------------------------------

    candidates = load_candidates(
        args.candidate_file
    )

    print("\n" + "=" * 80)
    print("CANDIDATE UNIVERSE")
    print("=" * 80)

    print(
        f"Candidate rows: {len(candidates):,}"
    )

    print(
        "ESRS datapoints:",
        candidates["esrs_datapoint_id"].nunique(),
    )

    test_candidates = candidates[
        candidates["brsr_id"].isin(gold_sets)
    ].copy()

    # ---------------------------------------------------------------
    # BASE ENCODER
    # ---------------------------------------------------------------

    print("\nLoading base retriever...")

    embedder = SentenceTransformer(
        args.base_model,
        device=device,
    )

    # ---------------------------------------------------------------
    # GLOBAL ESRS UNIVERSE
    # ---------------------------------------------------------------

    esrs_rows = (
        candidates[
            [
                "esrs_datapoint_id"
            ]
            + [
                c for c in [
                    "esrs_name",
                    "esrs_topic_code",
                    "esrs_dr",
                    "esrs_data_type",
                ]
                if c in candidates.columns
            ]
        ]
        .drop_duplicates(
            "esrs_datapoint_id"
        )
        .copy()
    )

    esrs_rows["candidate_text"] = (
        esrs_rows.apply(
            candidate_text,
            axis=1,
        )
    )

    esrs_ids = (
        esrs_rows[
            "esrs_datapoint_id"
        ].tolist()
    )

    esrs_texts = (
        esrs_rows[
            "candidate_text"
        ].tolist()
    )

    print(
        f"Encoding ESRS universe: {len(esrs_ids)} datapoints"
    )

    esrs_embeddings = encode_candidates(
        embedder,
        esrs_texts,
    )

    esrs_index = {
        x: i
        for i, x in enumerate(esrs_ids)
    }

    # ---------------------------------------------------------------
    # CROSS ENCODER
    # ---------------------------------------------------------------

    ce = None

    if args.ce_model:
        print("\nLoading cross encoder...")

        ce = CrossEncoder(
            args.ce_model,
            device=device,
        )

    # ---------------------------------------------------------------
    # KD MODELS
    # ---------------------------------------------------------------

    kd_gold = None
    kd_gold_medium = None

    if args.kd_gold:
        print("\nLoading KD GOLD...")

        kd_gold = SentenceTransformer(
            args.kd_gold,
            device=device,
        )

    if args.kd_gold_medium:
        print("\nLoading KD GOLD+MEDIUM...")

        kd_gold_medium = SentenceTransformer(
            args.kd_gold_medium,
            device=device,
        )

    # ---------------------------------------------------------------
    # KD UNIVERSE ENCODING  [FIX]
    # ---------------------------------------------------------------
    # BUG FIXED: the previous version of this script encoded the
    # QUERY with the KD model but scored it against `esrs_embeddings`,
    # which was encoded with the BASE embedder (`embedder`), never
    # re-run through the KD model. Query and document vectors were
    # therefore in two different, uncalibrated embedding spaces, so
    # every "multihop_kd_gold"/"multihop_kd_gold_medium" score in
    # previously generated results (data/results/zero_shot*) is
    # unreliable and should not be cited. Fixed by encoding the full
    # ESRS universe once per KD model, upfront, exactly as already
    # done for the base embedder above.

    kd_gold_esrs_embeddings = None
    kd_gold_medium_esrs_embeddings = None

    if kd_gold is not None:
        print("\nEncoding ESRS universe with KD GOLD...")

        kd_gold_esrs_embeddings = encode_candidates(
            kd_gold,
            esrs_texts,
        )

    if kd_gold_medium is not None:
        print("\nEncoding ESRS universe with KD GOLD+MEDIUM...")

        kd_gold_medium_esrs_embeddings = encode_candidates(
            kd_gold_medium,
            esrs_texts,
        )

    # ---------------------------------------------------------------
    # EVALUATION
    # ---------------------------------------------------------------

    methods = [
        "direct",
        "symbolic_multihop",
    ]

    if ce is not None:
        methods.append(
            "multihop_cross_encoder"
        )

    if kd_gold is not None:
        methods.append(
            "multihop_kd_gold"
        )

    if kd_gold_medium is not None:
        methods.append(
            "multihop_kd_gold_medium"
        )

    # NEW — true no-bridge baselines, scored against the FULL ESRS
    # universe (esrs_ids/esrs_texts), independent of the symbolic
    # BRSR -> GRI -> ESRS candidate restriction.

    if ce is not None:
        methods.append(
            "direct_full_cross_encoder"
        )

    if kd_gold is not None:
        methods.append(
            "direct_full_kd_gold"
        )

    if kd_gold_medium is not None:
        methods.append(
            "direct_full_kd_gold_medium"
        )

    per_query = []
    rankings = []

    print("\n" + "=" * 80)
    print("FROZEN ZERO-SHOT EVALUATION")
    print("=" * 80)

    for brsr_id in sorted(gold_sets):

        gold = gold_sets[brsr_id]

        group = test_candidates[
            test_candidates["brsr_id"] == brsr_id
        ].copy()

        if group.empty:
            print(
                f"[WARNING] No candidate rows for {brsr_id}"
            )
            continue

        # -----------------------------------------------------------
        # Query representation
        # -----------------------------------------------------------

        query_row = group.iloc[0]

        query_text = brsr_text(query_row)

        query_embedding = embedder.encode(
            query_text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        # -----------------------------------------------------------
        # DIRECT
        # -----------------------------------------------------------

        direct_ids, direct_scores = cosine_rank(
            query_embedding,
            esrs_embeddings,
            esrs_ids,
        )

        # -----------------------------------------------------------
        # SYMBOLIC
        # -----------------------------------------------------------

        symbolic_ids = build_symbolic_candidates(
            group
        )

        # Restrict symbolic ranking to the candidate universe.
        symbolic_set = set(symbolic_ids)

        symbolic_order = [
            x
            for x in direct_ids
            if x in symbolic_set
        ]

        # -----------------------------------------------------------
        # CE MULTIHOP
        # -----------------------------------------------------------

        ce_ids = None

        if ce is not None:

            ce_candidate_ids = symbolic_ids

            ce_text_map = {
                row["esrs_datapoint_id"]:
                candidate_text(row)
                for _, row in
                esrs_rows.iterrows()
            }

            ce_texts = [
                ce_text_map[x]
                for x in ce_candidate_ids
                if x in ce_text_map
            ]

            ce_candidate_ids = [
                x
                for x in ce_candidate_ids
                if x in ce_text_map
            ]

            ce_ids, _ = cross_encoder_rank(
                ce,
                query_text,
                ce_texts,
                ce_candidate_ids,
            )

        # -----------------------------------------------------------
        # KD MULTIHOP
        # -----------------------------------------------------------

        kd_gold_ids = None
        kd_medium_ids = None

        if kd_gold is not None:

            query_kd = kd_gold.encode(
                query_text,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

            symbolic_indices = [
                esrs_index[x]
                for x in symbolic_ids
                if x in esrs_index
            ]

            kd_emb = kd_gold_esrs_embeddings[
                symbolic_indices
            ]

            kd_candidate_ids = [
                esrs_ids[i]
                for i in symbolic_indices
            ]

            kd_gold_ids, _ = cosine_rank(
                query_kd,
                kd_emb,
                kd_candidate_ids,
            )

        if kd_gold_medium is not None:

            query_kd_medium = (
                kd_gold_medium.encode(
                    query_text,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
            )

            symbolic_indices = [
                esrs_index[x]
                for x in symbolic_ids
                if x in esrs_index
            ]

            kd_emb = kd_gold_medium_esrs_embeddings[
                symbolic_indices
            ]

            kd_candidate_ids = [
                esrs_ids[i]
                for i in symbolic_indices
            ]

            kd_medium_ids, _ = cosine_rank(
                query_kd_medium,
                kd_emb,
                kd_candidate_ids,
            )

        # -----------------------------------------------------------
        # DIRECT_FULL — no-bridge baselines over the full ESRS universe
        # -----------------------------------------------------------

        ce_full_ids = None
        kd_gold_full_ids = None
        kd_gold_medium_full_ids = None

        if ce is not None:

            ce_full_ids, _ = cross_encoder_rank(
                ce,
                query_text,
                esrs_texts,
                esrs_ids,
            )

        if kd_gold is not None:

            kd_gold_full_ids, _ = cosine_rank(
                query_kd,
                kd_gold_esrs_embeddings,
                esrs_ids,
            )

        if kd_gold_medium is not None:

            kd_gold_medium_full_ids, _ = cosine_rank(
                query_kd_medium,
                kd_gold_medium_esrs_embeddings,
                esrs_ids,
            )

        # -----------------------------------------------------------
        # METHOD RESULTS
        # -----------------------------------------------------------

        method_rankings = {
            "direct": direct_ids,
            "symbolic_multihop": symbolic_order,
        }

        if ce_ids is not None:
            method_rankings[
                "multihop_cross_encoder"
            ] = ce_ids

        if kd_gold_ids is not None:
            method_rankings[
                "multihop_kd_gold"
            ] = kd_gold_ids

        if kd_medium_ids is not None:
            method_rankings[
                "multihop_kd_gold_medium"
            ] = kd_medium_ids

        if ce_full_ids is not None:
            method_rankings[
                "direct_full_cross_encoder"
            ] = ce_full_ids

        if kd_gold_full_ids is not None:
            method_rankings[
                "direct_full_kd_gold"
            ] = kd_gold_full_ids

        if kd_gold_medium_full_ids is not None:
            method_rankings[
                "direct_full_kd_gold_medium"
            ] = kd_gold_medium_full_ids

        # -----------------------------------------------------------
        # PER-QUERY METRICS
        # -----------------------------------------------------------

        print("\n" + "-" * 80)
        print(f"BRSR: {brsr_id}")
        print(
            "Gold:",
            "|".join(sorted(gold)),
        )
        print(
            f"Symbolic candidates: {len(symbolic_ids)}"
        )

        bridge_gold = (
            set(symbolic_ids) & gold
        )

        print(
            f"Bridge gold reached: {len(bridge_gold)}"
        )

        for method in methods:

            ranking = method_rankings[
                method
            ]

            metrics = evaluate_ranking(
                ranking,
                gold,
            )

            print(
                f"{method:32s} "
                f"R@1={metrics['recall@1']:.4f} "
                f"R@5={metrics['recall@5']:.4f} "
                f"R@10={metrics['recall@10']:.4f} "
                f"R@20={metrics['recall@20']:.4f} "
                f"R@50={metrics['recall@50']:.4f} "
                f"MRR={metrics['mrr']:.4f}"
            )

            row = {
                "brsr_id": brsr_id,
                "method": method,
                "n_gold": len(gold),
                "n_symbolic_candidates": len(symbolic_ids),
                "bridge_gold_reached": len(
                    bridge_gold
                ),
                **metrics,
            }

            per_query.append(row)

            for rank, esrs_id in enumerate(
                ranking,
                start=1,
            ):

                rankings.append({
                    "brsr_id": brsr_id,
                    "method": method,
                    "rank": rank,
                    "esrs_datapoint_id": esrs_id,
                    "is_gold": int(
                        esrs_id in gold
                    ),
                })

    # -----------------------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------------------

    per_query_df = pd.DataFrame(
        per_query
    )

    summary_rows = []

    for method in methods:

        subset = per_query_df[
            per_query_df["method"] == method
        ]

        if subset.empty:
            continue

        summary_rows.append({
            "method": method,
            "evaluated_brsr_ids":
                subset["brsr_id"].nunique(),
            "recall@1":
                subset["recall@1"].mean(),
            "recall@5":
                subset["recall@5"].mean(),
            "recall@10":
                subset["recall@10"].mean(),
            "recall@20":
                subset["recall@20"].mean(),
            "recall@50":
                subset["recall@50"].mean(),
            "mrr":
                subset["mrr"].mean(),
        })

    summary_df = pd.DataFrame(
        summary_rows
    )

    # -----------------------------------------------------------------
    # EFFICIENCY
    # -----------------------------------------------------------------

    bridge_df = (
        per_query_df[
            per_query_df["method"]
            == "symbolic_multihop"
        ]
        .drop_duplicates("brsr_id")
    )

    efficiency = {
        "evaluated_brsr_ids":
            len(gold_sets),
        "mean_symbolic_candidates":
            float(
                bridge_df[
                    "n_symbolic_candidates"
                ].mean()
            )
            if not bridge_df.empty
            else 0.0,
        "mean_bridge_gold_reached":
            float(
                bridge_df[
                    "bridge_gold_reached"
                ].mean()
            )
            if not bridge_df.empty
            else 0.0,
        "mean_bridge_gold_coverage":
            float(
                np.mean([
                    len(
                        set(
                            build_symbolic_candidates(
                                test_candidates[
                                    test_candidates[
                                        "brsr_id"
                                    ] == brsr_id
                                ]
                            )
                        )
                        & gold_sets[brsr_id]
                    ) / len(gold_sets[brsr_id])
                    for brsr_id in gold_sets
                ])
            ),
        "mean_gri_nodes": (
            float(
                test_candidates[
                    test_candidates["brsr_id"]
                    .isin(gold_sets)
                ]
                .groupby("brsr_id")[
                    "gri_code"
                ]
                .nunique()
                .mean()
            )
            if "gri_code" in test_candidates.columns
            else None
        ),
        "full_esrs_universe":
            len(esrs_ids),
    }

    # -----------------------------------------------------------------
    # SAVE
    # -----------------------------------------------------------------

    per_query_path = (
        output_dir /
        "script16_zero_shot_per_query.csv"
    )

    summary_path = (
        output_dir /
        "script16_zero_shot_summary.csv"
    )

    rankings_path = (
        output_dir /
        "script16_zero_shot_rankings.csv"
    )

    efficiency_path = (
        output_dir /
        "script16_zero_shot_efficiency.json"
    )

    manifest_path = (
        output_dir /
        "script16_zero_shot_manifest.json"
    )

    per_query_df.to_csv(
        per_query_path,
        index=False,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    pd.DataFrame(
        rankings
    ).to_csv(
        rankings_path,
        index=False,
    )

    with open(
        efficiency_path,
        "w",
    ) as f:
        json.dump(
            efficiency,
            f,
            indent=2,
        )

    manifest = {
        "script": "16_large_scale_zero_shot_evaluation",
        "protocol": "frozen_zero_shot",
        "candidate_file": str(
            Path(args.candidate_file).resolve()
        ),
        "gold_file": str(
            Path(args.gold_file).resolve()
        ),
        "ranking_file": str(
            Path(args.ranking_file).resolve()
        ),
        "test_ids": sorted(
            gold_sets
        ),
        "kd_training_ids": sorted(
            kd_training_ids
        ),
        "kd_training_test_overlap": sorted(
            overlap
        ),
        "base_model": args.base_model,
        "cross_encoder": args.ce_model,
        "kd_gold": args.kd_gold,
        "kd_gold_medium": args.kd_gold_medium,
        "no_training": True,
        "no_calibration": True,
        "gold_column": "esrs_datapoint_id",
        "n_candidate_rows": len(
            candidates
        ),
        "n_esrs_datapoints": len(
            esrs_ids
        ),
        "methods": methods,
    }

    with open(
        manifest_path,
        "w",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    # -----------------------------------------------------------------
    # FINAL OUTPUT
    # -----------------------------------------------------------------

    print("\n" + "=" * 80)
    print("FINAL FROZEN ZERO-SHOT RESULTS")
    print("=" * 80)

    for _, row in summary_df.iterrows():

        print("\n" + "-" * 80)
        print(
            row["method"].upper()
        )

        print(
            f"evaluated_brsr_ids : "
            f"{int(row['evaluated_brsr_ids'])}"
        )

        print(
            f"recall@1            : "
            f"{row['recall@1']:.4f}"
        )

        print(
            f"recall@5            : "
            f"{row['recall@5']:.4f}"
        )

        print(
            f"recall@10           : "
            f"{row['recall@10']:.4f}"
        )

        print(
            f"recall@20           : "
            f"{row['recall@20']:.4f}"
        )

        print(
            f"recall@50           : "
            f"{row['recall@50']:.4f}"
        )

        print(
            f"mrr                 : "
            f"{row['mrr']:.4f}"
        )

    print("\n" + "=" * 80)
    print("ZERO-SHOT BRIDGE / EFFICIENCY")
    print("=" * 80)

    print(
        f"Mean symbolic candidates/query: "
        f"{efficiency['mean_symbolic_candidates']:.2f}"
    )

    print(
        f"Mean bridge gold reached/query: "
        f"{efficiency['mean_bridge_gold_reached']:.2f}"
    )

    print(
        f"Mean bridge gold coverage/query: "
        f"{efficiency['mean_bridge_gold_coverage']:.4f}"
    )

    if efficiency["mean_gri_nodes"] is not None:
        print(
            f"Mean GRI nodes/query: "
            f"{efficiency['mean_gri_nodes']:.2f}"
        )

    print(
        f"Full ESRS universe: "
        f"{efficiency['full_esrs_universe']}"
    )

    print("\n" + "=" * 80)
    print("PROTOCOL CHECKS")
    print("=" * 80)

    print(
        "[PASS] Authoritative final_test.csv used"
    )

    print(
        "[PASS] Fixed held-out zero-shot IDs evaluated"
    )

    print(
        "[PASS] No training performed"
    )

    print(
        "[PASS] KD checkpoints frozen"
    )

    print(
        "[PASS] KD training/test IDs disjoint"
    )

    print(
        "[PASS] Test labels used only for evaluation"
    )

    print("\nOutputs:")
    print(f"  {per_query_path}")
    print(f"  {summary_path}")
    print(f"  {rankings_path}")
    print(f"  {efficiency_path}")
    print(f"  {manifest_path}")

    print("\n" + "=" * 80)
    print("SCRIPT 16 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()