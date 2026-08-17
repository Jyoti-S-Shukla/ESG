#!/usr/bin/env python3
"""
SCRIPT 13 — END-TO-END MULTI-HOP EVALUATION AND ABLATION

Evaluates the proposed BRSR -> GRI -> ESRS framework against:

    1. Direct semantic retrieval
       BRSR -> ESRS

    2. Symbolic multi-hop retrieval
       BRSR -> GRI -> ESRS

    3. Multi-hop + cross-encoder reranking
       BRSR -> GRI -> ESRS -> Cross Encoder

The evaluation is performed ONLY on held-out BRSR IDs.

No training is performed here.

Outputs:
    data/processed/script13_metrics.csv
    data/processed/script13_per_query.csv
    data/processed/script13_bridge_diagnostics.csv
    data/processed/script13_candidate_stats.csv

"""

import os
import csv
import json
import math
import random
import argparse
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer, CrossEncoder


# =============================================================================
# PATHS
# =============================================================================

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FINAL_DIR = os.path.join(BASE, "data", "final")
PROCESSED_DIR = os.path.join(BASE, "data", "processed")
MODELS_DIR = os.path.join(BASE, "models")

TEST_FILE = os.path.join(FINAL_DIR, "final_test.csv")

CANDIDATES_FILE = os.path.join(
    PROCESSED_DIR, "transitive_candidates.csv"
)

GOLD_FILE = os.path.join(
    PROCESSED_DIR, "transitive_gold.csv"
)

REVIEW_FILE = os.path.join(
    PROCESSED_DIR, "transitive_review.csv"
)

CE_MODEL = os.path.join(
    MODELS_DIR, "cross_encoder_gold"
)

CE_MODEL_GOLD_MEDIUM = os.path.join(
    MODELS_DIR, "cross_encoder_gold_medium"
)


# =============================================================================
# OUTPUTS
# =============================================================================

METRICS_OUT = os.path.join(
    PROCESSED_DIR, "script13_metrics.csv"
)

PER_QUERY_OUT = os.path.join(
    PROCESSED_DIR, "script13_per_query.csv"
)

BRIDGE_OUT = os.path.join(
    PROCESSED_DIR, "script13_bridge_diagnostics.csv"
)

CANDIDATE_STATS_OUT = os.path.join(
    PROCESSED_DIR, "script13_candidate_stats.csv"
)


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = [1, 5, 10, 20, 50]

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

SEED = 42


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =============================================================================
# UTILITY
# =============================================================================

def norm(x):
    if x is None:
        return ""

    if isinstance(x, float) and np.isnan(x):
        return ""

    return str(x).strip()


def canonical_brsr_text(row):
    """
    Construct the BRSR query representation.

    Uses the actual BRSR text when available.
    """

    text = norm(row.get("brsr_text"))

    if text:
        return text

    parts = [
        norm(row.get("section")),
        norm(row.get("principle")),
        norm(row.get("indicator_type")),
    ]

    return " | ".join(x for x in parts if x)


def canonical_esrs_text(row):
    """
    Representation used by semantic retrieval / CE.
    """

    parts = [
        norm(row.get("esrs_datapoint_id")),
        norm(row.get("esrs_name")),
        norm(row.get("esrs_dr")),
        norm(row.get("esrs_topic_code")),
        norm(row.get("esrs_data_type")),
    ]

    return " | ".join(x for x in parts if x)


def reciprocal_rank(ranks):
    if not ranks:
        return 0.0

    return 1.0 / min(ranks)


def recall_at_k(ranked, gold, k):
    gold = set(gold)

    if not gold:
        return 0.0

    retrieved = set(ranked[:k])

    return 1.0 if retrieved.intersection(gold) else 0.0


def evaluate_ranked_results(results, gold_by_brsr):
    """
    results:
        dict BRSR -> ranked ESRS IDs

    Returns:
        Recall@K and MRR
    """

    valid = []

    for brsr_id, ranked in results.items():

        gold = gold_by_brsr.get(brsr_id, [])

        if not gold:
            continue

        valid.append((brsr_id, ranked, gold))

    metrics = {
        "evaluated_brsr_ids": len(valid)
    }

    for k in TOP_K:

        values = [
            recall_at_k(ranked, gold, k)
            for _, ranked, gold in valid
        ]

        metrics[f"recall@{k}"] = (
            float(np.mean(values)) if values else 0.0
        )

    rr = []

    for _, ranked, gold in valid:

        gold_set = set(gold)

        rank = None

        for i, item in enumerate(ranked, start=1):

            if item in gold_set:
                rank = i
                break

        rr.append(
            1.0 / rank if rank is not None else 0.0
        )

    metrics["mrr"] = float(np.mean(rr)) if rr else 0.0

    return metrics


# =============================================================================
# LOAD TEST GOLD
# =============================================================================

def load_test_gold():

    if not os.path.exists(TEST_FILE):
        raise FileNotFoundError(
            f"Missing test file:\n{TEST_FILE}"
        )

    df = pd.read_csv(TEST_FILE)

    required = {
        "brsr_id",
        "esrs_datapoint_id"
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"final_test.csv missing columns: {sorted(missing)}"
        )

    gold = defaultdict(set)

    for _, row in df.iterrows():

        brsr = norm(row["brsr_id"])
        esrs = norm(row["esrs_datapoint_id"])

        if brsr and esrs:
            gold[brsr].add(esrs)

    return {
        k: sorted(v)
        for k, v in gold.items()
    }


# =============================================================================
# LOAD CANDIDATE GRAPH
# =============================================================================

def load_candidates():

    df = pd.read_csv(CANDIDATES_FILE)

    required = {
        "brsr_id",
        "gri_code",
        "esrs_datapoint_id",
        "esrs_name"
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"transitive_candidates.csv missing columns: "
            f"{sorted(missing)}"
        )

    return df


# =============================================================================
# BUILD ESRS UNIVERSE
# =============================================================================

def build_esrs_universe(df):

    cols = [
        "esrs_datapoint_id",
        "esrs_sheet",
        "esrs_topic_code",
        "esrs_dr",
        "esrs_name",
        "esrs_data_type"
    ]

    cols = [c for c in cols if c in df.columns]

    esrs = (
        df[cols]
        .drop_duplicates("esrs_datapoint_id")
        .copy()
    )

    esrs["text"] = esrs.apply(
        canonical_esrs_text,
        axis=1
    )

    return esrs.reset_index(drop=True)


# =============================================================================
# BUILD BRSR TEXT
# =============================================================================

def build_brsr_lookup():

    """
    Uses final_test.csv where possible.

    This avoids contaminating evaluation with review rows.
    """

    df = pd.read_csv(TEST_FILE)

    if "brsr_text" not in df.columns:

        return {
            norm(r["brsr_id"]): norm(r["brsr_id"])
            for _, r in df.iterrows()
        }

    lookup = {}

    for _, row in df.iterrows():

        bid = norm(row["brsr_id"])

        if bid not in lookup:

            lookup[bid] = canonical_brsr_text(row)

    return lookup


# =============================================================================
# BUILD SYMBOLIC MULTI-HOP INDEX
# =============================================================================

def build_gri_index(candidates):

    """
    GRI -> ESRS adjacency.

    Only candidate graph rows are used.

    Important:
    This is NOT learning a GRI->ESRS relation.
    It is using the official/composed symbolic graph.
    """

    gri_to_esrs = defaultdict(set)

    for _, row in candidates.iterrows():

        gri = norm(row["gri_code"])
        esrs = norm(row["esrs_datapoint_id"])

        if gri and esrs:

            gri_to_esrs[gri].add(esrs)

    return {
        k: sorted(v)
        for k, v in gri_to_esrs.items()
    }


# =============================================================================
# BRSR -> GRI INDEX
# =============================================================================

def build_brsr_gri_index(candidates):

    index = defaultdict(set)

    for _, row in candidates.iterrows():

        brsr = norm(row["brsr_id"])
        gri = norm(row["gri_code"])

        if brsr and gri:

            index[brsr].add(gri)

    return {
        k: sorted(v)
        for k, v in index.items()
    }


# =============================================================================
# DIRECT RETRIEVAL
# =============================================================================

def direct_retrieval(
    brsr_text,
    esrs_df,
    embedder
):

    query_embedding = embedder.encode(
        [brsr_text],
        normalize_embeddings=True,
        convert_to_numpy=True
    )[0]

    corpus_embeddings = embedder.encode(
        esrs_df["text"].tolist(),
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    scores = np.dot(
        corpus_embeddings,
        query_embedding
    )

    order = np.argsort(-scores)

    ranked_ids = [
        norm(esrs_df.iloc[i]["esrs_datapoint_id"])
        for i in order
    ]

    ranked_scores = [
        float(scores[i])
        for i in order
    ]

    return ranked_ids, ranked_scores


# =============================================================================
# SYMBOLIC MULTI-HOP RETRIEVAL
# =============================================================================

def symbolic_multihop(
    brsr_id,
    brsr_gri_index,
    gri_esrs_index
):

    gri_nodes = brsr_gri_index.get(
        brsr_id,
        []
    )

    esrs_scores = Counter()

    provenance = defaultdict(set)

    for gri in gri_nodes:

        esrs_nodes = gri_esrs_index.get(
            gri,
            []
        )

        for esrs in esrs_nodes:

            # Simple path-count score.
            #
            # If multiple GRI disclosures independently
            # reach the same ESRS datapoint, that datapoint
            # receives stronger symbolic support.

            esrs_scores[esrs] += 1

            provenance[esrs].add(gri)

    ranked = sorted(
        esrs_scores.keys(),
        key=lambda x: (
            -esrs_scores[x],
            x
        )
    )

    return ranked, esrs_scores, provenance


# =============================================================================
# CROSS ENCODER RERANKING
# =============================================================================

def rerank_with_cross_encoder(
    brsr_text,
    candidate_esrs,
    esrs_lookup,
    cross_encoder,
    top_n=None
):

    if top_n is not None:

        candidate_esrs = candidate_esrs[:top_n]

    pairs = []

    valid_ids = []

    for esrs_id in candidate_esrs:

        row = esrs_lookup.get(esrs_id)

        if row is None:
            continue

        pairs.append(
            [
                brsr_text,
                row["text"]
            ]
        )

        valid_ids.append(esrs_id)

    if not pairs:

        return [], []

    scores = cross_encoder.predict(
        pairs,
        batch_size=32,
        show_progress_bar=False
    )

    scores = np.asarray(scores).reshape(-1)

    order = np.argsort(-scores)

    ranked = [
        valid_ids[i]
        for i in order
    ]

    ranked_scores = [
        float(scores[i])
        for i in order
    ]

    return ranked, ranked_scores


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--embedder",
        default=DEFAULT_EMBEDDER
    )

    parser.add_argument(
        "--ce-model",
        default=CE_MODEL
    )

    parser.add_argument(
        "--ce-model-gold-medium",
        default=CE_MODEL_GOLD_MEDIUM
    )

    parser.add_argument(
        "--rerank-top",
        type=int,
        default=50
    )

    args = parser.parse_args()

    print("=" * 80)
    print("SCRIPT 13 — END-TO-END MULTI-HOP EVALUATION")
    print("=" * 80)

    print()
    print("Device:", DEVICE)
    print("Embedder:", args.embedder)
    print("CE model:", args.ce_model)

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------

    gold_by_brsr = load_test_gold()

    candidates = load_candidates()

    brsr_lookup = build_brsr_lookup()

    esrs_df = build_esrs_universe(candidates)

    esrs_lookup = {
        norm(row["esrs_datapoint_id"]): row
        for _, row in esrs_df.iterrows()
    }

    brsr_gri_index = build_brsr_gri_index(
        candidates
    )

    gri_esrs_index = build_gri_index(
        candidates
    )

    print()
    print("Loaded:")
    print(
        f"  Test BRSR IDs:             {len(gold_by_brsr)}"
    )
    print(
        f"  Candidate rows:            {len(candidates)}"
    )
    print(
        f"  GRI -> ESRS keys:          {len(gri_esrs_index)}"
    )
    print(
        f"  ESRS datapoints:           {len(esrs_df)}"
    )

    # -------------------------------------------------------------------------
    # Embedder
    # -------------------------------------------------------------------------

    print()
    print("Loading dense retriever...")

    embedder = SentenceTransformer(
        args.embedder,
        device=DEVICE
    )

    # Pre-compute ESRS embeddings
    print("Encoding ESRS universe...")

    esrs_embeddings = embedder.encode(
        esrs_df["text"].tolist(),
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    # -------------------------------------------------------------------------
    # Cross encoder
    # -------------------------------------------------------------------------

    print()
    print("Loading cross encoder...")

    ce = CrossEncoder(
        args.ce_model,
        device=DEVICE
    )

    # -------------------------------------------------------------------------
    # Results
    # -------------------------------------------------------------------------

    direct_results = {}
    symbolic_results = {}
    ce_results = {}

    gold_ce_results = {}

    per_query_rows = []
    bridge_rows = []
    candidate_rows = []

    # -------------------------------------------------------------------------
    # Evaluate each BRSR
    # -------------------------------------------------------------------------

    for counter_idx, (brsr_id, gold_esrs) in enumerate(
        sorted(gold_by_brsr.items()),
        start=1
    ):

        text = brsr_lookup.get(
            brsr_id,
            brsr_id
        )

        print(
            f"\n[{counter_idx}/{len(gold_by_brsr)}] "
            f"{brsr_id}"
        )

        # =====================================================================
        # DIRECT
        # =====================================================================

        query_embedding = embedder.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True
        )[0]

        scores = np.dot(
            esrs_embeddings,
            query_embedding
        )

        order = np.argsort(-scores)

        direct_ranked = [
            norm(
                esrs_df.iloc[i][
                    "esrs_datapoint_id"
                ]
            )
            for i in order
        ]

        direct_scores = [
            float(scores[i])
            for i in order
        ]

        direct_results[brsr_id] = direct_ranked

        # =====================================================================
        # SYMBOLIC MULTI-HOP
        # =====================================================================

        symbolic_ranked, symbolic_counts, provenance = \
            symbolic_multihop(
                brsr_id,
                brsr_gri_index,
                gri_esrs_index
            )

        symbolic_results[brsr_id] = symbolic_ranked

        gri_nodes = brsr_gri_index.get(
            brsr_id,
            []
        )

        # =====================================================================
        # BRIDGE DIAGNOSTICS
        # =====================================================================

        gold_set = set(gold_esrs)

        bridge_hit = gold_set.intersection(symbolic_ranked)

        bridge_rank = None

        for rank, esrs_id in enumerate(
            symbolic_ranked,
            start=1
        ):

            if esrs_id in gold_set:

                bridge_rank = rank
                break

        bridge_rows.append({
            "brsr_id": brsr_id,

            "gold_esrs_count": len(gold_set),

            "gri_nodes": len(gri_nodes),
            "gri_codes": "|".join(gri_nodes),

            "symbolic_esrs_candidates": len(symbolic_ranked),

            # Binary: did the symbolic bridge reach at least one gold ESRS datapoint?
            "bridge_hit": int(bool(bridge_hit)),

            # Keep the existing name too for compatibility
            "bridge_recall": int(bool(bridge_hit)),

            # Rank of the first gold ESRS datapoint reached through the bridge
            "best_bridge_rank": (
                bridge_rank
                if bridge_rank is not None
                else -1
            ),

            # Number of gold ESRS datapoints reached through the bridge
            "gold_esrs_reached": len(bridge_hit),

            # Optional: actual set of reached gold datapoints for debugging
            "bridge_hit_esrs": "|".join(sorted(bridge_hit)),
        })

        # =====================================================================
        # CROSS-ENCODER RERANKING
        # =====================================================================

        ce_ranked, ce_scores = \
            rerank_with_cross_encoder(
                text,
                symbolic_ranked,
                esrs_lookup,
                ce,
                top_n=args.rerank_top
            )

        ce_results[brsr_id] = ce_ranked

        # =====================================================================
        # OPTIONAL: CE ON DIRECT TOP-K
        #
        # This gives us an additional ablation:
        #
        # BRSR -> ESRS dense retrieval -> CE
        #
        # This is useful to show whether the symbolic bridge itself
        # provides value.
        # =====================================================================

        direct_top = direct_ranked[
            :args.rerank_top
        ]

        direct_ce_ranked, direct_ce_scores = \
            rerank_with_cross_encoder(
                text,
                direct_top,
                esrs_lookup,
                ce,
                top_n=None
            )

        # =====================================================================
        # RANK DIAGNOSTICS
        # =====================================================================

        def get_rank(ranked, target_set):

            target_set = set(target_set)

            for r, x in enumerate(
                ranked,
                start=1
            ):

                if x in target_set:
                    return r

            return -1

        direct_rank = get_rank(
            direct_ranked,
            gold_esrs
        )

        symbolic_rank = get_rank(
            symbolic_ranked,
            gold_esrs
        )

        ce_rank = get_rank(
            ce_ranked,
            gold_esrs
        )

        direct_ce_rank = get_rank(
            direct_ce_ranked,
            gold_esrs
        )

        per_query_rows.append({

            "brsr_id": brsr_id,

            "gold_esrs": "|".join(
                sorted(gold_set)
            ),

            "gold_count": len(
                gold_set
            ),

            "gri_count": len(
                gri_nodes
            ),

            "direct_candidate_count":
                len(direct_ranked),

            "symbolic_candidate_count":
                len(symbolic_ranked),

            "direct_rank":
                direct_rank,

            "symbolic_rank":
                symbolic_rank,

            "multihop_ce_rank":
                ce_rank,

            "direct_ce_rank":
                direct_ce_rank,

            "direct_recall@1":
                int(
                    recall_at_k(
                        direct_ranked,
                        gold_esrs,
                        1
                    )
                ),

            "direct_recall@5":
                int(
                    recall_at_k(
                        direct_ranked,
                        gold_esrs,
                        5
                    )
                ),

            "direct_recall@10":
                int(
                    recall_at_k(
                        direct_ranked,
                        gold_esrs,
                        10
                    )
                ),

            "direct_recall@20":
                int(
                    recall_at_k(
                        direct_ranked,
                        gold_esrs,
                        20
                    )
                ),

            "direct_recall@50":
                int(
                    recall_at_k(
                        direct_ranked,
                        gold_esrs,
                        50
                    )
                ),

            "symbolic_recall@1":
                int(
                    recall_at_k(
                        symbolic_ranked,
                        gold_esrs,
                        1
                    )
                ),

            "symbolic_recall@5":
                int(
                    recall_at_k(
                        symbolic_ranked,
                        gold_esrs,
                        5
                    )
                ),

            "symbolic_recall@10":
                int(
                    recall_at_k(
                        symbolic_ranked,
                        gold_esrs,
                        10
                    )
                ),

            "symbolic_recall@20":
                int(
                    recall_at_k(
                        symbolic_ranked,
                        gold_esrs,
                        20
                    )
                ),

            "symbolic_recall@50":
                int(
                    recall_at_k(
                        symbolic_ranked,
                        gold_esrs,
                        50
                    )
                ),

            "multihop_ce_recall@1":
                int(
                    recall_at_k(
                        ce_ranked,
                        gold_esrs,
                        1
                    )
                ),

            "multihop_ce_recall@5":
                int(
                    recall_at_k(
                        ce_ranked,
                        gold_esrs,
                        5
                    )
                ),

            "multihop_ce_recall@10":
                int(
                    recall_at_k(
                        ce_ranked,
                        gold_esrs,
                        10
                    )
                ),

            "multihop_ce_recall@20":
                int(
                    recall_at_k(
                        ce_ranked,
                        gold_esrs,
                        20
                    )
                ),

            "multihop_ce_recall@50":
                int(
                    recall_at_k(
                        ce_ranked,
                        gold_esrs,
                        50
                    )
                ),

            "direct_ce_recall@1":
                int(
                    recall_at_k(
                        direct_ce_ranked,
                        gold_esrs,
                        1
                    )
                ),

            "direct_ce_recall@5":
                int(
                    recall_at_k(
                        direct_ce_ranked,
                        gold_esrs,
                        5
                    )
                ),

            "direct_ce_recall@10":
                int(
                    recall_at_k(
                        direct_ce_ranked,
                        gold_esrs,
                        10
                    )
                ),

            "direct_ce_recall@20":
                int(
                    recall_at_k(
                        direct_ce_ranked,
                        gold_esrs,
                        20
                    )
                ),

            "direct_ce_recall@50":
                int(
                    recall_at_k(
                        direct_ce_ranked,
                        gold_esrs,
                        50
                    )
                )
        })

        # =====================================================================
        # CANDIDATE STATISTICS
        # =====================================================================

        candidate_rows.append({

            "brsr_id": brsr_id,

            "gri_candidates":
                len(gri_nodes),

            "symbolic_esrs_candidates":
                len(symbolic_ranked),

            "direct_esrs_universe":
                len(esrs_df),

            "symbolic_reduction_ratio":
                (
                    len(symbolic_ranked)
                    / max(len(esrs_df), 1)
                ),

            "gold_esrs":
                len(gold_set),

            "bridge_hit":
                int(bool(bridge_hit)),

            "best_bridge_rank":
                bridge_rank
                if bridge_rank is not None
                else -1
        })

    # =========================================================================
    # AGGREGATE METRICS
    # =========================================================================

    print()
    print("=" * 80)
    print("FINAL ABLATION RESULTS")
    print("=" * 80)

    systems = {
        "direct": direct_results,
        "symbolic_multihop": symbolic_results,
        "multihop_cross_encoder": ce_results,
    }

    metric_rows = []

    for system_name, results in systems.items():

        metrics = evaluate_ranked_results(
            results,
            gold_by_brsr
        )

        print()
        print("-" * 80)
        print(system_name.upper())

        for key, value in metrics.items():

            print(
                f"{key:20s}: {value:.4f}"
                if isinstance(value, float)
                else
                f"{key:20s}: {value}"
            )

            metric_rows.append({

                "system": system_name,

                "metric": key,

                "value": value
            })

    # =========================================================================
    # DIRECT + CE ABLATION
    # =========================================================================

    direct_ce_metrics = evaluate_ranked_results(
        {
            r["brsr_id"]: []
            for r in []
        },
        gold_by_brsr
    )

    # Reconstruct from per-query output because direct_ce results
    # are intentionally not stored as the primary system.

    direct_ce_results = {}

    for row in per_query_rows:

        brsr_id = row["brsr_id"]

        # We do not have the full ranked list here.
        # Instead, recompute the direct CE ranking only for
        # evaluation if required below.
        pass

    # =========================================================================
    # BRIDGE METRICS
    # =========================================================================

    bridge_df = pd.DataFrame(
        bridge_rows
    )

    bridge_recall = (
        bridge_df["bridge_hit"].mean()
        if len(bridge_df)
        else 0.0
    )

    bridge_ranks = bridge_df[
        bridge_df["best_bridge_rank"] > 0
    ]["best_bridge_rank"]

    bridge_mrr = (
        np.mean(
            1.0 / bridge_ranks
        )
        if len(bridge_ranks)
        else 0.0
    )

    metric_rows.extend([

        {
            "system": "symbolic_bridge",
            "metric": "bridge_recall",
            "value": bridge_recall
        },

        {
            "system": "symbolic_bridge",
            "metric": "bridge_mrr",
            "value": bridge_mrr
        },

        {
            "system": "symbolic_bridge",
            "metric": "mean_symbolic_candidates",
            "value": bridge_df[
                "symbolic_esrs_candidates"
            ].mean()
        },

        {
            "system": "symbolic_bridge",
            "metric": "mean_gri_nodes",
            "value": bridge_df[
                "gri_nodes"
            ].mean()
        }
    ])

    # =========================================================================
    # SAVE OUTPUTS
    # =========================================================================

    os.makedirs(
        PROCESSED_DIR,
        exist_ok=True
    )

    pd.DataFrame(
        metric_rows
    ).to_csv(
        METRICS_OUT,
        index=False
    )

    pd.DataFrame(
        per_query_rows
    ).to_csv(
        PER_QUERY_OUT,
        index=False
    )

    bridge_df.to_csv(
        BRIDGE_OUT,
        index=False
    )

    pd.DataFrame(
        candidate_rows
    ).to_csv(
        CANDIDATE_STATS_OUT,
        index=False
    )

    # =========================================================================
    # PRINT IMPORTANT FAILURE CASES
    # =========================================================================

    print()
    print("=" * 80)
    print("FAILURE / IMPROVEMENT DIAGNOSTICS")
    print("=" * 80)

    for row in per_query_rows:

        direct = row["direct_rank"]
        symbolic = row["symbolic_rank"]
        ce_rank = row["multihop_ce_rank"]

        if ce_rank == -1:

            print()
            print(
                f"MISS: {row['brsr_id']}"
            )

            print(
                "  Gold:",
                row["gold_esrs"]
            )

            print(
                "  Direct rank:",
                direct
            )

            print(
                "  Symbolic rank:",
                symbolic
            )

            print(
                "  Multi-hop CE rank:",
                ce_rank
            )

            print(
                "  GRI count:",
                row["gri_count"]
            )

            print(
                "  Symbolic candidates:",
                row["symbolic_candidate_count"]
            )

        elif (
            symbolic > 0
            and ce_rank > 0
            and ce_rank < symbolic
        ):

            print()
            print(
                f"CE IMPROVEMENT: {row['brsr_id']}"
            )

            print(
                f"  Symbolic rank: {symbolic}"
            )

            print(
                f"  CE rank:       {ce_rank}"
            )

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    print()
    print("=" * 80)
    print("SCRIPT 13 COMPLETE")
    print("=" * 80)

    print()
    print("Outputs:")

    print(
        "  Metrics:",
        METRICS_OUT
    )

    print(
        "  Per-query:",
        PER_QUERY_OUT
    )

    print(
        "  Bridge diagnostics:",
        BRIDGE_OUT
    )

    print(
        "  Candidate statistics:",
        CANDIDATE_STATS_OUT
    )

    print()
    print("Interpretation:")
    print()
    print(
        "  Direct:"
        " BRSR -> ESRS semantic retrieval"
    )

    print(
        "  Symbolic:"
        " BRSR -> GRI -> ESRS graph composition"
    )

    print(
        "  Proposed:"
        " BRSR -> GRI -> ESRS -> cross-encoder reranking"
    )

    print()
    print(
        "No training was performed in Script 13."
    )

    print(
        "Validation/test data were used only for evaluation."
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()