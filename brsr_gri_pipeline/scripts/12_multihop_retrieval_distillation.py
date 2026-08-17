"""
SCRIPT 12 — MULTI-HOP RETRIEVAL + CROSS-ENCODER RERANKING

Framework:

    BRSR -> GRI -> ESRS

Two retrieval paths are evaluated:

1. Direct baseline:
       BRSR -> ESRS

2. Proposed symbolic multi-hop:
       BRSR -> GRI -> ESRS

The official BRSR-GRI linkage is used as the symbolic bridge.
The official GRI-ESRS datapoint mapping is used to expand GRI
candidates into ESRS datapoints.

A lightweight CrossEncoder is then used to rerank the ESRS
candidates.

Training:
    Gold-only CE
    Gold + medium-confidence CE

Validation/test are never used for CE training.

Outputs:
    data/processed/retrieval_direct_*.csv
    data/processed/retrieval_multihop_*.csv
    data/processed/retrieval_metrics.csv
    data/processed/cross_encoder_train.csv
    data/processed/cross_encoder_hard_negatives.csv
    data/processed/cross_encoder_predictions.csv
    models/cross_encoder_gold/
    models/cross_encoder_gold_medium/
"""

from __future__ import annotations

# ============================================================================
# IMPORTANT:
# If your machine has multiple GPUs, SentenceTransformers may automatically
# use DataParallel. Your installed SentenceTransformers version has a bug/
# incompatibility where BinaryCrossEntropyLoss expects model.device but
# DataParallel does not expose it.
#
# Therefore, select ONE GPU BEFORE importing torch.
#
# You can also set this from the shell:
#
#     CUDA_VISIBLE_DEVICES=0 python 12_retrieval_distillation.py
#
# If your cluster assigns a specific GPU, it is better to set it in the
# job script instead of hard-coding it here.
# ============================================================================

from __future__ import annotations

import os
# ============================================================
# IMPORTANT:
# Force CrossEncoder training to see ONLY ONE GPU.
#
# This MUST happen before importing torch or sentence_transformers.
# Otherwise CrossEncoder.fit() may automatically use DataParallel.
# ============================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import ast
import csv
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Set

import numpy as np
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import (
    CEBinaryClassificationEvaluator,
)
from torch.utils.data import DataLoader
from sentence_transformers import InputExample


# ============================================================================
# CONFIGURATION
# ============================================================================

SEED = 42

BASE = Path(__file__).resolve().parents[1]

FINAL_DIR = BASE / "data" / "final"
PROCESSED_DIR = BASE / "data" / "processed"
MODEL_DIR = BASE / "models"

TRAIN_FILE = FINAL_DIR / "final_train.csv"
VAL_FILE = FINAL_DIR / "final_validation.csv"
TEST_FILE = FINAL_DIR / "final_test.csv"

GOLD_FILE = FINAL_DIR / "gold_only_train.csv"

MEDIUM_FILE = FINAL_DIR / "medium_confidence_pool.csv"

CANDIDATE_FILE = (
    PROCESSED_DIR / "transitive_candidates.csv"
)

BRSR_GRI_FILE = (
    PROCESSED_DIR / "gold_pairs.csv"
)

GRI_ESRS_FILE = (
    BASE
    / "data"
    / "interim"
    / "gri_esrs_datapoint_mapping.csv"
)

INTEROP_FILE = (
    BASE
    / "data"
    / "interim"
    / "gri_esrs_interoperability.csv"
)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

BI_ENCODER = os.environ.get(
    "RETRIEVER_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

CROSS_ENCODER_MODEL = os.environ.get(
    "CROSS_ENCODER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------

TOP_K_GRIS = 20
TOP_K_ESRS_DIRECT = 50
TOP_K_ESRS_MULTIHOP = 50

MAX_ESRS_PER_GRI = 50

TOP_K_GRIS_FOR_MULTIHOP = 10


# --------------------------------------------------------------------------
# Cross encoder
# --------------------------------------------------------------------------

CE_EPOCHS = 3
CE_BATCH_SIZE = 16
CE_LR = 2e-5
CE_MAX_LENGTH = 256

HARD_NEGATIVES_PER_POSITIVE = 2

MAX_MEDIUM_PER_BRSR = 5

MAX_GOLD_PER_BRSR = 20


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def set_seed(seed: int = SEED):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# UTILITIES
# ============================================================================

def normalize_text(x) -> str:

    if x is None:
        return ""

    x = str(x)

    x = re.sub(r"\s+", " ", x)

    return x.strip()


def normalize_code(x) -> str:

    if x is None:
        return ""

    x = str(x).strip()

    x = x.upper()

    x = x.replace(" ", "")

    return x


def parse_list(value) -> List[str]:

    if value is None:
        return []

    if isinstance(value, list):

        return [
            normalize_code(x)
            for x in value
            if normalize_code(x)
        ]

    text = str(value).strip()

    if not text:
        return []

    try:

        parsed = ast.literal_eval(text)

        if isinstance(parsed, list):

            return [
                normalize_code(x)
                for x in parsed
                if normalize_code(x)
            ]

    except Exception:
        pass

    text = text.strip("[]")

    parts = re.split(
        r"[,;|]",
        text,
    )

    return [
        normalize_code(
            x.strip().strip("'").strip('"')
        )
        for x in parts
        if normalize_code(
            x.strip().strip("'").strip('"')
        )
    ]


def safe_float(x, default=0.0):

    try:
        return float(x)

    except Exception:
        return default


# ============================================================================
# DISCLOSURE TEXT BUILDERS
# ============================================================================

def build_brsr_text(row) -> str:

    fields = [
        row.get("brsr_text", ""),
        row.get("remarks", ""),
        row.get("mapping_semantics", ""),
    ]

    return " ".join(
        normalize_text(x)
        for x in fields
        if normalize_text(x)
    )


def build_gri_text(row) -> str:

    fields = [
        row.get("gri_code", ""),
        row.get("gri_disclosure", ""),
        row.get("gri_text_table1", ""),
        row.get("gri_text_table2", ""),
    ]

    return " ".join(
        normalize_text(x)
        for x in fields
        if normalize_text(x)
    )


def build_esrs_text(row) -> str:

    fields = [
        row.get("esrs_datapoint_id", ""),
        row.get("esrs_sheet", ""),
        row.get("esrs_topic_code", ""),
        row.get("esrs_dr", ""),
        row.get("esrs_name", ""),
        row.get("esrs_data_type", ""),
    ]

    return " ".join(
        normalize_text(x)
        for x in fields
        if normalize_text(x)
    )


# ============================================================================
# LOAD DATA
# ============================================================================

def load_csv(path: Path) -> pd.DataFrame:

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    # ========================================================================
    # CREATE DIRECTORIES
    # ========================================================================

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================================
    # SEED
    # ========================================================================

    set_seed()

    # ========================================================================
    # SCRIPT HEADER
    # ========================================================================

    print("=" * 80)
    print(
        "SCRIPT 12 — MULTI-HOP RETRIEVAL + CROSS-ENCODER"
    )
    print("=" * 80)

    # ========================================================================
    # GPU INFORMATION
    # ========================================================================

    print("\n" + "=" * 80)
    print("GPU CONFIGURATION")
    print("=" * 80)

    print(
        "CUDA_VISIBLE_DEVICES:",
        os.environ.get(
            "CUDA_VISIBLE_DEVICES",
            "<not set>",
        ),
    )

    print(
        "CUDA available:",
        torch.cuda.is_available(),
    )

    print(
        "Visible GPU count:",
        torch.cuda.device_count(),
    )

    if torch.cuda.is_available():

        device = "cuda"

        print(
            "Using GPU:",
            torch.cuda.get_device_name(0),
        )

        print(
            "IMPORTANT: CrossEncoder will use a single visible GPU."
        )

    else:

        device = "cpu"

        print(
            "CUDA unavailable. Using CPU."
        )

    # ========================================================================
    # LOAD DATA
    # ========================================================================

    train_df = load_csv(TRAIN_FILE)
    val_df = load_csv(VAL_FILE)
    test_df = load_csv(TEST_FILE)

    gold_train_df = load_csv(GOLD_FILE)

    brsr_gri_df = load_csv(BRSR_GRI_FILE)
    gri_esrs_df = load_csv(GRI_ESRS_FILE)
    interop_df = load_csv(INTEROP_FILE)

    print("\nLoaded:")

    print(
        f"  Train:                  {len(train_df)}"
    )

    print(
        f"  Validation:             {len(val_df)}"
    )

    print(
        f"  Test:                   {len(test_df)}"
    )

    print(
        f"  Gold training:          {len(gold_train_df)}"
    )

    print(
        f"  BRSR-GRI rows:          {len(brsr_gri_df)}"
    )

    print(
        f"  GRI-ESRS datapoints:    {len(gri_esrs_df)}"
    )

    print(
        f"  Interoperability rows:  {len(interop_df)}"
    )

    # ========================================================================
    # NORMALIZE BRSR-GRI LINKAGE
    # ========================================================================

    brsr_gri_df["brsr_id"] = (
        brsr_gri_df["brsr_id"]
        .astype(str)
        .str.strip()
    )

    if "gri_codes_union" in brsr_gri_df.columns:

        brsr_gri_df["gri_codes"] = (
            brsr_gri_df["gri_codes_union"]
            .apply(parse_list)
        )

    elif "gri_codes_consensus" in brsr_gri_df.columns:

        brsr_gri_df["gri_codes"] = (
            brsr_gri_df["gri_codes_consensus"]
            .apply(parse_list)
        )

    elif "gri_codes_table1" in brsr_gri_df.columns:

        brsr_gri_df["gri_codes"] = (
            brsr_gri_df["gri_codes_table1"]
            .apply(parse_list)
        )

    else:

        raise ValueError(
            "gold_pairs.csv does not contain a usable GRI code column."
        )

    # ========================================================================
    # BRSR TEXT INDEX
    # ========================================================================

    brsr_text_map = {}

    for _, row in brsr_gri_df.iterrows():

        bid = normalize_text(
            row.get("brsr_id", "")
        )

        if not bid:
            continue

        text = build_brsr_text(row)

        if bid not in brsr_text_map:

            brsr_text_map[bid] = text

        elif text:

            brsr_text_map[bid] += " " + text

    for bid in brsr_text_map:

        brsr_text_map[bid] = normalize_text(
            brsr_text_map[bid]
        )

    # ========================================================================
    # ESRS INDEX
    # ========================================================================

    required_esrs_columns = [
        "esrs_datapoint_id",
        "esrs_name",
    ]

    missing = [
        c
        for c in required_esrs_columns
        if c not in gri_esrs_df.columns
    ]

    if missing:

        raise ValueError(
            f"GRI-ESRS file missing columns: {missing}"
        )

    gri_esrs_df["esrs_datapoint_id"] = (
        gri_esrs_df["esrs_datapoint_id"]
        .astype(str)
        .str.strip()
    )

    esrs_unique = (
        gri_esrs_df
        .drop_duplicates(
            subset=["esrs_datapoint_id"]
        )
        .copy()
    )

    esrs_text_map = {
        row["esrs_datapoint_id"]:
            build_esrs_text(row)
        for _, row in esrs_unique.iterrows()
    }

    esrs_rows = {
        row["esrs_datapoint_id"]:
            row.to_dict()
        for _, row in esrs_unique.iterrows()
    }

    # ========================================================================
    # GRI INDEX
    # ========================================================================

    gri_exact_to_esrs = defaultdict(set)
    gri_disclosure_to_esrs = defaultdict(set)

    gri_code_text = {}

    for _, row in gri_esrs_df.iterrows():

        dp = normalize_text(
            row.get(
                "esrs_datapoint_id",
                "",
            )
        )

        if not dp:
            continue

        standard = normalize_code(
            row.get(
                "gri_standard",
                "",
            )
        )

        disclosure = normalize_code(
            row.get(
                "gri_disclosure",
                "",
            )
        )

        number = normalize_code(
            row.get(
                "gri_number",
                "",
            )
        )

        if not standard and not disclosure:
            continue

        disclosure_code = disclosure

        if not disclosure_code and number:

            standard_number = re.search(
                r"(\d+)",
                standard,
            )

            if standard_number:

                disclosure_code = (
                    f"{standard_number.group(1)}-{number}"
                )

        if disclosure_code:

            disclosure_code = normalize_code(
                disclosure_code
            )

            gri_disclosure_to_esrs[
                disclosure_code
            ].add(dp)

            gri_code_text[
                disclosure_code
            ] = build_gri_text(row)

        if disclosure_code and number:

            number_clean = normalize_code(
                number
            )

            if (
                number_clean
                and number_clean != disclosure_code
                and not number_clean.startswith(
                    disclosure_code + "-"
                )
            ):

                exact_code = (
                    disclosure_code
                    + "-"
                    + number_clean
                )

                gri_exact_to_esrs[
                    exact_code
                ].add(dp)

                gri_code_text[
                    exact_code
                ] = build_gri_text(row)

    # ========================================================================
    # INTEROPERABILITY SUPPORT
    # ========================================================================

    interop_by_gri = defaultdict(set)

    for _, row in interop_df.iterrows():

        code = normalize_code(
            row.get(
                "gri_disclosure_id",
                "",
            )
        )

        mapping_type = normalize_text(
            row.get(
                "mapping_type",
                "",
            )
        )

        if code:

            interop_by_gri[code].add(
                mapping_type
            )

    def get_interop_support(
        gri_code: str,
    ) -> str:

        gri_code = normalize_code(
            gri_code
        )

        if gri_code in interop_by_gri:

            values = sorted(
                interop_by_gri[gri_code]
            )

            return "; ".join(values)

        disclosure = re.match(
            r"^(\d+-\d+)",
            gri_code,
        )

        if disclosure:

            key = disclosure.group(1)

            if key in interop_by_gri:

                return "; ".join(
                    sorted(
                        interop_by_gri[key]
                    )
                )

        return "none"

    # ========================================================================
    # SYMBOLIC BRSR -> GRI GRAPH
    # ========================================================================

    brsr_to_gri = defaultdict(set)

    for _, row in brsr_gri_df.iterrows():

        bid = normalize_text(
            row.get(
                "brsr_id",
                "",
            )
        )

        if not bid:
            continue

        for code in row["gri_codes"]:

            code = normalize_code(code)

            if code:

                brsr_to_gri[
                    bid
                ].add(code)

    # ========================================================================
    # LOAD SENTENCE TRANSFORMER
    # ========================================================================

    print("\n" + "=" * 80)
    print("LOADING DENSE RETRIEVER")
    print("=" * 80)

    print(
        f"Retriever device: {device}"
    )

    print(
        f"Retriever model: {BI_ENCODER}"
    )

    bi_encoder = SentenceTransformer(
        BI_ENCODER,
        device=device,
    )

    # ========================================================================
    # BUILD GRI TEXT INDEX
    # ========================================================================

    all_gri_codes = sorted(
        set(
            gri_disclosure_to_esrs.keys()
        )
        |
        set(
            gri_exact_to_esrs.keys()
        )
    )

    gri_disclosure_codes = sorted(
        gri_disclosure_to_esrs.keys()
    )

    gri_corpus = [
        gri_code_text.get(
            code,
            code,
        )
        for code in gri_disclosure_codes
    ]

    print(
        f"GRI disclosure nodes indexed: "
        f"{len(gri_disclosure_codes)}"
    )

    # ========================================================================
    # BUILD ESRS TEXT INDEX
    # ========================================================================

    esrs_ids = sorted(
        esrs_text_map.keys()
    )

    esrs_corpus = [
        esrs_text_map[x]
        for x in esrs_ids
    ]

    print(
        f"ESRS datapoints indexed: "
        f"{len(esrs_ids)}"
    )

    # ========================================================================
    # PRECOMPUTE EMBEDDINGS
    # ========================================================================

    print("\nEncoding GRI corpus...")

    gri_embeddings = bi_encoder.encode(
        gri_corpus,
        batch_size=64,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    print("Encoding ESRS corpus...")

    esrs_embeddings = bi_encoder.encode(
        esrs_corpus,
        batch_size=64,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    # ========================================================================
    # GOLD TARGETS
    # ========================================================================

    def gold_targets(
        df: pd.DataFrame,
    ) -> Dict[str, Set[str]]:

        result = defaultdict(set)

        for _, row in df.iterrows():

            bid = normalize_text(
                row.get(
                    "brsr_id",
                    "",
                )
            )

            eid = normalize_text(
                row.get(
                    "esrs_datapoint_id",
                    "",
                )
            )

            if bid and eid:

                result[bid].add(eid)

        return dict(result)

    train_targets = gold_targets(
        train_df
    )

    val_targets = gold_targets(
        val_df
    )

    test_targets = gold_targets(
        test_df
    )

    # ========================================================================
    # RETRIEVAL HELPERS
    # ========================================================================

    def cosine_topk(
        query_embedding,
        corpus_embeddings,
        k: int,
    ):

        k = min(
            k,
            corpus_embeddings.shape[0],
        )

        scores = torch.matmul(
            corpus_embeddings,
            query_embedding,
        )

        values, indices = torch.topk(
            scores,
            k=k,
        )

        return [
            (
                int(idx),
                float(score),
            )
            for score, idx in zip(
                values,
                indices,
            )
        ]

    def retrieve_gri(
        brsr_id: str,
        top_k: int = TOP_K_GRIS,
    ):

        text = brsr_text_map.get(
            brsr_id,
            "",
        )

        if not text:
            return []

        query_embedding = bi_encoder.encode(
            text,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        results = cosine_topk(
            query_embedding,
            gri_embeddings,
            top_k,
        )

        return [
            {
                "gri_code":
                    gri_disclosure_codes[idx],
                "score":
                    score,
            }
            for idx, score in results
        ]

    def retrieve_esrs_direct(
        brsr_id: str,
        top_k: int = TOP_K_ESRS_DIRECT,
    ):

        text = brsr_text_map.get(
            brsr_id,
            "",
        )

        if not text:
            return []

        query_embedding = bi_encoder.encode(
            text,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        results = cosine_topk(
            query_embedding,
            esrs_embeddings,
            top_k,
        )

        return [
            {
                "esrs_datapoint_id":
                    esrs_ids[idx],
                "score":
                    score,
            }
            for idx, score in results
        ]

    # ========================================================================
    # MULTI-HOP RETRIEVAL
    # ========================================================================

    def expand_gri_to_esrs(
        gri_code: str,
    ) -> List[Tuple[str, str]]:

        gri_code = normalize_code(
            gri_code
        )

        results = []

        if gri_code in gri_exact_to_esrs:

            for eid in gri_exact_to_esrs[
                gri_code
            ]:

                results.append(
                    (
                        eid,
                        "requirement",
                    )
                )

        disclosure_match = re.match(
            r"^(\d+-\d+)",
            gri_code,
        )

        if disclosure_match:

            disclosure = (
                disclosure_match.group(1)
            )

            if (
                disclosure
                in gri_disclosure_to_esrs
            ):

                for eid in gri_disclosure_to_esrs[
                    disclosure
                ]:

                    if not any(
                        x[0] == eid
                        for x in results
                    ):

                        results.append(
                            (
                                eid,
                                "disclosure",
                            )
                        )

        return results

    def retrieve_multihop(
        brsr_id: str,
        top_k_gri: int = TOP_K_GRIS_FOR_MULTIHOP,
        top_k_esrs: int = TOP_K_ESRS_MULTIHOP,
    ):

        dense_gri = retrieve_gri(
            brsr_id,
            top_k=top_k_gri,
        )

        official_gris = brsr_to_gri.get(
            brsr_id,
            set(),
        )

        gri_candidates = {}

        for item in dense_gri:

            gri_candidates[
                normalize_code(
                    item["gri_code"]
                )
            ] = {
                "retrieval_score":
                    item["score"],
                "source":
                    "dense",
            }

        for code in official_gris:

            code = normalize_code(code)

            disclosure = re.match(
                r"^(\d+-\d+)",
                code,
            )

            disclosure = (
                disclosure.group(1)
                if disclosure
                else code
            )

            existing = gri_candidates.get(
                disclosure
            )

            if existing:

                existing["source"] = (
                    "dense+official"
                )

            else:

                gri_candidates[
                    disclosure
                ] = {
                    "retrieval_score":
                        1.0,
                    "source":
                        "official",
                }

        esrs_candidates = {}

        for gri_disclosure, meta in (
            gri_candidates.items()
        ):

            mappings = expand_gri_to_esrs(
                gri_disclosure
            )

            linked_requirements = [
                x
                for x in official_gris
                if re.match(
                    r"^(\d+-\d+)",
                    normalize_code(x),
                )
                and re.match(
                    r"^(\d+-\d+)",
                    normalize_code(x)
                ).group(1)
                == gri_disclosure
            ]

            for requirement in linked_requirements:

                mappings.extend(
                    expand_gri_to_esrs(
                        requirement
                    )
                )

            seen = set()

            for eid, level in mappings:

                key = (
                    eid,
                    level,
                )

                if key in seen:
                    continue

                seen.add(key)

                if (
                    len(esrs_candidates)
                    >= MAX_ESRS_PER_GRI
                    * max(
                        1,
                        len(gri_candidates),
                    )
                ):

                    break

                interop = get_interop_support(
                    gri_disclosure
                )

                previous = esrs_candidates.get(
                    eid
                )

                score = safe_float(
                    meta.get(
                        "retrieval_score",
                        0,
                    )
                )

                if previous is None:

                    esrs_candidates[eid] = {
                        "gri_code":
                            gri_disclosure,
                        "gri_score":
                            score,
                        "gri_esrs_match_level":
                            level,
                        "interop_support":
                            interop,
                        "source":
                            meta["source"],
                    }

                else:

                    if (
                        score
                        > previous["gri_score"]
                    ):

                        previous.update(
                            {
                                "gri_code":
                                    gri_disclosure,
                                "gri_score":
                                    score,
                                "gri_esrs_match_level":
                                    level,
                                "interop_support":
                                    interop,
                                "source":
                                    meta["source"],
                            }
                        )

        ranked = sorted(
            esrs_candidates.items(),
            key=lambda x: (
                x[1]["gri_score"],
                1
                if x[1][
                    "gri_esrs_match_level"
                ]
                == "requirement"
                else 0,
                1
                if x[1][
                    "interop_support"
                ]
                in {
                    "direct",
                    "broader_topic; direct",
                }
                else 0,
            ),
            reverse=True,
        )

        return [
            {
                "esrs_datapoint_id":
                    eid,
                **meta,
            }
            for eid, meta in ranked[
                :top_k_esrs
            ]
        ]

    # ========================================================================
    # RETRIEVAL METRICS
    # ========================================================================

    def evaluate_retrieval(
        results_by_brsr,
        targets_by_brsr,
        ks=(1, 5, 10, 20, 50),
    ):

        rows = []

        recalls = {
            k: []
            for k in ks
        }

        reciprocal_ranks = []

        evaluated = 0

        for bid, targets in (
            targets_by_brsr.items()
        ):

            if not targets:
                continue

            if bid not in results_by_brsr:
                continue

            evaluated += 1

            ranked = [
                x["esrs_datapoint_id"]
                for x in results_by_brsr[bid]
            ]

            rr = 0.0

            for rank, eid in enumerate(
                ranked,
                start=1,
            ):

                if eid in targets:

                    rr = 1.0 / rank

                    break

            reciprocal_ranks.append(rr)

            row = {
                "brsr_id":
                    bid,
                "target_count":
                    len(targets),
                "first_relevant_rank":
                    0
                    if rr == 0
                    else int(
                        round(1 / rr)
                    ),
            }

            for k in ks:

                hit = any(
                    eid in targets
                    for eid in ranked[:k]
                )

                recalls[k].append(
                    1.0 if hit else 0.0
                )

                row[
                    f"hit@{k}"
                ] = int(hit)

            rows.append(row)

        metrics = {
            "evaluated_brsr_ids":
                evaluated,
            "MRR":
                float(
                    np.mean(
                        reciprocal_ranks
                    )
                )
                if reciprocal_ranks
                else 0.0,
        }

        for k in ks:

            metrics[
                f"Recall@{k}"
            ] = (
                float(
                    np.mean(
                        recalls[k]
                    )
                )
                if recalls[k]
                else 0.0
            )

        return (
            metrics,
            pd.DataFrame(rows),
        )

    # ========================================================================
    # RUN RETRIEVAL
    # ========================================================================

    def run_retrieval(
        targets_by_brsr,
        split_name: str,
    ):

        print("\n" + "-" * 80)

        print(
            f"RETRIEVAL — {split_name}"
        )

        print("-" * 80)

        direct_results = {}
        multihop_results = {}

        detail_rows = []

        for i, bid in enumerate(
            sorted(
                targets_by_brsr
            ),
            start=1,
        ):

            print(
                f"\r[{i}/{len(targets_by_brsr)}] {bid}",
                end="",
            )

            direct = retrieve_esrs_direct(
                bid,
                TOP_K_ESRS_DIRECT,
            )

            multihop = retrieve_multihop(
                bid,
                TOP_K_GRIS_FOR_MULTIHOP,
                TOP_K_ESRS_MULTIHOP,
            )

            direct_results[bid] = direct
            multihop_results[bid] = multihop

            for rank, item in enumerate(
                multihop,
                start=1,
            ):

                eid = item[
                    "esrs_datapoint_id"
                ]

                target = int(
                    eid
                    in targets_by_brsr[
                        bid
                    ]
                )

                esrs_info = esrs_rows.get(
                    eid,
                    {},
                )

                detail_rows.append(
                    {
                        "split":
                            split_name,
                        "brsr_id":
                            bid,
                        "rank":
                            rank,
                        "esrs_datapoint_id":
                            eid,
                        "esrs_name":
                            esrs_info.get(
                                "esrs_name",
                                "",
                            ),
                        "gri_code":
                            item.get(
                                "gri_code",
                                "",
                            ),
                        "gri_score":
                            item.get(
                                "gri_score",
                                "",
                            ),
                        "gri_esrs_match_level":
                            item.get(
                                "gri_esrs_match_level",
                                "",
                            ),
                        "interop_support":
                            item.get(
                                "interop_support",
                                "",
                            ),
                        "source":
                            item.get(
                                "source",
                                "",
                            ),
                        "gold_target":
                            target,
                    }
                )

        print()

        direct_metrics, direct_rows = (
            evaluate_retrieval(
                direct_results,
                targets_by_brsr,
            )
        )

        multihop_metrics, multihop_rows = (
            evaluate_retrieval(
                multihop_results,
                targets_by_brsr,
            )
        )

        print("\nDIRECT BRSR -> ESRS")

        for k in (
            1,
            5,
            10,
            20,
            50,
        ):

            print(
                f"  Recall@{k}: "
                f"{direct_metrics[f'Recall@{k}']:.4f}"
            )

        print(
            f"  MRR: "
            f"{direct_metrics['MRR']:.4f}"
        )

        print(
            "\nMULTI-HOP BRSR -> GRI -> ESRS"
        )

        for k in (
            1,
            5,
            10,
            20,
            50,
        ):

            print(
                f"  Recall@{k}: "
                f"{multihop_metrics[f'Recall@{k}']:.4f}"
            )

        print(
            f"  MRR: "
            f"{multihop_metrics['MRR']:.4f}"
        )

        detail_path = (
            PROCESSED_DIR
            / f"retrieval_multihop_{split_name.lower()}.csv"
        )

        pd.DataFrame(
            detail_rows
        ).to_csv(
            detail_path,
            index=False,
        )

        direct_detail = []

        for bid, items in (
            direct_results.items()
        ):

            targets = targets_by_brsr[bid]

            for rank, item in enumerate(
                items,
                start=1,
            ):

                eid = item[
                    "esrs_datapoint_id"
                ]

                direct_detail.append(
                    {
                        "split":
                            split_name,
                        "brsr_id":
                            bid,
                        "rank":
                            rank,
                        "esrs_datapoint_id":
                            eid,
                        "score":
                            item["score"],
                        "gold_target":
                            int(
                                eid in targets
                            ),
                    }
                )

        direct_path = (
            PROCESSED_DIR
            / f"retrieval_direct_{split_name.lower()}.csv"
        )

        pd.DataFrame(
            direct_detail
        ).to_csv(
            direct_path,
            index=False,
        )

        return (
            direct_results,
            multihop_results,
            direct_metrics,
            multihop_metrics,
        )

    # ========================================================================
    # RUN ALL RETRIEVAL EVALUATIONS
    # ========================================================================

    all_retrieval_metrics = []

    (
        train_direct,
        train_multi,
        train_direct_m,
        train_multi_m,
    ) = run_retrieval(
        train_targets,
        "TRAIN",
    )

    (
        val_direct,
        val_multi,
        val_direct_m,
        val_multi_m,
    ) = run_retrieval(
        val_targets,
        "VALIDATION",
    )

    (
        test_direct,
        test_multi,
        test_direct_m,
        test_multi_m,
    ) = run_retrieval(
        test_targets,
        "TEST",
    )

    for split, direct_m, multi_m in [
        (
            "train",
            train_direct_m,
            train_multi_m,
        ),
        (
            "validation",
            val_direct_m,
            val_multi_m,
        ),
        (
            "test",
            test_direct_m,
            test_multi_m,
        ),
    ]:

        for method, metrics in [
            (
                "direct",
                direct_m,
            ),
            (
                "multihop",
                multi_m,
            ),
        ]:

            row = {
                "split":
                    split,
                "method":
                    method,
                **metrics,
            }

            all_retrieval_metrics.append(
                row
            )

    retrieval_metrics_path = (
        PROCESSED_DIR
        / "retrieval_metrics.csv"
    )

    pd.DataFrame(
        all_retrieval_metrics
    ).to_csv(
        retrieval_metrics_path,
        index=False,
    )

    # ========================================================================
    # CROSS-ENCODER TRAINING DATA
    # ========================================================================

    print("\n" + "=" * 80)
    print(
        "BUILDING CROSS-ENCODER TRAINING DATA"
    )
    print("=" * 80)

    train_gold_targets = train_targets

    ce_positive_rows = []

    for bid, targets in (
        train_gold_targets.items()
    ):

        if bid not in train_multi:
            continue

        positives_for_bid = 0

        for item in train_multi[bid]:

            eid = item[
                "esrs_datapoint_id"
            ]

            if eid not in targets:
                continue

            if (
                positives_for_bid
                >= MAX_GOLD_PER_BRSR
            ):
                break

            ce_positive_rows.append(
                {
                    "brsr_id":
                        bid,
                    "esrs_datapoint_id":
                        eid,
                    "label":
                        1.0,
                    "evidence":
                        "gold",
                    "gri_code":
                        item.get(
                            "gri_code",
                            "",
                        ),
                }
            )

            positives_for_bid += 1

    # ========================================================================
    # MEDIUM CONFIDENCE
    # ========================================================================

    medium_rows = []

    if MEDIUM_FILE.exists():

        medium_df = load_csv(
            MEDIUM_FILE
        )

        for _, row in (
            medium_df.iterrows()
        ):

            bid = normalize_text(
                row.get(
                    "brsr_id",
                    "",
                )
            )

            eid = normalize_text(
                row.get(
                    "esrs_datapoint_id",
                    "",
                )
            )

            if (
                not bid
                or not eid
                or bid not in train_targets
            ):
                continue

            medium_rows.append(
                {
                    "brsr_id":
                        bid,
                    "esrs_datapoint_id":
                        eid,
                    "label":
                        1.0,
                    "evidence":
                        "medium",
                    "gri_code":
                        row.get(
                            "gri_code",
                            "",
                        ),
                }
            )

    medium_unique = {}

    for row in medium_rows:

        key = (
            row["brsr_id"],
            row["esrs_datapoint_id"],
        )

        medium_unique[key] = row

    medium_rows = list(
        medium_unique.values()
    )

    # ========================================================================
    # HARD NEGATIVES
    # ========================================================================

    print(
        f"Gold positives: "
        f"{len(ce_positive_rows)}"
    )

    hard_negative_rows = []

    for bid, targets in (
        train_gold_targets.items()
    ):

        candidates = train_multi.get(
            bid,
            [],
        )

        negatives = []

        for item in candidates:

            eid = item[
                "esrs_datapoint_id"
            ]

            if eid in targets:
                continue

            negatives.append(
                (
                    eid,
                    item.get(
                        "gri_score",
                        0.0,
                    ),
                    item.get(
                        "gri_code",
                        "",
                    ),
                )
            )

        negatives = negatives[
            :HARD_NEGATIVES_PER_POSITIVE
        ]

        for eid, score, gri_code in (
            negatives
        ):

            hard_negative_rows.append(
                {
                    "brsr_id":
                        bid,
                    "esrs_datapoint_id":
                        eid,
                    "label":
                        0.0,
                    "evidence":
                        "hard_negative",
                    "gri_code":
                        gri_code,
                    "retrieval_score":
                        score,
                }
            )

    print(
        f"Hard negatives: "
        f"{len(hard_negative_rows)}"
    )

    # ========================================================================
    # BUILD CROSS-ENCODER EXAMPLES
    # ========================================================================

    def build_ce_examples(
        positives,
        negatives,
    ):

        examples = []

        for row in positives:

            bid = row[
                "brsr_id"
            ]

            eid = row[
                "esrs_datapoint_id"
            ]

            btext = brsr_text_map.get(
                bid,
                "",
            )

            etext = esrs_text_map.get(
                eid,
                "",
            )

            if not btext or not etext:
                continue

            examples.append(
                InputExample(
                    texts=[
                        btext,
                        etext,
                    ],
                    label=1.0,
                )
            )

        for row in negatives:

            bid = row[
                "brsr_id"
            ]

            eid = row[
                "esrs_datapoint_id"
            ]

            btext = brsr_text_map.get(
                bid,
                "",
            )

            etext = esrs_text_map.get(
                eid,
                "",
            )

            if not btext or not etext:
                continue

            examples.append(
                InputExample(
                    texts=[
                        btext,
                        etext,
                    ],
                    label=0.0,
                )
            )

        return examples

    # ========================================================================
    # GOLD-ONLY
    # ========================================================================

    gold_ce_examples = (
        build_ce_examples(
            ce_positive_rows,
            hard_negative_rows,
        )
    )

    # ========================================================================
    # GOLD + MEDIUM
    # ========================================================================

    gold_medium_positives = (
        ce_positive_rows
        + medium_rows
    )

    seen = set()

    gold_medium_unique = []

    for row in gold_medium_positives:

        key = (
            row["brsr_id"],
            row["esrs_datapoint_id"],
            row["label"],
        )

        if key in seen:
            continue

        seen.add(key)

        gold_medium_unique.append(
            row
        )

    gold_medium_ce_examples = (
        build_ce_examples(
            gold_medium_unique,
            hard_negative_rows,
        )
    )

    print(
        f"Gold CE examples: "
        f"{len(gold_ce_examples)}"
    )

    print(
        f"Gold + medium CE examples: "
        f"{len(gold_medium_ce_examples)}"
    )

    # ========================================================================
    # SAVE CE TRAINING DATA
    # ========================================================================

    ce_rows = []

    for row in ce_positive_rows:
        ce_rows.append(row)

    for row in medium_rows:
        ce_rows.append(
            dict(row)
        )

    for row in hard_negative_rows:
        ce_rows.append(row)

    ce_train_path = (
        PROCESSED_DIR
        / "cross_encoder_train.csv"
    )

    pd.DataFrame(
        ce_rows
    ).to_csv(
        ce_train_path,
        index=False,
    )

    pd.DataFrame(
        hard_negative_rows
    ).to_csv(
        PROCESSED_DIR
        / "cross_encoder_hard_negatives.csv",
        index=False,
    )

    # ========================================================================
    # CROSS-ENCODER TRAINING
    # ========================================================================

    print("\n" + "=" * 80)
    print(
        "CROSS-ENCODER TRAINING"
    )
    print("=" * 80)

    print(
        "IMPORTANT: using a SINGLE GPU."
    )

    print(
        "DataParallel is deliberately disabled by exposing only one GPU."
    )

    # ========================================================================
    # TRAIN CROSS ENCODER
    # ========================================================================

    def train_cross_encoder(
        examples,
        output_dir: Path,
        run_name: str,
    ):

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        if len(examples) < 10:

            print(
                f"Skipping {run_name}: "
                f"only {len(examples)} examples."
            )

            return None

        print("\n" + "=" * 80)
        print(f"TRAINING CROSS-ENCODER: {run_name}")
        print("=" * 80)

        print(
            f"Training examples: {len(examples)}"
        )

        print(
            f"CUDA_VISIBLE_DEVICES: "
            f"{os.environ.get('CUDA_VISIBLE_DEVICES')}"
        )

        print(
            f"torch.cuda.device_count(): "
            f"{torch.cuda.device_count()}"
        )

        if torch.cuda.is_available():

            print(
                f"GPU: "
                f"{torch.cuda.get_device_name(0)}"
            )

            train_device = "cuda"

        else:

            print("CUDA not available. Using CPU.")

            train_device = "cpu"

        # ================================================================
        # IMPORTANT
        #
        # DO NOT use:
        #
        #     torch.nn.DataParallel(...)
        #
        # DO NOT manually wrap the model.
        #
        # CUDA_VISIBLE_DEVICES=0 ensures CrossEncoder.fit() sees only
        # one GPU and therefore does not create DataParallel.
        # ================================================================

        model = CrossEncoder(
            CROSS_ENCODER_MODEL,
            num_labels=1,
            max_length=CE_MAX_LENGTH,
            device=train_device,
        )

        print(
            f"CrossEncoder device: {train_device}"
        )

        # ================================================================
        # VALIDATION DATA
        #
        # Validation is ONLY used by the evaluator.
        # It is NOT used for training.
        # ================================================================

        val_examples = []

        for bid, targets in val_targets.items():

            btext = brsr_text_map.get(
                bid,
                "",
            )

            if not btext:
                continue

            for eid in targets:

                etext = esrs_text_map.get(
                    eid,
                    "",
                )

                if not etext:
                    continue

                val_examples.append(
                    InputExample(
                        texts=[
                            btext,
                            etext,
                        ],
                        label=1.0,
                    )
                )

        evaluator = None

        if val_examples:

            evaluator = (
                CEBinaryClassificationEvaluator.from_input_examples(
                    val_examples,
                    name="validation",
                )
            )

            print(
                f"Validation examples: "
                f"{len(val_examples)}"
            )

        else:

            print(
                "No validation examples available."
            )

        # ================================================================
        # DATALOADER
        # ================================================================

        train_dataloader = DataLoader(
            examples,
            shuffle=True,
            batch_size=CE_BATCH_SIZE,
        )

        warmup_steps = max(
            1,
            int(
                len(train_dataloader) * 0.1
            ),
        )

        print(
            f"Batch size: {CE_BATCH_SIZE}"
        )

        print(
            f"Epochs: {CE_EPOCHS}"
        )

        print(
            f"Learning rate: {CE_LR}"
        )

        print(
            f"Warmup steps: {warmup_steps}"
        )

        # ================================================================
        # TRAIN
        # ================================================================

        model.fit(
            train_dataloader=train_dataloader,
            evaluator=evaluator,
            epochs=CE_EPOCHS,
            warmup_steps=warmup_steps,
            optimizer_params={
                "lr": CE_LR,
            },
            output_path=str(output_dir),
            show_progress_bar=True,
            use_amp=(
                train_device == "cuda"
            ),
        )

        print(
            f"\nSaved model -> {output_dir}"
        )

        return model

    # ========================================================================
    # TRAIN GOLD MODEL
    # ========================================================================

    gold_model = train_cross_encoder(
        gold_ce_examples,
        MODEL_DIR / "cross_encoder_gold",
        "GOLD-ONLY",
    )

    gold_model_dir = MODEL_DIR / "cross_encoder_gold"
    gold_model_dir.mkdir(parents=True, exist_ok=True)

    # Explicitly save the trained CrossEncoder
    if gold_model is not None:
        gold_model.save(str(gold_model_dir))

    print("\n" + "=" * 80)
    print("VERIFYING GOLD CROSS-ENCODER SAVE")
    print("=" * 80)

    required_files = [
        "config.json",
        "modules.json",
    ]

    for fname in required_files:
        path = gold_model_dir / fname
        print(f"{fname:20s}: {'FOUND' if path.exists() else 'MISSING'}")

    # Look for actual model weights
    weight_files = list(gold_model_dir.glob("*.safetensors"))
    weight_files += list(gold_model_dir.glob("*.bin"))

    print("Weight files:")
    for f in weight_files:
        print(f"  {f}")

    if not weight_files:
        raise RuntimeError(
            f"CrossEncoder weights were NOT saved to {gold_model_dir}"
        )

    print(f"\nGOLD MODEL SAVED SUCCESSFULLY -> {gold_model_dir}")

    # ========================================================================
    # TRAIN GOLD + MEDIUM MODEL
    # ========================================================================

    gold_medium_model = (
        train_cross_encoder(
            gold_medium_ce_examples,
            MODEL_DIR
            / "cross_encoder_gold_medium",
            "GOLD+MEDIUM",
        )
    )
    gold_medium_model_dir = MODEL_DIR / "cross_encoder_gold_medium"
    gold_medium_model_dir.mkdir(parents=True, exist_ok=True)

    if gold_medium_model is not None:
        gold_medium_model.save(str(gold_medium_model_dir))

    print("\n" + "=" * 80)
    print("VERIFYING GOLD+MEDIUM CROSS-ENCODER SAVE")
    print("=" * 80)

    required_files = [
        "config.json",
        "modules.json",
    ]

    for fname in required_files:
        path = gold_medium_model_dir / fname
        print(f"{fname:20s}: {'FOUND' if path.exists() else 'MISSING'}")

    weight_files = list(gold_medium_model_dir.glob("*.safetensors"))
    weight_files += list(gold_medium_model_dir.glob("*.bin"))

    print("Weight files:")
    for f in weight_files:
        print(f"  {f}")

    if not weight_files:
        raise RuntimeError(
            f"CrossEncoder weights were NOT saved to {gold_medium_model_dir}"
        )

    print(f"\nGOLD+MEDIUM MODEL SAVED SUCCESSFULLY -> {gold_medium_model_dir}")

    # ========================================================================
    # CROSS-ENCODER RERANKING
    # ========================================================================

    def rerank(
        model,
        brsr_id: str,
        candidates: List[dict],
    ):

        if model is None:
            return candidates

        btext = brsr_text_map.get(
            brsr_id,
            "",
        )

        if not btext:
            return candidates

        pairs = []

        valid_candidates = []

        for item in candidates:

            eid = item[
                "esrs_datapoint_id"
            ]

            etext = esrs_text_map.get(
                eid,
                "",
            )

            if not etext:
                continue

            pairs.append(
                [
                    btext,
                    etext,
                ]
            )

            valid_candidates.append(
                item
            )

        if not pairs:
            return candidates

        scores = model.predict(
            pairs,
            batch_size=32,
            show_progress_bar=False,
        )

        reranked = []

        for item, score in zip(
            valid_candidates,
            scores,
        ):

            x = dict(item)

            x[
                "cross_encoder_score"
            ] = float(score)

            reranked.append(x)

        reranked.sort(
            key=lambda x:
                x["cross_encoder_score"],
            reverse=True,
        )

        return reranked

    # ========================================================================
    # RERANKER EVALUATION
    # ========================================================================

    def evaluate_reranker(
        model,
        retrieval_results,
        targets,
        split_name,
        model_name,
    ):

        results = {}

        detail = []

        for bid, candidates in (
            retrieval_results.items()
        ):

            reranked = rerank(
                model,
                bid,
                candidates,
            )

            results[bid] = reranked

            for rank, item in enumerate(
                reranked,
                start=1,
            ):

                eid = item[
                    "esrs_datapoint_id"
                ]

                detail.append(
                    {
                        "split":
                            split_name,
                        "model":
                            model_name,
                        "brsr_id":
                            bid,
                        "rank":
                            rank,
                        "esrs_datapoint_id":
                            eid,
                        "cross_encoder_score":
                            item.get(
                                "cross_encoder_score",
                                "",
                            ),
                        "gri_code":
                            item.get(
                                "gri_code",
                                "",
                            ),
                        "gold_target":
                            int(
                                eid
                                in targets[bid]
                            ),
                    }
                )

        metrics, _ = evaluate_retrieval(
            results,
            targets,
        )

        return (
            results,
            metrics,
            detail,
        )

    # ========================================================================
    # RERANK MULTIHOP CANDIDATES
    # ========================================================================

    reranker_metrics = []

    reranker_details = []

    for model_name, model in [
        (
            "gold",
            gold_model,
        ),
        (
            "gold_medium",
            gold_medium_model,
        ),
    ]:

        if model is None:
            continue

        for split_name, results, targets in [
            (
                "validation",
                val_multi,
                val_targets,
            ),
            (
                "test",
                test_multi,
                test_targets,
            ),
        ]:

            (
                reranked,
                metrics,
                detail,
            ) = evaluate_reranker(
                model,
                results,
                targets,
                split_name,
                model_name,
            )

            reranker_details.extend(
                detail
            )

            reranker_metrics.append(
                {
                    "split":
                        split_name,
                    "method":
                        f"multihop+ce_{model_name}",
                    **metrics,
                }
            )

            print(
                "\n"
                + "-" * 80
            )

            print(
                f"{model_name.upper()} CE — "
                f"{split_name.upper()}"
            )

            for k in (
                1,
                5,
                10,
                20,
                50,
            ):

                print(
                    f"  Recall@{k}: "
                    f"{metrics[f'Recall@{k}']:.4f}"
                )

            print(
                f"  MRR: "
                f"{metrics['MRR']:.4f}"
            )

    # ========================================================================
    # SAVE RERANK RESULTS
    # ========================================================================

    pd.DataFrame(
        reranker_metrics
    ).to_csv(
        PROCESSED_DIR
        / "cross_encoder_metrics.csv",
        index=False,
    )

    pd.DataFrame(
        reranker_details
    ).to_csv(
        PROCESSED_DIR
        / "cross_encoder_predictions.csv",
        index=False,
    )

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================

    print("\n")

    print("=" * 80)

    print(
        "SCRIPT 12 COMPLETE"
    )

    print("=" * 80)

    print(
        "\nRetrieval metrics:"
    )

    print(
        retrieval_metrics_path
    )

    print(
        "\nCross-encoder training:"
    )

    print(
        ce_train_path
    )

    print(
        "\nCross-encoder metrics:"
    )

    print(
        PROCESSED_DIR
        / "cross_encoder_metrics.csv"
    )

    print(
        "\nModels:"
    )

    print(
        MODEL_DIR
        / "cross_encoder_gold"
    )

    print(
        MODEL_DIR
        / "cross_encoder_gold_medium"
    )

    print(
        "\nFramework stages completed:"
    )

    print(
        "  [1] Direct BRSR -> ESRS retrieval"
    )

    print(
        "  [2] Symbolic BRSR -> GRI -> ESRS retrieval"
    )

    print(
        "  [3] Hard-negative mining"
    )

    print(
        "  [4] Gold-only cross-encoder"
    )

    print(
        "  [5] Gold + medium cross-encoder"
    )

    print(
        "  [6] Multi-hop cross-encoder reranking"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "  The review pool was NOT used as positive supervision."
    )

    print(
        "  Validation/test BRSR IDs were NOT used for CE training."
    )

    print(
        "  DataParallel was NOT used."
    )

    print(
        "  The official BRSR-GRI linkage remains the symbolic bridge."
    )

    print(
        "\nNext step:"
    )

    print(
        "  Compare direct vs multi-hop retrieval and CE reranking."
    )

    print(
        "  Only after this should symbolic knowledge distillation "
        "be implemented."
    )

    print("=" * 80)


# ============================================================================
# PYTHON ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()