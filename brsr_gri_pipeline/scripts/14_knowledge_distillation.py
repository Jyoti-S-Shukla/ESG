"""
SCRIPT 14 — SYMBOLIC KNOWLEDGE DISTILLATION

Framework
---------
    BRSR -> GRI -> ESRS

The official BRSR->GRI linkage is the symbolic bridge.
The official GRI->ESRS datapoint mapping expands that bridge
into an ESRS candidate set.

A trained CrossEncoder is used as the teacher:

    Teacher(BRSR text, ESRS text) -> relevance score

A lightweight bi-encoder student is then trained to reproduce the
teacher's per-BRSR relevance distribution using KL-divergence.

Important design constraints
----------------------------
1. ONLY TRAIN BRSR IDs are used to construct KD training candidates.
2. Validation/test BRSR IDs are never used for student training.
3. The review pool is NOT used as positive supervision.
4. BRSR text is loaded from the same gold_pairs.csv source used by
   Script 12; it is NOT reconstructed from the BRSR->ESRS edge table.
5. Candidate generation is symbolic first:
       BRSR -> official GRI -> official GRI->ESRS
6. Gold train edges may be added only as official train supervision
   when they are absent from the symbolic candidate set. They are
   explicitly marked as "gold_added".
7. Teacher scores are converted to a per-query soft distribution.
8. Student training uses:
       L = lambda_kd * KL(p_teacher || p_student)
         + lambda_gold * supervised BCE
9. The student is saved as a normal SentenceTransformer model.
10. Validation is used only for checkpoint selection/early stopping.
    Test is evaluated only after training is complete.

Outputs
-------
data/processed/kd_symbolic_candidates.csv
data/processed/kd_teacher_scores_gold.csv
data/processed/kd_teacher_scores_gold_medium.csv
data/processed/kd_training_data_gold.csv
data/processed/kd_training_data_gold_medium.csv
data/processed/kd_training_summary.csv
data/processed/kd_validation_metrics.csv
data/processed/kd_test_metrics.csv

models/kd_student_gold/
models/kd_student_gold_medium/
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# GPU: expose only one GPU BEFORE importing torch / sentence-transformers.
# This avoids the DataParallel issue encountered in Script 12.
# Override from shell if required:
#     CUDA_VISIBLE_DEVICES=0 python scripts/14_knowledge_distillation.py
# ---------------------------------------------------------------------------
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import ast
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers import models as st_models


# ============================================================================
# CONFIGURATION
# ============================================================================

SEED = 42

BASE = Path(__file__).resolve().parents[1]
FINAL_DIR = BASE / "data" / "final"
PROCESSED_DIR = BASE / "data" / "processed"
INTERIM_DIR = BASE / "data" / "interim"
MODEL_DIR = BASE / "models"

TRAIN_FILE = FINAL_DIR / "final_train.csv"
VAL_FILE = FINAL_DIR / "final_validation.csv"
TEST_FILE = FINAL_DIR / "final_test.csv"

GOLD_FILE = FINAL_DIR / "gold_only_train.csv"
MEDIUM_FILE = FINAL_DIR / "medium_confidence_pool.csv"

BRSR_GRI_FILE = PROCESSED_DIR / "gold_pairs.csv"
GRI_ESRS_FILE = INTERIM_DIR / "gri_esrs_datapoint_mapping.csv"
INTEROP_FILE = INTERIM_DIR / "gri_esrs_interoperability.csv"

# Dense student / teacher models
STUDENT_MODEL = os.environ.get(
    "KD_STUDENT_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

CROSS_ENCODER_GOLD = MODEL_DIR / "cross_encoder_gold"
CROSS_ENCODER_GOLD_MEDIUM = MODEL_DIR / "cross_encoder_gold_medium"

# Candidate generation
MAX_GRI_PER_BRSR = 20
MAX_ESRS_PER_GRI = 50
MAX_CANDIDATES_PER_BRSR = 100

# Optional addition of official train gold targets that are not recovered
# by symbolic expansion. These are still TRAIN-only and are marked.
ADD_MISSING_GOLD_TARGETS = True

# Teacher scoring
TEACHER_BATCH_SIZE = 32
CE_MAX_LENGTH = 256

# Distillation
TEMPERATURE = 2.0
LAMBDA_KD = 1.0
LAMBDA_GOLD = 0.25

EPOCHS = 10
BATCH_QUERIES = 4
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_EPOCHS = 1
PATIENCE = 3
MAX_GRAD_NORM = 1.0

# Student encoding
STUDENT_MAX_LENGTH = 256

# Evaluation
EVAL_KS = (1, 5, 10, 20, 50)

# Medium confidence pool
MAX_MEDIUM_PER_BRSR = 5


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# BASIC UTILITIES
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

    x = str(x).strip().upper()
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

    parts = re.split(r"[,;|]", text)

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


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found:\n{path}")

    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
    )


# ============================================================================
# TEXT BUILDERS
# These intentionally match the construction used in Script 12.
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


def build_gri_text(row) -> str:
    fields = [
        row.get("gri_standard", ""),
        row.get("gri_disclosure", ""),
        row.get("gri_number", ""),
        row.get("gri_text", ""),
        row.get("gri_text_table1", ""),
        row.get("gri_text_table2", ""),
    ]

    return " ".join(
        normalize_text(x)
        for x in fields
        if normalize_text(x)
    )


# ============================================================================
# GOLD TARGET LOADER
# ============================================================================

def gold_targets(df: pd.DataFrame) -> Dict[str, Set[str]]:
    result = defaultdict(set)

    if "brsr_id" not in df.columns:
        raise ValueError(
            "Expected brsr_id in edge file. "
            f"Available columns: {list(df.columns)}"
        )

    if "esrs_datapoint_id" not in df.columns:
        raise ValueError(
            "Expected esrs_datapoint_id in edge file. "
            f"Available columns: {list(df.columns)}"
        )

    for _, row in df.iterrows():
        bid = normalize_text(row.get("brsr_id", ""))
        eid = normalize_text(row.get("esrs_datapoint_id", ""))

        if bid and eid:
            result[bid].add(eid)

    return dict(result)


# ============================================================================
# BRSR / GRI / ESRS INDEXES
# ============================================================================

def load_indexes():
    print("\n" + "=" * 80)
    print("LOADING OFFICIAL SYMBOLIC SOURCES")
    print("=" * 80)

    brsr_gri_df = load_csv(BRSR_GRI_FILE)
    gri_esrs_df = load_csv(GRI_ESRS_FILE)
    interop_df = load_csv(INTEROP_FILE)

    # ------------------------------------------------------------------------
    # BRSR text
    # ------------------------------------------------------------------------
    if "brsr_id" not in brsr_gri_df.columns:
        raise ValueError(
            f"{BRSR_GRI_FILE} has no brsr_id column.\n"
            f"Columns: {list(brsr_gri_df.columns)}"
        )

    brsr_gri_df["brsr_id"] = (
        brsr_gri_df["brsr_id"].astype(str).str.strip()
    )

    if "gri_codes_union" in brsr_gri_df.columns:
        gri_col = "gri_codes_union"
    elif "gri_codes_consensus" in brsr_gri_df.columns:
        gri_col = "gri_codes_consensus"
    elif "gri_codes_table1" in brsr_gri_df.columns:
        gri_col = "gri_codes_table1"
    elif "gri_codes" in brsr_gri_df.columns:
        gri_col = "gri_codes"
    else:
        raise ValueError(
            "No usable GRI-code column found in gold_pairs.csv.\n"
            f"Columns: {list(brsr_gri_df.columns)}"
        )

    brsr_gri_df["_gri_codes"] = (
        brsr_gri_df[gri_col].apply(parse_list)
    )

    brsr_text_map: Dict[str, str] = {}

    for _, row in brsr_gri_df.iterrows():
        bid = normalize_text(row.get("brsr_id", ""))

        if not bid:
            continue

        text = build_brsr_text(row)

        if not text:
            continue

        if bid in brsr_text_map:
            if text not in brsr_text_map[bid]:
                brsr_text_map[bid] = normalize_text(
                    brsr_text_map[bid] + " " + text
                )
        else:
            brsr_text_map[bid] = text

    print(f"BRSR text entries: {len(brsr_text_map)}")

    if len(brsr_text_map) == 0:
        raise RuntimeError(
            "\nNO BRSR TEXT WAS FOUND.\n"
            "Script 14 cannot continue because the teacher would receive "
            "empty BRSR queries.\n\n"
            "Script 12 constructs BRSR text from gold_pairs.csv using:\n"
            "  brsr_text + remarks + mapping_semantics\n\n"
            "Check the columns in:\n"
            f"  {BRSR_GRI_FILE}\n"
        )

    # ------------------------------------------------------------------------
    # BRSR -> GRI symbolic graph
    # ------------------------------------------------------------------------
    brsr_to_gri: Dict[str, Set[str]] = defaultdict(set)

    for _, row in brsr_gri_df.iterrows():
        bid = normalize_text(row.get("brsr_id", ""))

        if not bid:
            continue

        for code in row["_gri_codes"]:
            code = normalize_code(code)

            if code:
                brsr_to_gri[bid].add(code)

    # ------------------------------------------------------------------------
    # ESRS index
    # ------------------------------------------------------------------------
    required = {"esrs_datapoint_id", "esrs_name"}
    missing = required - set(gri_esrs_df.columns)

    if missing:
        raise ValueError(
            f"GRI-ESRS mapping is missing columns: {sorted(missing)}"
        )

    gri_esrs_df["esrs_datapoint_id"] = (
        gri_esrs_df["esrs_datapoint_id"]
        .astype(str)
        .str.strip()
    )

    esrs_unique = (
        gri_esrs_df
        .drop_duplicates("esrs_datapoint_id")
        .copy()
    )

    esrs_text_map = {}

    for _, row in esrs_unique.iterrows():
        eid = normalize_text(
            row.get("esrs_datapoint_id", "")
        )

        if eid:
            esrs_text_map[eid] = build_esrs_text(row)

    esrs_rows = {
        normalize_text(row["esrs_datapoint_id"]): row.to_dict()
        for _, row in esrs_unique.iterrows()
    }

    print(f"ESRS text entries: {len(esrs_text_map)}")

    if not esrs_text_map:
        raise RuntimeError("No ESRS datapoint text was constructed.")

    # ------------------------------------------------------------------------
    # GRI -> ESRS indexes
    #
    # We keep both:
    #   disclosure-level mapping
    #   requirement/sub-point mapping
    # ------------------------------------------------------------------------
    gri_disclosure_to_esrs: Dict[str, Set[str]] = defaultdict(set)
    gri_requirement_to_esrs: Dict[str, Set[str]] = defaultdict(set)

    gri_code_text: Dict[str, str] = {}

    def add_mapping(code, eid, row):
        code = normalize_code(code)

        if not code or not eid:
            return

        if re.match(r"^\d+-\d+$", code):
            gri_disclosure_to_esrs[code].add(eid)
        else:
            gri_requirement_to_esrs[code].add(eid)

        text = build_gri_text(row)

        if text:
            gri_code_text[code] = text

    for _, row in gri_esrs_df.iterrows():
        eid = normalize_text(
            row.get("esrs_datapoint_id", "")
        )

        if not eid:
            continue

        standard = normalize_code(
            row.get("gri_standard", "")
        )

        disclosure = normalize_code(
            row.get("gri_disclosure", "")
        )

        number = normalize_code(
            row.get("gri_number", "")
        )

        # Most official workbook rows already contain a disclosure code.
        if disclosure:
            add_mapping(disclosure, eid, row)

        # If the workbook stores standard + number separately, reconstruct
        # the disclosure code where possible.
        if not disclosure and standard and number:
            m = re.search(r"(\d+)", standard)

            if m:
                reconstructed = (
                    f"{m.group(1)}-{number}"
                )
                add_mapping(
                    reconstructed,
                    eid,
                    row,
                )

        # Some versions contain a finer-grained GRI requirement code.
        if number and disclosure:
            number_clean = normalize_code(number)

            if (
                number_clean != disclosure
                and not number_clean.startswith(
                    disclosure + "-"
                )
            ):
                requirement = (
                    disclosure + "-" + number_clean
                )

                add_mapping(
                    requirement,
                    eid,
                    row,
                )

    # ------------------------------------------------------------------------
    # Interoperability metadata
    # ------------------------------------------------------------------------
    interop_by_gri = defaultdict(set)

    if "gri_disclosure_id" in interop_df.columns:
        for _, row in interop_df.iterrows():
            code = normalize_code(
                row.get("gri_disclosure_id", "")
            )

            mapping_type = normalize_text(
                row.get("mapping_type", "")
            )

            if code:
                interop_by_gri[code].add(
                    mapping_type
                )

    print(
        f"GRI disclosure nodes: "
        f"{len(gri_disclosure_to_esrs)}"
    )

    print(
        f"GRI requirement nodes: "
        f"{len(gri_requirement_to_esrs)}"
    )

    return {
        "brsr_to_gri": brsr_to_gri,
        "brsr_text_map": brsr_text_map,
        "gri_disclosure_to_esrs": gri_disclosure_to_esrs,
        "gri_requirement_to_esrs": gri_requirement_to_esrs,
        "gri_code_text": gri_code_text,
        "interop_by_gri": interop_by_gri,
        "esrs_text_map": esrs_text_map,
        "esrs_rows": esrs_rows,
    }


# ============================================================================
# SYMBOLIC EXPANSION
# ============================================================================

def disclosure_prefix(code: str) -> str:
    code = normalize_code(code)

    m = re.match(r"^(\d+-\d+)", code)

    if m:
        return m.group(1)

    return code


def expand_gri_to_esrs(
    gri_code: str,
    gri_disclosure_to_esrs,
    gri_requirement_to_esrs,
) -> List[Tuple[str, str]]:
    """
    Expand one official GRI code into ESRS datapoints.

    Returns:
        [(esrs_datapoint_id, match_level), ...]
    """
    gri_code = normalize_code(gri_code)

    results = []
    seen = set()

    # Exact/fine-grained requirement first.
    if gri_code in gri_requirement_to_esrs:
        for eid in gri_requirement_to_esrs[gri_code]:
            if eid not in seen:
                results.append(
                    (eid, "requirement")
                )
                seen.add(eid)

    # Disclosure-level mapping.
    disclosure = disclosure_prefix(gri_code)

    if disclosure in gri_disclosure_to_esrs:
        for eid in gri_disclosure_to_esrs[disclosure]:
            if eid not in seen:
                results.append(
                    (eid, "disclosure")
                )
                seen.add(eid)

    return results


def build_symbolic_candidates(
    brsr_ids: Set[str],
    brsr_to_gri,
    gri_disclosure_to_esrs,
    gri_requirement_to_esrs,
    gold_targets_map,
    max_gri_per_brsr=MAX_GRI_PER_BRSR,
    max_esrs_per_gri=MAX_ESRS_PER_GRI,
    max_candidates=MAX_CANDIDATES_PER_BRSR,
):
    """
    Construct candidates exclusively through:

        BRSR -> official GRI -> official ESRS mapping

    Dense BRSR->ESRS retrieval is deliberately NOT used here.
    """

    rows = []

    for bid in sorted(brsr_ids):
        official_gris = sorted(
            brsr_to_gri.get(bid, set())
        )

        if not official_gris:
            continue

        # Keep the official bridge. There is no semantic GRI retriever
        # involved in KD candidate construction.
        official_gris = official_gris[
            :max_gri_per_brsr
        ]

        candidates = {}

        for gri_code in official_gris:
            mappings = expand_gri_to_esrs(
                gri_code,
                gri_disclosure_to_esrs,
                gri_requirement_to_esrs,
            )

            mappings = mappings[
                :max_esrs_per_gri
            ]

            for eid, match_level in mappings:
                if eid not in candidates:
                    candidates[eid] = {
                        "gri_code": gri_code,
                        "match_level": match_level,
                        "candidate_source": "symbolic",
                    }

        # Optional: official train gold edges that were not recovered by
        # the transitive mapping are added ONLY for TRAIN BRSR IDs.
        # This does not use validation/test supervision.
        if ADD_MISSING_GOLD_TARGETS:
            targets = gold_targets_map.get(
                bid,
                set(),
            )

            for eid in sorted(targets):
                if eid not in candidates:
                    candidates[eid] = {
                        "gri_code": "",
                        "match_level": "gold_added",
                        "candidate_source": "gold_added",
                    }

        # Keep all symbolic candidates, but cap extremely large candidate
        # sets deterministically. Gold targets are always retained.
        target_ids = gold_targets_map.get(
            bid,
            set(),
        )

        ordered = []

        # First preserve gold targets.
        for eid in sorted(target_ids):
            if eid in candidates:
                ordered.append(
                    (eid, candidates[eid])
                )

        # Then symbolic candidates.
        for eid in sorted(candidates):
            if eid in target_ids:
                continue

            ordered.append(
                (eid, candidates[eid])
            )

            if len(ordered) >= max_candidates:
                break

        for eid, meta in ordered:
            rows.append(
                {
                    "brsr_id": bid,
                    "esrs_datapoint_id": eid,
                    **meta,
                    "gold_target": int(
                        eid in target_ids
                    ),
                }
            )

    return pd.DataFrame(rows)


# ============================================================================
# TEACHER SCORING
# ============================================================================

def score_with_teacher(
    teacher: CrossEncoder,
    candidates_df: pd.DataFrame,
    brsr_text_map: Dict[str, str],
    esrs_text_map: Dict[str, str],
    output_path: Path,
):
    """
    Score every symbolic candidate with the CE teacher.
    """

    if candidates_df.empty:
        raise RuntimeError(
            "No KD candidates were generated."
        )

    pairs = []
    valid_indices = []

    for idx, row in candidates_df.iterrows():
        bid = normalize_text(row["brsr_id"])
        eid = normalize_text(row["esrs_datapoint_id"])

        btext = brsr_text_map.get(bid, "")
        etext = esrs_text_map.get(eid, "")

        if not btext or not etext:
            continue

        pairs.append([btext, etext])
        valid_indices.append(idx)

    if not pairs:
        raise RuntimeError(
            "No candidate pairs contain both BRSR and ESRS text."
        )

    print(
        f"Scoring {len(pairs)} candidate pairs..."
    )

    scores = teacher.predict(
        pairs,
        batch_size=TEACHER_BATCH_SIZE,
        show_progress_bar=True,
    )

    out = candidates_df.loc[
        valid_indices
    ].copy()

    out["teacher_score"] = [
        float(x)
        for x in scores
    ]

    # ------------------------------------------------------------------------
    # Per-BRSR teacher distribution
    # ------------------------------------------------------------------------
    out["teacher_probability"] = 0.0

    for bid, group in out.groupby(
        "brsr_id",
        sort=False,
    ):
        idx = group.index

        logits = torch.tensor(
            group["teacher_score"].astype(float).values,
            dtype=torch.float32,
        )

        probs = torch.softmax(
            logits / TEMPERATURE,
            dim=0,
        ).numpy()

        out.loc[
            idx,
            "teacher_probability"
        ] = probs

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Saved teacher scores -> {output_path}"
    )

    return out


# ============================================================================
# STUDENT MODEL
# ============================================================================

def load_student_model(device: str):
    """
    Load a standard SentenceTransformer bi-encoder.

    We deliberately keep the student lightweight and use the same
    MiniLM family as the retrieval baseline.
    """
    print(
        f"\nLoading student: {STUDENT_MODEL}"
    )

    student = SentenceTransformer(
        STUDENT_MODEL,
        device=device,
    )

    return student


def encode_texts(
    model: SentenceTransformer,
    texts: List[str],
):
    return model.encode(
        texts,
        batch_size=32,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


# ============================================================================
# STUDENT KD LOSS
# ============================================================================

def _forward_sentence_embeddings(
    student: SentenceTransformer,
    texts: List[str],
):
    """
    Gradient-preserving SentenceTransformer forward pass.

    DO NOT replace this with SentenceTransformer.encode() during training:
    encode() runs inference/no-grad internally and detaches the graph.
    """
    features = student.tokenize(texts)
    features = {
        key: value.to(student.device) if isinstance(value, torch.Tensor) else value
        for key, value in features.items()
    }

    output = student(features)
    embeddings = output["sentence_embedding"]

    return F.normalize(
        embeddings,
        p=2,
        dim=1,
    )


def student_scores_for_query(
    student: SentenceTransformer,
    btext: str,
    candidate_texts: List[str],
    require_grad: bool = False,
):
    """Return cosine scores between one BRSR query and ESRS candidates."""
    if require_grad:
        q_emb = _forward_sentence_embeddings(
            student,
            [btext],
        )[0]
        d_emb = _forward_sentence_embeddings(
            student,
            candidate_texts,
        )
    else:
        with torch.no_grad():
            q_emb = _forward_sentence_embeddings(
                student,
                [btext],
            )[0]
            d_emb = _forward_sentence_embeddings(
                student,
                candidate_texts,
            )

    return torch.matmul(d_emb, q_emb)


def _prepare_group(
    group: pd.DataFrame,
    brsr_text_map,
    esrs_text_map,
):
    """Extract text, teacher probabilities and gold labels for one query."""
    bid = normalize_text(group.iloc[0]["brsr_id"])
    btext = brsr_text_map.get(bid, "")

    candidate_ids = []
    candidate_texts = []
    teacher_probs = []
    gold_labels = []

    for _, row in group.iterrows():
        eid = normalize_text(row["esrs_datapoint_id"])
        etext = esrs_text_map.get(eid, "")

        if not eid or not etext:
            continue

        candidate_ids.append(eid)
        candidate_texts.append(etext)
        teacher_probs.append(
            safe_float(row.get("teacher_probability", 0.0))
        )
        gold_labels.append(
            safe_float(row.get("gold_target", 0.0))
        )

    return bid, btext, candidate_ids, candidate_texts, teacher_probs, gold_labels


def train_student(
    training_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    brsr_text_map,
    esrs_text_map,
    output_dir: Path,
    run_name: str,
    device: str,
):
    """
    Robust train-only KD with validation-only checkpoint selection.

    Training:
        KL(teacher || student) + optional supervised BCE on TRAIN gold edges.

    Validation:
        Teacher soft targets are used only to calculate validation KL.
        No validation labels are used in the optimization objective.
        No validation gradients are computed.

    The best checkpoint is selected exclusively by validation KL.
    """
    required = {
        "brsr_id",
        "esrs_datapoint_id",
        "teacher_probability",
        "gold_target",
    }

    if training_df.empty:
        raise RuntimeError(
            f"{run_name}: training dataframe is empty."
        )

    missing = required - set(training_df.columns)
    if missing:
        raise RuntimeError(
            f"{run_name}: missing training columns: {sorted(missing)}"
        )

    if validation_df.empty:
        raise RuntimeError(
            f"{run_name}: validation KD dataframe is empty. "
            "A validation-isolated KD run is required for reliable checkpoint selection."
        )

    missing_val = required - set(validation_df.columns)
    if missing_val:
        raise RuntimeError(
            f"{run_name}: missing validation columns: {sorted(missing_val)}"
        )

    student = load_student_model(device)
    student.max_seq_length = STUDENT_MAX_LENGTH

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    train_groups = {
        str(bid): group.copy()
        for bid, group in training_df.groupby("brsr_id", sort=False)
    }
    val_groups = {
        str(bid): group.copy()
        for bid, group in validation_df.groupby("brsr_id", sort=False)
    }

    print(f"\n{run_name} student training groups: {len(train_groups)}")
    print(f"{run_name} validation groups: {len(val_groups)}")

    if len(train_groups) == 0 or len(val_groups) == 0:
        raise RuntimeError(
            f"{run_name}: insufficient train/validation groups for KD."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    steps_per_epoch = max(
        1,
        math.ceil(len(train_groups) / BATCH_QUERIES),
    )
    total_steps = max(1, EPOCHS * steps_per_epoch)
    warmup_steps = min(
        total_steps - 1,
        WARMUP_EPOCHS * steps_per_epoch,
    )

    global_step = 0
    best_val = float("inf")
    best_epoch = -1
    patience_counter = 0
    history = []

    def update_lr(step: int):
        if warmup_steps > 0 and step <= warmup_steps:
            scale = step / float(warmup_steps)
        else:
            remaining = max(1, total_steps - warmup_steps)
            progress = (step - warmup_steps) / float(remaining)
            scale = max(0.0, 1.0 - progress)

        lr = LEARNING_RATE * scale
        for group in optimizer.param_groups:
            group["lr"] = lr

    def validation_kl():
        student.eval()
        losses = []
        used_groups = 0

        with torch.no_grad():
            for bid, group in val_groups.items():
                bid, btext, _, candidate_texts, teacher_probs, _ = _prepare_group(
                    group, brsr_text_map, esrs_text_map
                )

                if not btext or len(candidate_texts) < 2:
                    continue

                scores = student_scores_for_query(
                    student,
                    btext,
                    candidate_texts,
                    require_grad=False,
                )

                teacher_t = torch.tensor(
                    teacher_probs,
                    dtype=torch.float32,
                    device=scores.device,
                )
                teacher_t = teacher_t.clamp_min(1e-8)
                teacher_t = teacher_t / teacher_t.sum().clamp_min(1e-12)

                student_log_probs = F.log_softmax(
                    scores / TEMPERATURE,
                    dim=0,
                )

                kl = F.kl_div(
                    student_log_probs,
                    teacher_t,
                    reduction="sum",
                ) * (TEMPERATURE ** 2)

                losses.append(float(kl.cpu()))
                used_groups += 1

        student.train()

        if not losses:
            return float("inf"), 0

        return float(np.mean(losses)), used_groups

    train_ids = list(train_groups.keys())

    for epoch in range(1, EPOCHS + 1):
        student.train()
        random.shuffle(train_ids)

        epoch_losses = []
        epoch_kd = []
        epoch_gold = []
        used_train_groups = 0

        for start in range(0, len(train_ids), BATCH_QUERIES):
            batch_ids = train_ids[start:start + BATCH_QUERIES]
            optimizer.zero_grad(set_to_none=True)

            batch_loss = None
            batch_kd_values = []
            batch_gold_values = []
            batch_used = 0

            for bid in batch_ids:
                group = train_groups[bid]
                bid, btext, _, candidate_texts, teacher_probs, gold_labels = _prepare_group(
                    group, brsr_text_map, esrs_text_map
                )

                if not btext or len(candidate_texts) < 2:
                    continue

                scores = student_scores_for_query(
                    student,
                    btext,
                    candidate_texts,
                    require_grad=True,
                )

                if not scores.requires_grad:
                    raise RuntimeError(
                        "Student scores do not require gradients. "
                        "The KD forward path must not use SentenceTransformer.encode()."
                    )

                teacher_t = torch.tensor(
                    teacher_probs,
                    dtype=torch.float32,
                    device=scores.device,
                )
                teacher_t = teacher_t.clamp_min(1e-8)
                teacher_t = teacher_t / teacher_t.sum().clamp_min(1e-12)

                student_log_probs = F.log_softmax(
                    scores / TEMPERATURE,
                    dim=0,
                )

                kd_loss = F.kl_div(
                    student_log_probs,
                    teacher_t,
                    reduction="sum",
                ) * (TEMPERATURE ** 2)

                gold_t = torch.tensor(
                    gold_labels,
                    dtype=torch.float32,
                    device=scores.device,
                )

                # Cosine scores are in roughly [-1, 1]; scaling gives BCE
                # a usable logit range without introducing another model.
                gold_loss = F.binary_cross_entropy_with_logits(
                    scores * 10.0,
                    gold_t,
                )

                loss = (
                    LAMBDA_KD * kd_loss
                    + LAMBDA_GOLD * gold_loss
                )

                batch_loss = loss if batch_loss is None else batch_loss + loss
                batch_kd_values.append(float(kd_loss.detach().cpu()))
                batch_gold_values.append(float(gold_loss.detach().cpu()))
                batch_used += 1

            if batch_loss is None:
                continue

            batch_loss = batch_loss / float(batch_used)
            batch_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                student.parameters(),
                MAX_GRAD_NORM,
            )

            global_step += 1
            update_lr(global_step)
            optimizer.step()

            used_train_groups += batch_used
            epoch_losses.append(float(batch_loss.detach().cpu()))
            epoch_kd.extend(batch_kd_values)
            epoch_gold.extend(batch_gold_values)

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("inf")
        train_kd = float(np.mean(epoch_kd)) if epoch_kd else float("inf")
        train_gold = float(np.mean(epoch_gold)) if epoch_gold else float("inf")
        val_loss, val_used = validation_kl()

        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "teacher": run_name.lower(),
            "epoch": epoch,
            "train_loss": train_loss,
            "train_kd_loss": train_kd,
            "train_gold_loss": train_gold,
            "validation_kl": val_loss,
            "train_groups_used": used_train_groups,
            "validation_groups_used": val_used,
            "learning_rate": current_lr,
        }
        history.append(row)

        print(f"\n{run_name} | Epoch {epoch}/{EPOCHS}")
        print(f"  train_loss      : {train_loss:.6f}")
        print(f"  train_KD_loss   : {train_kd:.6f}")
        print(f"  train_gold_loss : {train_gold:.6f}")
        print(f"  validation_KL   : {val_loss:.6f}")
        print(f"  validation_groups_used: {val_used}")

        if not math.isfinite(val_loss) or val_used == 0:
            raise RuntimeError(
                f"{run_name}: validation KD loss could not be computed. "
                "Check symbolic validation candidate coverage and teacher scores."
            )

        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_epoch = epoch
            patience_counter = 0

            best_dir = output_dir / "best"
            if best_dir.exists():
                import shutil
                shutil.rmtree(best_dir)

            student.save(str(best_dir))
            print(f"  -> saved BEST student: {best_dir}")
        else:
            patience_counter += 1
            print(f"  -> no validation improvement ({patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                print("  -> early stopping")
                break

    if best_epoch < 0:
        raise RuntimeError(
            f"{run_name}: no valid validation checkpoint was saved."
        )

    history_path = PROCESSED_DIR / f"kd_epoch_history_{run_name.lower()}.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)

    best_dir = output_dir / "best"
    final_student = SentenceTransformer(str(best_dir), device=device)
    final_student.max_seq_length = STUDENT_MAX_LENGTH

    final_dir = output_dir / "final"
    if final_dir.exists():
        import shutil
        shutil.rmtree(final_dir)
    final_student.save(str(final_dir))

    metadata = {
        "run_name": run_name,
        "student_model": STUDENT_MODEL,
        "student_max_length": STUDENT_MAX_LENGTH,
        "temperature": TEMPERATURE,
        "lambda_kd": LAMBDA_KD,
        "lambda_gold": LAMBDA_GOLD,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "epochs_requested": EPOCHS,
        "best_epoch": best_epoch,
        "best_validation_kl": best_val,
        "train_brsr_ids": len(train_groups),
        "validation_brsr_ids": len(val_groups),
        "train_candidate_rows": int(len(training_df)),
        "validation_candidate_rows": int(len(validation_df)),
        "history_file": str(history_path),
    }

    with open(output_dir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved final KD student -> {final_dir}")
    print(f"Saved epoch history -> {history_path}")

    return final_student, metadata


# ============================================================================
# LEAKAGE CHECK
# ============================================================================

def verify_training_isolation(
    training_df: pd.DataFrame,
    train_ids: Set[str],
    validation_ids: Set[str],
    test_ids: Set[str],
):
    print("\n" + "=" * 80)
    print("KD DATA LEAKAGE CHECK")
    print("=" * 80)

    if training_df is None:
        raise RuntimeError(
            "KD training dataframe is None."
        )

    if training_df.empty:
        raise RuntimeError(
            "KD training dataframe is EMPTY."
        )

    if "brsr_id" not in training_df.columns:
        raise RuntimeError(
            "KD training dataframe has no brsr_id column.\n"
            f"Columns: {list(training_df.columns)}"
        )

    used_ids = set(
        training_df["brsr_id"]
        .astype(str)
        .str.strip()
    )

    train_ids = {
        str(x).strip()
        for x in train_ids
    }

    validation_ids = {
        str(x).strip()
        for x in validation_ids
    }

    test_ids = {
        str(x).strip()
        for x in test_ids
    }

    train_overlap = used_ids & train_ids
    val_overlap = used_ids & validation_ids
    test_overlap = used_ids & test_ids

    print(
        f"Training BRSR IDs used: {len(used_ids)}"
    )

    print(
        f"Train IDs present: {len(train_overlap)}"
    )

    print(
        f"Validation IDs present: {len(val_overlap)}"
    )

    print(
        f"Test IDs present: {len(test_overlap)}"
    )

    if val_overlap:
        raise RuntimeError(
            "KD LEAKAGE: validation BRSR IDs found:\n"
            + "\n".join(sorted(val_overlap))
        )

    if test_overlap:
        raise RuntimeError(
            "KD LEAKAGE: test BRSR IDs found:\n"
            + "\n".join(sorted(test_overlap))
        )

    if not used_ids.issubset(train_ids):
        unexpected = used_ids - train_ids

        raise RuntimeError(
            "KD training contains BRSR IDs outside the train split:\n"
            + "\n".join(sorted(unexpected))
        )

    print("STATUS: PASS")


# ============================================================================
# STUDENT RETRIEVAL EVALUATION
# ============================================================================

def student_rank_candidates(
    student: SentenceTransformer,
    brsr_id: str,
    candidates: pd.DataFrame,
    brsr_text_map,
    esrs_text_map,
):
    btext = brsr_text_map.get(
        brsr_id,
        "",
    )

    if not btext or candidates.empty:
        return []

    candidate_ids = []
    candidate_texts = []

    for _, row in candidates.iterrows():
        eid = normalize_text(
            row["esrs_datapoint_id"]
        )

        etext = esrs_text_map.get(
            eid,
            "",
        )

        if not etext:
            continue

        candidate_ids.append(eid)
        candidate_texts.append(etext)

    if not candidate_texts:
        return []

    scores = student_scores_for_query(
        student,
        btext,
        candidate_texts,
    )

    scores = scores.detach().cpu().numpy()

    ranked = sorted(
        zip(candidate_ids, scores),
        key=lambda x: float(x[1]),
        reverse=True,
    )

    return [
        {
            "esrs_datapoint_id": eid,
            "score": float(score),
        }
        for eid, score in ranked
    ]


def evaluate_rankings(
    rankings: Dict[str, List[dict]],
    targets: Dict[str, Set[str]],
):
    rows = []

    recalls = {
        k: []
        for k in EVAL_KS
    }

    rrs = []

    for bid, target_set in targets.items():
        if not target_set:
            continue

        ranked = [
            x["esrs_datapoint_id"]
            for x in rankings.get(bid, [])
        ]

        rr = 0.0

        for rank, eid in enumerate(
            ranked,
            start=1,
        ):
            if eid in target_set:
                rr = 1.0 / rank
                break

        rrs.append(rr)

        row = {
            "brsr_id": bid,
            "target_count": len(target_set),
            "first_relevant_rank": (
                0
                if rr == 0
                else int(round(1.0 / rr))
            ),
        }

        for k in EVAL_KS:
            hit = any(
                eid in target_set
                for eid in ranked[:k]
            )

            recalls[k].append(
                float(hit)
            )

            row[f"hit@{k}"] = int(hit)

        rows.append(row)

    metrics = {
        "evaluated_brsr_ids": len(rrs),
        "mrr": (
            float(np.mean(rrs))
            if rrs
            else 0.0
        ),
    }

    for k in EVAL_KS:
        metrics[
            f"recall@{k}"
        ] = (
            float(np.mean(recalls[k]))
            if recalls[k]
            else 0.0
        )

    return metrics, pd.DataFrame(rows)


def evaluate_student_on_symbolic_candidates(
    student,
    candidate_df,
    targets,
    brsr_text_map,
    esrs_text_map,
    split_name,
):
    rankings = {}

    for bid in sorted(targets):
        subset = candidate_df[
            candidate_df["brsr_id"].astype(str)
            == str(bid)
        ]

        rankings[bid] = student_rank_candidates(
            student,
            bid,
            subset,
            brsr_text_map,
            esrs_text_map,
        )

    metrics, detail = evaluate_rankings(
        rankings,
        targets,
    )

    detail["split"] = split_name

    return metrics, detail, rankings


# ============================================================================
# MAIN
# ============================================================================

def main():
    set_seed()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SCRIPT 14 — ROBUST SYMBOLIC KNOWLEDGE DISTILLATION")
    print("=" * 80)

    if torch.cuda.is_available():
        device = "cuda"
        print(f"\nDevice: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("\nDevice: CPU")

    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>"))
    print("Visible GPU count:", torch.cuda.device_count())

    # ----------------------------------------------------------------------
    # 1. Load split-level gold edges ONLY to define the split IDs and for
    #    final retrieval evaluation. Validation/test labels never enter the
    #    student optimization objective.
    # ----------------------------------------------------------------------
    train_df = load_csv(TRAIN_FILE)
    val_df = load_csv(VAL_FILE)
    test_df = load_csv(TEST_FILE)

    train_targets = gold_targets(train_df)
    val_targets = gold_targets(val_df)
    test_targets = gold_targets(test_df)

    train_ids = set(train_targets)
    val_ids = set(val_targets)
    test_ids = set(test_targets)

    print("\nLoaded:")
    print(f"  Train edges:       {len(train_df)}")
    print(f"  Validation edges:  {len(val_df)}")
    print(f"  Test edges:        {len(test_df)}")
    print(f"  Train BRSR IDs:    {len(train_ids)}")
    print(f"  Validation IDs:    {len(val_ids)}")
    print(f"  Test IDs:          {len(test_ids)}")

    # ----------------------------------------------------------------------
    # 2. Official indexes.
    # ----------------------------------------------------------------------
    indexes = load_indexes()
    brsr_to_gri = indexes["brsr_to_gri"]
    brsr_text_map = indexes["brsr_text_map"]
    gri_disclosure_to_esrs = indexes["gri_disclosure_to_esrs"]
    gri_requirement_to_esrs = indexes["gri_requirement_to_esrs"]
    esrs_text_map = indexes["esrs_text_map"]

    # Every split must have BRSR text for a meaningful query.
    split_missing = {
        "train": sorted(train_ids - set(brsr_text_map)),
        "validation": sorted(val_ids - set(brsr_text_map)),
        "test": sorted(test_ids - set(brsr_text_map)),
    }
    for split_name, missing in split_missing.items():
        if missing:
            raise RuntimeError(
                f"{split_name.upper()} BRSR IDs without text: {missing}"
            )

    print(f"\nUsable train BRSR text entries: {len(train_ids)}")
    print(f"Usable validation BRSR text entries: {len(val_ids)}")
    print(f"Usable test BRSR text entries: {len(test_ids)}")

    # ----------------------------------------------------------------------
    # 3. TRAIN symbolic candidates.
    #    Gold labels may be used ONLY for TRAIN candidate completion.
    # ----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("BUILDING TRAIN SYMBOLIC KD CANDIDATES")
    print("=" * 80)

    train_candidates = build_symbolic_candidates(
        brsr_ids=train_ids,
        brsr_to_gri=brsr_to_gri,
        gri_disclosure_to_esrs=gri_disclosure_to_esrs,
        gri_requirement_to_esrs=gri_requirement_to_esrs,
        gold_targets_map=train_targets,
    )

    if train_candidates.empty:
        raise RuntimeError("No TRAIN symbolic KD candidates were produced.")

    train_candidates.to_csv(
        PROCESSED_DIR / "kd_symbolic_candidates_train.csv",
        index=False,
    )

    print(f"Train symbolic candidate rows: {len(train_candidates)}")
    print(f"Train candidate BRSR IDs: {train_candidates['brsr_id'].nunique()}")
    print(f"Train gold targets in candidate pool: {int(train_candidates['gold_target'].sum())}")
    print(
        "Train gold-added candidates:",
        int((train_candidates["candidate_source"] == "gold_added").sum()),
    )

    verify_training_isolation(
        train_candidates,
        train_ids,
        val_ids,
        test_ids,
    )

    # ----------------------------------------------------------------------
    # 4. VALIDATION symbolic candidates.
    #    IMPORTANT: no validation gold targets are passed into candidate
    #    construction. This prevents label-dependent candidate injection.
    # ----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("BUILDING VALIDATION SYMBOLIC KD CANDIDATES")
    print("=" * 80)

    val_candidates = build_symbolic_candidates(
        brsr_ids=val_ids,
        brsr_to_gri=brsr_to_gri,
        gri_disclosure_to_esrs=gri_disclosure_to_esrs,
        gri_requirement_to_esrs=gri_requirement_to_esrs,
        gold_targets_map={},
    )

    val_candidates.to_csv(
        PROCESSED_DIR / "kd_symbolic_candidates_validation.csv",
        index=False,
    )

    print(f"Validation symbolic candidate rows: {len(val_candidates)}")
    print(
        f"Validation candidate BRSR IDs: "
        f"{val_candidates['brsr_id'].nunique() if not val_candidates.empty else 0}"
    )

    missing_val_candidates = sorted(
        val_ids - set(val_candidates["brsr_id"].astype(str))
        if not val_candidates.empty
        else val_ids
    )
    if missing_val_candidates:
        print("WARNING: validation IDs with no symbolic ESRS candidates:")
        for bid in missing_val_candidates:
            print(f"  {bid}")

    if val_candidates.empty:
        raise RuntimeError(
            "Validation symbolic candidate pool is empty. "
            "Cannot perform reliable KD checkpoint selection."
        )

    # ----------------------------------------------------------------------
    # 5. Optional medium-confidence TRAIN pool.
    #    Never used for validation/test.
    # ----------------------------------------------------------------------
    medium_targets = {}
    if MEDIUM_FILE.exists():
        try:
            medium_df = load_csv(MEDIUM_FILE)
            raw_medium = gold_targets(medium_df)
            restricted = defaultdict(set)
            for bid, eids in raw_medium.items():
                if bid not in train_ids:
                    continue
                for eid in sorted(eids):
                    if len(restricted[bid]) >= MAX_MEDIUM_PER_BRSR:
                        break
                    if eid in esrs_text_map:
                        restricted[bid].add(eid)
            medium_targets = dict(restricted)
            print(f"\nMedium-confidence train BRSR IDs: {len(medium_targets)}")
        except Exception as exc:
            print(f"\nWARNING: could not load medium pool: {exc}")
            medium_targets = {}
    else:
        print("\nNo medium-confidence pool found.")

    # ----------------------------------------------------------------------
    # 6. Teacher scoring.
    #    For each teacher, score TRAIN and VALIDATION separately.
    #    TEST is deliberately not teacher-scored here.
    # ----------------------------------------------------------------------
    teacher_specs = [
        ("gold", CROSS_ENCODER_GOLD),
        ("gold_medium", CROSS_ENCODER_GOLD_MEDIUM),
    ]

    teacher_outputs = {}
    teacher_paths = {}

    for teacher_name, teacher_path in teacher_specs:
        print("\n" + "=" * 80)
        print(f"LOADING {teacher_name.upper()} CROSS-ENCODER TEACHER")
        print("=" * 80)

        if not teacher_path.exists():
            print(f"WARNING: teacher not found: {teacher_path}")
            continue

        teacher = CrossEncoder(
            str(teacher_path),
            device=device,
            max_length=CE_MAX_LENGTH,
        )
        print(f"Teacher: {teacher_path}")

        train_scored = score_with_teacher(
            teacher,
            train_candidates,
            brsr_text_map,
            esrs_text_map,
            PROCESSED_DIR / f"kd_teacher_scores_{teacher_name}_train.csv",
        )

        val_scored = score_with_teacher(
            teacher,
            val_candidates,
            brsr_text_map,
            esrs_text_map,
            PROCESSED_DIR / f"kd_teacher_scores_{teacher_name}_validation.csv",
        )

        teacher_outputs[teacher_name] = {
            "train": train_scored,
            "validation": val_scored,
        }
        teacher_paths[teacher_name] = teacher_path

        del teacher
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not teacher_outputs:
        raise RuntimeError("No cross-encoder teacher was available.")

    # ----------------------------------------------------------------------
    # 7. Add medium TRAIN candidates only to the gold_medium teacher dataset.
    #    They are soft-teacher candidates, not hard positives.
    # ----------------------------------------------------------------------
    training_tables = {}

    for teacher_name, split_tables in teacher_outputs.items():
        train_scored = split_tables["train"].copy()

        if teacher_name == "gold_medium" and medium_targets:
            medium_rows = []
            for bid, eids in medium_targets.items():
                for eid in sorted(eids):
                    exists = (
                        (train_scored["brsr_id"].astype(str) == str(bid))
                        & (train_scored["esrs_datapoint_id"].astype(str) == str(eid))
                    )
                    if exists.any():
                        continue
                    medium_rows.append(
                        {
                            "brsr_id": bid,
                            "esrs_datapoint_id": eid,
                            "gri_code": "",
                            "match_level": "medium_added",
                            "candidate_source": "medium_added",
                            "gold_target": 0,
                        }
                    )

            if medium_rows:
                medium_extra = pd.DataFrame(medium_rows)
                teacher = CrossEncoder(
                    str(teacher_paths[teacher_name]),
                    device=device,
                    max_length=CE_MAX_LENGTH,
                )
                extra_scored = score_with_teacher(
                    teacher,
                    medium_extra,
                    brsr_text_map,
                    esrs_text_map,
                    PROCESSED_DIR / "kd_teacher_scores_gold_medium_extra_train.csv",
                )
                train_scored = pd.concat(
                    [train_scored, extra_scored],
                    ignore_index=True,
                )
                del teacher
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        training_tables[teacher_name] = train_scored
        train_scored.to_csv(
            PROCESSED_DIR / f"kd_training_data_{teacher_name}.csv",
            index=False,
        )

        # Validation table remains untouched by medium/gold augmentation.
        split_tables["validation"].to_csv(
            PROCESSED_DIR / f"kd_validation_data_{teacher_name}.csv",
            index=False,
        )

        print(
            f"\n{teacher_name}: train rows={len(train_scored)}, "
            f"validation rows={len(split_tables['validation'])}"
        )

    # Strict isolation on TRAIN tables.
    for teacher_name, df in training_tables.items():
        print(f"\nChecking isolation for {teacher_name} KD training dataset...")
        verify_training_isolation(df, train_ids, val_ids, test_ids)

    # ----------------------------------------------------------------------
    # 8. Train students. Validation teacher scores are used ONLY for
    #    validation KL/checkpoint selection.
    # ----------------------------------------------------------------------
    students = {}
    summaries = []
    validation_metrics = []
    test_metrics = []
    validation_details = []
    test_details = []

    # Build validation/test symbolic candidate pools once.
    test_candidates = build_symbolic_candidates(
        brsr_ids=test_ids,
        brsr_to_gri=brsr_to_gri,
        gri_disclosure_to_esrs=gri_disclosure_to_esrs,
        gri_requirement_to_esrs=gri_requirement_to_esrs,
        gold_targets_map={},
    )
    test_candidates.to_csv(
        PROCESSED_DIR / "kd_symbolic_candidates_test.csv",
        index=False,
    )

    for teacher_name, train_kd_df in training_tables.items():
        print("\n" + "=" * 80)
        print(f"TRAINING STUDENT FROM {teacher_name.upper()} TEACHER")
        print("=" * 80)

        validation_kd_df = teacher_outputs[teacher_name]["validation"].copy()

        # Explicitly assert that validation candidates are disjoint from train.
        val_overlap = set(validation_kd_df["brsr_id"].astype(str)) & set(train_kd_df["brsr_id"].astype(str))
        if val_overlap:
            raise RuntimeError(
                f"{teacher_name}: train/validation BRSR overlap detected: {sorted(val_overlap)}"
            )

        output_dir = MODEL_DIR / f"kd_student_{teacher_name}"

        student, metadata = train_student(
            training_df=train_kd_df,
            validation_df=validation_kd_df,
            brsr_text_map=brsr_text_map,
            esrs_text_map=esrs_text_map,
            output_dir=output_dir,
            run_name=teacher_name.upper(),
            device=device,
        )

        # --------------------------------------------------------------
        # Validation retrieval is now a post-training report, not a
        # checkpoint-selection signal. Gold labels are used here only
        # because this is evaluation.
        # --------------------------------------------------------------
        val_m, val_d, _ = evaluate_student_on_symbolic_candidates(
            student,
            val_candidates,
            val_targets,
            brsr_text_map,
            esrs_text_map,
            "validation",
        )

        # Test is evaluated exactly once per student after training.
        test_m, test_d, _ = evaluate_student_on_symbolic_candidates(
            student,
            test_candidates,
            test_targets,
            brsr_text_map,
            esrs_text_map,
            "test",
        )

        print("\n" + "-" * 80)
        print(f"{teacher_name.upper()} KD STUDENT — VALIDATION")
        for k in EVAL_KS:
            print(f"  Recall@{k}: {val_m[f'recall@{k}']:.4f}")
        print(f"  MRR: {val_m['mrr']:.4f}")

        print("\n" + "-" * 80)
        print(f"{teacher_name.upper()} KD STUDENT — TEST")
        for k in EVAL_KS:
            print(f"  Recall@{k}: {test_m[f'recall@{k}']:.4f}")
        print(f"  MRR: {test_m['mrr']:.4f}")

        students[teacher_name] = student

        summaries.append({
            "teacher": teacher_name,
            **metadata,
            "validation_recall@1": val_m["recall@1"],
            "validation_recall@5": val_m["recall@5"],
            "validation_recall@10": val_m["recall@10"],
            "validation_recall@20": val_m["recall@20"],
            "validation_recall@50": val_m["recall@50"],
            "validation_mrr": val_m["mrr"],
            "test_recall@1": test_m["recall@1"],
            "test_recall@5": test_m["recall@5"],
            "test_recall@10": test_m["recall@10"],
            "test_recall@20": test_m["recall@20"],
            "test_recall@50": test_m["recall@50"],
            "test_mrr": test_m["mrr"],
        })

        validation_metrics.append({"teacher": teacher_name, **val_m})
        test_metrics.append({"teacher": teacher_name, **test_m})

        val_d["teacher"] = teacher_name
        test_d["teacher"] = teacher_name
        validation_details.append(val_d)
        test_details.append(test_d)

        del student
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ----------------------------------------------------------------------
    # 9. Save all auditable outputs.
    # ----------------------------------------------------------------------
    pd.DataFrame(summaries).to_csv(
        PROCESSED_DIR / "kd_training_summary.csv",
        index=False,
    )
    pd.DataFrame(validation_metrics).to_csv(
        PROCESSED_DIR / "kd_validation_metrics.csv",
        index=False,
    )
    pd.DataFrame(test_metrics).to_csv(
        PROCESSED_DIR / "kd_test_metrics.csv",
        index=False,
    )

    if validation_details:
        pd.concat(validation_details, ignore_index=True).to_csv(
            PROCESSED_DIR / "kd_validation_predictions.csv",
            index=False,
        )
    if test_details:
        pd.concat(test_details, ignore_index=True).to_csv(
            PROCESSED_DIR / "kd_test_predictions.csv",
            index=False,
        )

    # Candidate coverage report.
    coverage_rows = []
    for split_name, ids, candidate_df in [
        ("train", train_ids, train_candidates),
        ("validation", val_ids, val_candidates),
        ("test", test_ids, test_candidates),
    ]:
        candidate_ids = set(candidate_df["brsr_id"].astype(str)) if not candidate_df.empty else set()
        for bid in sorted(ids):
            subset = candidate_df[candidate_df["brsr_id"].astype(str) == str(bid)] if not candidate_df.empty else pd.DataFrame()
            coverage_rows.append({
                "split": split_name,
                "brsr_id": bid,
                "has_symbolic_candidates": int(bid in candidate_ids),
                "candidate_count": int(len(subset)),
                "gold_target_count": int(len((train_targets if split_name == "train" else val_targets if split_name == "validation" else test_targets).get(bid, set()))),
                "gold_targets_in_symbolic_pool": int(subset.get("gold_target", pd.Series(dtype=int)).sum()) if not subset.empty else 0,
            })

    pd.DataFrame(coverage_rows).to_csv(
        PROCESSED_DIR / "kd_candidate_coverage.csv",
        index=False,
    )

    print("\n" + "=" * 80)
    print("SCRIPT 14 COMPLETE")
    print("=" * 80)
    print("\nKey outputs:")
    for name in [
        "kd_symbolic_candidates_train.csv",
        "kd_symbolic_candidates_validation.csv",
        "kd_symbolic_candidates_test.csv",
        "kd_teacher_scores_gold_train.csv",
        "kd_teacher_scores_gold_validation.csv",
        "kd_teacher_scores_gold_medium_train.csv",
        "kd_teacher_scores_gold_medium_validation.csv",
        "kd_training_summary.csv",
        "kd_validation_metrics.csv",
        "kd_test_metrics.csv",
        "kd_candidate_coverage.csv",
    ]:
        print("  ", PROCESSED_DIR / name)

    print("\nStudent models:")
    for teacher_name in students:
        print("  ", MODEL_DIR / f"kd_student_{teacher_name}" / "final")

    print("\nProtocol:")
    print("  [1] Official BRSR -> GRI symbolic bridge")
    print("  [2] Official GRI -> ESRS symbolic expansion")
    print("  [3] TRAIN and VALIDATION candidates built independently")
    print("  [4] CE teacher scores TRAIN + VALIDATION separately")
    print("  [5] Student optimized on TRAIN only")
    print("  [6] Validation KL used only for checkpoint selection")
    print("  [7] Validation gold labels used only for post-training retrieval evaluation")
    print("  [8] TEST is evaluated only after training")
    print("  [9] No DataParallel; one visible GPU")
    print("  [10] Student forward pass preserves autograd")
    print("=" * 80)


if __name__ == "__main__":
    main()