"""
Module for processing and standardizing datasets into the 8-category taxonomy.
Includes logic for mapping labels from various raw sources.
"""

import logging
import os
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm

from src.sentinel.utils.taxonomy import CATEGORY_NAMES, MAPPING_RULES, Category

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.getcwd()
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
SYNTHETIC_FILE = os.path.join(PROJECT_ROOT, "data", "synthetic", "synthetic_data.jsonl")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def process_beavertails() -> List[Dict[str, Any]]:
    """Process BeaverTails dataset."""
    path = os.path.join(RAW_DIR, "beavertails_30k.parquet")
    if not os.path.exists(path):
        logger.warning(f"Skipping BeaverTails (not found): {path}")
        return []

    logger.info("Processing BeaverTails...")
    df = pd.read_parquet(path)
    processed_data = []

    rules = MAPPING_RULES["beaver_tails"]

    for _, row in tqdm(df.iterrows(), total=len(df), desc="BeaverTails"):
        text = row.get("prompt", "") + " " + row.get("response", "")
        labels = set()
        is_safe = row.get("is_safe", True)

        # Check categories (usually a dict or boolean columns)
        raw_cats = row.get("category", {})

        for raw_label, mapped_cat in rules.items():
            # Handle if dictionary or direct column access
            if isinstance(raw_cats, dict) and raw_cats.get(raw_label, False):
                labels.add(mapped_cat)
            elif raw_label in row and row[raw_label]:
                labels.add(mapped_cat)

        if is_safe and not labels:
            labels.add(Category.SAFE)

        processed_data.append({"text": text, "labels": list(labels), "source": "beaver_tails"})
    return processed_data


def process_jailbreak() -> List[Dict[str, Any]]:
    """Process JailbreakBench dataset."""
    path = os.path.join(RAW_DIR, "jailbreak_bench.parquet")
    if not os.path.exists(path):
        logger.warning(f"Skipping JailbreakBench (not found): {path}")
        return []

    logger.info("Processing JailbreakBench...")
    df = pd.read_parquet(path)
    processed_data = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="JailbreakBench"):
        text = row.get("prompt", "")
        labels = set()

        # 'type' column: 'jailbreak' or 'benign'
        if row.get("type") == "jailbreak":
            labels.add(Category.PROMPT_ATTACK)
        else:
            labels.add(Category.SAFE)

        processed_data.append({"text": text, "labels": list(labels), "source": "jailbreak_bench"})
    return processed_data


def process_civil_comments() -> List[Dict[str, Any]]:
    """Process Civil Comments dataset."""
    path = os.path.join(RAW_DIR, "civil_comments_sample.parquet")
    if not os.path.exists(path):
        logger.warning(f"Skipping Civil Comments (not found): {path}")
        return []

    logger.info("Processing Civil Comments...")
    df = pd.read_parquet(path)
    processed_data = []

    rules = MAPPING_RULES["civil_comments"]

    for _, row in tqdm(df.iterrows(), total=len(df), desc="CivilComments"):
        text = row.get("text", "")
        labels = set()
        is_safe = True

        for col, cat in rules.items():
            if col in row and row[col] >= 0.5:
                labels.add(cat)
                is_safe = False

        if is_safe:
            labels.add(Category.SAFE)

        processed_data.append({"text": text, "labels": list(labels), "source": "civil_comments"})
    return processed_data


def process_jigsaw_clean() -> List[Dict[str, Any]]:
    """Process Jigsaw Clean dataset."""
    path = os.path.join(RAW_DIR, "jigsaw_clean.parquet")
    if not os.path.exists(path):
        logger.warning(f"Skipping Jigsaw Clean (not found): {path}")
        return []

    logger.info("Processing Jigsaw Clean...")
    df = pd.read_parquet(path)
    processed_data = []

    rules = MAPPING_RULES.get("jigsaw_clean", {})

    for _, row in tqdm(df.iterrows(), total=len(df), desc="JigsawClean"):
        text = row.get("comment_text", row.get("text", ""))
        labels = set()
        is_safe = True

        for col, cat in rules.items():
            if col in row and row[col] >= 0.5:
                labels.add(cat)
                is_safe = False

        if is_safe:
            labels.add(Category.SAFE)

        processed_data.append({"text": text, "labels": list(labels), "source": "jigsaw_clean"})
    return processed_data


def process_koala() -> List[Dict[str, Any]]:
    """Process KoalaAI Multilingual dataset."""
    path = os.path.join(RAW_DIR, "koala_multilingual.parquet")
    if not os.path.exists(path):
        logger.warning(f"Skipping KoalaAI (not found): {path}")
        return []

    logger.info("Processing KoalaAI...")
    df = pd.read_parquet(path)
    processed_data = []

    rules = MAPPING_RULES["koala"]

    # Koala columns are 'prompt' (text) and labels 'S', 'H', etc (0/1)
    for _, row in tqdm(df.iterrows(), total=len(df), desc="KoalaAI"):
        text = row.get("prompt", "")
        # Fallback if prompt is missing but text exists
        if not isinstance(text, str):
            text = row.get("text", "")
        if not text:
            continue

        labels = set()
        is_safe = True

        for col, cat in rules.items():
            if col in row and row[col] == 1:
                labels.add(cat)
                is_safe = False

        if is_safe:
            labels.add(Category.SAFE)

        processed_data.append(
            {"text": text, "labels": list(labels), "source": "koala_multilingual"}
        )
    return processed_data


def process_multijail() -> List[Dict[str, Any]]:
    """Process MultiJail dataset (Multilingual)."""
    path = os.path.join(RAW_DIR, "multijail.parquet")
    if not os.path.exists(path):
        logger.warning(f"Skipping MultiJail (not found): {path}")
        return []

    logger.info("Processing MultiJail...")
    df = pd.read_parquet(path)
    processed_data = []

    # Iterate over likely language columns
    # MultiJail cols: 'en', 'zh', 'it', 'vi', 'ar', 'ko', 'th', 'bn', 'sw', 'jv'
    # We will try to grab all string columns that look like prompts

    for _, row in tqdm(df.iterrows(), total=len(df), desc="MultiJail"):
        for col in df.columns:
            text = row[col]
            if isinstance(text, str) and len(text) > 5:
                processed_data.append(
                    {"text": text, "labels": [Category.PROMPT_ATTACK], "source": "damo_multijail"}
                )

    return processed_data


def map_all_raw_data() -> pd.DataFrame:
    """Processing pipeline for all raw data."""
    all_data = []
    all_data.extend(process_beavertails())
    all_data.extend(process_jailbreak())
    all_data.extend(process_civil_comments())
    all_data.extend(process_jigsaw_clean())
    all_data.extend(process_koala())
    all_data.extend(process_multijail())

    if not all_data:
        logger.warning("No data processed.")
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    ensure_dir(PROCESSED_DIR)

    out_path = os.path.join(PROCESSED_DIR, "unified_dataset.parquet")
    df.to_parquet(out_path)
    logger.info(f"Saved Unified Data: {len(df)} samples into {out_path}")
    return df


def merge_synthetic(df_main: pd.DataFrame) -> pd.DataFrame:
    """Merge synthetic data JSONL into the main DataFrame."""
    import json

    if not os.path.exists(SYNTHETIC_FILE):
        return df_main

    logger.info("Merging synthetic data...")
    synth_data = []
    # Helper to map string label back to ID
    name_to_id = {v: k for k, v in CATEGORY_NAMES.items()}

    with open(SYNTHETIC_FILE, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                label_str = item.get("label", "").strip()
                cat_id = name_to_id.get(label_str, Category.SAFE)

                # Robust partial matching if exact fail
                if label_str not in name_to_id:
                    if "Sexual" in label_str:
                        cat_id = Category.SEXUAL
                    elif "Child" in label_str:
                        cat_id = Category.CHILD_SAFETY
                    elif "Harassment" in label_str:
                        cat_id = Category.HARASSMENT
                    elif "Hate" in label_str:
                        cat_id = Category.HATE_EXTREMISM
                    elif "Prompt" in label_str:
                        cat_id = Category.PROMPT_ATTACK

                synth_data.append(
                    {
                        "text": item.get("text", ""),
                        "labels": [cat_id],
                        "source": "synthetic_openrouter",
                    }
                )
            except (json.JSONDecodeError, KeyError, ValueError):
                pass
            except Exception:
                pass

    if synth_data:
        df_synth = pd.DataFrame(synth_data)
        df_final = pd.concat([df_main, df_synth], ignore_index=True)
        logger.info(f"Added {len(df_synth)} synthetic samples.")
    else:
        logger.info("No synthetic data found (or empty). Using main dataset only.")
        df_final = df_main

    final_path = os.path.join(PROCESSED_DIR, "final_augmented_dataset.parquet")
    df_final.to_parquet(final_path)
    logger.info(f"Saved Final Augmented Dataset: {len(df_final)} samples to {final_path}")
    return df_final


if __name__ == "__main__":
    df = map_all_raw_data()
    merge_synthetic(df)
