"""
Utilities for validating and cleaning dataset records.
"""

import logging
from typing import Any, Iterable, List

import pandas as pd

from src.sentinel.utils.taxonomy import Category

logger = logging.getLogger(__name__)


def normalize_text(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    return " ".join(text.split()).strip()


def _flatten_labels(labels: Any) -> Iterable[int]:
    if labels is None:
        return []
    if isinstance(labels, (int, float)):
        return [int(labels)]
    if isinstance(labels, (list, tuple, set)):
        out: List[int] = []
        for item in labels:
            if isinstance(item, (list, tuple, set)):
                out.extend([int(x) for x in item])
            elif item is not None:
                out.append(int(item))
        return out
    return []


def sanitize_labels(labels: Any) -> List[int]:
    valid_ids = {cat.value for cat in Category}
    cleaned = [label for label in _flatten_labels(labels) if label in valid_ids]
    cleaned = sorted(set(cleaned))

    if not cleaned:
        return [Category.SAFE.value]

    if Category.SAFE.value in cleaned and len(cleaned) > 1:
        cleaned = [label for label in cleaned if label != Category.SAFE.value]

    return cleaned


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    working = df.copy()
    before = len(working)

    if "text" not in working.columns or "labels" not in working.columns:
        logger.warning("Expected 'text' and 'labels' columns. Skipping cleaning.")
        return working

    working["text"] = working["text"].map(normalize_text)
    working["labels"] = working["labels"].map(sanitize_labels)

    working = working[working["text"].str.len() > 0].reset_index(drop=True)
    after = len(working)

    if after != before:
        logger.info("Removed %s empty text rows during cleaning.", before - after)

    return working


def log_label_distribution(df: pd.DataFrame, title: str) -> None:
    if df.empty or "labels" not in df.columns:
        return

    all_labels: List[int] = []
    for labels in df["labels"]:
        all_labels.extend(labels)

    counts = pd.Series(all_labels).value_counts().sort_index()
    logger.info("%s label distribution:\n%s", title, counts.to_string())
