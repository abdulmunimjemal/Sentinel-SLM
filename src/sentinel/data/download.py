"""
Module for downloading and caching public datasets.
Handles BeaverTails, JailbreakBench, Civil Comments, and Jigsaw.
"""

import logging
import os

from datasets import load_dataset  # type: ignore
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Use relative pathing from project root usually, but let's be robust
# Assuming run from root
PROJECT_ROOT = os.getcwd()
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


def ensure_dir(path: str) -> None:
    """Ensure that the directory exists."""
    os.makedirs(path, exist_ok=True)


def download_beavertails(split: str = "train") -> None:
    """
    Download BeaverTails dataset (30k subset).

    Args:
        split: Dataset split to download.
    """
    ensure_dir(RAW_DATA_DIR)
    path = os.path.join(RAW_DATA_DIR, "beavertails_30k.parquet")
    if os.path.exists(path):
        logger.info(f"BeaverTails already exists at {path}")
        return

    logger.info("Downloading BeaverTails (30k)...")
    try:
        # User confirmed ID: PKU-Alignment/BeaverTails
        ds = load_dataset("PKU-Alignment/BeaverTails", split="30k_train", token=HF_TOKEN)
        ds.to_parquet(path)
        logger.info(f"Saved to {path}")
    except Exception as e:
        logger.warning(f"Skipping BeaverTails (Auth/Access required): {e}")


def download_jailbreak() -> None:
    """Download JailbreakBench dataset."""
    ensure_dir(RAW_DATA_DIR)
    path = os.path.join(RAW_DATA_DIR, "jailbreak_bench.parquet")
    if os.path.exists(path):
        logger.info(f"JailbreakBench already exists at {path}")
        return

    logger.info("Downloading JailbreakBench...")
    try:
        from datasets import concatenate_datasets

        # Splits are 'harmful' and 'benign'
        ds_harmful = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
        ds_benign = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="benign")

        # Add type column to distinguish
        ds_harmful = ds_harmful.add_column("type", ["jailbreak"] * len(ds_harmful))
        ds_benign = ds_benign.add_column("type", ["benign"] * len(ds_benign))

        # Concatenate
        ds_combined = concatenate_datasets([ds_harmful, ds_benign])
        ds_combined.to_parquet(path)
        logger.info(f"Saved to {path} (Combined {len(ds_combined)} samples)")
    except Exception as e:
        logger.error(f"Failed to download JailbreakBench: {e}")


def download_civil_comments(split_percent: str = "10%") -> None:
    """
    Download a subset of Civil Comments for harassment data.

    Args:
        split_percent: Percentage of training data to fetch (e.g. '10%').
    """
    ensure_dir(RAW_DATA_DIR)
    path = os.path.join(RAW_DATA_DIR, "civil_comments_sample.parquet")
    if os.path.exists(path):
        logger.info(f"Civil Comments already exists at {path}")
        return

    logger.info(f"Downloading Civil Comments (train[:{split_percent}])...")
    try:
        ds = load_dataset("google/civil_comments", split=f"train[:{split_percent}]")
        ds.to_parquet(path)
        logger.info(f"Saved to {path}")
    except Exception as e:
        logger.error(f"Failed to download Civil Comments: {e}")


def download_jigsaw_clean() -> None:
    """Download Jigsaw Toxicity dataset (clean version if available)."""
    ensure_dir(RAW_DATA_DIR)
    path = os.path.join(RAW_DATA_DIR, "jigsaw_clean.parquet")
    if os.path.exists(path):
        logger.info(f"Jigsaw Clean already exists at {path}")
        return

    logger.info("Downloading Jigsaw (Clean Version via fallback)...")
    try:
        # Note: 'google/jigsaw_toxicity_pred' is often gated/requires manual script.
        # Check specific user uploads or fallbacks if main fails.
        ds = load_dataset("google/jigsaw_toxicity_pred", split="train[:5%]")
        ds.to_parquet(path)
        logger.info(f"Saved to {path}")
    except Exception as e:
        logger.warning(f"Failed to download Jigsaw: {e}")


def download_koala() -> None:
    """Download KoalaAI Multilingual Moderation dataset."""
    ensure_dir(RAW_DATA_DIR)
    path = os.path.join(RAW_DATA_DIR, "koala_multilingual.parquet")
    if os.path.exists(path):
        logger.info(f"KoalaAI already exists at {path}")
        return

    logger.info("Downloading KoalaAI Multilingual...")
    try:
        # Check if train split is available, usually 'train'
        ds = load_dataset("KoalaAI/Text-Moderation-Multilingual", split="train")
        ds.to_parquet(path)
        logger.info(f"Saved to {path}")
    except Exception as e:
        logger.error(f"Failed to download KoalaAI: {e}")


def download_multijail() -> None:
    """Download MultiJail dataset (Multilingual Jailbreaks)."""
    ensure_dir(RAW_DATA_DIR)
    path = os.path.join(RAW_DATA_DIR, "multijail.parquet")
    if os.path.exists(path):
        logger.info(f"MultiJail already exists at {path}")
        return

    logger.info("Downloading MultiJail...")
    try:
        # DAMO-NLP-SG/MultiJail usually loads as CSV/Text.
        # It's small so we can load 'train'.
        ds = load_dataset("DAMO-NLP-SG/MultiJail", split="train")
        ds.to_parquet(path)
        logger.info(f"Saved to {path}")
    except Exception as e:
        logger.error(f"Failed to download MultiJail: {e}")


def download_all() -> None:
    """Download all required datasets."""
    download_beavertails()
    download_jailbreak()
    download_civil_comments()
    download_jigsaw_clean()
    download_koala()
    download_multijail()


if __name__ == "__main__":
    download_all()
