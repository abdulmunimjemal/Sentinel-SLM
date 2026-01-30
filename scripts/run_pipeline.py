#!/usr/bin/env python3
"""
Sentinel-SLM Data Preparation Pipeline.
Orchestrates downloading, processing, and synthetic generation.

Usage:
    python scripts/run_pipeline.py --all
    python scripts/run_pipeline.py --download --process
    python scripts/run_pipeline.py --synthetic --count 50
"""

import argparse
import logging
import os
import sys

# Add repo root to path if running closely
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.sentinel.data.download import download_all
from src.sentinel.data.processing import map_all_raw_data, merge_synthetic
from src.sentinel.data.synthetic import generate_synthetic_data
from src.sentinel.utils.taxonomy import Category

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("sentinel_pipeline")


def main():
    parser = argparse.ArgumentParser(description="Sentinel-SLM Data Pipeline")
    parser.add_argument("--all", action="store_true", help="Run entire pipeline")
    parser.add_argument("--download", action="store_true", help="Run downloader")
    parser.add_argument("--process", action="store_true", help="Run mapping and processing")
    parser.add_argument("--synthetic", action="store_true", help="Run synthetic generation")
    parser.add_argument(
        "--count", type=int, default=10, help="Number of synthetic samples per category"
    )

    args = parser.parse_args()

    if not any([args.all, args.download, args.process, args.synthetic]):
        parser.print_help()
        return

    if args.all or args.download:
        logger.info("--- Step 1: Downloading Data ---")
        download_all()

    if args.synthetic:  # Run before process to include it, or separate?
        # Usually generate then merge.
        logger.info(f"--- Step: Synthetic Generation (Count={args.count}) ---")
        # Target specific gaps
        targets = [Category.CHILD_SAFETY, Category.PROMPT_ATTACK]
        generate_synthetic_data(targets, count_per_cat=args.count)

    if args.all or args.process:
        logger.info("--- Step 2: Processing & Merging ---")
        df = map_all_raw_data()
        merge_synthetic(df)


if __name__ == "__main__":
    main()
