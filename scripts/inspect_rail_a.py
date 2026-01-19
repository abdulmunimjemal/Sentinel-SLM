#!/usr/bin/env python3
"""
Inspect Rail A Dataset Statistics

This utility script analyzes the Rail A dataset to provide insights on:
- Attack vs Safe sample distribution
- Source breakdown
- Language distribution (if available)
- Text length statistics
- Sample examples

Usage:
    python scripts/inspect_rail_a.py
    python scripts/inspect_rail_a.py --path data/processed/rail_a_external.parquet

Author: Sentinel-SLM Team
"""

import pandas as pd
import numpy as np
import sys
import os
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.append(os.getcwd())
from src.sentinel.utils.taxonomy import Category


def inspect_rail_a(data_path: str) -> None:
    """
    Analyze and display statistics for a Rail A dataset.
    
    Args:
        data_path: Path to the dataset parquet file
    """
    logger.info(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    
    # Check if this is a processed Rail A dataset (has 'target') or main dataset (has 'labels')
    if 'target' in df.columns:
        # Processed Rail A dataset
        logger.info("\n" + "=" * 60)
        logger.info("RAIL A DATASET STATISTICS")
        logger.info("=" * 60)
        
        logger.info(f"\nTotal samples: {len(df)}")
        logger.info(f"\nBy target (0=Safe, 1=Attack):")
        logger.info(f"{df['target'].value_counts().to_string()}")
        
        if 'source' in df.columns:
            logger.info(f"\nBy source:")
            logger.info(f"{df['source'].value_counts().to_string()}")
        
        if 'lang' in df.columns:
            logger.info(f"\nTop languages:")
            logger.info(f"{df['lang'].value_counts().head(10).to_string()}")
        
        logger.info(f"\nText length statistics (chars):")
        logger.info(f"{df['text'].str.len().describe().to_string()}")
        
        logger.info("\n[Sample Attacks]")
        for _, row in df[df['target'] == 1].sample(min(5, len(df[df['target']==1]))).iterrows():
            logger.info(f" - {row['text'][:100]}...")
            
        logger.info("\n[Sample Safe]")
        for _, row in df[df['target'] == 0].sample(min(5, len(df[df['target']==0]))).iterrows():
            logger.info(f" - {row['text'][:100]}...")
    else:
        # Main dataset with 'labels' column
        rail_a_attacks = df[df['labels'].apply(lambda x: Category.PROMPT_ATTACK.value in x)]
        
        logger.info("\n" + "=" * 60)
        logger.info("RAIL A ATTACK ANALYSIS (from main dataset)")
        logger.info("=" * 60)
        logger.info(f"Total Attack Samples: {len(rail_a_attacks)}")
        
        logger.info("\n[Sources]")
        logger.info(f"{rail_a_attacks['source'].value_counts().to_string()}")
        
        if 'lang' in df.columns:
            logger.info("\n[Languages - Top 15]")
            logger.info(f"{rail_a_attacks['lang'].value_counts().head(15).to_string()}")

        logger.info("\n[Text Length Stats]")
        logger.info(f"{rail_a_attacks['text'].str.len().describe().to_string()}")
        
        logger.info("\n[Sample Attacks]")
        for _, row in rail_a_attacks.sample(min(5, len(rail_a_attacks))).iterrows():
            logger.info(f" - [{row['source']}] {row['text'][:100]}...")

        safe_docs = df[df['labels'].apply(lambda x: len(x) == 1 and x[0] == 0)]
        logger.info(f"\nSafe candidates available: {len(safe_docs)}")

def main():
    parser = argparse.ArgumentParser(description="Inspect Rail A dataset statistics")
    parser.add_argument(
        '--path',
        type=str,
        default=None,
        help='Path to dataset (auto-detects if not specified)'
    )
    args = parser.parse_args()
    
    # Auto-detect dataset path
    if args.path:
        path = args.path
    else:
        # Try Rail A external first, then main dataset
        candidates = [
            "data/processed/rail_a_external.parquet",
            "data/processed/rail_a_clean.parquet",
            "data/processed/final_augmented_dataset_enriched.parquet",
            "data/processed/final_augmented_dataset.parquet"
        ]
        path = next((p for p in candidates if os.path.exists(p)), None)
        if not path:
            logger.error("No dataset found. Please specify --path")
            return
            
    inspect_rail_a(path)


if __name__ == "__main__":
    main()
