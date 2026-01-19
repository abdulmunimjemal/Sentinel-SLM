#!/usr/bin/env python3
"""
Prepare Rail A Dataset from Main Sentinel Dataset

This script extracts prompt injection samples (Category 8) from the main
unified Sentinel dataset and balances them with safe samples (Category 0).

Use this when you want to leverage the main dataset (which includes 
BeaverTails, Jigsaw, etc.) rather than downloading external sources.

Input:
    data/processed/final_augmented_dataset_enriched.parquet
    (or data/processed/final_augmented_dataset.parquet)

Output:
    data/processed/rail_a_jailbreak.parquet

Label Convention:
    0 = Safe (benign prompts)
    1 = Attack (prompt injection/jailbreak)

Usage:
    python scripts/prepare_rail_a_data.py

Author: Sentinel-SLM Team
"""

import pandas as pd
import numpy as np
import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(os.getcwd())
from src.sentinel.utils.taxonomy import Category


def prepare_rail_a(input_path: str, output_path: str, seed: int = 42) -> None:
    """
    Prepare Rail A dataset by extracting attacks and balancing with safe samples.
    
    Args:
        input_path: Path to the source dataset (enriched or base)
        output_path: Path to save the output dataset
        seed: Random seed for reproducibility
    """
    logger.info(f"Loading from {input_path}...")
    try:
        df = pd.read_parquet(input_path)
    except FileNotFoundError:
        logger.error(f"File {input_path} not found.")
        return

    # 1. Extract Attacks (Category 8 - PROMPT_ATTACK)
    attack_mask = df['labels'].apply(lambda x: Category.PROMPT_ATTACK.value in x)
    attacks_df = df[attack_mask].copy()
    num_attacks = len(attacks_df)
    logger.info(f"Found {num_attacks} Prompt Attacks (Category 8).")
    
    # 2. Extract Safe (Category 0 only, no multi-labels)
    safe_mask = df['labels'].apply(lambda x: len(x) == 1 and x[0] == Category.SAFE.value)
    candidates_safe_df = df[safe_mask]
    logger.info(f"Found {len(candidates_safe_df)} Safe candidates.")

    # 3. Sample Safe to balance (1.1x ratio - slightly more safe to prevent False Positives)
    target_safe = int(num_attacks * 1.1)
    
    if 'lang' in candidates_safe_df.columns:
        logger.info(f"Sampling {target_safe} Safe examples (Language-aware)...")
        attack_langs = set(attacks_df['lang'].unique())
        logger.info(f"Attack dataset has {len(attack_langs)} languages")
        safe_df = candidates_safe_df.sample(n=target_safe, random_state=seed).copy()
    else:
        logger.info(f"Sampling {target_safe} Safe examples (Random)...")
        safe_df = candidates_safe_df.sample(n=target_safe, random_state=seed).copy()

    # 4. Labeling for Binary Classification (Attack=1, Safe=0)
    attacks_df['target'] = 1
    safe_df['target'] = 0
    
    # 5. Merge and shuffle
    final_df = pd.concat([attacks_df, safe_df])
    final_df = final_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Report statistics
    logger.info("\n" + "=" * 50)
    logger.info("FINAL RAIL A DATASET")
    logger.info("=" * 50)
    logger.info(f"Total: {len(final_df)}")
    logger.info(f"\nBy target (0=Safe, 1=Attack):")
    logger.info(f"{final_df['target'].value_counts().to_string()}")
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_parquet(output_path)
    logger.info(f"\nSaved to {output_path}")

if __name__ == "__main__":
    IN_FILE = "data/processed/final_augmented_dataset_enriched.parquet"
    if not os.path.exists(IN_FILE):
        IN_FILE = "data/processed/final_augmented_dataset.parquet"
        
    OUT_FILE = "data/processed/rail_a_jailbreak.parquet"
    
    prepare_rail_a(IN_FILE, OUT_FILE)
