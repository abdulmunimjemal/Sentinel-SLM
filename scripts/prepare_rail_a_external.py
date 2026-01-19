#!/usr/bin/env python3
"""
Prepare Rail A Dataset from External Jailbreak Sources

This script downloads and processes prompt injection/jailbreak datasets from
HuggingFace to create the training data for Rail A (Input Guard).

Data Sources (5 datasets):
    1. deepset/prompt-injections - Labeled prompt injection dataset
    2. JailbreakBench/JBB-Behaviors - Jailbreak behavior goals
    3. TrustAIRLab/in-the-wild-jailbreak-prompts - Real-world jailbreak prompts
    4. Simsonsun/JailbreakPrompts - Curated jailbreak prompts
    5. yanismiraoui/prompt_injections - Additional injection samples

Safe Data Sources (for balancing):
    - databricks/databricks-dolly-15k - Safe instructions
    - tatsu-lab/alpaca - Safe instruction-following examples

Label Convention:
    0 = Safe (benign prompts)
    1 = Attack (prompt injection/jailbreak)

Output:
    data/processed/rail_a_external.parquet

Usage:
    python scripts/prepare_rail_a_external.py
    python scripts/prepare_rail_a_external.py --output data/processed/custom_rail_a.parquet

Author: Sentinel-SLM Team
"""

import os
import sys
import argparse
import logging
from typing import List, Tuple

import pandas as pd
from datasets import load_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_attack_datasets() -> List[pd.DataFrame]:
    """
    Download and normalize attack datasets from HuggingFace.
    
    Returns:
        List of DataFrames, each with columns: ['text', 'target', 'source']
    """
    all_data = []
    
    logger.info("=" * 60)
    logger.info("DOWNLOADING JAILBREAK/PROMPT INJECTION DATASETS")
    logger.info("=" * 60)
    
    # 1. deepset/prompt-injections
    logger.info("\n[1/5] deepset/prompt-injections")
    try:
        for split in ['train', 'test']:
            ds = load_dataset("deepset/prompt-injections", split=split)
            df = ds.to_pandas()
            # label: 1=injection, 0=safe
            df['target'] = df['label'].astype(int)
            df['source'] = 'deepset'
            all_data.append(df[['text', 'target', 'source']])
            logger.info(f"   Added {len(df)} samples from {split} split")
    except Exception as e:
        logger.warning(f"   Failed: {e}")
    
    # 2. JailbreakBench/JBB-Behaviors
    logger.info("\n[2/5] JailbreakBench/JBB-Behaviors")
    try:
        ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="train")
        df = ds.to_pandas()
        df['text'] = df['Goal']  # Use 'Goal' as attack prompt
        df['target'] = 1  # All are attacks
        df['source'] = 'jailbreakbench'
        all_data.append(df[['text', 'target', 'source']])
        logger.info(f"   Added {len(df)} attack samples")
    except Exception as e:
        logger.warning(f"   Failed: {e}")
    
    # 3. TrustAIRLab/in-the-wild-jailbreak-prompts
    logger.info("\n[3/5] TrustAIRLab/in-the-wild-jailbreak-prompts")
    for config in ['jailbreak_2023_05_07', 'jailbreak_2023_12_25']:
        try:
            ds = load_dataset("TrustAIRLab/in-the-wild-jailbreak-prompts", config, split="train")
            df = ds.to_pandas()
            text_col = 'prompt' if 'prompt' in df.columns else df.columns[0]
            df['text'] = df[text_col]
            df['target'] = 1
            df['source'] = f'trustailab_{config}'
            all_data.append(df[['text', 'target', 'source']])
            logger.info(f"   Added {len(df)} from {config}")
        except Exception as e:
            logger.warning(f"   {config}: {e}")
    
    # 4. Simsonsun/JailbreakPrompts
    logger.info("\n[4/5] Simsonsun/JailbreakPrompts")
    for split in ['Dataset_1', 'Dataset_2']:
        try:
            ds = load_dataset("Simsonsun/JailbreakPrompts", split=split)
            df = ds.to_pandas()
            text_cols = [c for c in df.columns if any(kw in c.lower() for kw in ['prompt', 'text', 'jailbreak'])]
            df['text'] = df[text_cols[0]] if text_cols else df.iloc[:, 0]
            df['target'] = 1
            df['source'] = f'simsonsun_{split}'
            all_data.append(df[['text', 'target', 'source']])
            logger.info(f"   Added {len(df)} from {split}")
        except Exception as e:
            logger.warning(f"   {split}: {e}")
    
    # 5. yanismiraoui/prompt_injections
    logger.info("\n[5/5] yanismiraoui/prompt_injections")
    try:
        ds = load_dataset("yanismiraoui/prompt_injections", split="train")
        df = ds.to_pandas()
        if 'label' in df.columns:
            df['target'] = df['label'].astype(int)
        else:
            df['target'] = 1
        text_col = 'text' if 'text' in df.columns else 'prompt'
        df['text'] = df[text_col]
        df['source'] = 'yanismiraoui'
        all_data.append(df[['text', 'target', 'source']])
        logger.info(f"   Added {len(df)} samples")
    except Exception as e:
        logger.warning(f"   Failed: {e}")
    
    return all_data


def download_safe_datasets(target_count: int) -> pd.DataFrame:
    """
    Download safe/benign instruction datasets for balancing.
    
    Args:
        target_count: Number of safe samples needed
        
    Returns:
        DataFrame with columns: ['text', 'target', 'source']
    """
    logger.info("\n" + "=" * 60)
    logger.info("DOWNLOADING SAFE INSTRUCTION DATASETS")
    logger.info("=" * 60)
    
    safe_samples = []
    
    # 1. Dolly-15k
    logger.info("\n[1/2] databricks/databricks-dolly-15k")
    try:
        ds = load_dataset("databricks/databricks-dolly-15k", split="train")
        df = ds.to_pandas()
        samples = df['instruction'].dropna().tolist()
        safe_samples.extend(samples[:min(len(samples), target_count // 2)])
        logger.info(f"   Added {min(len(samples), target_count // 2)} samples")
    except Exception as e:
        logger.warning(f"   Failed: {e}")
    
    # 2. Alpaca
    logger.info("\n[2/2] tatsu-lab/alpaca")
    try:
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        df = ds.to_pandas()
        samples = df['instruction'].dropna().tolist()
        remaining = target_count - len(safe_samples)
        safe_samples.extend(samples[:min(len(samples), remaining)])
        logger.info(f"   Added {min(len(samples), remaining)} samples")
    except Exception as e:
        logger.warning(f"   Failed: {e}")
    
    df_safe = pd.DataFrame({
        'text': safe_samples[:target_count],
        'target': 0,
        'source': 'dolly_alpaca'
    })
    
    return df_safe


def clean_and_merge(dataframes: List[pd.DataFrame], seed: int = 42) -> pd.DataFrame:
    """
    Clean, deduplicate, and merge all dataframes.
    
    Args:
        dataframes: List of DataFrames to merge
        seed: Random seed for shuffling
        
    Returns:
        Cleaned and merged DataFrame
    """
    logger.info("\n" + "=" * 60)
    logger.info("CLEANING AND MERGING DATA")
    logger.info("=" * 60)
    
    df = pd.concat(dataframes, ignore_index=True)
    
    # Clean
    df = df.dropna(subset=['text', 'target'])
    df['text'] = df['text'].astype(str).str.strip()
    df = df[df['text'].str.len() > 10]  # Remove very short texts
    df = df.drop_duplicates(subset=['text'])
    df['target'] = df['target'].astype(int)
    
    # Shuffle
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Prepare Rail A dataset from external sources")
    parser.add_argument(
        '--output', 
        type=str, 
        default='data/processed/rail_a_external.parquet',
        help='Output path for the dataset'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Download attack datasets
    attack_dfs = download_attack_datasets()
    
    # Merge attack data first to get count
    df_attacks = clean_and_merge(attack_dfs, args.seed)
    n_attacks = (df_attacks['target'] == 1).sum()
    n_safe_existing = (df_attacks['target'] == 0).sum()
    
    logger.info(f"\nAttacks collected: {n_attacks}")
    logger.info(f"Safe from attack sources: {n_safe_existing}")
    
    # Calculate how many safe samples we need
    needed_safe = n_attacks - n_safe_existing
    if needed_safe > 0:
        logger.info(f"Need {needed_safe} more safe samples for balance")
        df_safe = download_safe_datasets(needed_safe)
        df_final = pd.concat([df_attacks, df_safe], ignore_index=True)
    else:
        df_final = df_attacks
    
    # Final clean and shuffle
    df_final = df_final.drop_duplicates(subset=['text'])
    df_final = df_final.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    
    # Report
    logger.info("\n" + "=" * 60)
    logger.info("FINAL DATASET STATISTICS")
    logger.info("=" * 60)
    logger.info(f"\nTotal samples: {len(df_final)}")
    logger.info(f"\nBy target (0=Safe, 1=Attack):")
    logger.info(f"{df_final['target'].value_counts().to_string()}")
    logger.info(f"\nBy source:")
    logger.info(f"{df_final['source'].value_counts().to_string()}")
    
    # Save
    df_final.to_parquet(args.output)
    logger.info(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
