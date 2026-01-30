"""
Rail B Data Preparation Script

Converts the multi-label dataset into training-ready format:
- Binary label vectors (8 categories, excluding Safe=0)
- Train/Val/Test splits (80/10/10)
- Optional: Create balanced 10k prototype for rapid iteration

Usage:
    python scripts/prepare_rail_b_data.py --full  # Full dataset
    python scripts/prepare_rail_b_data.py --prototype  # 10k sample
"""

import pandas as pd
import numpy as np
import os
import argparse
from sklearn.model_selection import train_test_split

# Category mapping (excluding Safe=0 which is implicit)
# PromptAttack is EXCLUDED - handled by Rail A
CATEGORIES = {
    1: "Hate & Extremism",
    2: "Harassment & Bullying", 
    3: "Sexual Content",
    4: "Child Safety",
    5: "Violence & Gore",
    6: "Illegal Activities",
    7: "Privacy Violations",
    # 8: "Prompt Attack" - EXCLUDED (handled by Rail A)
}

NUM_LABELS = 7  # Categories 1-7 only (no PromptAttack)

def labels_to_binary_vector(labels_list, exclude_safe=True):
    """
    Convert list of category IDs to binary vector.
    
    Args:
        labels_list: List of category IDs (e.g., [1, 5])
        exclude_safe: If True, ignore category 0 (Safe)
    
    Returns:
        Binary vector of length 7 (one-hot encoded for categories 1-7)
        PromptAttack (8) is always excluded - handled by Rail A
    """
    vector = np.zeros(NUM_LABELS, dtype=np.float32)
    
    for label in labels_list:
        if exclude_safe and label == 0:
            continue
        if label == 8:  # Skip PromptAttack - handled by Rail A
            continue
        if 1 <= label <= 7:
            vector[label - 1] = 1.0  # Shift index (1->0, 2->1, ..., 7->6)
    
    return vector

def prepare_rail_b(input_path, output_path, prototype=False, prototype_size=10000, seed=42):
    """
    Prepare Rail B dataset with multi-label encoding.
    """
    print("=" * 60)
    print("RAIL B DATA PREPARATION")
    print("=" * 60)
    
    # Load
    print(f"\nLoading from {input_path}...")
    df = pd.read_parquet(input_path)
    print(f"Total samples: {len(df):,}")
    
    # Separate Safe and Violation samples
    pure_safe = df[df['labels'].apply(lambda x: len(x) == 1 and x[0] == 0)]
    has_violations = df[df['labels'].apply(lambda x: not (len(x) == 1 and x[0] == 0))]
    
    print(f"Original Pure Safe: {len(pure_safe):,}")
    print(f"Original Violations: {len(has_violations):,}")
    
    # --- BALANCING STRATEGY ---
    print("\n⚖️  Applying Rarest-Label Balancing...")
    
    # 1. Map labels to counts to find rarity
    all_labels = [l for sublist in has_violations['labels'] for l in sublist if l != 0]
    from collections import Counter
    label_counts = Counter(all_labels)
    print("  Original Label Counts:", dict(sorted(label_counts.items())))
    
    # 2. Assign each sample to its rarest label bucket
    def get_rarest_label(labels):
        valid_labels = [l for l in labels if l != 0 and l <= 7] # 1-7 (PromptAttack=8 excluded)
        if not valid_labels: return -1
        # Sort by count (ascending), return the one with lowest count
        return sorted(valid_labels, key=lambda l: label_counts[l])[0]

    has_violations['rarest_label'] = has_violations['labels'].apply(get_rarest_label)
    
    # 3. Determine target size (e.g., 15k per class, capped at reasonable limit)
    # We want to upsample rare classes (Privacy~1.4k) and downsample common (Harassment~400k)
    # Target: 15,000 per class (gives ~105k violations total)
    TARGET_PER_CLASS = 15000 
    
    balanced_dfs = []
    unique_rarest = sorted(has_violations['rarest_label'].unique())
    
    for label in unique_rarest:
        if label == -1: continue
        subset = has_violations[has_violations['rarest_label'] == label]
        count = len(subset)
        
        if count < TARGET_PER_CLASS:
            # Upsample (replace=True)
            resampled = subset.sample(TARGET_PER_CLASS, replace=True, random_state=seed)
        else:
            # Downsample (replace=False)
            resampled = subset.sample(TARGET_PER_CLASS, replace=False, random_state=seed)
            
        balanced_dfs.append(resampled)
        cat_name = CATEGORIES.get(label, f"Cat {label}")
        print(f"  - {cat_name:25s}: {count:>7,} -> {len(resampled):,}")
        
    balanced_violations = pd.concat(balanced_dfs, ignore_index=True).drop(columns=['rarest_label'])
    
    # 4. Match Safe samples to Violation count (50/50 split)
    n_safe = len(balanced_violations)
    print(f"\n  Violations (Balanced): {len(balanced_violations):,}")
    
    if len(pure_safe) >= n_safe:
        balanced_safe = pure_safe.sample(n_safe, random_state=seed)
    else:
        # Upsample safe if needed (unlikely given we have 1M+)
        balanced_safe = pure_safe.sample(n_safe, replace=True, random_state=seed)
        
    print(f"  Safe (Matched):       {len(balanced_safe):,}")
    
    # Combine
    df = pd.concat([balanced_violations, balanced_safe], ignore_index=True)
    print(f"  Total Balanced Data:  {len(df):,}")
    
    # --- END BALANCING ---
    
    # Convert labels to binary vectors
    print("\nEncoding labels to binary vectors...")
    df['label_vector'] = df['labels'].apply(labels_to_binary_vector)
    
    # Note: Safe samples will have all-zero vectors [0,0,0,0,0,0,0,0]
    # This is CORRECT - the model needs to learn what "safe" looks like
    
    # Truncate text (optimization)
    print("Truncating long texts...")
    df['text'] = df['text'].str.slice(0, 2000)
    
    # Label statistics (Recalculate)
    print("\n=== LABEL DISTRIBUTION (Balanced) ===")
    label_matrix = np.vstack(df['label_vector'].values)
    for i in range(NUM_LABELS):
        count = label_matrix[:, i].sum()
        pct = (count / len(df)) * 100
        cat_name = CATEGORIES.get(i + 1, f"Category {i+1}")
        print(f"{cat_name:30s}: {int(count):,} ({pct:.1f}%)")
    
    # Multi-label stats
    labels_per_sample = label_matrix.sum(axis=1)
    print(f"\nAvg labels per sample: {labels_per_sample.mean():.2f}")
    
    # Split: 90/5/5 for larger datasets, or 80/10/10
    print("\nSplitting into train/val/test...")
    train_df, temp_df = train_test_split(df, test_size=0.1, random_state=seed) # 90/10 split
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed)
    
    print(f"  Train: {len(train_df):,}")
    print(f"  Val:   {len(val_df):,}")
    print(f"  Test:  {len(test_df):,}")
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as separate files
    base_path = output_path.replace('.parquet', '')
    train_df.to_parquet(f"{base_path}_train.parquet")
    val_df.to_parquet(f"{base_path}_val.parquet")
    test_df.to_parquet(f"{base_path}_test.parquet")
    
    print(f"\n✅ Saved to:")
    print(f"   {base_path}_train.parquet")
    print(f"   {base_path}_val.parquet")
    print(f"   {base_path}_test.parquet")
    
    return train_df, val_df, test_df

def main():
    parser = argparse.ArgumentParser(description="Prepare Rail B dataset")
    parser.add_argument("--full", action="store_true", help="Prepare full dataset")
    parser.add_argument("--prototype", action="store_true", help="Create 10k prototype")
    parser.add_argument("--medium", action="store_true", help="Create 100k medium dataset")
    parser.add_argument("--input", type=str, default="data/processed/final_augmented_dataset_enriched.parquet")
    parser.add_argument("--output", type=str, default="data/processed/rail_b_policy.parquet")
    
    args = parser.parse_args()
    
    if not any([args.full, args.prototype, args.medium]):
        print("Please specify --full, --medium, or --prototype")
        return
    
    # Determine size and output path
    size = 10000
    is_prototype = True
    
    if args.full:
        is_prototype = False
        args.output = "data/processed/rail_b_full.parquet"
    elif args.medium:
        size = 100000
        args.output = "data/processed/rail_b_medium.parquet"
    else: # prototype
        size = 10000
        args.output = "data/processed/rail_b_prototype.parquet"
    
    prepare_rail_b(
        input_path=args.input,
        output_path=args.output,
        prototype=is_prototype,
        prototype_size=size
    )

if __name__ == "__main__":
    main()
