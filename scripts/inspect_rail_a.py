import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.getcwd())
from src.sentinel.utils.taxonomy import Category

def inspect_rail_a(data_path):
    print("Loading data...")
    df = pd.read_parquet(data_path)
    
    # Filter for Category 8 (Prompt Attack)
    # Check if 8 is in the list of labels
    rail_a_attacks = df[df['labels'].apply(lambda x: Category.PROMPT_ATTACK.value in x)]
    
    print(f"\n--- Rail A: Input Guard (Attack Class) ---")
    print(f"Total Attack Samples: {len(rail_a_attacks)}")
    
    # Source Breakdown
    print("\n[Sources]")
    print(rail_a_attacks['source'].value_counts())
    
    # Language Breakdown (if available)
    if 'lang' in df.columns:
        print("\n[Languages - Top 15]")
        print(rail_a_attacks['lang'].value_counts().head(15))
    else:
        print("\n[Languages] 'lang' column not found in dataset.")

    # Length Analysis
    print("\n[Text Length Stats (Chars)]")
    print(rail_a_attacks['text'].str.len().describe())
    
    # Sample Attacks (to see quality)
    print("\n[Sample Attacks]")
    for i, row in rail_a_attacks.sample(5).iterrows():
        print(f" - [{row['source']} / {row.get('lang', '?')}] {row['text'][:100]}...")

    # Safe Data Analysis (for Balance)
    print(f"\n--- Rail A: Safe Class (Background) ---")
    # Category 0 only
    # Category 0 only (Safe) - Ensure x is a list/array
    safe_docs = df[df['labels'].apply(lambda x: len(x) == 1 and x[0] == 0)]
    print(f"Total Safe Candidates: {len(safe_docs)}")
    print("We need to sample ~4k from these to balance the 3.8k attacks.")

if __name__ == "__main__":
    # Prefer enriched
    import os
    path = "data/processed/final_augmented_dataset_enriched.parquet"
    if not os.path.exists(path):
        path = "data/processed/final_augmented_dataset.parquet"
        
    inspect_rail_a(path)
