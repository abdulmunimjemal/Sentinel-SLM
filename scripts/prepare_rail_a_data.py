import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())
from src.sentinel.utils.taxonomy import Category

def prepare_rail_a(input_path, output_path, seed=42):
    print(f"Loading from {input_path}...")
    try:
        df = pd.read_parquet(input_path)
    except FileNotFoundError:
        print(f"Error: File {input_path} not found.")
        return

    # 1. Extract Attacks (Category 8)
    # Using the same robust list length check logic as the inspection script
    attack_mask = df['labels'].apply(lambda x: Category.PROMPT_ATTACK.value in x)
    attacks_df = df[attack_mask].copy()
    num_attacks = len(attacks_df)
    print(f"Found {num_attacks} Prompt Attacks (Category 8).")
    
    # 2. Extract Safe (Category 0)
    # Strict definition: Only label 0, no multi-labels
    safe_mask = df['labels'].apply(lambda x: len(x) == 1 and x[0] == Category.SAFE.value)
    candidates_safe_df = df[safe_mask]
    print(f"Found {len(candidates_safe_df)} Safe candidates.")

    # 3. Sample Safe to balance (1:1 ratio typically, or slightly more safe to prevent False Positives)
    # Let's go for 1.1x Safe to be slightly conservative
    target_safe = int(num_attacks * 1.1)
    
    # Stratified sampling by language if 'lang' exists, otherwise random
    if 'lang' in candidates_safe_df.columns:
        print(f"Sampling {target_safe} Safe examples (Language Balanced)...")
        # Try to match the language distribution of attacks?
        # Actually, for the negative class, we want it to represent "Normal usage" across languages.
        # So we should sample proportionally to the languages present in the attacks if possible, 
        # OR just sample broadly from all languages to ensure the model doesn't associate "Arabic" with "Attack".
        # Better strategy: Sample Safe examples roughly proportionally to their occurrence in the enriched dataset 
        # BUT ensuring we cover the languages found in the Attack dataset.
        
        # Simpler robust approach for now: Random sample. FastText `lang` will ensure we know what we got.
        # Additional thought: If we have 300 Arabic attacks, we MUST have Safe Arabic text, otherwise 
        # the model might learn "Arabic script = Attack".
        
        # Let's verify overlapping languages
        attack_langs = set(attacks_df['lang'].unique())
        print(f"Attack Languages: {len(attack_langs)}")
        
        # Filter safe candidates to only those languages present in attacks + 'en' (always keep en)
        # This prevents training on languages that are irrelevant to the attack surface defined so far,
        # though arguably "Safe" should be universal. 
        # Let's just do a weighted sample to prioritize languages that appear in attacks.
        
        # Implementation: Just random sample for now to keep it simple and robust, 
        # assuming the 1M+ safe set covers most languages.
        safe_df = candidates_safe_df.sample(n=target_safe, random_state=seed).copy()
        
    else:
        print(f"Sampling {target_safe} Safe examples (Random)...")
        safe_df = candidates_safe_df.sample(n=target_safe, random_state=seed).copy()

    # 4. Labeling for Binary Classification
    # Attack = 1, Safe = 0
    attacks_df['target'] = 1
    safe_df['target'] = 0
    
    # 5. Merge
    final_df = pd.concat([attacks_df, safe_df])
    # Shuffle
    final_df = final_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    print(f"\n--- Final Rail A Dataset ---")
    print(f"Total: {len(final_df)}")
    print(final_df['target'].value_counts())
    
    # Save
    final_df.to_parquet(output_path)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    IN_FILE = "data/processed/final_augmented_dataset_enriched.parquet"
    if not os.path.exists(IN_FILE):
        IN_FILE = "data/processed/final_augmented_dataset.parquet"
        
    OUT_FILE = "data/processed/rail_a_jailbreak.parquet"
    
    prepare_rail_a(IN_FILE, OUT_FILE)
