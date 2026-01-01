import pandas as pd
import sys
import os
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.taxonomy import Category, CATEGORY_NAMES

PROCESSED_DIR = "data/processed"
SYNTHETIC_FILE = "data/synthetic/synthetic_data.jsonl"
FINAL_FILE = "data/processed/final_augmented_dataset.parquet"

# Reverse mapping for synthetic strings
NAME_TO_CAT = {v: k for k, v in CATEGORY_NAMES.items()}
# Handle potential variations in generated labels if the model wasn't strict
# But our prompt instructed strictly. Let's add robustness.
def get_cat_id(label_str):
    label_str = label_str.strip()
    # Try exact match
    if label_str in NAME_TO_CAT:
        return NAME_TO_CAT[label_str]
    # Try partial match or standard variations
    if "Sexual" in label_str: return Category.SEXUAL
    if "Child" in label_str: return Category.CHILD_SAFETY
    if "Harassment" in label_str: return Category.HARASSMENT
    if "Hate" in label_str: return Category.HATE_EXTREMISM
    if "Prompt" in label_str: return Category.PROMPT_ATTACK
    if "Privacy" in label_str: return Category.PRIVACY
    return Category.SAFE # Default fallback if completely unknown

def main():
    # 1. Load Main
    main_path = os.path.join(PROCESSED_DIR, "unified_dataset.parquet")
    if not os.path.exists(main_path):
        print("Main dataset not found!")
        return
    
    df_main = pd.read_parquet(main_path)
    print(f"Main Data: {len(df_main)} samples")
    
    # 2. Load Synthetic
    if not os.path.exists(SYNTHETIC_FILE):
        print("No synthetic data found. Skipping merge.")
        return
        
    synth_data = []
    with open(SYNTHETIC_FILE, 'r') as f:
        for line in f:
            if not line.strip(): continue
            try:
                item = json.loads(line)
                # Structure: {"text": "...", "label": "...", "lang": "..."}
                text = item.get("text", "")
                raw_label = item.get("label", "")
                
                cat_id = get_cat_id(raw_label)
                
                synth_data.append({
                    "text": text,
                    "labels": [cat_id], # List of ints
                    "source": "synthetic_openrouter"
                })
            except Exception as e:
                print(f"Skipping bad JSON line: {e}")
                
    if synth_data:
        df_synth = pd.DataFrame(synth_data)
        print(f"Synthetic Data: {len(df_synth)} samples")
        
        # 3. Concatenate
        df_final = pd.concat([df_main, df_synth], ignore_index=True)
    else:
        df_final = df_main
        
    # 4. Save
    df_final.to_parquet(FINAL_FILE)
    print(f"Saved Final Augmented Dataset: {len(df_final)} samples to {FINAL_FILE}")
    
    # 5. Audit
    print("\nFinal Distribution:")
    exploded = df_final.explode("labels")
    print(exploded["labels"].value_counts().rename(index=CATEGORY_NAMES))

if __name__ == "__main__":
    main()
