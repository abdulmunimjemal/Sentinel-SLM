import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.taxonomy import Category, CATEGORY_NAMES, MAPPING_RULES

# Re-import logic (copy-paste of logic or import if refactored, 
# here we essentially verify the logic by running it on samples)
# importing directly from scripts.map_labels might be messy if not structured as module, 
# but let's try to simulate the logic to verify correctness.

def verify_beavertails():
    print("\n--- Verifying BeaverTails ---")
    path = "data/raw/beavertails_30k.parquet"
    if not os.path.exists(path): return
    df = pd.read_parquet(path)
    
    # Find a safe and unsafe example
    safe_rows = df[df['is_safe'] == True].head(1)
    unsafe_rows = df[df['is_safe'] == False].head(5)
    
    rules = MAPPING_RULES["beaver_tails"]
    
    print("Rules keys:", rules.keys())
    
    for i, row in unsafe_rows.iterrows():
        print(f"\nRaw Category: {row['category']}")
        mapped = []
        for raw_label, mapped_cat in rules.items():
            if raw_label in row and row[raw_label]: # If column access
                mapped.append(CATEGORY_NAMES[mapped_cat])
            elif "category" in row and isinstance(row["category"], dict) and row["category"].get(raw_label, False):
                mapped.append(CATEGORY_NAMES[mapped_cat])
        print(f"Mapped: {mapped}")

def verify_civil():
    print("\n--- Verifying Civil Comments ---")
    path = "data/raw/civil_comments_sample.parquet"
    if not os.path.exists(path): return
    df = pd.read_parquet(path)
    
    # Get high tox examples
    unsafe = df[df['toxicity'] > 0.8].head(3)
    
    rules = MAPPING_RULES["civil_comments"]
    
    for i, row in unsafe.iterrows():
        print(f"\nText: {row['text'][:50]}...")
        print(f"Scores: tox={row['toxicity']}, threat={row['threat']}, insult={row['insult']}")
        mapped = []
        for col, cat in rules.items():
            if col in row and row[col] >= 0.5:
                mapped.append(CATEGORY_NAMES[cat])
        print(f"Mapped: {mapped}")

def verify_jailbreak():
    print("\n--- Verifying JailbreakBench ---")
    path = "data/raw/jailbreak_bench.parquet"
    if not os.path.exists(path): return
    df = pd.read_parquet(path)
    
    jailbreaks = df[df['type'] == 'jailbreak'].head(3)
    for i, row in jailbreaks.iterrows():
        print(f"\nPrompt: {row['prompt'][:50]}...")
        print(f"Type: {row['type']}")
        # Logic check
        if str(row['type']).lower() == 'jailbreak':
            print(f"Mapped: [{CATEGORY_NAMES[Category.PROMPT_ATTACK]}]")
        else:
            print("Mapped: [Safe]")

if __name__ == "__main__":
    verify_beavertails()
    verify_civil()
    verify_jailbreak()
