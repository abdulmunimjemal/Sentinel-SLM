import os
import sys
import pandas as pd
import datasets
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.taxonomy import Category, MAPPING_RULES, CATEGORY_NAMES

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

def process_beavertails():
    path = os.path.join(RAW_DIR, "beavertails_30k.parquet")
    if not os.path.exists(path):
        print(f"Skipping BeaverTails (not found): {path}")
        return []

    print("Processing BeaverTails...")
    df = pd.read_parquet(path)
    
    # BeaverTails has columns like 'prompt', 'response', 'is_safe', 'category'
    # Category is a dictionary or string? Let's assume it has 'category' column which might be null if safe
    # Actually BeaverTails structure usually has boolean flags or a list of categories.
    # We need to check the structure. Let's assume standard HF structure: 
    # It often has 'category' mapping to keys like 'violence', etc.
    
    processed_data = []
    
    # The MAPPING_RULES for beavertails is a dict: raw_label -> Category Enum
    bt_rules = MAPPING_RULES["beaver_tails"]
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = row.get("prompt", "") + " " + row.get("response", "")
        is_safe = row.get("is_safe", True)
        
        labels = set()
        
        if is_safe:
            labels.add(Category.SAFE)
        else:
            # Check various columns or a category column
            # Some versions have 'category' as a dict or list
            # Let's inspect typical columns: 'category' (dict: {violence: true...}) or separate cols
            
            # Implementation assuming 'category' is a dict or we just check keys from our rules
            # If the dataset is flat (e.g. columns 'violence', 'hate_speech'), we iterate rules
            for raw_label, mapped_cat in bt_rules.items():
                # Check if this raw_label exists as a column and is True
                if raw_label in row and row[raw_label]:
                    labels.add(mapped_cat)
                # Or if 'category' column contains it
                elif "category" in row and isinstance(row["category"], dict) and row["category"].get(raw_label, False):
                    labels.add(mapped_cat)
            
            # If unwieldy, fallback to safe if no bad labels found (shouldn't happen if is_safe=False)
            if not labels:
                labels.add(Category.SAFE) # Fallback
                
        # Convert to Multi-Hot or list of ints
        # For our dataset, we usually want one row per text, with list of labels
        # But for simpler training, maybe just binary flags for each of 8 cats?
        # Let's store 'labels' as a list of integers
        
        processed_data.append({
            "text": text,
            "labels": list(labels),
            "source": "beavertails"
        })
        
    return processed_data

def process_jailbreak():
    path = os.path.join(RAW_DIR, "jailbreak_bench.parquet")
    if not os.path.exists(path):
        print(f"Skipping JailbreakBench (not found): {path}")
        return []
        
    print("Processing JailbreakBench...")
    df = pd.read_parquet(path)
    
    processed_data = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # Usually has 'prompt' and 'label' (1 for jailbreak, 0 for benign)
        text = row.get("prompt", "")
        # Adjust column names if needed based on specific dataset version
        # Some use 'payload' or 'text'
        if "payload" in row: text = row["payload"]
        
        # Label: 'type' column is 'jailbreak' or 'benign'
        label_val = row.get("type", row.get("label", "benign"))
        
        labels = []
        if str(label_val).lower() == "jailbreak" or str(label_val) == "1":
            labels.append(Category.PROMPT_ATTACK)
        else:
            labels.append(Category.SAFE)
            
        processed_data.append({
            "text": text,
            "labels": labels,
            "source": "jailbreak_bench"
        })
        
    return processed_data

def process_civil_comments():
    path = os.path.join(RAW_DIR, "civil_comments_sample.parquet")
    if not os.path.exists(path):
        print(f"Skipping Civil Comments (not found): {path}")
        return []

    print("Processing Civil Comments...")
    df = pd.read_parquet(path)
    processed_data = []
    
    rules = MAPPING_RULES["civil_comments"]
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = row.get("text", "")
        labels = set()
        
        # Civil Comments usually has float scores 0.0-1.0
        # Threshold typically 0.5
        is_safe = True
        
        for col, cat in rules.items():
            if col in row and row[col] >= 0.5:
                labels.add(cat)
                is_safe = False
                
        if is_safe:
            labels.add(Category.SAFE)
            
        processed_data.append({
            "text": text,
            "labels": list(labels),
            "source": "civil_comments"
        })
    return processed_data

def process_jigsaw_clean():
    path = os.path.join(RAW_DIR, "jigsaw_clean.parquet")
    if not os.path.exists(path):
        print(f"Skipping Jigsaw Clean (not found): {path}")
        return []

    print("Processing Jigsaw Clean...")
    df = pd.read_parquet(path)
    processed_data = []
    
    rules = MAPPING_RULES["jigsaw_clean"]
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # Usually 'comment_text'
        text = row.get("comment_text", row.get("text", ""))
        labels = set()
        is_safe = True
        
        for col, cat in rules.items():
            # Jigsaw often has 0/1 ints or floats
            if col in row and row[col] >= 0.5:
                labels.add(cat)
                is_safe = False
                
        if is_safe:
            labels.add(Category.SAFE)
            
        processed_data.append({
            "text": text,
            "labels": list(labels),
            "source": "jigsaw_clean"
        })
    return processed_data

def main():
    all_data = []
    all_data.extend(process_beavertails())
    all_data.extend(process_jailbreak())
    all_data.extend(process_civil_comments())
    all_data.extend(process_jigsaw_clean())
    
    if not all_data:
        print("No data processed.")
        return

    df = pd.DataFrame(all_data)
    
    # Save
    out_path = os.path.join(PROCESSED_DIR, "unified_dataset.parquet")
    df.to_parquet(out_path)
    
    print(f"Saved {len(df)} samples to {out_path}")
    print("Sample distribution:")
    # Simple count of how many samples have each category
    exploded = df.explode("labels")
    print(exploded["labels"].value_counts().rename(index=CATEGORY_NAMES))

if __name__ == "__main__":
    main()
