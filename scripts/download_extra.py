import os
from datasets import load_dataset

DATA_DIR = "data/raw"
os.makedirs(DATA_DIR, exist_ok=True)

def download_child_guard():
    print("Downloading ChildGuard (via safe alternative if direct requires access)...")
    # Real ChildGuard often requires gated access. 
    # Alternative: Use "Hate-Speech-CNERG/hate_speech_incidents" or similar if ChildGuard fails.
    # Let's try to find a public HF version or similar child-safety relevant data.
    # "ethz-spylab/civic_toxicity" is good for harassment.
    
    # For now, let's try a known safe harassment dataset to fill the "0" gap in Harassment
    try:
        print("Fetching Civil Comments (for Harassment)...")
        ds = load_dataset("google/civil_comments", split="train[:10%]") # 10% is huge already
        ds.to_parquet(os.path.join(DATA_DIR, "civil_comments_sample.parquet"))
        print(" Saved civil_comments_sample.parquet")
    except Exception as e:
        print(f" Failed Civil Comments: {e}")

def download_jigsaw_clean():
    print("Downloading Jigsaw (Clean Version)...")
    try:
        # This is a user-uploaded version that often works without the script issue
        ds = load_dataset("google/jigsaw_toxicity_pred", split="train[:5%]") 
        ds.to_parquet(os.path.join(DATA_DIR, "jigsaw_clean.parquet"))
        print(" Saved jigsaw_clean.parquet")
    except Exception as e:
        print(f" Failed Jigsaw Clean: {e}")
        # Secondary fallback
        try:
             ds = load_dataset("mediabiasgroup/mbi_dataset", split="train") # Example placeholder
             # Actually, let's stick to standard ones we can likely get.
             pass
        except: pass

if __name__ == "__main__":
    download_child_guard()
    download_jigsaw_clean()
