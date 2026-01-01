import os
import datasets
from datasets import load_dataset

# Configuration
DATA_DIR = "data/raw"
os.makedirs(DATA_DIR, exist_ok=True)

def download_beavertails():
    print("Downloading PKU-Alignment/BeaverTails...")
    try:
        # 30k subset for efficiency, can scale up
        ds = load_dataset("PKU-Alignment/BeaverTails", split="30k_train")
        ds.to_parquet(os.path.join(DATA_DIR, "beavertails_30k.parquet"))
        print(" Saved beavertails_30k.parquet")
    except Exception as e:
        print(f" Failed to download BeaverTails: {e}")

def download_jigsaw():
    print("Downloading Jigsaw Toxicity Severity...")
    try:
        # Note: Jigsaw data on HF often requires accepting terms/gated access.
        # Fallback to a mirror or public version if main is gated.
        # Using 'jigsaw_toxicity_pred' which is a common public upload
        ds = load_dataset("jigsaw_toxicity_pred", split="train[:10%]") # 10% for quick dev
        ds.to_parquet(os.path.join(DATA_DIR, "jigsaw_sample.parquet"))
        print(" Saved jigsaw_sample.parquet")
    except Exception as e:
        print(f" Failed to download Jigsaw: {e}")

def download_jailbreak():
    print("Downloading JailbreakBench (Jackhhao/jailbreak-classification)...")
    try:
        ds = load_dataset("jackhhao/jailbreak-classification", split="train")
        ds.to_parquet(os.path.join(DATA_DIR, "jailbreak_bench.parquet"))
        print(" Saved jailbreak_bench.parquet")
    except Exception as e:
        print(f" Failed to download JailbreakBench: {e}")

if __name__ == "__main__":
    download_beavertails()
    download_jigsaw()
    download_jailbreak()
    print("Download complete.")
