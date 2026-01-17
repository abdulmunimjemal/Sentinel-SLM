import fasttext
import pandas as pd
import numpy as np
import os
import urllib.request
from tqdm import tqdm
import logging
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("EnrichLang")

DATA_DIR = "data/processed"
OLD_FILE = os.path.join(DATA_DIR, "final_augmented_dataset.parquet")
NEW_FILE = os.path.join(DATA_DIR, "final_augmented_dataset_enriched.parquet")
MODEL_DIR = "data/models"
MODEL_PATH = os.path.join(MODEL_DIR, "lid.176.bin")
MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"

def download_model():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    if not os.path.exists(MODEL_PATH):
        logger.info(f"Downloading FastText model from {MODEL_URL}...")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            logger.info("Download complete.")
        except Exception as e:
            logger.error(f"Download failed: {e}")
            raise

def predict_language(text: str, model) -> str:
    if not isinstance(text, str) or not text.strip():
        return "unknown"
    # FastText expects single line, no newlines
    clean_text = text.replace("\n", " ")[:1000] # Limit context for speed, usually enough
    try:
        predictions = model.predict(clean_text)
        # Format: (('__label__en',), array([0.98]))
        lang = predictions[0][0].replace("__label__", "")
        return lang
    except Exception as e:
        if np.random.rand() < 0.001: # Log rarely to avoid spam
             logger.error(f"Prediction error: {e}")
        return "unknown"

def main():
    if not os.path.exists(OLD_FILE):
        logger.error(f"Dataset not found at {OLD_FILE}")
        return

    download_model()
    
    logger.info("Loading FastText model...")
    model = fasttext.load_model(MODEL_PATH)
    
    logger.info("Loading dataset...")
    df = pd.read_parquet(OLD_FILE)
    logger.info(f"Loaded {len(df):,} rows.")
    
    logger.info("Predicting languages (FastText)...")
    tqdm.pandas()
    # Apply prediction
    df['lang'] = df['text'].progress_apply(lambda x: predict_language(x, model))
    
    # Save
    logger.info(f"Saving enriched dataset to {NEW_FILE}...")
    df.to_parquet(NEW_FILE)
    
    # Report
    logger.info("\n--- Language Distribution (Top 20) ---")
    print(df['lang'].value_counts().head(20))
    
    logger.info(f"\nSaved enriched file to: {NEW_FILE}")

if __name__ == "__main__":
    main()
