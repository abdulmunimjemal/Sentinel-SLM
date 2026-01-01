import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.taxonomy import CATEGORY_NAMES

path = "data/processed/unified_dataset.parquet"
if not os.path.exists(path):
    print("No unified dataset found.")
    sys.exit(1)

df = pd.read_parquet(path)
print(f"Total Samples: {len(df)}")

exploded = df.explode("labels")
counts = exploded["labels"].value_counts().rename(index=CATEGORY_NAMES)

print("\nCategory Distribution:")
print(counts)

print("\nMissing/Low Data Categories (< 500 samples):")
for cat_id, name in CATEGORY_NAMES.items():
    count = counts.get(name, 0)
    if count < 500:
        print(f" - {name}: {count}")
