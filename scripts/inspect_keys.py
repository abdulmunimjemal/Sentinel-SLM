import pandas as pd

df = pd.read_parquet("data/raw/beavertails_30k.parquet")
first_row = df.iloc[0]
cat_dict = first_row["category"]
print("Is dict?", isinstance(cat_dict, dict))
print("Keys:", list(cat_dict.keys()))
