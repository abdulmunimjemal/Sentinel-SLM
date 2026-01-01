import pandas as pd

df = pd.read_parquet("data/raw/beavertails_30k.parquet")
print("Columns:", df.columns.tolist())
print("-" * 20)
print(df.iloc[0])
print("-" * 20)
print(df.iloc[1])
