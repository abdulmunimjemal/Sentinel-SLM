import pandas as pd

df = pd.read_parquet("data/raw/jailbreak_bench.parquet")
print("Columns:", df.columns.tolist())
print("-" * 20)
print(df.iloc[0])
print("-" * 20)
print(df.iloc[1])
print("-" * 20)
print("Unique labels:", df.iloc[:, 1].unique()) # Assuming label is 2nd col or inspect by name
