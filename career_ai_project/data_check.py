from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "data" / "career_prediction_multi_industry.csv"

df = pd.read_csv(DATA_PATH)

print(f"File: {DATA_PATH.name}")
print(f"Số dòng: {len(df)}")
print(f"Số nghề (Career): {df['Career'].nunique()}")
print(f"Số lĩnh vực (Field): {df['Field'].nunique()}")
print("\nFields:")
for field in sorted(df["Field"].unique()):
    n = (df["Field"] == field).sum()
    print(f"  - {field}: {n} rows")
