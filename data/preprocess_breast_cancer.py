import pandas as pd
import os

df = pd.read_csv("data/raw/breast_cancer/data.csv")
print("Columns:", df.columns.tolist())

FEATURES_SRC = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave points_mean", "symmetry_mean", "fractal_dimension_mean"
]

# Verify all exist
missing = [f for f in FEATURES_SRC if f not in df.columns]
if missing:
    print(f"Missing: {missing}. Available: {df.columns.tolist()}")
    raise SystemExit("Fix column names above before continuing.")

df["label"] = (df["diagnosis"] == "M").astype(int)
out = df[FEATURES_SRC + ["label"]].dropna().copy()

# Rename to FEM-NET standard names (positional mapping)
FEMNET_NAMES = ["age","bmi","fsh_level","lh_level","amh_level",
                "tsh_level","cycle_length_days","follicle_count",
                "fatigue_score","weight_gain_kg"]
rename_map = dict(zip(FEATURES_SRC, FEMNET_NAMES))
out = out.rename(columns=rename_map)

os.makedirs("data/processed", exist_ok=True)
out.to_csv("data/processed/breast_cancer.csv", index=False)
print(f"Saved: {len(out)} rows, {out['label'].mean():.1%} malignant")
print(out.describe())