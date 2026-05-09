import pandas as pd
import numpy as np
import os

# --- Load both PCOS datasets ---
# Print columns first to verify names, then adjust the column_map below
files = [f for f in os.listdir("data/raw/pcos") if f.endswith(".csv")]
print("Files found:", files)

dfs = []
for f in files:
    try:
        dfs.append(pd.read_csv(f"data/raw/pcos/{f}"))
        print(f"Loaded {f}: {dfs[-1].shape}")
    except Exception as e:
        print(f"Skip {f}: {e}")

df = pd.concat(dfs, ignore_index=True)
print("\nAll columns:", df.columns.tolist())

# Strip spaces from column names first
df.columns = df.columns.str.strip()

# --- Map to FEM-NET schema ---
column_map = {
    "Age (yrs)":            "age",
    "BMI":                  "bmi",
    "FSH(mIU/mL)":          "fsh_level",
    "LH(mIU/mL)":           "lh_level",
    "AMH(ng/mL)":           "amh_level",
    "TSH (mIU/L)":          "tsh_level",
    "Cycle length(days)":   "cycle_length_days",
    "Follicle No. (R)":     "follicle_count",
    "Fatigue (Y/N)":        "fatigue_score",
    "Weight gain(Y/N)":     "weight_gain_kg",
    "PCOS (Y/N)":           "label",
}
df = df.rename(columns=column_map)

FEATURES = ["age","bmi","fsh_level","lh_level","amh_level",
            "tsh_level","cycle_length_days","follicle_count",
            "fatigue_score","weight_gain_kg"]

# Keep only columns that exist after renaming
available = [c for c in FEATURES if c in df.columns]
missing = [c for c in FEATURES if c not in df.columns]
if missing:
    print(f"WARNING: missing features {missing} — filling with median")
    for m in missing:
        df[m] = 0.0

# --- Clean ---
df = df[FEATURES + ["label"]].copy()
df = df.apply(pd.to_numeric, errors="coerce")
df["label"] = df["label"].fillna(0).astype(int)
df = df.dropna(thresh=int(len(df.columns) * 0.8))
df = df.fillna(df.median(numeric_only=True))
df = df[(df["label"] == 0) | (df["label"] == 1)]

os.makedirs("data/processed", exist_ok=True)
df[FEATURES + ["label"]].to_csv("data/processed/pcos.csv", index=False)
print(f"\nSaved: {len(df)} rows, {df['label'].mean():.1%} positive")
print(df[FEATURES].describe())