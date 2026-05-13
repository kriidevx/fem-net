import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.constants import FEATURES

df = pd.read_csv("data/raw/cervical/cervical_cancer.csv", na_values="?")
print("Columns:", df.columns.tolist())
print("Shape:", df.shape)

df = df.apply(pd.to_numeric, errors="coerce")

# Drop columns with >40% missing
thresh = len(df) * 0.6
df = df.dropna(thresh=thresh, axis=1)

# Target column — check what exists
for candidate in ["Biopsy", "biopsy", "Ca cervix", "dx"]:
    if candidate in df.columns:
        label_col = candidate
        break
else:
    print("Label column not found. Available:", df.columns.tolist())
    raise SystemExit

df = df.dropna(subset=[label_col])
df[label_col] = df[label_col].astype(int)

X = df.drop(columns=[label_col])
y = df[label_col]

X = X.fillna(X.median())

# Select top 10 features
k = min(10, X.shape[1])
selector = SelectKBest(f_classif, k=k)
X_sel = selector.fit_transform(X, y)
selected_names = X.columns[selector.get_support()].tolist()
print("Selected features:", selected_names)

out = pd.DataFrame(X_sel, columns=FEATURES[:k])
# Pad missing columns with 0 if k < 10
for name in FEATURES[k:]:
    if name not in out.columns:
        out[name] = 0.0
out["label"] = y.values

os.makedirs("data/processed", exist_ok=True)
out.to_csv("data/processed/cervical.csv", index=False)
print(f"Saved: {len(out)} rows, {out['label'].mean():.1%} positive")