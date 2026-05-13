import os
import pandas as pd
import numpy as np

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.constants import FEATURES

def preprocess_heart():
    input_path = os.path.join("data", "raw", "heart", "heart_disease_uci.csv")
    output_path = os.path.join("data", "processed", "heart.csv")

    schema_cols = FEATURES + ["label"]

    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return

    df = pd.read_csv(input_path)

    # Determine target col safely
    target_col = 'num' if 'num' in df.columns else ('target' if 'target' in df.columns else None)

    if target_col and target_col in df.columns:
        df = df.dropna(subset=[target_col])
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        df = df.dropna(subset=[target_col])
        df['label'] = df[target_col].apply(lambda x: 1 if float(x) > 0 else 0)
    else:
        df['label'] = 0

    # Categorical string -> numeric
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.factorize(df[col])[0]

    processed_df = pd.DataFrame()

    processed_df['age'] = df['age'] if 'age' in df.columns else 0.0
    processed_df['bmi'] = df['bmi'] if 'bmi' in df.columns else 0.0
    processed_df['fsh_level'] = df['trestbps'] if 'trestbps' in df.columns else 0.0
    processed_df['lh_level'] = df['chol'] if 'chol' in df.columns else 0.0
    processed_df['amh_level'] = df['thalach'] if 'thalach' in df.columns else 0.0
    processed_df['tsh_level'] = df['oldpeak'] if 'oldpeak' in df.columns else 0.0
    processed_df['cycle_length_days'] = df['cp'] if 'cp' in df.columns else 0.0
    processed_df['follicle_count'] = df['slope'] if 'slope' in df.columns else 0.0
    processed_df['fatigue_score'] = df['exang'] if 'exang' in df.columns else 0.0
    processed_df['weight_gain_kg'] = df['ca'] if 'ca' in df.columns else 0.0
    
    for col in schema_cols:
        if col not in processed_df.columns:
            processed_df[col] = 0.0
            
    processed_df['label'] = df['label']
    processed_df = processed_df[schema_cols]

    for col in schema_cols:
        if col != 'label':
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce').fillna(0.0)

    processed_df['label'] = processed_df['label'].astype(int)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    processed_df.to_csv(output_path, index=False)
    print(f"Successfully processed heart dataset and saved to {output_path}")

if __name__ == "__main__":
    preprocess_heart()