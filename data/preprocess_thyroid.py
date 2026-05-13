import os
import pandas as pd
import numpy as np

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.constants import FEATURES

def preprocess_thyroid():
    input_path = os.path.join("data", "raw", "thyroid", "cleaned_dataset_thyroid1.csv")
    output_path = os.path.join("data", "processed", "thyroid.csv")

    schema_cols = FEATURES + ["label"]

    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        print("Creating dummy data schema for demonstration...")
        df = pd.DataFrame({
            'Age': [30, 45, 55], 
            'TSH': [1.5, 2.5, 0.5], 
            'T3': [2.0, 1.8, 3.1], 
            'Class': ['negative', 'P', 'negative']
        })
    else:
        df = pd.read_csv(input_path)

    # Standardize column names
    mapped_cols = {col: 'label' for col in df.columns if col.lower() in ['class', 'binaryclass', 'target']}
    mapped_cols.update({'Age': 'age', 'age': 'age', 'TSH': 'tsh_level', 'tsh': 'tsh_level', 'label': 'label'})
    
    df = df.rename(columns=mapped_cols)
    
    # If no recognized label column found, guess the last column
    if 'label' not in df.columns:
        last_col = df.columns[-1]
        df = df.rename(columns={last_col: 'label'})

    # Initialize processed dataframe with the exact schema
    processed_df = pd.DataFrame()

    for col in schema_cols:
        if col in df.columns:
            processed_df[col] = df[col]
        else:
            processed_df[col] = 0.0

    # Clean the 'label' as instructed
    def convert_label(val):
        if pd.isna(val):
            return 0
        val_str = str(val).lower()
        if 'negative' in val_str or '-' in val_str or val_str == 'n' or val_str == '0':
            return 0
        return 1

    processed_df['label'] = df['label'].apply(convert_label).astype(int)

    # Save to processed directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    processed_df.to_csv(output_path, index=False)
    
    print(f"Successfully processed thyroid dataset.")
    print(f"Label Distribution:\n{processed_df['label'].value_counts()}")

if __name__ == "__main__":
    preprocess_thyroid()