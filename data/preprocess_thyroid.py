import os
import pandas as pd
import numpy as np

def preprocess_thyroid():
    input_path = os.path.join("data", "raw", "thyroid", "cleaned_dataset_thyroid1.csv")
    output_path = os.path.join("data", "processed", "thyroid.csv")

    schema_cols = [
        'age', 'bmi', 'fsh_level', 'lh_level', 'amh_level', 'tsh_level', 
        'cycle_length_days', 'follicle_count', 'fatigue_score', 'weight_gain_kg', 'label'
    ]

    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        # Create dummy data for now so the script can still run and generate the target format
        print("Creating dummy data schema for demonstration...")
        df = pd.DataFrame(columns=['Age', 'TSH', 'T3', 'Class'])
    else:
        df = pd.read_csv(input_path)

    # Standardize original column names for mapping
    # Assuming standard representations in the raw file
    col_mapping = {
        'Age': 'age',
        'age': 'age',
        'TSH': 'tsh_level',
        'tsh': 'tsh_level',
        'Class': 'label',
        'target': 'label',
        'label': 'label'
    }
    
    # Apply mapping
    df = df.rename(columns=col_mapping)

    # Initialize processed dataframe with the exact FEM-NET schema structure
    processed_df = pd.DataFrame()

    for col in schema_cols:
        if col in df.columns:
            processed_df[col] = df[col]
        else:
            # Fill missing columns with 0.0 as requested
            processed_df[col] = 0.0

    # Clean the 'label' to ensure it's binary 0/1
    if 'label' in processed_df.columns:
        # Convert any existing labels to numeric, filling NaNs with 0
        processed_df['label'] = pd.to_numeric(processed_df['label'], errors='coerce').fillna(0)
        # Binarize: anything > 0 becomes 1
        processed_df['label'] = processed_df['label'].apply(lambda x: 1 if float(x) > 0 else 0).astype(int)

    # Make sure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to processed directory
    processed_df.to_csv(output_path, index=False)
    print(f"Successfully processed thyroid dataset and saved to {output_path}")
    print(f"Columns: {list(processed_df.columns)}")

if __name__ == "__main__":
    preprocess_thyroid()
