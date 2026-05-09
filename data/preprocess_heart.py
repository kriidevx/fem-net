import os
import pandas as pd
import numpy as np

def preprocess_heart():
    input_path = os.path.join("data", "raw", "heart", "heart_disease_uci.csv")
    output_path = os.path.join("data", "processed", "heart.csv")

    schema_cols = [
        'age', 'bmi', 'fsh_level', 'lh_level', 'amh_level', 'tsh_level', 
        'cycle_length_days', 'follicle_count', 'fatigue_score', 'weight_gain_kg', 'label'
    ]

    # Create dummy DataFrame if file does not exist, for robust demonstration and testing
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        print("Creating dummy dataset schema for demonstration...")
        df = pd.DataFrame({
            'age': [63, 67, 67, 37], 
            'sex': [1, 1, 1, np.nan], 
            'cp': [1, 4, 4, 3], 
            'trestbps': [145, 160, 120, 130], 
            'chol': [233, 286, 229, 250], 
            'fbs': [1, 0, 0, 0], 
            'restecg': [2, 2, 2, 0], 
            'thalach': [150, 108, 129, 187], 
            'exang': [0, 1, 1, 0], 
            'oldpeak': [2.3, 1.5, 2.6, 3.5], 
            'slope': [3, 2, 2, 3], 
            'ca': [0.0, 3.0, 2.0, 0.0], 
            'thal': [6.0, 3.0, 7.0, 3.0], 
            'num': [0, 2, 1, np.nan] # num target included with a NaN to test dropping
        })
    else:
        df = pd.read_csv(input_path)

    # Determine original target column (typically 'num', sometimes 'target')
    target_col = 'num' if 'num' in df.columns else ('target' if 'target' in df.columns else None)

    # Handle missing labels: drop rows with NaNs in the target
    if target_col and target_col in df.columns:
        df = df.dropna(subset=[target_col])
        # Binarize the target column so that any value > 0 becomes 1
        df['label'] = df[target_col].apply(lambda x: 1 if float(x) > 0 else 0)
    else:
        df['label'] = 0

    # Initialize processed dataframe
    processed_df = pd.DataFrame()

    # Map the UCI attributes to the 10-feature FEM-NET schema
    processed_df['age'] = df['age'] if 'age' in df.columns else 0.0
    processed_df['bmi'] = df['bmi'] if 'bmi' in df.columns else 0.0 # Most UCI heart lists do not have BMI
    processed_df['fsh_level'] = df['trestbps'] if 'trestbps' in df.columns else 0.0
    processed_df['lh_level'] = df['chol'] if 'chol' in df.columns else 0.0
    processed_df['amh_level'] = df['thalach'] if 'thalach' in df.columns else 0.0
    processed_df['tsh_level'] = df['oldpeak'] if 'oldpeak' in df.columns else 0.0
    processed_df['cycle_length_days'] = df['cp'] if 'cp' in df.columns else 0.0
    processed_df['follicle_count'] = df['slope'] if 'slope' in df.columns else 0.0
    processed_df['fatigue_score'] = df['exang'] if 'exang' in df.columns else 0.0
    processed_df['weight_gain_kg'] = df['ca'] if 'ca' in df.columns else 0.0
    
    # Fill remaining columns with 0.0 if any are absent
    for col in schema_cols:
        if col not in processed_df.columns:
            processed_df[col] = 0.0
            
    # Assign our preprocessed target label
    processed_df['label'] = df['label']

    # Reorder fully to match schema perfectly
    processed_df = processed_df[schema_cols]

    # Fill any arbitrary remaining NaNs with 0.0
    processed_df = processed_df.fillna(0.0)

    # Make target an int
    processed_df['label'] = processed_df['label'].astype(int)

    # Save to processed directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    processed_df.to_csv(output_path, index=False)
    print(f"Successfully processed heart dataset and saved to {output_path}")

if __name__ == "__main__":
    preprocess_heart()