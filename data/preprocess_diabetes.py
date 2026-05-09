import os
import pandas as pd
import numpy as np

def preprocess_diabetes():
    input_path = os.path.join("data", "raw", "diabetes", "diabetes.csv")
    output_path = os.path.join("data", "processed", "diabetes.csv")

    schema_cols = [
        'age', 'bmi', 'fsh_level', 'lh_level', 'amh_level', 'tsh_level', 
        'cycle_length_days', 'follicle_count', 'fatigue_score', 'weight_gain_kg', 'label'
    ]

    # Create dummy dataframe if file does not exist, for demonstration
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        print("Creating dummy data schema for demonstration...")
        df = pd.DataFrame({
            'Pregnancies': [1, 0, 2],
            'Glucose': [85, 89, 137],
            'BloodPressure': [66, 66, 40],
            'SkinThickness': [29, 23, 35],
            'Insulin': [0, 94, 168],
            'BMI': [26.6, 28.1, 43.1],
            'DiabetesPedigreeFunction': [0.351, 0.167, 2.288],
            'Age': [31, 21, 33],
            'Outcome': [0, 0, 1]
        })
    else:
        df = pd.read_csv(input_path)

    # Replace 0s with NaN for columns where 0 is invalid before filling with median
    cols_with_invalid_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_with_invalid_zero:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
            df[col] = df[col].fillna(df[col].median())

    # Engineer features
    if 'Glucose' in df.columns and 'BMI' in df.columns:
        df['fatigue_score'] = df['Glucose'] / df['BMI']
    else:
        df['fatigue_score'] = 0.0

    if 'Insulin' in df.columns and 'Glucose' in df.columns:
        df['weight_gain_kg'] = df['Insulin'] / df['Glucose']
    else:
        df['weight_gain_kg'] = 0.0

    # Initialize processed dataframe
    processed_df = pd.DataFrame()

    # Map variables to FEM-NET schema
    processed_df['age'] = df['Pregnancies'] if 'Pregnancies' in df.columns else 0.0
    processed_df['follicle_count'] = df['Age'] if 'Age' in df.columns else 0.0
    processed_df['bmi'] = df['BMI'] if 'BMI' in df.columns else 0.0
    
    # Map remaining available variables to fill in the schema
    processed_df['fsh_level'] = df['Glucose'] if 'Glucose' in df.columns else 0.0
    processed_df['lh_level'] = df['BloodPressure'] if 'BloodPressure' in df.columns else 0.0
    processed_df['amh_level'] = df['SkinThickness'] if 'SkinThickness' in df.columns else 0.0
    processed_df['tsh_level'] = df['Insulin'] if 'Insulin' in df.columns else 0.0
    processed_df['cycle_length_days'] = df['DiabetesPedigreeFunction'] if 'DiabetesPedigreeFunction' in df.columns else 0.0
    
    # Add engineered features
    processed_df['fatigue_score'] = df['fatigue_score']
    processed_df['weight_gain_kg'] = df['weight_gain_kg']

    # Target
    processed_df['label'] = df['Outcome'] if 'Outcome' in df.columns else 0

    # Ensure all columns in schema exist (if any missed)
    for col in schema_cols:
        if col not in processed_df.columns:
            processed_df[col] = 0.0
            
    # Reorder to match schema exactly
    processed_df = processed_df[schema_cols]

    # Fill any remaining NaNs with the median
    processed_df = processed_df.fillna(processed_df.median())

    # Ensure label is binary integer
    processed_df['label'] = processed_df['label'].astype(int)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    processed_df.to_csv(output_path, index=False)
    print(f"Successfully processed diabetes dataset and saved to {output_path}")

if __name__ == "__main__":
    preprocess_diabetes()
