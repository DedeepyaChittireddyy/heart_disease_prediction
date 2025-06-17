import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import joblib

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from config import CLEANED_DATA, OUTPUT_DIR
except ImportError:
    print("Error: Could not import from config.py. Please ensure:")
    print("1. config.py exists in your project root")
    print("2. The project root is in your Python path")
    sys.exit(1)

# Create feature engineering output directory
FEATURE_DIR = Path(OUTPUT_DIR) / "feature_engineering"
os.makedirs(FEATURE_DIR, exist_ok=True)

def engineer_features(data: pd.DataFrame, save_artifacts: bool = True) -> pd.DataFrame:
    """
    Perform feature engineering on the dataset and save processed data
    
    Args:
        data: Input cleaned DataFrame
        save_artifacts: Whether to save processed data and encoders
        
    Returns:
        DataFrame with engineered features
    """
    print("\n=== Feature Engineering ===")
    
    # Columns to drop
    cols_to_drop = [
        'PriorityArea2', 'PriorityArea4', 'DataSource', 'Category',
        'Data_Value_Type', 'LocationAbbr', 'LocationDesc', 'TopicId',
        'BreakOutId', 'BreakOutCategoryId', 'CategoryId', 'LocationID',
        'Data_Value_TypeID', 'IndicatorID'
    ]
    data = data.drop(columns=cols_to_drop, errors='ignore')
    
    # Feature engineering for Priority Area
    print("Engineering PriorityArea feature...")
    data['PriorityArea1'] = data['PriorityArea1'].fillna('Missing')
    data['PriorityArea3'] = data['PriorityArea3'].fillna('Missing')
    
    # Create dummy variables with proper prefixes
    priority1_dummies = pd.get_dummies(data['PriorityArea1'], prefix='Priority1')
    priority3_dummies = pd.get_dummies(data['PriorityArea3'], prefix='Priority3')
    
    # Combine with original data
    data = pd.concat([data, priority1_dummies, priority3_dummies], axis=1)
    
    # Create combined PriorityArea feature
    if 'Priority1_Million Hearts' in data.columns and 'Priority3_Healthy People 2020' in data.columns:
        data["PriorityArea"] = (
            data["Priority1_Million Hearts"].astype(int) + 
            data["Priority3_Healthy People 2020"].astype(int)
        )
    else:
        print("Warning: Required columns for PriorityArea not found")
        data["PriorityArea"] = 0
    
    # Drop intermediate columns
    cols_to_drop = ['PriorityArea1', 'PriorityArea3']
    if 'Priority1_Million Hearts' in data.columns:
        cols_to_drop.append('Priority1_Million Hearts')
    if 'Priority3_Healthy People 2020' in data.columns:
        cols_to_drop.append('Priority3_Healthy People 2020')
    
    data = data.drop(cols_to_drop, axis=1)
    
    # Label Encoding with artifact saving
    print("Performing Label Encoding...")
    encoders = {}
    categorical_cols = [
        'Topic', 'Indicator', 'Data_Value_Unit',
        'Break_Out_Category', 'Break_Out'
    ]
    
    for col in categorical_cols:
        if col in data.columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            encoders[col] = le
            
            # Save mapping for interpretation
            mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            print(f"{col} mapping: {mapping}")
        else:
            print(f"Warning: Column {col} not found for encoding")
    
    # Save artifacts if requested
    if save_artifacts:
        # Save processed data
        processed_path = os.path.join(FEATURE_DIR, "heart_disease_processed.csv")
        data.to_csv(processed_path, index=False)
        print(f"\nSaved processed data to {processed_path}")
        
        # Save encoders
        encoder_path = os.path.join(FEATURE_DIR, "label_encoders.pkl")
        joblib.dump(encoders, encoder_path)
        print(f"Saved label encoders to {encoder_path}")
    
    # Final info
    print("\n=== Feature Engineering Complete ===")
    print("Final Columns:", data.columns.tolist())
    print("Data Shape:", data.shape)
    
    return data

if __name__ == "__main__":
    try:
        # Load cleaned data
        print(f"Loading cleaned data from {CLEANED_DATA}")
        if not os.path.exists(CLEANED_DATA):
            raise FileNotFoundError(f"Cleaned data not found at {CLEANED_DATA}")
            
        cleaned_data = pd.read_csv(CLEANED_DATA)
        
        # Perform feature engineering
        engineered_data = engineer_features(cleaned_data)
        
        # Sample output
        print("\nSample of processed data:")
        print(engineered_data.head())
        
    except Exception as e:
        print(f"\nError in feature engineering: {str(e)}")
        sys.exit(1)