import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path
from typing import List

# Configure paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from config import OUTPUT_DIR
except ImportError:
    print("Error: Could not import from config.py")
    sys.exit(1)

# Model artifacts
MODEL_DIR = Path(OUTPUT_DIR) / "modeling"
MODEL_PATH = MODEL_DIR / "logistic_regression.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
PCA_PATH = MODEL_DIR / "pca.pkl"

def load_artifacts() -> tuple:
    """Load trained model and preprocessing objects"""
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        pca = joblib.load(PCA_PATH)
        return model, scaler, pca
    except Exception as e:
        print(f"Error loading artifacts: {str(e)}")
        sys.exit(1)

def predict_disease(model, scaler, pca, features: np.ndarray) -> str:
    """Make prediction with proper feature processing"""
    try:
        # Features should already be 2D array at this point
        scaled_data = scaler.transform(features)
        x_pca = pca.transform(scaled_data)
        prediction = model.predict(x_pca)[0]
        
        disease_mapping = {
            0: 'Coronary Heart Disease',
            1: 'Heart Attack',
            2: 'Heart Failure',
            3: 'Stroke',
            4: 'Cardiovascular Diseases'
        }
        return disease_mapping.get(prediction, 'Unknown Condition')
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        raise

def get_user_input() -> pd.DataFrame:
    """Collect user input with detailed descriptions"""
    print("\n=== Heart Disease Risk Assessment ===")
    print("Please enter the following information:\n")
    
    # Feature descriptions with validation hints
    features = {
        'Year': {
            'description': 'Year of data collection (e.g., 2020)',
            'type': 'integer'
        },
        'Indicator': {
            'description': 'Disease indicator (0-9):\n'
                         '0=PrevAllHeart, 1=PrevHF, 2=PrevCerebro,\n'
                         '3=PrevStroke, 4=PrevCHD, 5=PrevAngina,\n'
                         '6=PrevMI, 7=PrevPVD, 8=PrevAFib, 9=PrevAAA',
            'type': 'integer',
            'range': (0, 9)
        },
        'Data_Value_Unit': {
            'description': 'Measurement unit:\n'
                          '1=Percentage, 2=Rate per 1000',
            'type': 'integer',
            'range': (1, 2)
        },
        'Data_Value_Alt': {
            'description': 'Numerical measurement value',
            'type': 'float'
        },
        'LowConfidenceLimit': {
            'description': 'Lower confidence bound',
            'type': 'float'
        },
        'HighConfidenceLimit': {
            'description': 'Upper confidence bound',
            'type': 'float'
        },
        'Break_Out_Category': {
            'description': 'Demographic category:\n'
                         '0=Age, 1=Gender, 2=Overall, 3=Race',
            'type': 'integer',
            'range': (0, 3)
        },
        'Break_Out': {
            'description': 'Demographic breakdown:\n'
                          '0=18-44, 1=45-64, 2=65-74, 3=75+,\n'
                          '4=Male, 5=Female, 6=White, 7=Black,\n'
                          '8=Hispanic, 9=Asian/Pacific Islander',
            'type': 'integer',
            'range': (0, 9)
        },
        'Priority1_Missing': {
            'description': 'Million Hearts priority (0=No, 1=Yes)',
            'type': 'integer',
            'range': (0, 1)
        },
        'Priority3_Missing': {
            'description': 'Healthy People 2020 priority (0=No, 1=Yes)',
            'type': 'integer',
            'range': (0, 1)
        },
        'PriorityArea': {
            'description': 'Priority classification:\n'
                         '0=None, 1=Million Hearts, 2=Healthy People 2020',
            'type': 'integer',
            'range': (0, 2)
        }
    }
    
    feature_values = {}
    for feature, meta in features.items():
        while True:
            try:
                print(f"\n{feature}: {meta['description']}")
                value = input("Enter value: ").strip()
                
                if meta['type'] == 'integer':
                    value = int(value)
                else:
                    value = float(value)
                
                if 'range' in meta:
                    if not (meta['range'][0] <= value <= meta['range'][1]):
                        print(f"Value must be between {meta['range'][0]} and {meta['range'][1]}")
                        continue
                
                feature_values[feature] = value
                break
            except ValueError:
                print(f"Please enter a valid {meta['type']}")
    
    return pd.DataFrame([feature_values])

def main():
    """Main execution"""
    print("\nLoading prediction artifacts...")
    model, scaler, pca = load_artifacts()
    
    # Get user input as DataFrame with proper feature names
    user_data = get_user_input()
    
    # Make and display prediction
    prediction = predict_disease(model, scaler, pca, user_data)
    print(f"\n=== Prediction Result ===")
    print(f"Predicted Condition: {prediction}")
    print("\nNote: This is a statistical prediction. Please consult a healthcare professional for medical advice.")

if __name__ == "__main__":
    main()