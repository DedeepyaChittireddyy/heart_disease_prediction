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
    from config import (RAW_DATA, CLEANED_DATA, FEATURE_DATA, 
                       MODEL_DIR, OUTPUT_DIR, EDA_DIR)
    from data_loading import load_data, initial_data_info
    from data_cleaning import clean_data, remove_outliers_iqr
    from eda import run_eda_pipeline
    from feature_engineering import engineer_features
    from modeling import run_modeling_pipeline
    from prediction import predict_disease, get_user_input
except ImportError as e:
    print(f"Error importing modules: {str(e)}")
    print("Please ensure:")
    print("1. All required modules exist")
    print("2. The project root is in Python path")
    print(f"Current Python path: {sys.path}")
    sys.exit(1)

def main():
    """End-to-end data science pipeline for heart disease prediction"""
    try:
        # Create output directory structure
        for directory in [OUTPUT_DIR, EDA_DIR, MODEL_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

        # Step 1: Data Loading with validation
        print("\n" + "="*50)
        print("STEP 1: DATA LOADING")
        print("="*50)
        try:
            if not RAW_DATA.exists():
                raise FileNotFoundError(f"Data file not found at {RAW_DATA}\n"
                                      f"Please ensure the file exists at this location")
            
            data = load_data(RAW_DATA)
            initial_data_info(data)
        except Exception as e:
            print(f"Data loading failed: {str(e)}")
            print("\nPossible solutions:")
            print(f"1. Place your data file at: {RAW_DATA}")
            print("2. Update the RAW_DATA path in config.py")
            sys.exit(1)

        # Step 2: Data Cleaning
        print("\n" + "="*50)
        print("STEP 2: DATA CLEANING")
        print("="*50)
        try:
            cleaned_data = clean_data(data)
            final_data = remove_outliers_iqr(cleaned_data)
            final_data.to_csv(CLEANED_DATA, index=False)
            print(f"✓ Cleaned data saved to {CLEANED_DATA}")
        except Exception as e:
            print(f"Data cleaning failed: {str(e)}")
            raise

        # Step 3: Exploratory Data Analysis
        print("\n" + "="*50)
        print("STEP 3: EXPLORATORY DATA ANALYSIS")
        print("="*50)
        try:
            run_eda_pipeline(CLEANED_DATA)
            print("✓ EDA completed successfully")
        except Exception as e:
            print(f"EDA failed: {str(e)}")
            raise

        # Step 4: Feature Engineering
        print("\n" + "="*50)
        print("STEP 4: FEATURE ENGINEERING")
        print("="*50)
        try:
            engineered_data = engineer_features(final_data)
            engineered_data.to_csv(FEATURE_DATA, index=False)
            print(f"✓ Engineered features saved to {FEATURE_DATA}")
        except Exception as e:
            print(f"Feature engineering failed: {str(e)}")
            raise

        # Step 5: Model Training
        print("\n" + "="*50)
        print("STEP 5: MODEL TRAINING")
        print("="*50)
        try:
            run_modeling_pipeline()
            print("✓ Model training completed")
        except Exception as e:
            print(f"Model training failed: {str(e)}")
            raise

        # Step 6: Prediction Demo
        print("\n" + "="*50)
        print("STEP 6: PREDICTION DEMONSTRATION")
        print("="*50)
        
        try:
            # Load the best model with fallback
            try:
                model_comparison = pd.read_csv(MODEL_DIR / "model_comparison.csv")
                best_model_name = model_comparison.iloc[0]['Model'].replace(' ', '_').lower()
                model_path = MODEL_DIR / f"{best_model_name}.pkl"
                print(f"✓ Loading best model: {best_model_name}")
            except Exception as e:
                print("Couldn't determine best model, defaulting to logistic regression")
                model_path = MODEL_DIR / "logistic_regression.pkl"
            
            model = joblib.load(model_path)
            scaler = joblib.load(MODEL_DIR / "scaler.pkl")
            pca = joblib.load(MODEL_DIR / "pca.pkl")
            
            # Example prediction with proper DataFrame structure
            print("\nMaking example prediction with sample data...")
            example_features = pd.DataFrame([[
                2006, 4, 0, 5.2, 5.1, 5.3, 3, 7, 1, 1, 0
            ]], columns=[
                'Year', 'Indicator', 'Data_Value_Unit', 'Data_Value_Alt',
                'LowConfidenceLimit', 'HighConfidenceLimit', 'Break_Out_Category',
                'Break_Out', 'Priority1_Missing', 'Priority3_Missing', 'PriorityArea'
            ])
            
            prediction = predict_disease(model, scaler, pca, example_features)
            print("\n" + "-"*50)
            print(f"EXAMPLE PREDICTION RESULT: {prediction}")
            print("-"*50)
            
            # Interactive prediction
            print("\nWould you like to make a custom prediction? (y/n)")
            if input().strip().lower() == 'y':
                print("\n" + "-"*50)
                print("INTERACTIVE PREDICTION")
                user_features = get_user_input()
                prediction = predict_disease(model, scaler, pca, user_features)
                print("\n" + "-"*50)
                print(f"PREDICTION RESULT: {prediction}")
                print("-"*50)
                
        except Exception as e:
            print(f"\n⚠️ Prediction demo failed: {str(e)}")
            print("Continuing without prediction demonstration...")

        print("\n" + "="*50)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"All outputs saved to: {OUTPUT_DIR}")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR in pipeline execution: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()