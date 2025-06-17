import pandas as pd
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import List, Dict, Optional
from pathlib import Path

# Create output directory if it doesn't exist
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by handling missing values and removing unnecessary columns.
    Saves missing values visualization to output folder.
    
    Args:
        data: Input DataFrame containing heart disease data
        
    Returns:
        Cleaned DataFrame with missing values handled and columns dropped
    """
    print("\n=== Cleaning Data ===")
    
    # Columns to drop
    columns_to_drop = [
        'Data_Value_Footnote_Symbol',
        'Data_Value_Footnote',
        'GeoLocation',
        'Data_Value'
    ]
    
    # Drop columns in a single operation
    data = data.drop(columns=columns_to_drop, errors='ignore')
    
    # Fill missing values using median for specified columns
    fill_cols = ['HighConfidenceLimit', 'LowConfidenceLimit']
    fill_values = {col: data[col].median() for col in fill_cols}
    data = data.fillna(fill_values)
    
    # Visualize and save missing values plot
    plt.figure(figsize=(12, 12))
    msno.matrix(data)
    plt.title("Missing Values Visualization")
    plt.tight_layout()
    missing_values_path = Path(OUTPUT_DIR) / "missing_values.png"
    plt.savefig(missing_values_path)
    plt.close()
    print(f"Saved missing values visualization to {missing_values_path}")
    
    return data

def remove_outliers_iqr(data: pd.DataFrame, 
                       numeric_cols: Optional[List[str]] = None,
                       iqr_factor: float = 1.5) -> pd.DataFrame:
    """
    Remove outliers from numerical columns using IQR method with visualization.
    Saves distribution visualizations to output folder.
    
    Args:
        data: Input DataFrame
        numeric_cols: List of numeric columns to process
        iqr_factor: Multiplier for IQR range (default 1.5)
        
    Returns:
        DataFrame with outliers removed
    """
    print("\n=== Removing Outliers ===")
    
    if numeric_cols is None:
        numeric_cols = ['Data_Value_Alt', 'LowConfidenceLimit', 'HighConfidenceLimit']
    
    # Calculate bounds for each column
    bounds = {}
    for col in numeric_cols:
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        iqr = q3 - q1
        bounds[col] = (q1 - iqr_factor * iqr, q3 + iqr_factor * iqr)
    
    # Identify outliers
    outlier_mask = pd.Series(False, index=data.index)
    for col, (lower, upper) in bounds.items():
        outlier_mask |= (data[col] < lower) | (data[col] > upper)
    
    # Remove outliers
    data_clean = data[~outlier_mask].copy()
    
    print(f"Original shape: {data.shape}")
    print(f"Shape after removing outliers: {data_clean.shape}")
    print(f"Number of outliers removed: {outlier_mask.sum()}")
    
    # Create visualization subdirectory
    viz_dir = Path(OUTPUT_DIR) / "distribution_plots"
    os.makedirs(viz_dir, exist_ok=True)
    
    # Visualize and save distributions after outlier removal
    for col in numeric_cols:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Histogram with KDE
        sns.histplot(data=data_clean, x=col, kde=True, ax=ax1)
        ax1.set_title(f'Distribution of {col}')
        
        # Boxplot
        sns.boxplot(data=data_clean, x=col, ax=ax2)
        ax2.set_title(f'Boxplot of {col}')
        
        plt.tight_layout()
        
        # Save figure
        plot_path = viz_dir / f"{col}_distribution.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved {col} visualization to {plot_path}")
    
    return data_clean

def main():
    """Main execution function that saves all outputs to the output folder"""
    try:
        # Create output directory structure
        Path(OUTPUT_DIR).mkdir(exist_ok=True)
        
        # Load data
        input_path = "data/heart_disease_data.csv"
        print(f"\nLoading data from {input_path}")
        data = pd.read_csv(input_path)
        
        # Data cleaning pipeline
        cleaned_data = clean_data(data)
        final_data = remove_outliers_iqr(cleaned_data)
        
        # Save cleaned data
        output_path = Path(OUTPUT_DIR) / "heart_disease_cleaned.csv"
        final_data.to_csv(output_path, index=False)
        print(f"\nSaved cleaned data to {output_path}")
        
        print("\nData processing completed successfully!")
        print(f"All outputs saved to {OUTPUT_DIR} directory")
        
    except FileNotFoundError as e:
        print(f"\nError: Input file not found at {e.filename}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()