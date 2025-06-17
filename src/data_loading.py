import pandas as pd
import numpy as np

def load_data(filepath):
    """
    Load the heart disease data from CSV file
    """
    data = pd.read_csv(filepath)
    print("Data loaded successfully. Shape:", data.shape)
    return data

def initial_data_info(data):
    """
    Display initial information about the dataset
    """
    print("\n=== Initial Data Info ===")
    print("\nFirst 5 rows:")
    print(data.head())
    
    print("\nLast 5 rows:")
    print(data.tail())
    
    print("\nData Info:")
    print(data.info())
    
    print("\nData Shape:", data.shape)
    print("\nColumns:", data.columns.tolist())
    
    print("\nDescriptive Statistics:")
    print(data.describe().T)
    
    print("\nMissing Values:")
    print(data.isnull().sum())
    
    print("\nUnique Value Counts:")
    print(data.nunique())

if __name__ == "__main__":
    data = load_data('data/heart_disease_data.csv')
    initial_data_info(data)