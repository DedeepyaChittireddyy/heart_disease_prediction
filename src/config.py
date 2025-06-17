# config.py
from pathlib import Path

# Base directories - use the correct project structure
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"  # Data should be in project folder
OUTPUT_DIR = PROJECT_DIR / "outputs"

# Subdirectories
EDA_DIR = OUTPUT_DIR / "eda"
MODEL_DIR = OUTPUT_DIR / "modeling"

# File paths - now relative to project
RAW_DATA = DATA_DIR / "heart_disease_data.csv"
CLEANED_DATA = OUTPUT_DIR / "cleaned_data.csv"
FEATURE_DATA = OUTPUT_DIR / "features.csv"

# Create directories
DATA_DIR.mkdir(parents=True, exist_ok=True)  # Will create data directory if missing
OUTPUT_DIR.mkdir(exist_ok=True)
EDA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)