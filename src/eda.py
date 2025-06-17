import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from pathlib import Path

# Configuration
OUTPUT_DIR = Path("output/eda_visualizations")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_plot(fig, filename: str):
    """Helper function to save plots"""
    path = OUTPUT_DIR / filename
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved plot: {path}")

def univariate_analysis(data: pd.DataFrame):
    """
    Perform univariate analysis and save visualizations
    """
    print("\n=== Univariate Analysis ===")
    
    # 1. Pie chart of Break_Out_Category
    review_percent = data.Break_Out_Category.value_counts().reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.pie(review_percent['count'], labels=review_percent['Break_Out_Category'], 
           autopct='%1.1f%%', startangle=140)
    ax.axis('equal')
    ax.set_title("Break Out Category Distribution")
    save_plot(fig, "breakout_category_pie.png")
    
    # 2. Count plots
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))
    sns.countplot(data=data, x="Break_Out_Category", ax=axes[0])
    sns.countplot(data=data, x="Break_Out", ax=axes[1])
    plt.tight_layout()
    save_plot(fig, "breakout_counts.png")
    
    # 3. Histogram of Topic
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(y="Topic", data=data, color="darkblue", edgecolor='black')
    ax.set_title('Topic Distribution')
    save_plot(fig, "topic_distribution.png")
    
    # 4. Numerical features histograms
    freqgraph = data.select_dtypes(include=['float64', 'int64'])
    fig = freqgraph.hist(figsize=(10, 8), xlabelsize=8, ylabelsize=8)[0][0].figure
    plt.suptitle('Numerical Features Distribution')
    plt.tight_layout()
    save_plot(fig, "numerical_distributions.png")

def multivariate_analysis(data: pd.DataFrame):
    """
    Perform multivariate analysis and save visualizations
    """
    print("\n=== Multivariate Analysis ===")
    
    # 1. Correlation Heatmap
    numeric_data = data.select_dtypes(include=['number'])
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric_data.corr(), annot=True, cmap='GnBu', ax=ax, fmt=".2f")
    ax.set_title('Feature Correlation Heatmap')
    save_plot(fig, "correlation_heatmap.png")

def analyze_priority_areas(data: pd.DataFrame):
    """
    Analyze and visualize Priority Areas
    """
    print("\n=== Priority Areas Analysis ===")
    
    # Prepare data
    priority_cols = ['PriorityArea1', 'PriorityArea2', 'PriorityArea3', 'PriorityArea4']
    for col in priority_cols:
        data[col] = data[col].fillna('Missing')
    
    # Create plot
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 12))
    for i, col in enumerate(priority_cols):
        sns.countplot(data=data, x=col, ax=axes[i])
        axes[i].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    save_plot(fig, "priority_areas.png")

def run_eda_pipeline(data_path: str):
    """Full EDA pipeline execution"""
    try:
        data = pd.read_csv(data_path)
        
        # Create visualization directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Run analysis
        univariate_analysis(data)
        multivariate_analysis(data)
        analyze_priority_areas(data)
        
        print(f"\nEDA complete! All visualizations saved to {OUTPUT_DIR}")
    except Exception as e:
        print(f"Error during EDA: {str(e)}")

if __name__ == "__main__":
    # Use the cleaned data from previous steps
    cleaned_data_path = "output/heart_disease_cleaned.csv"
    run_eda_pipeline(cleaned_data_path)