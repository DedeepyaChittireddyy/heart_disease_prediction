import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, classification_report, 
                           confusion_matrix, ConfusionMatrixDisplay,
                           precision_recall_fscore_support)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from config import OUTPUT_DIR, FEATURE_DATA
except ImportError:
    print("Error: Could not import from config.py. Please ensure:")
    print("1. config.py exists in your project root")
    print("2. The project root is in your Python path")
    sys.exit(1)

# Create modeling output directories
MODEL_DIR = Path(OUTPUT_DIR) / "modeling"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR = MODEL_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

class ModelTrainer:
    """Class to handle model training and evaluation"""
    
    def __init__(self):
        self.scaler = None
        self.pca = None
        self.models = {}  # Now stores dicts with model and metrics
        self.results = {}  # For backward compatibility
        
    def prepare_data(self, data: pd.DataFrame) -> Tuple:
        """Prepare data with scaling and PCA"""
        print("\n=== Data Preparation ===")
        
        # Validate input
        if 'Topic' not in data.columns:
            raise ValueError("Target column 'Topic' not found in data")
            
        y = data['Topic']
        X = data.drop('Topic', axis=1)
        
        # Check class distribution
        print("\nClass distribution:")
        print(y.value_counts())
        
        if len(y.unique()) < 2:
            raise ValueError("Need at least 2 classes for classification")
        
        # Standard Scaling
        self.scaler = StandardScaler()
        scaled_data = self.scaler.fit_transform(X)
        
        # PCA Transformation
        self.pca = PCA(n_components=2)
        x_pca = self.pca.fit_transform(scaled_data)
        
        # Visualize and save PCA
        self._plot_pca(x_pca, y)
        
        # Train-Test Split with stratification
        return train_test_split(
            x_pca, y, test_size=0.2, random_state=42, stratify=y
        )
    
    def _plot_pca(self, x_pca: np.ndarray, y: pd.Series):
        """Visualize PCA results"""
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(x_pca[:, 0], x_pca[:, 1], c=y, alpha=0.6, 
                            cmap='viridis')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title('PCA Visualization of Heart Disease Data')
        plt.colorbar(scatter, label='Topic Class')
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "pca_visualization.png", dpi=300, 
                   bbox_inches='tight')
        plt.close()
    
    def train_models(self, X_train: np.ndarray, y_train: pd.Series, 
                    X_test: np.ndarray, y_test: pd.Series):
        """Train and evaluate multiple models"""
        print("\n=== Model Training ===")
        
        # Define model configurations with stable parameters
        model_configs = {
            "logistic_regression": {
                "model": LogisticRegression(max_iter=1000, random_state=42),
                "params": {
                    'C': [0.1, 1, 10],
                    'penalty': ['l2'],
                    'solver': ['lbfgs']
                }
            },
            "decision_tree": {
                "model": DecisionTreeClassifier(random_state=42),
                "params": {
                    'max_depth': [5, 10, None],
                    'min_samples_split': [2, 5]
                }
            },
            "knn": {
                "model": KNeighborsClassifier(),
                "params": {
                    'n_neighbors': range(3, 21, 2),
                    'weights': ['uniform', 'distance']
                }
            },
            "xgboost": {
                "model": XGBClassifier(
                    eval_metric='logloss',
                    random_state=42
                ),
                "params": {
                    'learning_rate': [0.01, 0.1],
                    'n_estimators': [100, 200]
                }
            }
        }
        
        # Train each model with error handling
        for name, config in model_configs.items():
            try:
                print(f"\nTraining {name.replace('_', ' ').title()}...")
                model, train_metrics, test_metrics = self._train_single_model(
                    config["model"], X_train, y_train, X_test, y_test, 
                    config["params"], name
                )
                
                # Store model and metrics together
                self.models[name] = {
                    "model_object": model,
                    "train_metrics": train_metrics,
                    "test_metrics": test_metrics
                }
                
                # Save the trained model
                joblib.dump(model, MODEL_DIR / f"{name}.pkl")
                print(f"Successfully trained and saved {name}")
                
            except Exception as e:
                print(f"\nERROR training {name}: {str(e)}")
                print("Skipping this model and continuing...")
                continue
        
        # Save preprocessing objects
        joblib.dump(self.scaler, MODEL_DIR / "scaler.pkl")
        joblib.dump(self.pca, MODEL_DIR / "pca.pkl")
        
        # Compare and visualize results
        if self.models:  # Only compare if we have results
            self._compare_models()
        else:
            print("\nNo models were successfully trained")
    
    def _train_single_model(self, model, X_train, y_train, X_test, y_test, 
                          params: Optional[Dict], name: str) -> Tuple:
        """Train and evaluate a single model"""
        # Hyperparameter tuning if parameters provided
        if params:
            print(f"Performing grid search for {name}...")
            grid = GridSearchCV(
                estimator=model,
                param_grid=params,
                scoring='accuracy',
                cv=5,
                n_jobs=-1,
                verbose=1
            )
            grid.fit(X_train, y_train)
            model = grid.best_estimator_
            print(f"Best parameters: {grid.best_params_}")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate model
        train_metrics = self._evaluate_model(model, X_train, y_train, "train")
        test_metrics = self._evaluate_model(model, X_test, y_test, "test")
        
        # Save visualizations
        self._save_model_plots(model, X_test, y_test, name)
        
        return model, train_metrics, test_metrics
    
    def _evaluate_model(self, model, X, y, dataset_type: str) -> Dict:
        """Evaluate model performance"""
        y_pred = model.predict(X)
        
        accuracy = accuracy_score(y, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, y_pred, average='weighted'
        )
        report = classification_report(y, y_pred, output_dict=True)
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "report": report
        }
    
    def _save_model_plots(self, model, X_test, y_test, name: str):
        """Save model evaluation plots"""
        # Confusion matrix
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(f"{name.replace('_', ' ').title()} Confusion Matrix")
        plt.savefig(PLOT_DIR / f"{name}_confusion_matrix.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        pd.DataFrame(report).transpose().to_csv(
            PLOT_DIR / f"{name}_classification_report.csv"
        )
    
    def _compare_models(self):
        """Compare and visualize model performance"""
        print("\n=== Model Comparison ===")
        
        # Prepare comparison data
        comparison_data = []
        for name, model_data in self.models.items():
            comparison_data.append({
                "Model": name.replace('_', ' ').title(),
                "Train Accuracy": model_data["train_metrics"]["accuracy"],
                "Test Accuracy": model_data["test_metrics"]["accuracy"],
                "Test Precision": model_data["test_metrics"]["precision"],
                "Test Recall": model_data["test_metrics"]["recall"],
                "Test F1": model_data["test_metrics"]["f1"]
            })
        
        comparison_df = pd.DataFrame(comparison_data).sort_values(
            "Test Accuracy", ascending=False
        )
        
        # Save comparison table
        comparison_df.to_csv(MODEL_DIR / "model_comparison.csv", index=False)
        print("\nModel Performance Comparison:")
        print(comparison_df.to_markdown(index=False))
        
        # Plot comparison
        self._plot_model_comparison(comparison_df)
    
    def _plot_model_comparison(self, comparison_df: pd.DataFrame):
        """Visualize model comparison"""
        metrics = ["Test Accuracy", "Test Precision", "Test Recall", "Test F1"]
        
        plt.figure(figsize=(12, 8))
        comparison_df.set_index("Model")[metrics].plot(
            kind='bar', rot=45, colormap='viridis'
        )
        plt.title('Model Performance Comparison')
        plt.ylabel('Score')
        plt.ylim(0.5, 1.0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "model_comparison.png", dpi=300, 
                   bbox_inches='tight')
        plt.close()

def run_modeling_pipeline():
    """Complete modeling pipeline execution"""
    try:
        # Initialize model trainer
        trainer = ModelTrainer()
        
        # Load feature-engineered data
        print(f"\nLoading processed data from {FEATURE_DATA}")
        if not os.path.exists(FEATURE_DATA):
            raise FileNotFoundError(
                f"Feature-engineered data not found at {FEATURE_DATA}\n"
                "Please run feature_engineering.py first"
            )
            
        data = pd.read_csv(FEATURE_DATA)
        
        # Prepare data
        X_train, X_test, y_train, y_test = trainer.prepare_data(data)
        
        # Train and evaluate models
        trainer.train_models(X_train, y_train, X_test, y_test)
        
        print(f"\nModeling complete! All artifacts saved to {MODEL_DIR}")
        
    except Exception as e:
        print(f"\nFATAL ERROR in modeling pipeline: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    run_modeling_pipeline()