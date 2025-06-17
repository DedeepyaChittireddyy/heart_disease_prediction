# 🫀 Heart Disease Prediction

This project applies an end-to-end machine learning pipeline to predict different types of heart disease using real-world health surveillance data. It demonstrates data cleaning, EDA, feature engineering, model training, and prediction workflows.

---

## 📁 Project Structure

```
heart_disease_prediction/
├── src/                    # Python source code for each pipeline step
│   ├── config.py
│   ├── main.py
│   ├── data_loading.py
│   ├── data_cleaning.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   └── prediction.py
├── data/                   # (You must place raw data here)
├── sample_outputs/         # (Optional) Saved plots and model comparison charts
├── requirements.txt        # List of Python dependencies
└── README.md               # Project overview and usage guide
```

---

## 🧪 Features
- Automated pipeline with `main.py`
- Visual EDA: distribution plots, heatmaps, and priority breakdowns
- Feature encoding and transformation with `PCA`
- Model training using:
  - Logistic Regression
  - Decision Tree
  - k-NN
  - XGBoost
- GridSearchCV-based hyperparameter tuning
- Evaluation with accuracy, precision, recall, F1, and confusion matrices
- Optional CLI-based prediction for new inputs

---

## ⚙️ How to Run

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Place your dataset
Put your `heart_disease_data.csv` in the `/data` folder. Update `config.py` if paths differ.

### Step 3: Run the pipeline
```bash
python src/main.py
```

### Step 4: (Optional) Make a prediction
After model training:
```bash
python src/prediction.py
```

---

## 📊 Output
All outputs are saved to the `output/` folder created during runtime:
- Cleaned dataset
- Feature-engineered data
- Visualizations (EDA + PCA)
- Trained models (.pkl)
- Model evaluation reports

---

## 📝 Notes
- This project is for educational and demonstrative purposes only.
- Do **not** use for real medical diagnosis without proper validation and regulatory compliance.

---

## 📬 Contact

Made by [Dedeepya Chittireddy](https://www.linkedin.com/in/dedeepya-ch/).  
Feel free to reach out for suggestions or collaboration!
