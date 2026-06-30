# ✈️ Flight Price & Travel Time Prediction

An end-to-end Machine Learning project that predicts **Flight Price** and **Travel Time** simultaneously using **multi-output regression**. The project compares different multi-output learning strategies, performs hyperparameter optimization with **Optuna**, tracks experiments with **MLflow**, provides an interactive **Streamlit** application, and includes **data drift monitoring** through GitHub Actions.

## 📊 Dataset

**Source:** Kaggle - Flight Price Prediction Dataset

https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction

Only the following datasets are used:

- `economy.csv`
- `business.csv`

---

## 🚀 Features

- Multi-output regression (Price & Travel Time)
- Comparison of two multi-output strategies
  - `RegressorChain`
  - `MultiOutputRegressor`
- Model comparison
  - XGBoost
  - Random Forest
- Hyperparameter tuning using Optuna
- Experiment tracking using MLflow
- Streamlit prediction application
- Performance dashboard
- A/B testing using user feedback
- Data drift detection using Chi-Square Test
- Automated notebook execution with GitHub Actions

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Random Forest
- Optuna
- MLflow
- Streamlit
- Joblib
- SciPy

---

# 📂 Project Workflow

## 1. Data Preprocessing

**Notebook:** `test1.ipynb`

- Load and combine Economy and Business datasets
- Clean missing values
- Feature engineering
- Save processed dataset as:

```
clean_data.csv
```

---

## 2. RegressorChain Models

**Notebook:** `test2.ipynb`

Uses:

```python
sklearn.multioutput.RegressorChain
```

Models trained:

- RandomForestRegressor
- XGBRegressor

### XGBoost Objective

```python
objective = "count:poisson"
```

This objective helps prevent negative predictions for Price and Travel Time.

### Includes

- Optuna Hyperparameter Tuning
- MLflow Experiment Tracking
- Evaluation Metrics
  - R² Score
  - RMSE
  - MAE
  - MSE
- Saves trained models using Joblib

---

## 3. MultiOutputRegressor Models

Uses:

```python
sklearn.multioutput.MultiOutputRegressor
```

Models trained:

- RandomForestRegressor
- XGBRegressor

Also includes:

- Optuna Hyperparameter Tuning
- MLflow Logging
- R²
- RMSE
- MAE
- MSE
- Model Saving

---

# 🌐 Streamlit Applications

## app.py

Prediction interface for the trained models.

Features:

- Predict Flight Price
- Predict Travel Time
- Compare the two best models
- Collect user feedback
- Store reviews in:

```
prediction_reviews.csv
```

---

## dashboard.py

Interactive dashboard for model evaluation.

Includes:

- Model Performance Metrics 
- Error Analysis
- A/B Testing using collected user reviews

---

## flight.py

Launches both:

- Prediction App (`app.py`)
- Dashboard (`dashboard.py`)

---

# 📈 Data Drift Monitoring

**Script:** `Data_drift.py`

Evaluates model performance on new incoming data (`test.csv`).

Checks:

- R² Score
- RMSE
- MAE
- MSE
- Chi-Square Test for Data Drift

Assumes `test.csv` represents newly collected production data.

---

# ⚙️ GitHub Actions

**Workflow:** `test.yml`

Automatically executes the Data Drift notebook.

Workflow includes:

- Install project dependencies
- Execute `Data_drift.ipynb`
- Generate an executed notebook with outputs
- Upload the notebook as a GitHub Actions artifact

Can be triggered:

- On every push to `main`
- Manually using **workflow_dispatch**

---

# 📁 Project Structure

```
.
├── app.py
├── dashboard.py
├── flight.py
├── Data_drift.ipynb
├── test1.ipynb
├── test2.ipynb
├── clean_data.csv
├── prediction_reviews.csv
├── requirements.txt
├── test.yml
└── README.md
```

---

# 📊 Evaluation Metrics

Models are evaluated using:

- R² Score
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

---



<img width="1907" height="686" alt="Screenshot 2026-06-30 142928" src="https://github.com/user-attachments/assets/3ecc2cb9-f63a-4a65-b8d6-c087296d63eb" />
