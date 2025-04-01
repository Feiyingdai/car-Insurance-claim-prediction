## 🚗 Car Insurance Claim Prediction with XGBoost

This project builds an machine learning model to predict car insurance claims using domain-specific features from driver demographics, vehicle details, region-based risk indicators, and behavioral patterns.

### 🌟 Objective

Predict whether a driver will make a claim (target = 1) based on historical and behavioral features.

✅ Helps improve:

- Risk stratification

- Premium pricing


### 📊 Dataset Overview

Training set: 590,000+ rows

Test set: 890,000+ rows

Target variable: target (1 = made a claim, 0 = no claim)

### 🧩 Feature Groups

1. Individual Info

ps_ind_* : Age group, license type, profession, training participation

2. Car Features

ps_car_* : Vehicle type, battery health, charging data, Autopilot

3. Region Features

ps_reg_* : Climate, traffic, charger density

4. Driving Behavior

ps_calc_* : Acceleration/braking aggressiveness, high mileage behavior

### 🧠 Model Architecture

✅ Model: XGBoost

✅ Data Clean:

Data Imbalance

Outlier and missing value handling

✅ Feature Engineering:

Target Encoding with smoothed priors + noise

Polynomial feature interactions

Gain/weight-based feature selection

✅ Tuning:

Hyperparameter tuning via Optuna

Target encoding parameter tuning via nested Optuna

5-fold Stratified Cross-Validation

✅ Final Performance:

AUC: 0.73

Identified ~65% of high-risk drivers

### 📦 Tech Stack

Python 3.10+

XGBoost 1.7+

Optuna 3.0+

scikit-learn, pandas, seaborn, matplotlib

### 📁 Project Structure
```
.
├── data/                   # Raw and processed datasets (not uploaded)
├── notebooks/              # Code
└── README.md               # Project overview
```

### 📊 Results

Final AUC: 0.73

High-risk user detection: 65%

Dimensionality reduced via gain-based feature selection

Tuned encoding parameters with nested Optuna


### 🔒 Notes

Dataset not uploaded due to privacy constraints
