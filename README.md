# 🚗 Car Insurance Claim Prediction with XGBoost

Predicting whether a driver will file a car insurance claim based on demographic, vehicle, and behavioral data. This project involves a full ML pipeline: from preprocessing and feature engineering to model tuning and business-oriented risk stratification.

---

## 📚 Table of Contents

- [🌟 Project Overview](#-project-overview)
- [📊 Dataset Overview](#-dataset-overview)
- [🚀 Quick Start](#-quick-start)
- [📦 Project Phases & To-Do List](#-project-phases--to-do-list)
- [🧹 Data Cleaning](#-data-cleaning)
- [📌 Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)
- [🧩 Feature Engineering & Selection](#-feature-engineering--selection)
- [🧠 Model Building](#-model-building)
- [📊 Model Evaluation](#-model-evaluation)
- [🔍 SHAP Feature Importance](#-shap-feature-importance)
- [💼 Business Interpretation](#-business-interpretation)
- [📦 Tech Stack](#-tech-stack)
- [🚫 Notes](#-notes)
- [📄 License & Contribution](#-license--contribution)

---

## 🌟 Project Overview

### Backgorund:

In recent years, Tesla's intelligent systems and high performance have gained significant market traction. However, some models such as **Model Y and Cybertruck exhibit elevated claim frequencies and costs**, attracting attention from insurers and regulators. These trends are driven by:
- High repair costs associated with intelligent systems
- Complex Autopilot-related accident factors
- Heterogeneous driver behaviors and vehicle usage scenarios

**This poses increased underwriting and payout risks for insurers**. The project proposes building a **machine learning-based dynamic risk prediction model** to modernize the traditional actuarial approach, leveraging data-driven modeling to quantify risk more accurately.

### Objective:

1. Identify high-risk vehicle models, driving behavior patterns and predict the likelihood of claims over a defined period.
2. Provide quantitative basis for dynamic pricing and risk stratification.

### Business Applications:

💰 Risk-based Premium Adjustment
Adjust insurance pricing based on predicted claim probabilities

🛡️ Tailored Underwriting Rules
Define acceptance thresholds or policy conditions based on risk profiles

🚘 Proactive Behavioral Interventions
Identify high-risk drivers early and offer training, incentives, or personalized feedback to reduce claim likelihood

---

## 📊 Dataset Overview

**400,000+** training data, **178,000+** testing data for prediction.

The dataset contains rich, multi-dimensional features covering **driver profiles**, **vehicle specifications**, **regional context**, and **driving behavior**, structured as follows:

### 🧍 Individual Features (`ps_ind_*`)

| Feature         | Description |
|----------------|-------------|
| `ps_ind_01`     | Driver age group (ordered, e.g., 18–25, 26–35...) |
| `ps_ind_02_cat` | License type (categorical: C1, C2, A2, etc.) |
| `ps_ind_03`     | Driving experience (years) |
| `ps_ind_04_cat` | Occupation risk level (e.g., white-collar, high-risk) |
| `ps_ind_06_bin` | Prior accident history (0 = No, 1 = Yes) |
| `ps_ind_07_bin` | Has Tesla dashcam installed (0/1) |
| `ps_ind_16_bin` | Participated in Tesla safe driving training (0/1) |

---

### 🚗 Vehicle Features (`ps_car_*`)

| Feature         | Description |
|----------------|-------------|
| `ps_car_01_cat` | Vehicle model (e.g., Model 3, Y, S, X) |
| `ps_car_03_cat` | Battery type (e.g., LFP, NCM/NCA) |
| `ps_car_05_cat` | Charging method (home charger, Supercharger, 3rd-party) |
| `ps_car_11_cat` | Battery health score (0–100) |
| `ps_car_12`     | Avg. charging time per session (hours) |
| `ps_car_15`     | Autopilot usage frequency (avg. daily minutes) |

---

### 🌍 Regional Features (`ps_reg_*`)

| Feature         | Description |
|----------------|-------------|
| `ps_reg_01`     | Climate risk index of the registered location (e.g., high rain/snow) |
| `ps_reg_02`     | Traffic congestion index (continuous) |
| `ps_reg_03`     | EV charging station density (chargers per km²) |

---

### 🛣️ Driving Behavior Features (`ps_calc_*`)

| Feature         | Description |
|----------------|-------------|
| `ps_calc_01`     | Aggressive driving score (frequency of harsh braking/acceleration) |
| `ps_calc_04`     | Nighttime driving ratio (share of total driving time) |
| `ps_calc_10`     | Long-distance trip frequency (e.g., monthly trips >300km) |
| `ps_calc_15_bin` | Sentry Mode enabled (0/1) |

---

## 🧩 Feature Groups

To streamline feature engineering and modeling, the dataset was grouped into logical feature domains:

- **Individual Attributes**: age, license type, driving history  
- **Vehicle Characteristics**: model, battery type, charging habits, Autopilot usage  
- **Regional Attributes**: environmental and infrastructure context  
- **Behavioral Indicators**: aggressive driving patterns, night driving, sentry mode  
- **Interaction Features**: engineered terms such as `traffic congestion × autopilot usage frequency`, `battery health × aggresive driving score`

### 🛠 Feature Processing Techniques

- **Encoding**
  - One-Hot Encoding for low-cardinality categorical variables
  - Target Encoding for high-cardinality categorical features
- **Correlation Filtering**
  - Visualized pairwise correlations
- **Interaction Construction**
  - Filter numerical features with correlation coefficient(>=0.02) and create interactive features between them
  - Example: `Autopilot usage frequency × region congestion level`
- **Feature Selection**
  - Used shallow XGBoost with strong regularization (high λ, γ)
  - Selected features with both **gain** and **weight** above the median

---

## 🚀 Quick Start

To get started quickly, follow these steps:


#### 1. Clone the repository

```bash
git clone https://github.com/Feiyingdai/Car-Insurance-Claim-Prediction-Final.git
cd Car-Insurance-Claim-Prediction
```


#### 2. Install required packages

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install pandas numpy scikit-learn xgboost optuna shap matplotlib seaborn
```

#### 3. Run the notebook

Open the notebook using Jupyter or Colab:

```bash
jupyter notebook notebooks/car_insurance_claim_prediction.ipynb
```


#### 4. Use the sample dataset

A small sample dataset `sample_data.csv` (100 rows) is included in the `data/` folder for testing and quick experimentation.

```python
# Load sample data
import pandas as pd
df = pd.read_csv('data/sample_data.csv')
df.head()
```
---

## 📦 Project Phases & To-Do List

### ✅ Completed

- [x] Dataset understanding and schema review
- [x] Data cleaning and type formatting
- [x] Exploratory data analysis (EDA)
- [x] Feature engineering and interaction terms
- [x] Feature selection via regularized XGBoost
- [x] Imbalance handling (SMOTE / iterative undersampling / scale_pos_weight)
- [x] Model training with XGBoost
- [x] Optuna-based hyperparameter tuning (10-fold CV)
- [x] Threshold optimization for risk segmentation
- [x] SHAP-based model interpretation
- [x] Business Insights

### 📝 To-Do (Potential Future Work)

- [ ] Compare other models (e.g., Logistic Regression, LightGBM)
- [ ] Model ensemble and stacking
---

## 🧹 Data Cleaning

The raw dataset underwent extensive preprocessing to improve quality and ensure modeling readiness. Key steps are outlined below:

### 🧼 Missing Value Handling

- **Missing Value Matrix & Missing Value Heatmap**
  - Use missing value matrix and heatmap to see if these  data are in **MCAR(missing at random)**  or **MAR(missing at random)** pattern.
    <img width="1066" height="629" alt="image" src="https://github.com/user-attachments/assets/ae0042c1-d344-47ce-8439-7a38b4bbb894" />

    <img width="737" height="612" alt="image" src="https://github.com/user-attachments/assets/da773ed9-7785-4b32-a2f6-06ef5699b748" />

- **Numerical Variables:**
  - 'ps_reg_03' and 'ps_car_14', 'ps_car_11', 'ps_car_12' seems **MCAR** since the correlation coeffient in missing value heatmap is almost 0, we will impute them with mean or medium based on their data distribution.
  - Median imputation was applied to these missing variables since they had skewed distributions.
<img width="1201" height="593" alt="image" src="https://github.com/user-attachments/assets/c9d7197b-ff17-4978-bbc7-a348206e7d52" />


- **Categorical Variables:**
  - 'ps_car_03_cat','ps_car_05_cat' had over 40% missing values, so we'll drop them first.
  - 'ps_car_07_cat','ps_ind_05_cat', 'ps_car_09_cat', 'ps_ind_02_cat', 'ps_car_01_cat', 'ps_ind_04_cat', we’ll keep -1 as a **separate category**, as EDA showed:
    
    - Missing values had **non-random patterns**
    - Their presence was predictive of **higher/lower claim risk**
<img width="1205" height="594" alt="image" src="https://github.com/user-attachments/assets/47d28298-5ba2-40e2-975d-fa190be5ed99" />

  - For the column 'ps_car_02_cat', it seems MCAR type, we'll **impute using the mode**.



### 🔍 Outlier Detection & Clipping

- Used the **IQR method** to clip extreme values in:
  - `ps_car_12` (charging time per session)
  - `ps_car_15` (Autopilot usage frequency)
- This avoided distortion of downstream feature scaling and model training.


### ⚖️ Initial Class Imbalance Check

- The target variable (`claim_filed`) was highly imbalanced (~3% positive class).
<img width="439" height="376" alt="image" src="https://github.com/user-attachments/assets/e17e5ef6-b51f-404d-a589-5fc03fec3ae8" />


- Multiple strategies were later evaluated:
  - **`scale_pos_weight`**
  - **Borderline-SMOTE**
  - **Iterative undersampling(best performer)**

---
## 📌 Exploratory Data Analysis (EDA)

We performed a thorough exploratory analysis to better understand data distribution, and explore relationships between key features and the target variable.

**Key EDA Highlights:**
To understand feature-target relationships and identify predictive patterns, we visualized:

- **Continuous features** using boxplots, grouped by target (claim filed vs not)
- **Categorical features** using aggregated claim rates per category level, by calculating **claim rate by category**, we avoided repeating the 0/1 bars and enhanced interpretability for categorical features.

### 🔍 Key Observations:
1. An **unexpected pattern** was observed with `ps_ind_01` (Driver Age), `ps_ind_03` (Driving Experience): claimants appear to have **higher driver age and driving experience values** on average.
This may suggest:
- Experience is not linearly related to safe driving
- More experienced drivers may take more risks or drive longer distances

2. Also **contrary to expectations**, `ps_car_15` (Autopilot usage) shows **slightly higher values among claimants**.
This observation suggests:
- Heavier reliance on Autopilot may **not reduce claim risk**, and might even be associated with riskier driving contexts
- Possible explanations include:
  - Overconfidence when using semi-autonomous systems
  - More highway driving exposure (where Autopilot is typically used)

3. Tesla Dashcam installation（`ps_ind_07_bin = 1`）shows **a higher claim rate** among users who have installed the device.
This suggests:
- Dashcam-equipped drivers are more likely to file a claim
- Possible explanations include:
  - Drivers with dashcams may be more assertive in reporting incidents, knowing they have video evidence
  - Dashcam installation could be more common among risk-conscious or previously involved drivers
  - Alternatively, it might reflect reverse causality: drivers with prior incidents tend to install dashcams afterward

4.Interestingly, `ps_ind_06_bin` (accident history) shows **lower claim rates** for users with prior accident records.
This may reflect:
- **Behavioral correction**: Drivers with prior incidents may drive more cautiously afterward
- **Claims aversion**: Experienced claimants might avoid filing for minor damages due to higher premiums or claim fatigue
- **Policy design effects**: Insurers may impose stricter rules (e.g., higher deductibles) on drivers with prior accidents, reducing future claims

5. Comletelt safety training (`ps_ind_16_bin = 1`) show lower claim rates
   
6. High traffic congestion(`ps_reg_02`) show higher claim rates

<img width="1207" height="605" alt="image" src="https://github.com/user-attachments/assets/60ed6a98-20a9-4432-b375-a6a169781197" />


<img width="1192" height="617" alt="image" src="https://github.com/user-attachments/assets/07fcea5b-9f67-4966-9cdf-5624975dcd62" />


### 🔍 Several counter-intuitive insights emerged during EDA：
- Highlight the **limitations of linear modeling**, where such non-monotonic or interaction-driven effects are difficult to capture.
- As a result, we chose to build a **tree-based model (XGBoost)** to automatically learn complex **non-linear relationships** and **feature interactions**, rather than assuming a linear form.

---


## 🧩 Feature Engineering & Selection

Feature engineering focused on encoding, interaction, and dimensionality reduction.

### 🧩 Encoding Strategy

- **XGBoost Native Categorical Handling (`enable_categorical=True`)**: Applied to low-cardinality categorical variables
- **Target Encoding**: Used for high-cardinality features `ps_car_11_cat`(battery health score)
   - To **prevent data leakage and avoid overfitting**, we used:
    - **10-fold cross-validation**: computed encodings from out-of-fold data
    - **Smoothing**: blended category mean with global mean based on frequency to avoid overfitting for rare categories

### 📊 Correlation Heatmap
As part of exploratory analysis, we plotted a correlation heatmap to detect highly linear relationships between features.

While this provides useful insights into data structure (e.g., identifying pairs like `ps_reg_03` and `ps_reg_02` with corr > 0.7), we did **not rely on it for feature elimination**.
<img width="982" height="865" alt="image" src="https://github.com/user-attachments/assets/5cecbedc-ad47-46b5-bd0a-c88191c0f472" />

- Correlation matrices—especially Pearson or even Spearman—are limited to capturing linear or monotonic patterns. They may miss nonlinear redundancy or overstate the relationship between unrelated variables.  
- Therefore, final feature selection was based on **model-based importance** rather than correlation alone.



### 🧬 Interaction Features

- Engineered combinations for numerical features like:
  - `Autopilot usage × region congestion`
  - `Autopilot usage × driving age`

### 🪓 Feature Selection Process
-  **Variance Filtering**
  - Removed features with **near-zero variance**
  - These features contribute little to model differentiation and often introduce noise

- Used a **shallow XGBoost model with strong regularization** (high λ, γ) to evaluate feature importance.

  **Selection Criteria:**
  - `Gain > median(gain)`
  - `Weight > median(weight)`
  - Final model retained **122 features** based on intersection of both


## 🧠 Model Building
We trained an XGBoost model to predict claim probability using individual, vehicle, regional, and behavioral features. Key steps include:

### Class Imbalance Handling
The dataset was highly imbalanced, with only ~3% positive class. We experimented with:
- scale_pos_weight (XGBoost internal re-weighting)
- Borderline-SMOTE (oversampling near decision boundaries)
- ✅Iterative Undersampling (best result)
  
### Iterative Undersampling Strategy:
**Steps:**
- Train multiple models on different undersampled subsets, for each base model:
  - Use all positive samples
  - Randomly sample a 1:2 ratio of negative samples
- Repeat this process iteratively until all negative samples are used across the ensemble
- Train a separate model on each subset
- Final predictions are averaged across all models

**Benefits:**
- Preserves full signal from minority class, without introducing synthetic examples- ensuring the real data distribution is maintained
- Provides balanced training sets to reduce model bias
- Helps prevent overfitting and increases model robustness via ensemble diversity

### Hyperparameter Tuning
- Used Optuna for 10-fold cross-validation to tune max_depth, n_estimators, learning_rate, etc.
- Regularization (lambda, gamma) applied to prevent overfitting
  
---

## 📊 Model Evaluation

### 🔍 Model Performance
The final XGBoost model was trained on iteratively undersampled data (1:2 positive-to-negative sampling, rotating through the entire negative class) to preserve the full signal of positive (claim) cases without introducing synthetic noise.
- Cross Validation AUC: ~0.64 
  
  <img width="602" height="449" alt="image" src="https://github.com/user-attachments/assets/60c7599d-7ce4-4b20-baea-ad0c61f310f1" />

- Test AUC: ~0.70
  
  <img width="574" height="444" alt="image" src="https://github.com/user-attachments/assets/05bf77e0-d5a9-479e-abb1-a5909a35a66d" />

  
This indicates a reasonably strong ability to rank drivers by claim risk, despite severe class imbalance in the raw data.

### 🎯 Threshold-Based Risk Segmentation
To convert predicted probabilities into actionable business segments, thresholds were tuned using the cross validation set to:
- Ensure high recall for claimants (minimize missed high-risk individuals)
- Maintain a realistic distribution across Low / Medium / High risk groups

**Precision & Recall vs Threshold Plot on CV set**

<img width="581" height="453" alt="image" src="https://github.com/user-attachments/assets/accf8e74-2178-4261-9eb8-ec644ca5cb36" />


**Precision & Recall vs Threshold Table on CV set**

<img width="582" height="551" alt="image" src="https://github.com/user-attachments/assets/65846926-1a38-4c08-8ec8-cbbc3e499bac" />



By balancing the recall vs. operational cost tradeoff, we set:
**High-Risk Threshold (0.36)**
**Low-Risk Threshold (0.23)**


| Metric                      | CV Set                  | Testing Set                |
| --------------------------- | ---------------------------- | -------------------------- |
| **AUC**                        | **0.64**                         | **0.70**                       |
| **Medium + High Risk Users [0.23, 1]** | **82% of users, Recall = 92%** | **85% of users, Recall = 96.5%**|
| - High-Risk Users [0.36, 1] | 29% of users, Recall = 47% | 37% of users, Recall = 65% |
| - Medium-Risk Users [0.23, 0.36) | 53% of users, Recall = 45% | 48% of users, Recall = 31.5%|
| Low-Risk Users [0,0.23)  | 18% of users, Recall = 8%   | 15% of users, Recall = 3.5% |

In testing dataset, **96.5% of all claimants** were captured in the **Medium and High risk groups**, suggesting that the Low-Risk group is truly low-risk and well suited for business strategies like faster approval or preferential pricing.

### ✅ Key Insights
- Iterative undersampling allowed efficient use of all negative samples while fully retaining rare positive cases.
- The model supports actionable segmentation, enabling risk-based policy adjustments.
- Thresholding strategy is business-aligned: it balances customer volume with risk sensitivity.

---


## 🔍 SHAP Feature Importance

To better understand the final XGBoost model's decision-making process, we computed SHAP (SHapley Additive exPlanations) values on the holdout dataset.

### 🔧 Why SHAP?

Traditional feature importance metrics like **Gain** or **Weight** are biased in the presence of multicollinearity — where only the first selected variable captures the importance.  
SHAP overcomes this by **fairly distributing contributions** across correlated features and explaining the **marginal contribution** of each feature to the prediction.

### 📈 SHAP Summary Plot
Note: Due to privacy constraints, full feature names have been masked or abbreviated.

Most important features are shown below, ranked by average absolute SHAP value:

<img width="846" height="877" alt="image" src="https://github.com/user-attachments/assets/5872d384-d60d-4bc7-b07c-09731237d0cc" />

- ps_ind_05_cat and `ps_car_01_cat` (car attributes) are among the most predictive.
- A composite encoded feature combining `ps_car_11_cat`(battery health) and `ps_car_13` shows significant importance, highlighting the benefit of targeted feature engineering.
- ps_ind_17_bin (binary behavioral feature) and ps_ind_07_bin (Tesla Dashcam usage) also contributed substantially to the model’s prediction. Model shows Dashcam-equipped drivers are more likely to file a claim, supporting the finding from EDA.

---  

## 💼 Business Interpretation
Based on the predicted claim probabilities and segmented risk thresholds, we identified three distinct customer groups:
- High Risk (≥ 0.36): ~29% of customers
- Low Risk (< 0.23): ~18% of customers
- Medium Risk (in-between): Remaining ~53%
  
By tailoring insurance strategies to these segments, the company can optimize underwriting decisions, balance portfolio risk, and improve operational efficiency.

### 🟥 High-Risk Customers
These customers are significantly more likely to file a claim. To mitigate financial risk, insurers can:

**Goal**:Reduce future claim exposure by proactively managing the riskiest segment.

**Action**：
- Cap maximum coverage limits: Prevent large losses by limiting payout ceilings.
- Set higher deductibles: Shift small-claim risk to customers and discourage frequent minor claims.
- Restrict optional coverages: Limit access to high-cost add-ons such as rental car reimbursement, windshield coverage, etc.
- Apply stricter underwriting rules: Consider conditional acceptance, higher premiums, or require additional documentation.

### 🟨 Medium-Risk Customers
This group represents a large portion of the portfolio with moderate claim probability. Strategies should focus on risk management and behavior improvement:

**Goal**: Shift medium-risk customers to a lower risk group through targeted interventions.

**Action**：
- Usage-based insurance (UBI): Dynamically price premiums based on real-time driving behavior.
- Incentivized safety programs: Offer discounts for completing safety training or defensive driving courses.
- Customized engagement: Monitor over time and adjust pricing or policies as behaviors change.

### 🟩 Low-Risk Customers
Customers predicted to have very low claim probability. 

**Goal**: Retain valuable customers and offer rewards.

**Action**:
- Offer loyalty incentives: Discounts, extended coverage, or rewards for claim-free history.
- Bundle products: Cross-sell life, home, or umbrella insurance to improve customer lifetime value.
- Fast-track underwriting: Simplify approval and offer streamlined digital experiences.


---

## 📦 Tech Stack

Language: Python 3.10+

Modeling: XGBoost 1.7+

Tuning: Optuna 3.0+

Libraries: pandas, scikit-learn, seaborn, matplotlib

---

## 🚫 Notes

Dataset not uploaded due to privacy constraints.

A small sample dataset **sample_data.csv** is available in data/ folder for reproducibility.

---

### 📄 License & Contribution

**📝 License**

MIT License

**👥 Contributors**

This project was developed by Feiying Dai, as part of my personal portfolio to demonstrate risk modeling in real-world insurance applications.

