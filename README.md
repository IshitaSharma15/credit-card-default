# Credit Card Default Prediction

A machine learning project to predict whether a credit card client will default on their next payment, using the [UCI Credit Card Dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients) (Taiwan, 2005).

---

## Project Overview

Credit card default is a major financial risk for banks. This project builds and evaluates three machine learning models to identify customers likely to default, enabling early intervention such as credit limit adjustments or repayment assistance.

The dataset contains **30,000 customers** with 6 months of payment history, bill amounts, and demographic information. The target variable is whether the customer defaulted the following month.

---


---

## Dataset

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)
- **Rows:** 30,000 customers (29,601 after cleaning)
- **Features:** 23 input features + 3 engineered features = 28 total
- **Target:** `default.payment.next.month` (1 = defaulted, 0 = did not)
- **Class Distribution:** 77.7% non-default / 22.3% default

---



---

## Models Used

| Model | Type | Why Used |
|---|---|---|
| Logistic Regression | Linear classifier | Fast, interpretable baseline |
| Random Forest | Ensemble (Bagging) | Handles non-linearity, robust to outliers |
| XGBoost | Ensemble (Boosting) | State-of-the-art on tabular data, best overall performance |

---

## Feature Engineering

Three domain-knowledge features were created to capture credit risk signals not present in the raw data:

| Feature | Formula | What It Captures |
|---|---|---|
| `AVG_UTIL_RATIO` | mean(BILL_AMTi / LIMIT_BAL) | Average credit utilization — high ratio signals financial stress |
| `AVG_PAY_RATIO` | mean(PAY_AMTi / \|BILL_AMTi\|) | Fraction of bill actually paid — low ratio signals debt management issues |
| `DELAY_TREND` | PAY_1 − PAY_6 | Whether payment delays are worsening over time |

All three features appear in the top 15 most important features in both XGBoost and Random Forest, confirming they added genuine predictive value.

---

## Results

### Pre-Tuning (Default Hyperparameters + Optimised Threshold)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7936 | 0.5383 | 0.5269 | 0.5325 | 0.7309 | 0.5195 |
| Random Forest | 0.7904 | 0.5308 | 0.5216 | 0.5262 | 0.7621 | 0.5162 |
| XGBoost | 0.7759 | 0.4980 | 0.5519 | 0.5235 | 0.7456 | 0.5212 |

### After Hyperparameter Tuning + Threshold Optimisation

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7891 | 0.5265 | 0.5420 | 0.5341 | 0.7283 | 0.5196 |
| Random Forest | 0.7977 | 0.5468 | 0.5443 | 0.5455 | 0.7730 | 0.5475 |
| **XGBoost** ✅ | **0.8117** | **0.5908** | **0.5072** | **0.5458** | **0.7764** | **0.5548** |

**Best Model: XGBoost** — highest accuracy (81.2%), precision (59.1%), ROC-AUC (0.776), and PR-AUC (0.555).

---

## Feature Importance

Both XGBoost and Random Forest agree on the top predictors:

**XGBoost Top Features:** PAY_1 (dominant) → PAY_2 → PAY_3 → DELAY_TREND → PAY_4 → PAY_6 → PAY_5 → PAY_AMT2 → PAY_AMT4 → AVG_UTIL_RATIO

**Random Forest Top Features:** PAY_1 → PAY_2 → DELAY_TREND → PAY_3 → PAY_4 → AVG_UTIL_RATIO → PAY_5 → PAY_AMT1 → LIMIT_BAL → AVG_PAY_RATIO

**Key takeaways:**
- `PAY_1` (most recent payment status) is by far the strongest predictor in both models — recent behaviour predicts future default better than any demographic variable
- `DELAY_TREND` ranks 3rd–4th in both models, validating the feature engineering
- `AVG_UTIL_RATIO` consistently appears in the top 10, confirming that credit utilization is a meaningful risk signal
- Demographics (AGE, SEX, EDUCATION, MARRIAGE) do not appear in the top 15 of either model — payment behaviour matters far more than who the customer is

---

## Learning Curve Analysis

| Model | Diagnosis | Detail |
|---|---|---|
| Logistic Regression | **Underfitting** | Training and validation F1 both plateau at ~0.51 — the linear model is too simple for this problem |
| Random Forest | **Severe Overfitting** | Training F1 starts at 0.88, validation at 0.51 — large persistent gap even at full data size |
| XGBoost | **Moderate Overfitting** | Training F1 starts at 0.81, validation at 0.51 — gap is smaller than RF, validation curve more stable |

All three models plateau around F1 ≈ 0.55 on the validation set, suggesting more data alone will not significantly improve performance. The next step would be richer features (e.g. credit bureau data, spending categories) or a stacked ensemble.

---

## Key Findings

- Recent payment behaviour (`PAY_1`, `PAY_2`) dominates prediction — demographics have almost no predictive power
- XGBoost performs best because credit default risk is driven by **non-linear feature interactions** (e.g. high utilisation AND recent delays together signal extreme risk) that boosted trees capture naturally
- Threshold optimisation via Precision-Recall curve improved F1 across all models compared to the default 0.5 threshold

---
##   Getting Started

```bash
git clone https://github.com/IshitaSharma15/credit-card-default-prediction.git
cd credit-card-default-prediction
pip install -r requirements.txt
jupyter notebook Credit_Card_Default_Improved.ipynb
```

Download the dataset from the UCI Machine Learning Repository and place it in the project root directory before running the notebook.

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
joblib
jupyter
```

Install all at once:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib jupyter
```
---

## Limitations

- **Interpretability:** XGBoost is a black-box model. For regulated banking environments, SHAP values should be added to explain individual predictions
- **Temporal validity:** Dataset is from 2005 Taiwan. Economic conditions and consumer behaviour have changed significantly — the model would need retraining on recent data before any real-world deployment
- **Performance ceiling:** All models plateau around F1 ≈ 0.55 on the minority class. Richer features (credit bureau scores, transaction-level data) would be needed to push higher
- **PAY_1 over-reliance:** XGBoost places extreme weight on a single feature. If PAY_1 data is delayed or missing in production, model performance would degrade significantly
- **No deployment pipeline:** This is a study/research project — production use would require API wrapping, input validation, data drift monitoring, and periodic retraining

---

## Author

**ISHITA**  
[LinkedIn](https://www.linkedin.com/in/ishita-sharma-125671260) · [GitHub](https://github.com/IshitaSharma15)

---
