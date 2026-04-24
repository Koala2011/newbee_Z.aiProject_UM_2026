from Data_preprocessing import df
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Feature engineering

df=df.copy()
df["profit_rm"] = df["current_offer_rm"] - df["fuel_cost_rm"]

df["earning_efficiency"] = (
    df["current_offer_rm"] / df["task_duration_min"]
)

df["tip_ratio"] = (
    df["expected_tip_rm"] / df["current_offer_rm"]
)

df["traffic_distance_burden"] = (
    df["traffic_index"] * df["distance_km"]
)

df["demand_urgency_score"] = (
    df["demand_score"] * df["urgency_level"]
)

df["cost_burden_ratio"] = (
    df["fuel_cost_rm"] / df["current_offer_rm"]
)

# Remove inf values caused by division
df = df.replace([np.inf, -np.inf], 0)
df = df.fillna(0)



# Drop unnecessary / leakage columns
drop_cols = [
    "task_id",
    "date",

    # leakage columns for accept/reject model
    "suggested_price_rm",
    "net_income_rm",
    "hourly_income_rm",
    "best_work_period"
]

df_model = df.drop(columns=drop_cols,errors="ignore")

# Define X and y
X = df_model.drop(columns=["accept_recommendation"])
y = df_model["accept_recommendation"]

print("\nX shape before encoding:", X.shape)
print("y shape:", y.shape)

# Encode target
le = LabelEncoder()
y = le.fit_transform(y)

print("\nTarget classes:")
print(le.classes_)

print("\nEncoded target values:")
print(np.unique(y))

# One-hot encode categorical features
X = pd.get_dummies(X, drop_first=True)

print("\nX shape after encoding:", X.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SMOTE
from imblearn.over_sampling import SMOTE

smote = SMOTE(
    random_state=42
)

X_train_res, y_train_res = smote.fit_resample(
    X_train_scaled,
    y_train
)

print(pd.Series(y_train_res).value_counts())

# =========================
# MODEL 1: TASK SELECTION MODEL
# Classification
# Target: accept_recommendation
# =========================

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000, class_weight="balanced")

lr_model.fit(X_train_res, y_train_res)

# predict on TEST SET, not X_train_res
y_pred_lr = lr_model.predict(X_test_scaled)
y_prob_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

print("\n===== Logistic Regression Result =====")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_lr))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
print("Classification Report:\n", classification_report(
    y_test,
    y_pred_lr,
    target_names=[str(c) for c in le.classes_]
))


# Random Forest Classifier
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

rf_model.fit(X_train_res, y_train_res)

y_pred_rf = rf_model.predict(X_test_scaled)
y_prob_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

print("\n===== Random Forest Classifier Result =====")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(
    y_test,
    y_pred_rf,
    target_names=[str(c) for c in le.classes_]
))


# Feature Importance
rf_importance = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nTop 10 Features for Task Selection:")
print(rf_importance.head(10))

# =========================
# MODEL 2: SMART PRICING MODEL
# Regression
# Target: suggested_price_rm
# =========================

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df_price = df.copy()

drop_cols_price = [
    "accept_recommendation",
    "net_income_rm",
    "hourly_income_rm",
    "best_work_period"
]

df_price = df_price.drop(columns=drop_cols_price, errors="ignore")

X_price = df_price.drop(columns=["suggested_price_rm"])
y_price = df_price["suggested_price_rm"]

X_price = pd.get_dummies(X_price, drop_first=True)

X_train_price, X_test_price, y_train_price, y_test_price = train_test_split(
    X_price,
    y_price,
    test_size=0.2,
    random_state=42
)

price_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

price_model.fit(X_train_price, y_train_price)

y_pred_price = price_model.predict(X_test_price)

print("\n===== Smart Pricing Model Result =====")
print("MAE:", mean_absolute_error(y_test_price, y_pred_price))
print("RMSE:", np.sqrt(mean_squared_error(y_test_price, y_pred_price)))
print("R2 Score:", r2_score(y_test_price, y_pred_price))


price_importance = pd.Series(
    price_model.feature_importances_,
    index=X_price.columns
).sort_values(ascending=False)

print("\nTop 10 Features for Smart Pricing:")
print(price_importance.head(10))

# =========================
# MODEL 3: INCOME PREDICTION MODEL
# Regression
# Target: hourly_income_rm
# =========================

df_income = df.copy()

drop_cols_income = [
    "accept_recommendation",
    "suggested_price_rm",
    "net_income_rm",
    "best_work_period"
]

df_income = df_income.drop(columns=drop_cols_income, errors="ignore")

X_income = df_income.drop(columns=["hourly_income_rm"])
y_income = df_income["hourly_income_rm"]

X_income = pd.get_dummies(X_income, drop_first=True)

X_train_income, X_test_income, y_train_income, y_test_income = train_test_split(
    X_income,
    y_income,
    test_size=0.2,
    random_state=42
)

income_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

income_model.fit(X_train_income, y_train_income)

y_pred_income = income_model.predict(X_test_income)

print("\n===== Income Prediction Model Result =====")
print("MAE:", mean_absolute_error(y_test_income, y_pred_income))
print("RMSE:", np.sqrt(mean_squared_error(y_test_income, y_pred_income)))
print("R2 Score:", r2_score(y_test_income, y_pred_income))


income_importance = pd.Series(
    income_model.feature_importances_,
    index=X_income.columns
).sort_values(ascending=False)

print("\nTop 10 Features for Income Prediction:")
print(income_importance.head(10))