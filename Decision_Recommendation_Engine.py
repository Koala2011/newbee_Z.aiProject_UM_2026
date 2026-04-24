from Model_Training import (
    rf_model,
    price_model,
    income_model,
    scaler,
    X,
    X_price,
    X_income
)

import pandas as pd
import numpy as np


# =========================
# Prepare Single Task
# =========================
def prepare_single_task(task_row, model_columns, scaler=None):
    task_df = pd.DataFrame([task_row])

    # Feature engineering
    task_df["profit_rm"] = task_df["current_offer_rm"] - task_df["fuel_cost_rm"]

    task_df["earning_efficiency"] = (
        task_df["current_offer_rm"] / task_df["task_duration_min"]
    )

    task_df["tip_ratio"] = (
        task_df["expected_tip_rm"] / task_df["current_offer_rm"]
    )

    task_df["traffic_distance_burden"] = (
        task_df["traffic_index"] * task_df["distance_km"]
    )

    task_df["demand_urgency_score"] = (
        task_df["demand_score"] * task_df["urgency_level"]
    )

    task_df["cost_burden_ratio"] = (
        task_df["fuel_cost_rm"] / task_df["current_offer_rm"]
    )

    # Handle infinity / missing values
    task_df = task_df.replace([np.inf, -np.inf], 0)
    task_df = task_df.fillna(0)

    # One-hot encode
    task_df = pd.get_dummies(task_df, drop_first=True)

    # Align columns with model training columns
    task_df = task_df.reindex(columns=model_columns, fill_value=0)

    # Scale if needed
    if scaler is not None:
        task_df = scaler.transform(task_df)

    return task_df


# =========================
# Task Acceptance Decision
# =========================
def recommend_task_acceptance(task_row):
    task_processed = prepare_single_task(
        task_row,
        X.columns,
        scaler=scaler
    )

    accept_probability = rf_model.predict_proba(task_processed)[0][1]

    if accept_probability >= 0.75:
        decision = "Accept"
    elif accept_probability >= 0.50:
        decision = "Consider / Negotiate"
    else:
        decision = "Reject"

    return decision, accept_probability


# =========================
# Smart Pricing Decision
# =========================
def recommend_smart_price(task_row):
    task_processed = prepare_single_task(
        task_row,
        X_price.columns,
        scaler=None
    )

    suggested_price = price_model.predict(task_processed)[0]
    current_offer = task_row["current_offer_rm"]
    price_gap = suggested_price - current_offer

    if price_gap > 5:
        pricing_advice = "Negotiate higher price"
    elif price_gap < -5:
        pricing_advice = "Current offer is already good"
    else:
        pricing_advice = "Price is reasonable"

    return suggested_price, price_gap, pricing_advice


# =========================
# Income Prediction Decision
# =========================
def predict_income(task_row):
    task_processed = prepare_single_task(
        task_row,
        X_income.columns,
        scaler=None
    )

    predicted_hourly_income = income_model.predict(task_processed)[0]

    if predicted_hourly_income >= 40:
        income_level = "High income potential"
    elif predicted_hourly_income >= 25:
        income_level = "Moderate income potential"
    else:
        income_level = "Low income potential"

    return predicted_hourly_income, income_level


# =========================
# Full Decision Engine
# =========================
def full_task_recommendation(task_row):
    decision, accept_probability = recommend_task_acceptance(task_row)

    suggested_price, price_gap, pricing_advice = recommend_smart_price(task_row)

    predicted_income, income_level = predict_income(task_row)

    reasons = []

    earning_efficiency = task_row["current_offer_rm"] / task_row["task_duration_min"]
    cost_burden_ratio = task_row["fuel_cost_rm"] / task_row["current_offer_rm"]

    if earning_efficiency > 0.8:
        reasons.append("Good earning efficiency")

    if cost_burden_ratio > 0.35:
        reasons.append("Fuel cost is high compared to offer")

    if task_row["demand_score"] >= 75:
        reasons.append("High demand area")

    if task_row["traffic_index"] >= 70:
        reasons.append("Traffic risk is high")

    if price_gap > 5:
        reasons.append("Current offer appears underpriced")

    if predicted_income >= 40:
        reasons.append("Strong hourly income potential")

    final_output = {
        "acceptance_decision": decision,
        "accept_probability": round(accept_probability, 3),
        "suggested_price_rm": round(suggested_price, 2),
        "price_gap_rm": round(price_gap, 2),
        "pricing_advice": pricing_advice,
        "predicted_hourly_income_rm": round(predicted_income, 2),
        "income_level": income_level,
        "reasons": reasons
    }

    return final_output


# =========================
# WHAT-IF SCENARIO SIMULATOR
# =========================

def apply_what_if_scenario(task_row, scenario):
    simulated_task = task_row.copy()

    if "fuel_cost_change_pct" in scenario:
        simulated_task["fuel_cost_rm"] *= (1 + scenario["fuel_cost_change_pct"] / 100)

    if "current_offer_change_pct" in scenario:
        simulated_task["current_offer_rm"] *= (1 + scenario["current_offer_change_pct"] / 100)

    if "traffic_index_change_pct" in scenario:
        simulated_task["traffic_index"] *= (1 + scenario["traffic_index_change_pct"] / 100)

    if "demand_score_change_pct" in scenario:
        simulated_task["demand_score"] *= (1 + scenario["demand_score_change_pct"] / 100)

    if "task_duration_change_pct" in scenario:
        simulated_task["task_duration_min"] *= (1 + scenario["task_duration_change_pct"] / 100)

    return simulated_task