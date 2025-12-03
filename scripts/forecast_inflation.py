import pandas as pd
import joblib
import numpy as np

# ==========================================================
# CONFIG
# ==========================================================
MODEL_PATH = "../models/best_model.joblib"
SCALER_PATH = "../models/scaler.joblib"
DATA_PATH = "../data/processed/merged_preprocessed.csv"


def build_features(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Same feature engineering used in training and Streamlit app:
    lags 1,3,6 and 3‑month rolling means with shift(1).
    """
    df = df_in.copy().sort_values("DATE").reset_index(drop=True)

    lags = [1, 3, 6]
    rolls = [3]

    # CPI features
    for lag in lags:
        df[f"CPI_lag_{lag}"] = df["CPI_VALUE"].shift(lag)
    for w in rolls:
        df[f"CPI_roll_{w}"] = df["CPI_VALUE"].shift(1).rolling(w).mean()

    # Exchange Rate features
    for lag in lags:
        df[f"ER_lag_{lag}"] = df["EXCHANGE_RATE"].shift(lag)
    for w in rolls:
        df[f"ER_roll_{w}"] = df["EXCHANGE_RATE"].shift(1).rolling(w).mean()

    # Interest Rate features
    for lag in lags:
        df[f"IR_lag_{lag}"] = df["INTEREST_RATE"].shift(lag)
    for w in rolls:
        df[f"IR_roll_{w}"] = df["INTEREST_RATE"].shift(1).rolling(w).mean()

    return df


FEATURES = [
    "CPI_lag_1", "CPI_lag_3", "CPI_lag_6",
    "CPI_roll_3",
    "ER_lag_1", "ER_lag_3", "ER_lag_6",
    "ER_roll_3",
    "IR_lag_1", "IR_lag_3", "IR_lag_6",
    "IR_roll_3",
]

# ================================
# LOAD MODEL AND SCALER
# ================================
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ================================
# LOAD LATEST MERGED DATA
# ================================
data = pd.read_csv(DATA_PATH)
data["DATE"] = pd.to_datetime(data["DATE"])
data = data.sort_values("DATE")

print(f"Using latest data for prediction: {data['DATE'].iloc[-1]}")

# ================================================
# BUILD FEATURES FOR PREDICTION
# ================================================
df = build_features(data)
df = df.dropna().reset_index(drop=True)

if df.empty:
    raise RuntimeError("Not enough rows after feature engineering to make a prediction.")

# ================================================
# SELECT THE LAST ROW AS PREDICTION INPUT
# ================================================
last_row = df.iloc[-1]

input_data = last_row[FEATURES].values.reshape(1, -1)

# Scale using training scaler
scaled_input = scaler.transform(input_data)

# ================================================
# PREDICT NEXT MONTH CPI
# ================================================
predicted_cpi = model.predict(scaled_input)[0]

print("\n===== INFLATION FORECAST =====")
print(f"Next Month CPI Prediction: {predicted_cpi:.3f}")

# Compute next month date
next_date = last_row["DATE"] + pd.DateOffset(months=1)
print(f"Forecast Date: {next_date.date()}")

