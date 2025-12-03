import pandas as pd
import numpy as np
import os
import joblib
import json
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

# ==========================================================
# CONFIG
# ==========================================================
DATA_DIR = "../data/processed"
RAW_MERGED_FILE = "merged_preprocessed.csv"
TRAIN_FILE = "train_dataset.csv"
TEST_FILE = "test_dataset.csv"

MODEL_DIR = "../models"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# ==========================================================
# Helper: build features (must match app/CpiForecast.py)
# ==========================================================
def build_features(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Given df with columns DATE, CPI_VALUE, EXCHANGE_RATE, INTEREST_RATE,
    compute lags (1,3,6) and rolling (3) features and return a new df.

    IMPORTANT: This logic must match the one used in `app/CpiForecast.py`
    so that the Streamlit app and this training script use identical
    features.
    """
    df = df_in.copy().sort_values("DATE").reset_index(drop=True)

    lags = [1, 3, 6]
    rolls = [3]

    # CPI features
    for lag in lags:
        df[f"CPI_lag_{lag}"] = df["CPI_VALUE"].shift(lag)
    for w in rolls:
        # shift(1) so rolling does not include the current row
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


# ==========================================================
# 1. Load merged preprocessed dataset
# ==========================================================
merged_path = os.path.join(DATA_DIR, RAW_MERGED_FILE)
print(f"Loading merged dataset: {merged_path}")

if not os.path.exists(merged_path):
    raise FileNotFoundError(
        f"File '{merged_path}' not found. "
        f"Run 'scripts/build_training_dataset.py' first."
    )

df_raw = pd.read_csv(merged_path, parse_dates=["DATE"])
df_raw = df_raw.sort_values("DATE").reset_index(drop=True)

print("Raw merged shape:", df_raw.shape)

# ==========================================================
# 2. Feature engineering (lags & rolling)
# ==========================================================
df_fe = build_features(df_raw)

# drop rows with NaNs from lags/rolls
df_fe = df_fe.dropna().reset_index(drop=True)
print("Post-feature-engineering shape:", df_fe.shape)

if df_fe.empty:
    raise RuntimeError(
        "After feature engineering, no rows remain. "
        "Need at least 6 months of data."
    )

# ==========================================================
# 3. Train / test split (time-based)
# ==========================================================
FEATURES = [
    "CPI_lag_1", "CPI_lag_3", "CPI_lag_6",
    "CPI_roll_3",
    "ER_lag_1", "ER_lag_3", "ER_lag_6",
    "ER_roll_3",
    "IR_lag_1", "IR_lag_3", "IR_lag_6",
    "IR_roll_3",
]
TARGET = "CPI_VALUE"

if not set(FEATURES).issubset(df_fe.columns):
    missing = [c for c in FEATURES if c not in df_fe.columns]
    raise RuntimeError(f"Missing engineered feature columns: {missing}")

n_rows = len(df_fe)
default_test_size = max(12, int(n_rows * 0.2))  # at least 1 year or 20%
test_size = min(default_test_size, n_rows // 3)  # keep some for train

if test_size == 0:
    raise RuntimeError("Not enough rows to create a test set.")

train_df = df_fe.iloc[:-test_size].copy()
test_df = df_fe.iloc[-test_size:].copy()

print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")
print(f"Train period: {train_df['DATE'].iloc[0]} → {train_df['DATE'].iloc[-1]}")
print(f"Test period:  {test_df['DATE'].iloc[0]} → {test_df['DATE'].iloc[-1]}")

# Save train/test datasets (for inspection / reuse)
train_path = os.path.join(DATA_DIR, TRAIN_FILE)
test_path = os.path.join(DATA_DIR, TEST_FILE)
train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)
print(f"Saved train dataset → {train_path}")
print(f"Saved test dataset  → {test_path}")

# ==========================================================
# 4. Build inputs
# ==========================================================
X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]

# ==========================================================
# 5. Scaling
# ==========================================================
scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
joblib.dump(scaler, scaler_path)
print(f"\nScaler saved → {scaler_path}")

# ==========================================================
# 6. Linear Regression
# ==========================================================
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

lr_pred = lr.predict(X_test_scaled)
lr_rmse = sqrt(mean_squared_error(y_test, lr_pred))
print("\nLinear Regression RMSE:", lr_rmse)

# ==========================================================
# 7. Random Forest
# ==========================================================
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    random_state=42,
)
rf.fit(X_train_scaled, y_train)

rf_pred = rf.predict(X_test_scaled)
rf_rmse = sqrt(mean_squared_error(y_test, rf_pred))
print("Random Forest RMSE:", rf_rmse)

# ==========================================================
# 8. Save best model
# ==========================================================
best_model = rf if rf_rmse < lr_rmse else lr
model_name = "RandomForest" if rf_rmse < lr_rmse else "LinearRegression"

model_path = os.path.join(MODEL_DIR, "best_model.joblib")
joblib.dump(best_model, model_path)
print(f"\nSaved best model → {model_name} @ {model_path}")

# ==========================================================
# 9. Save metadata.json (for Streamlit app)
# ==========================================================
metadata = {
    "feature_names": FEATURES,
    "target_column": TARGET,
    "date_column": "DATE",
    "model_used": model_name,
    "n_rows_raw": int(len(df_raw)),
    "n_rows_features": int(len(df_fe)),
    "n_train_rows": int(len(train_df)),
    "n_test_rows": int(len(test_df)),
    "source_file": RAW_MERGED_FILE,
}

metadata_path = os.path.join(MODEL_DIR, "metadata.json")
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nMetadata saved → {metadata_path}")
print("\nTraining complete!")

