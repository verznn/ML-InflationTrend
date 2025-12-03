# CpiForecast.py (Scenario Simulator version)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px


# -----------------------------
# PATHS (robust)
# -----------------------------
APP_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
LATEST_DATA_FILE = "merged_preprocessed.csv"

DATA_PATH = os.path.join(DATA_DIR, LATEST_DATA_FILE)
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "metadata.json")  # optional, fallback to FEATURES below

st.set_page_config(page_title="CPI Scenario Simulator", layout="wide")
st.title("CPI Scenario Simulator")
st.markdown(
    "Explore how changes in **exchange rates** and **interest rates** may shape "
    "Philippines CPI over the coming months, with forecasts, uncertainty bands, "
    "and scenario analysis to support financial planning."
)

# -----------------------------
# Helper: Build features (lags & rolling) – must match training
# -----------------------------
def build_features(df_in):
    """Given df with columns DATE, CPI_VALUE, EXCHANGE_RATE, INTEREST_RATE,
    compute lags (1,3,6) and rolling (3) features and return a new df.

    This logic MUST stay in sync with `scripts/train_models.py`.
    """
    df = df_in.copy().sort_values("DATE").reset_index(drop=True)

    lags = [1, 3, 6]
    rolls = [3]

    # CPI features
    for lag in lags:
        df[f"CPI_lag_{lag}"] = df["CPI_VALUE"].shift(lag)
    for w in rolls:
        # shift(1) so rolling doesn't include the current row (we want last months)
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


# -----------------------------
# Feature list (must match training)
# -----------------------------
FEATURES = [
    "CPI_lag_1", "CPI_lag_3", "CPI_lag_6", "CPI_roll_3",
    "ER_lag_1", "ER_lag_3", "ER_lag_6", "ER_roll_3",
    "IR_lag_1", "IR_lag_3", "IR_lag_6", "IR_roll_3"
]


# -----------------------------
# Forecast helpers: multi-step & uncertainty
# -----------------------------
def multi_step_forecast(df_hist, model, scaler, feature_cols, horizon):
    """
    Simple multi-step (1,3,6-month) CPI forecast.

    Assumptions:
    - Uses latest cleaned history (Philippines) as a starting point.
    - For horizons > 1, predicted CPI for each step is fed back in as history.
    - EXCHANGE_RATE and INTEREST_RATE are held constant at their latest values
      (or the user-simulated latest values) through the horizon.
    """
    if horizon < 1:
        raise ValueError("Horizon must be at least 1 month.")

    current_df = df_hist.copy().sort_values("DATE").reset_index(drop=True)
    base_cols = [
        c for c in current_df.columns
        if c not in feature_cols
    ]
    current_df = current_df[base_cols].copy()

    all_dates = []
    all_preds = []

    for _ in range(horizon):
        if current_df.empty:
            raise ValueError("Historical dataframe is empty.")

        last_row = current_df.iloc[-1]
        last_date = last_row["DATE"]
        next_date = last_date + pd.DateOffset(months=1)

        # create placeholder row for next month
        new_row = last_row.copy()
        new_row["DATE"] = next_date
        # placeholder CPI so row survives dropna; overwritten after prediction
        new_row["CPI_VALUE"] = last_row["CPI_VALUE"]
        current_df = pd.concat(
            [current_df, pd.DataFrame([new_row])],
            ignore_index=True,
        )

        df_fe = build_features(current_df)
        df_fe = df_fe.dropna().reset_index(drop=True)
        # select row corresponding to next_date
        next_rows = df_fe[df_fe["DATE"] == next_date]
        if next_rows.empty:
            raise ValueError("Could not build features for forecast horizon.")
        latest_row = next_rows.iloc[-1].copy()

        X = latest_row[feature_cols].values.reshape(1, -1)
        X_scaled = scaler.transform(X)
        y_pred = float(model.predict(X_scaled)[0])

        # update placeholder CPI with predicted value for future iterations
        current_df.loc[current_df["DATE"] == next_date, "CPI_VALUE"] = y_pred

        all_dates.append(next_date)
        all_preds.append(y_pred)

    return all_dates, all_preds


def estimate_rmse(model, scaler, df_eval, feature_cols, target_col="CPI_VALUE"):
    """
    Crude RMSE estimate for uncertainty bands.

    Uses in-sample error on the available (cleaned, featured) history.
    """
    cols_needed = list(feature_cols) + [target_col]
    for c in cols_needed:
        if c not in df_eval.columns:
            return None

    df_tmp = df_eval.dropna(subset=cols_needed).copy()
    if df_tmp.empty:
        return None

    X = df_tmp[feature_cols].values
    y_true = df_tmp[target_col].values

    try:
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)
    except Exception:
        return None

    if len(y_true) != len(y_pred):
        return None

    mse = np.mean((y_true - y_pred) ** 2)
    return float(np.sqrt(mse))

# -----------------------------
# Load data (logic)
# -----------------------------
if not os.path.exists(DATA_PATH):
    st.error(f"Data file not found at: {DATA_PATH}")
    st.stop()

df = pd.read_csv(DATA_PATH, parse_dates=["DATE"])
# ensure columns exist
required_cols = {"COUNTRY", "DATE", "CPI_VALUE", "EXCHANGE_RATE", "INTEREST_RATE"}
if not required_cols.issubset(set(df.columns)):
    st.error(f"Dataset missing required columns. Required: {required_cols}")
    st.stop()

# -----------------------------
# Load model & scaler
# -----------------------------
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.error("Model or scaler not found in models/. Run training script first.")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    st.error(f"Failed to load model/scaler: {e}")
    st.stop()

st.sidebar.success("Model & scaler loaded")

# Try to load metadata feature names if available
feature_names_meta = None
if os.path.exists(METADATA_PATH):
    try:
        meta = pd.read_json(METADATA_PATH, orient="index") if False else None
    except Exception:
        meta = None
if os.path.exists(METADATA_PATH):
    try:
        import json
        with open(METADATA_PATH, "r") as f:
            j = json.load(f)
        feature_names_meta = j.get("feature_names", None)
    except Exception:
        feature_names_meta = None

# If metadata feature names exist, validate they match our FEATURES
if feature_names_meta:
    # basic check
    if set(feature_names_meta) != set(FEATURES):
        st.warning("metadata.json feature_names differ from FEATURES in app. Using FEATURES defined in app.")
    else:
        # Use metadata order if provided (so scaler.feature names match)
        FEATURES = feature_names_meta

# -----------------------------
# Baseline feature engineering
# -----------------------------
df_fe = build_features(df)
# remove rows with NaNs created by lags/rolls
df_fe = df_fe.dropna().reset_index(drop=True)

if df_fe.empty:
    st.error("Not enough historical rows to compute lag/rolling features. Need at least 6 months of data.")
    st.stop()

# We'll focus on a single country (Philippines) — if the dataset contains multiple countries,
# filter to latest country or let user choose.
countries = df_fe["COUNTRY"].unique().tolist()
country = st.sidebar.selectbox("Select country (dataset)", countries, index=0)
df_country = df_fe[df_fe["COUNTRY"] == country].copy().reset_index(drop=True)

if df_country.empty:
    st.error(f"No data for country {country}")
    st.stop()

# Ensure sorted by DATE
df_country = df_country.sort_values("DATE").reset_index(drop=True)

# Forecast settings (horizon & alert)
st.sidebar.header("Forecast Settings")
horizon = st.sidebar.selectbox(
    "Forecast horizon (months ahead)",
    options=[1, 3, 6],
    index=0,
)
alert_threshold = st.sidebar.slider(
    "Inflation alert threshold (CPI level)",
    min_value=0.0,
    max_value=20.0,
    value=5.0,
    step=0.1,
)

# Validate all FEATURES present in dataset
missing = [c for c in FEATURES if c not in df_country.columns]
if missing:
    st.error(f"Model features missing from dataset after feature engineering: {missing}")
    st.stop()

# Estimate RMSE for (approximate) confidence intervals
rmse_est = estimate_rmse(model, scaler, df_country, FEATURES)
if rmse_est is None:
    st.info(
        "Could not estimate prediction uncertainty (RMSE). "
        "Confidence intervals will not be shown."
    )

# Baseline multi-step forecast (no simulation)
try:
    dates_base, preds_base = multi_step_forecast(
        df_country, model, scaler, FEATURES, horizon=horizon
    )
    pred_base = float(preds_base[-1])
    pred_date = dates_base[-1]
except Exception as e:
    st.error(f"Baseline forecast failed: {e}")
    st.stop()

lower_base = upper_base = None
if rmse_est is not None:
    std_h = rmse_est * np.sqrt(horizon)
    z = 1.96  # ~95% CI
    lower_base = pred_base - z * std_h
    upper_base = pred_base + z * std_h

# -----------------------------
# Scenario simulator inputs
# -----------------------------
st.sidebar.header("Scenario Simulator")
# show current/latest values for user context
current_exch = df_country["EXCHANGE_RATE"].iloc[-1]
current_ir = df_country["INTEREST_RATE"].iloc[-1]

st.sidebar.markdown("Adjust the **latest available month** (not the forecast month). The model will re-calculate features based on this adjusted latest row and produce a new next-month CPI forecast.")

# sliders ranges: reasonable defaults
exch_min = float(max(0.5 * current_exch, 0.0001))
exch_max = float(max(2.0 * current_exch, current_exch + 1.0))
ir_min = 0.0
ir_max = max(20.0, current_ir * 3)

sim_exch = st.sidebar.slider("Simulated Exchange Rate (PHP per USD)", float(exch_min), float(exch_max), float(current_exch), step=0.0001, format="%.6f")
sim_ir = st.sidebar.slider("Simulated Interest Rate (%)", float(ir_min), float(ir_max), float(current_ir), step=0.25, format="%.2f")

# Option to run sensitivity scan (ER only or IR only)
st.sidebar.markdown("### Sensitivity scan")
do_scan = st.sidebar.checkbox("Show sensitivity scan for Exchange Rate (ER) and Interest Rate (IR)", value=False)

# -----------------------------
# Build simulated dataset (copy history and replace last row ER/IR)
# -----------------------------
df_sim = df_country.copy().reset_index(drop=True)
df_sim.loc[df_sim.index[-1], "EXCHANGE_RATE"] = sim_exch
df_sim.loc[df_sim.index[-1], "INTEREST_RATE"] = sim_ir

# Scenario multi-step forecast (with simulated ER/IR path)
try:
    dates_sim, preds_sim = multi_step_forecast(
        df_sim, model, scaler, FEATURES, horizon=horizon
    )
    pred_sim = float(preds_sim[-1])
except Exception as e:
    st.error(f"Scenario forecast failed: {e}")
    st.stop()

lower_sim = upper_sim = None
if rmse_est is not None:
    std_h = rmse_est * np.sqrt(horizon)
    z = 1.96
    lower_sim = pred_sim - z * std_h
    upper_sim = pred_sim + z * std_h

# -----------------------------
# Display results
# -----------------------------
st.subheader("Inflation forecast for financial planning")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(
        label=f"Baseline CPI forecast ({horizon}-month ahead, {pred_date.strftime('%Y-%m')})",
        value=f"{pred_base:.3f}",
    )
with col2:
    st.metric(
        label=f"Scenario CPI forecast ({horizon}-month ahead, {pred_date.strftime('%Y-%m')})",
        value=f"{pred_sim:.3f}",
    )
with col3:
    delta = pred_sim - pred_base
    pct = (delta / pred_base * 100) if pred_base != 0 else np.nan
    st.metric(
        label="Scenario vs Baseline (Δ CPI)",
        value=f"{delta:+.3f}",
        delta=f"{pct:+.2f}%" if not np.isnan(pct) else "n/a",
    )

if rmse_est is not None and lower_base is not None:
    st.caption(
        f"Approx. 95% confidence interval for baseline {horizon}-month forecast: "
        f"{lower_base:.3f} – {upper_base:.3f} (based on in-sample RMSE ≈ {rmse_est:.3f})."
    )
    st.caption(
        f"Approx. 95% confidence interval for scenario {horizon}-month forecast: "
        f"{lower_sim:.3f} – {upper_sim:.3f}."
    )

# Alerts if inflation exceeds threshold
if pred_base >= alert_threshold or pred_sim >= alert_threshold:
    which = "Scenario" if pred_sim >= alert_threshold and pred_sim >= pred_base else "Baseline"
    exceed_val = pred_sim if which == "Scenario" else pred_base
    st.warning(
        f"{which} CPI forecast ({exceed_val:.2f}) exceeds the alert threshold "
        f"of {alert_threshold:.1f}. This may indicate elevated inflation risk "
        "for the selected horizon."
    )

# Impact analysis – simple comparison table
impact_df = pd.DataFrame(
    {
        "metric": [
            "current_EXCHANGE_RATE",
            "current_INTEREST_RATE",
            f"baseline_CPI_{horizon}m",
            f"scenario_CPI_{horizon}m",
            "delta_CPI",
            "delta_pct",
        ],
        "value": [
            current_exch,
            current_ir,
            pred_base,
            pred_sim,
            pred_sim - pred_base,
            pct,
        ],
    }
)
st.table(impact_df)

# Visual impact: forecast path (baseline vs scenario)
last_actual_date = df_country["DATE"].iloc[-1]
last_actual_cpi = df_country["CPI_VALUE"].iloc[-1]
path_dates = [last_actual_date] + dates_base
path_base = [last_actual_cpi] + preds_base
path_scenario = [last_actual_cpi] + preds_sim
path_df = pd.DataFrame(
    {
        "DATE": path_dates,
        "Baseline_CPI": path_base,
        "Scenario_CPI": path_scenario,
    }
)
fig_path = px.line(
    path_df,
    x="DATE",
    y=["Baseline_CPI", "Scenario_CPI"],
    title=f"{horizon}-month ahead CPI forecast path (baseline vs scenario)",
    color_discrete_map={
        "Baseline_CPI": "#1f77b4",  # blue
        "Scenario_CPI": "#d62728",  # red
    },
)
st.plotly_chart(fig_path, use_container_width=True)

# -----------------------------
# Sensitivity scan (optional)
# -----------------------------
if do_scan:
    st.subheader("Sensitivity scan — ER (varying) with IR fixed to simulated value (1‑month ahead)")
    # build a range around current_exch
    low = current_exch * 0.9
    high = current_exch * 1.1
    er_vals = np.linspace(low, high, 25)

    preds_er = []
    for v in er_vals:
        df_tmp = df_country.copy().reset_index(drop=True)
        df_tmp.loc[df_tmp.index[-1], "EXCHANGE_RATE"] = float(v)
        df_tmp.loc[df_tmp.index[-1], "INTEREST_RATE"] = float(sim_ir)  # keep IR at chosen sim
        df_tmp_fe = build_features(df_tmp).dropna().reset_index(drop=True)
        if df_tmp_fe.empty:
            preds_er.append(np.nan)
            continue
        x = df_tmp_fe.iloc[-1][FEATURES].values.reshape(1, -1)
        try:
            preds_er.append(model.predict(scaler.transform(x))[0])
        except Exception:
            preds_er.append(np.nan)

    scan_df = pd.DataFrame({"ER": er_vals, "Predicted_CPI": preds_er})
    fig_scan = px.line(scan_df, x="ER", y="Predicted_CPI", title="ER sensitivity (IR fixed)")
    st.plotly_chart(fig_scan, use_container_width=True)

    st.subheader("Sensitivity scan — IR (varying) with ER fixed to simulated value (1‑month ahead)")
    low_ir = max(0.0, current_ir * 0.9)
    high_ir = current_ir * 1.1 + 1.0
    ir_vals = np.linspace(low_ir, high_ir, 25)

    preds_ir = []
    for v in ir_vals:
        df_tmp = df_country.copy().reset_index(drop=True)
        df_tmp.loc[df_tmp.index[-1], "EXCHANGE_RATE"] = float(sim_exch)
        df_tmp.loc[df_tmp.index[-1], "INTEREST_RATE"] = float(v)
        df_tmp_fe = build_features(df_tmp).dropna().reset_index(drop=True)
        if df_tmp_fe.empty:
            preds_ir.append(np.nan)
            continue
        x = df_tmp_fe.iloc[-1][FEATURES].values.reshape(1, -1)
        try:
            preds_ir.append(model.predict(scaler.transform(x))[0])
        except Exception:
            preds_ir.append(np.nan)

    scan_ir_df = pd.DataFrame({"IR": ir_vals, "Predicted_CPI": preds_ir})
    fig_ir_scan = px.line(scan_ir_df, x="IR", y="Predicted_CPI", title="IR sensitivity (ER fixed)")
    st.plotly_chart(fig_ir_scan, use_container_width=True)

# -----------------------------
# Historical charts (main)
# -----------------------------
st.header("Historical data & context")
st.markdown(
    "Review past movements in **CPI**, **exchange rates**, and **interest rates**. "
    "These historical patterns underpin the model’s forecasts and help you see how "
    "recent trends compare with the projected inflation path."
)
left, right = st.columns(2)

with left:
    st.subheader("CPI (historic)")
    fig1 = px.line(df_country, x="DATE", y="CPI_VALUE", title="CPI Value")
    st.plotly_chart(fig1, use_container_width=True)

with right:
    st.subheader("Exchange Rate & Interest Rate (historic)")
    fig2 = px.line(df_country, x="DATE", y="EXCHANGE_RATE", title="Exchange Rate (PHP per USD)")
    fig3 = px.line(df_country, x="DATE", y="INTEREST_RATE", title="Interest Rate (%)")
    st.plotly_chart(fig2, use_container_width=True)
    st.plotly_chart(fig3, use_container_width=True)

# -----------------------------
# Data & model info / raw preview (bottom)
# -----------------------------
st.header("Data & model details")
st.sidebar.header("Data & Model")
st.write(f"Using cleaned dataset from: `{DATA_PATH}`")
with st.expander("Raw data preview (first / last 5 rows)"):
    st.write(df.head())
    st.write(df.tail())

# -----------------------------
# Allow user to download scenario row used for prediction
# -----------------------------
st.subheader("Download scenario input")
scenario_row = df_sim.iloc[-1][["DATE", "COUNTRY", "CPI_VALUE", "EXCHANGE_RATE", "INTEREST_RATE"]]
scenario_df = pd.DataFrame([scenario_row])
csv = scenario_df.to_csv(index=False)
st.download_button("Download scenario input CSV", data=csv, file_name="scenario_input.csv", mime="text/csv")