import os

import pandas as pd

# ------------------------------------------------------------------
# CONFIG – change these if you want to use different IMF series
# ------------------------------------------------------------------
RAW_DIR = "../data/raw"
OUTPUT_DIR = "../data/processed"

# CPI index (2010 = 100, all items, monthly)
CPI_SERIES_CODE = "PHL.CPI._T.IX.M"

# FX: **PHP per USD** (domestic currency per US dollar), period‑average
# If you instead want USD per PHP, switch to: "PHL.USD_XDC.PA_RT.M"
FX_SERIES_CODE = "PHL.XDC_USD.PA_RT.M"

# Interest rate: Monetary policy‑related rate, percent per annum
IR_SERIES_CODE = "PHL.MFS166_RT_PT_A_PT.M"


def _long_from_imf_row(df: pd.DataFrame, series_code: str, value_name: str) -> pd.DataFrame:
    """
    Helper to take a wide IMF row (2016-M01, 2016-M02, ...) and turn it into
    a long DataFrame with DATE + value_name.
    """
    row = df.loc[df["SERIES_CODE"] == series_code]
    if row.empty:
        raise ValueError(f"Series code '{series_code}' not found in file.")

    row = row.iloc[0]

    # Month columns look like "2016-M01", "2016-M02", ...
    month_cols = [c for c in df.columns if "-M" in c]
    series = row[month_cols]

    # Build proper month start dates
    dates = (
        pd.to_datetime(series.index.str.replace("-M", "-") + "-01", format="%Y-%m-%d")
    )

    out = pd.DataFrame(
        {
            "DATE": dates,
            value_name: pd.to_numeric(series.values, errors="coerce"),
        }
    ).sort_values("DATE")

    return out


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -------------------------------------------------------
    # 1. LOAD RAW DATASETS
    # -------------------------------------------------------
    print("📌 Loading raw datasets...")

    cpi_df = pd.read_csv(os.path.join(RAW_DIR, "cpi.csv"))
    fx_df = pd.read_csv(os.path.join(RAW_DIR, "ExchangeRates.csv"))
    ir_df = pd.read_csv(os.path.join(RAW_DIR, "InterestRates.csv"))

    # -------------------------------------------------------
    # 2. CLEAN CPI DATA  (CPI index, 2010 = 100)
    # -------------------------------------------------------
    print("📌 Cleaning CPI (index 2010=100)...")
    cpi_clean = _long_from_imf_row(cpi_df, CPI_SERIES_CODE, "CPI_VALUE")

    # -------------------------------------------------------
    # 3. CLEAN EXCHANGE RATES  (PHP per USD)
    # -------------------------------------------------------
    print("📌 Cleaning Exchange Rates (PHP per USD)...")
    fx_clean = _long_from_imf_row(fx_df, FX_SERIES_CODE, "EXCHANGE_RATE")

    # -------------------------------------------------------
    # 4. CLEAN INTEREST RATES  (policy rate)
    # -------------------------------------------------------
    print("📌 Cleaning Interest Rates (policy rate)...")
    ir_clean = _long_from_imf_row(ir_df, IR_SERIES_CODE, "INTEREST_RATE")

    # -------------------------------------------------------
    # 5. MERGE ALL DATASETS
    # -------------------------------------------------------
    print("📌 Merging datasets...")

    merged = (
        cpi_clean
        .merge(fx_clean, on="DATE", how="left")
        .merge(ir_clean, on="DATE", how="left")
    )

    # Add constant COUNTRY column
    merged["COUNTRY"] = "Philippines"

    # Reorder columns
    merged = merged[["COUNTRY", "DATE", "CPI_VALUE", "EXCHANGE_RATE", "INTEREST_RATE"]]

    # -------------------------------------------------------
    # 6. SAVE FINAL OUTPUT
    # -------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "merged_preprocessed.csv")
    merged.to_csv(output_path, index=False)

    print(f"\n✅ DONE! Saved merged dataset to:\n{output_path}")
    print("📊 Final dataset shape:", merged.shape)
    print(merged.head())


if __name__ == "__main__":
    main()
