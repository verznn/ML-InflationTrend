# &#10132; CPI Scenario Simulator

An interactive Streamlit web application to explore **Philippines CPI forecasts** and perform **scenario analysis**. Designed to help **investors, policymakers, and households** make informed financial decisions by projecting inflation trends under various exchange rate (ER) and interest rate (IR) assumptions.

---

## &#10132; Features

- Select **forecast horizon**: 1, 3, or 6 months ahead  
- Adjust the **latest month’s ER and IR** to simulate scenarios  
- Shows **baseline vs scenario CPI forecasts** side-by-side  
- Alerts when CPI exceeds a **user-defined threshold**  
- Displays **approximate 95% confidence intervals**  
- Provides **impact tables and forecast path charts**  
- Optional **sensitivity scan** for ER and IR variations  
- Review **historical CPI, ER, and IR data** for context  
- Download the **scenario input row** as CSV  

---

## &#8258; About CPI Forecasting

CPI forecasts help users anticipate inflation trends to plan financial decisions. This app uses a **machine learning model** trained on historical CPI, ER, and IR data to produce:

- **Baseline CPI:** forecast using the latest observed data, assuming no changes in ER or IR  
- **Scenario CPI:** forecast after user-simulated changes in ER/IR  

### &#10132; Key Concepts:

- **CPI Alert Threshold:**  
  - A user-defined trigger for elevated inflation risk.  
  - Example uses:  
    - Policymakers: near 4% inflation target  
    - Households: personal cost-of-living thresholds (5–6%)  
    - Investors: historical highs or stress-test levels  
  - When the forecasted CPI ≥ threshold, a warning is shown.

- **Baseline vs Scenario CPI:**  
  - **Baseline:** Model projection with observed data.  
  - **Scenario:** Projection under user-adjusted ER/IR.  
  - The **delta (absolute & %)** shows how the scenario differs from baseline.  
  - Forecast path chart:  
    - Blue line = baseline CPI  
    - Red line = scenario CPI  
    - Highlights divergence over the selected horizon.

---

## &#8258; How to Run

1. **Prepare Your Environment**

   - Install **Python 3.8+**  
   - Install dependencies using pip:

     ```bash
     pip install streamlit pandas numpy plotly joblib
     ```

2. **Extract the Files**

   - Ensure the folder contains:  
     - `CpiForecast.py` (main app)  
     - `data/processed/merged_preprocessed.csv`  
     - `models/best_model.joblib` and `models/scaler.joblib`  
     - Optional: `models/metadata.json`  

3. **Run the Application**

   - Open a terminal or command prompt  
   - Navigate to the folder containing `CpiForecast.py`  
   - Run the app:

     ```bash
     streamlit run CpiForecast.py
     ```

4. **Use the Application**

   - A new browser tab opens with the CPI Scenario Simulator interface  
   - Adjust **forecast horizon**, **alert threshold**, and **scenario sliders**  
   - Review metrics, charts, and tables  
   - Optional: run **sensitivity scans**  
   - Download scenario input CSV if needed  

---

## &#8984; Preview / Procedures

### 1. Select forecast horizon, country, and CPI alert threshold

---

### 2. Adjust scenario sliders for ER and IR to simulate “what-if” conditions

---

### 3. Review baseline vs scenario metrics, forecast path chart, and impact table

---

### 4. Optional: Run sensitivity scan to see effect of ER or IR variations on CPI

---

### 5. Historical data charts provide context for past CPI, ER, and IR trends

---

## Built With

- [Streamlit](https://streamlit.io/) – Interactive web interface  
- [Plotly](https://plotly.com/) – Interactive charts  
- [Pandas](https://pandas.pydata.org/) – Data manipulation  
- [NumPy](https://numpy.org/) – Numerical operations  
- [Joblib](https://joblib.readthedocs.io/) – Model and scaler persistence  

---

## Members

**Rhiane Miguel Veron Dalumpines**  
**Nino Renzonald Driz**  
**John Caleb Restituto**  

CPE - 3A
