# Air Quality Forecasting in Vancouver (PM2.5)

**Author:** Isaac Castillo Reyes  
**Course:** Special Topics in Data Analytics – April 2025  

This project forecasts PM2.5 pollution levels in Vancouver using weather, air quality, and wildfire data. Two models are used: **SARIMAX** (a time-series statistical model) and **XGBoost** (a machine learning model).

---

## Project Structure

```plaintext
├── Final_Dataset.csv             # Final cleaned dataset with features
├── sarimax_prediction.png       # PM2.5 forecast from SARIMAX
├── xgboost_prediction_fixed.png # PM2.5 forecast from XGBoost
├── xgboost_feature_importance.png # Feature importance from XGBoost
├── parameter_*.csv              # Monthly air quality parameter files
├── YVR_2024.csv                 # Weather data from NOAA
├── Fire_bc.csv                  # Fire events from VIIRS/NASA
├── scripts/                     # Your modeling and preprocessing scripts
└── README.md                    # This file
```

---

## Project Objective

The goal is to predict hourly PM2.5 levels in Vancouver for the year 2024, leveraging:
- Weather features (temperature, wind, dew point, sea-level pressure)
- Air pollutant levels (CO, NO₂)
- Wildfire activity (via FRP and fire event flags)

---

##  Data Sources

- **Weather Data**: NOAA (YVR station)
- **Air Quality**: OpenAQ (PM2.5, CO, NO₂)
- **Wildfires**: NASA VIIRS FIRMS (BC region)

---

## Data Preprocessing

1. **Pivot** monthly air quality parameters into tabular format
2. **Merge** weather and pollution data using timestamps
3. **Convert** temperature from °F to °C and normalize wind/dew/pressure
4. **Flag fire events** based on bounding box and acquisition dates
5. **Handle missing values** with interpolation
6. **Extract time features**: hour, day, month, etc.

Final dataset includes:
- `TMP`, `WND`, `CIG`, `DEW`, `SLP`, `co`, `no2`, `pm25`
- `hour`, `month`, `fire_event`

---

##  Models

### 1. SARIMAX (Seasonal ARIMA with Exogenous Variables)
- Models time-series patterns with external inputs
- Seasonal order: `(1, 0, 1, 24)` (captures daily seasonality)
- Exogenous variables: temperature, pressure, CO, NO₂, fire events
- Slight overfitting noted; performed better on training than test

### 2. XGBoost
- Gradient Boosted Decision Trees for regression
- Trained on full feature set with lagged features and fire flags
- Performed consistently on both training and test data
- Feature importance showed fire events had strong predictive power

---

## Results

- **SARIMAX vs Actual:**  
  ![SARIMAX](sarimax_prediction.png)

- **XGBoost vs Actual:**  
  ![XGBoost](xgboost_prediction_fixed.png)

- **Feature Importance (XGBoost):**  
  ![Importance](xgboost_feature_importance.png)

**Performance Summary:**

| Model     | Train MAE | Test MAE | Train RMSE | Test RMSE |
|-----------|-----------|----------|------------|-----------|
| SARIMAX   | 8.97      | 12.34    | 2.99       | 3.89      |
| XGBoost   | 2.31      | 3.38     | 3.21       | 3.76      |

*(Values above are examples — replace with your actual printed values.)*

---

## Running the Code

Make sure to install required libraries:
```bash
pip install pandas numpy matplotlib seaborn statsmodels xgboost scikit-learn
```

Run the script in order or use a Jupyter Notebook for modular execution.  
Ensure all CSV files (monthly params, fire, weather) are in the same directory.

---

## Key Learnings

- SARIMAX is strong for interpreting time-based trends but may overfit
- XGBoost generalizes better and captures nonlinear effects
- Fire events significantly impact PM2.5 spikes
- Feature engineering (time lags, fire flags) improves prediction accuracy

---
