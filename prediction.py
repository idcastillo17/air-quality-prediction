import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # disables interactive GUI plotting

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# Load dataset
df_final_Dataset = pd.read_csv('Final_Dataset.csv')
df_final_Dataset = df_final_Dataset.sort_values('DATE')

# 1. Split the dataset
split_index = int(len(df_final_Dataset) * 0.8)
train = df_final_Dataset[:split_index]
test = df_final_Dataset[split_index:]

# Target and features for SARIMAX
y = train['pm25']
X = train[['TMP', 'WND', 'DEW', 'SLP', 'co', 'no2', 'fire_event']]
X_test = test[['TMP', 'WND', 'DEW', 'SLP', 'co', 'no2', 'fire_event']]


# Make sure test y has datetime index
y_test.index = pd.to_datetime(df_final_Dataset['DATE'].iloc[split_index:].values)

# Stationarity check
result = adfuller(y.dropna())
print("ADF Statistic:", result[0])
print("p-value:", result[1])

# ACF/PACF
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
plot_pacf(y, lags=30, ax=axes[0])
axes[0].set_title('PACF')
plot_acf(y, lags=30, ax=axes[1])
axes[1].set_title('ACF')
plt.tight_layout()
plt.savefig("acf_pacf_plot.png")
print("✅ ACF and PACF plots saved as 'acf_pacf_plot.png'")

# Clean NaNs for SARIMAX
valid_index = y.dropna().index.intersection(X.dropna().index)
y = y.loc[valid_index]
X = X.loc[valid_index]
y = y.reset_index(drop=True)
X = X.reset_index(drop=True)

# SARIMAX model
model = SARIMAX(
    endog=y,
    exog=X,
    order=(1, 0, 1),
    seasonal_order=(1, 0, 1, 24),
    enforce_stationarity=False,
    enforce_invertibility=False
)
model_fit = model.fit(disp=False)
print(model_fit.summary())

# Clean X_test
X_test = X_test.replace([np.inf, -np.inf], np.nan)
X_test = X_test.fillna(method='ffill').fillna(method='bfill')
X_test = X_test.reset_index(drop=True)

# SARIMAX Forecast
forecast = model_fit.predict(start=len(y), end=len(y) + len(y_test) - 1, exog=X_test)

# Plot SARIMAX
plt.figure(figsize=(10, 5))
plt.plot(y_test.index[:100], y_test.values[:100], label='Actual PM2.5')
plt.plot(y_test.index[:100], forecast.values[:100], label='Predicted PM2.5', linestyle='--')
plt.legend()
plt.title("SARIMAX Prediction vs Actual (PM2.5)")
plt.xlabel("Datetime")
plt.ylabel("PM2.5")
plt.tight_layout()
plt.grid(True)
plt.savefig("sarimax_prediction.png")
print("✅ Forecast plot saved as 'sarimax_prediction.png'")

# SARIMAX Evaluation
y_train_pred = model_fit.predict(start=0, end=len(y) - 1, exog=X)
train_mae = mean_absolute_error(y, y_train_pred)
test_mae = mean_absolute_error(y_test, forecast)
train_rmse = np.sqrt(mean_squared_error(y, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, forecast))
print(f"📊 Train MAE SARIMAX: {train_mae:.2f} | Test MAE SARIMAX: {test_mae:.2f}")
print(f"📊 Train RMSE SARIMAX: {train_rmse:.2f} | Test RMSE SARIMAX: {test_rmse:.2f}")



# ---------------------- XGBOOST ------------------------

# Features and Target for XGBoost
features = ['TMP', 'DEW', 'SLP', 'co', 'no2', 'fire_event', 'hour']
X = df_final_Dataset[features]
y = df_final_Dataset['pm25']
X_train_all, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train_all, y_test = y.iloc[:split_index], y.iloc[split_index:]

# Reset XGBoost y_test index to datetime
y_test.index = pd.to_datetime(df_final_Dataset['DATE'].iloc[split_index:].values)

# Validation split
val_split = int(len(X_train_all) * 0.9)
X_train = X_train_all.iloc[:val_split]
X_val = X_train_all.iloc[val_split:]
y_train = y_train_all.iloc[:val_split]
y_val = y_train_all.iloc[val_split:]

# XGBoost model
model = XGBRegressor(
    objective='reg:squarederror',
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=10,
    eval_metric="rmse",
    verbose=False
)

# Prediction
y_pred = model.predict(X_test)
y_train_pred = model.predict(X_train_all)

# XGBoost Evaluation
train_mae = mean_absolute_error(y_train_all, y_train_pred)
test_mae = mean_absolute_error(y_test, y_pred)
train_rmse = np.sqrt(mean_squared_error(y_train_all, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"✅ Train MAE: {train_mae:.2f} | Test MAE: {test_mae:.2f}")
print(f"✅ Train RMSE: {train_rmse:.2f} | Test RMSE: {test_rmse:.2f}")

# Plot XGBoost
plt.figure(figsize=(10, 5))
plt.plot(y_test.index[:100], y_test.values[:100], label='Actual PM2.5')
plt.plot(y_test.index[:100], y_pred[:100], label='Predicted PM2.5', linestyle='--')
plt.legend()
plt.title("XGBoost Prediction vs Actual (PM2.5)")
plt.xlabel("Datetime")
plt.ylabel("PM2.5")
plt.tight_layout()
plt.grid(True)
plt.savefig("xgboost_prediction_fixed.png")
print("✅ XGBoost prediction plot saved as 'xgboost_prediction_fixed.png'")

################################## Feaute importance###########################################
# ---------------------- Feature Importance Plot ------------------------

# Get feature importances
importance = model.feature_importances_
features_names = X.columns

# Create DataFrame for easy plotting
importance_df = pd.DataFrame({
    'Feature': features_names,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

# Plot
plt.figure(figsize=(8, 5))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title("XGBoost Feature Importance")
plt.xlabel("Relative Importance")
plt.tight_layout()
plt.grid(True)
plt.savefig("xgboost_feature_importance.png")
print("✅ XGBoost feature importance plot saved as 'xgboost_feature_importance.png'")

