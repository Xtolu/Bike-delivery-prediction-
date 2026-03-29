# Bike Sharing Demand Prediction - Intermediate ML Project
# This project predicts hourly bike rental demand using historical data.
# Features: weather, time, season. Models: Random Forest, XGBoost.
# Perfect for GitHub portfolio - includes EDA, modeling, evaluation.

# Step 1: Install required libraries (run in terminal)
# pip install pandas numpy scikit-learn matplotlib seaborn xgboost ydata-profiling

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Step 2: Load the dataset (download from Kaggle: https://www.kaggle.com/datasets/city-of-seattle/bike-sharing-dataset)
# Save as 'hour.csv' in your project folder
df = pd.read_csv('hour.csv')

print("Dataset shape:", df.shape)
print(df.head())

# Step 3: Exploratory Data Analysis
print(df.describe())
print(df['cnt'].hist(bins=50))
plt.title('Distribution of Bike Rental Demand')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation')
plt.show()

# Step 4: Feature Engineering
df['dtime'] = pd.to_datetime(df['dteday'] + ' ' + df['hr'].astype(str) + ':00:00')
df['hour'] = df['hr']
df['dayofweek'] = df['dtime'].dt.dayofweek
df['month'] = df['dtime'].dt.month
df.drop(['instant', 'dteday', 'dtime'], axis=1, inplace=True)

# Select features (exclude casual/registered as they sum to cnt)
features = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 
            'temp', 'atemp', 'hum', 'windspeed', 'dayofweek', 'month']
X = df[features]
y = df['cnt']

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

# Step 6: Model 1 - Random Forest with Hyperparameter Tuning
rf = RandomForestRegressor(random_state=42)
param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20]}
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
rf_pred = best_rf.predict(X_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

print(f"Random Forest - RMSE: {rf_rmse:.2f}, R2: {rf_r2:.4f}")
print("Best params:", grid_search.best_params_)

# Step 7: Model 2 - XGBoost
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
xgb_r2 = r2_score(y_test, xgb_pred)

print(f"XGBoost - RMSE: {xgb_rmse:.2f}, R2: {xgb_r2:.4f}")

# Step 8: Feature Importance (Random Forest)
importances = best_rf.feature_importances_
feat_imp = pd.DataFrame({'feature': features, 'importance': importances}).sort_values('importance', ascending=False)
print(feat_imp)

plt.figure(figsize=(10,6))
sns.barplot(data=feat_imp, x='importance', y='feature')
plt.title('Feature Importance - Random Forest')
plt.show()

# Step 9: Predictions vs Actual Plot
plt.figure(figsize=(10,6))
plt.scatter(y_test, rf_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Demand')
plt.ylabel('Predicted Demand')
plt.title('Actual vs Predicted (Random Forest)')
plt.show()

# Step 10: Save models (optional)
import joblib
joblib.dump(best_rf, 'bike_demand_rf_model.pkl')
joblib.dump(xgb_model, 'bike_demand_xgb_model.pkl')

print("Models saved! Project complete.")
print("To run: Download dataset, run this script, check plots and metrics.")
