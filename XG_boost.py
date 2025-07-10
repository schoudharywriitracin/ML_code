import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from matplotlib import rcParams
import seaborn as sns

# Plotting settings
rcParams['font.family'] = 'Times New Roman'
sns.set_style("white")

# Load data
data = pd.read_csv(r'G:\OTHER_FOLDERS\USERS_DIFF\yogesh\MERRA_2\output\final_dataset_cropped.csv')
data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y %H:%M')
data_cleaned = data.dropna()

X = data_cleaned.drop(columns=['Date', 'Station', 'Moisture_soil'])
y = data_cleaned['Moisture_soil']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Manual parameter grid
param_grid = {
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 200],
    'max_depth': [3, 6],
    'subsample': [0.8, 1.0]
}

# Manual grid search
best_score = float('inf')
best_params = {}

for lr in param_grid['learning_rate']:
    for est in param_grid['n_estimators']:
        for md in param_grid['max_depth']:
            for subs in param_grid['subsample']:
                model = XGBRegressor(learning_rate=lr, n_estimators=est,
                                     max_depth=md, subsample=subs,
                                     verbosity=0, use_label_encoder=False)
                model.fit(X_scaled, y)
                y_pred = model.predict(X_scaled)
                mse = mean_squared_error(y, y_pred)

                print(f"Params: lr={lr}, est={est}, max_depth={md}, subsample={subs}, MSE={mse:.4f}")

                if mse < best_score:
                    best_score = mse
                    best_params = {'learning_rate': lr, 'n_estimators': est,
                                   'max_depth': md, 'subsample': subs}

# Best model
print("\nBest Parameters:", best_params)

final_model = XGBRegressor(**best_params, verbosity=0, use_label_encoder=False)
final_model.fit(X_scaled, y)
y_pred = final_model.predict(X_scaled)

# Final evaluation
print(f"\nFinal MSE: {mean_squared_error(y, y_pred):.4f}")
print(f"Final RMSE: {np.sqrt(mean_squared_error(y, y_pred)):.4f}")
print(f"Final MAE: {mean_absolute_error(y, y_pred):.4f}")
print(f"Final R²: {r2_score(y, y_pred):.4f}")
print(f"Final MAPE: {np.mean(np.abs((y - y_pred) / y)) * 100:.2f}%")
