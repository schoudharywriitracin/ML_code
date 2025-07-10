# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 01:26:02 2025

@author: SOURAV
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.model_selection import GridSearchCV, learning_curve, validation_curve
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Plot style settings
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 12
rcParams['axes.labelweight'] = 'bold'
rcParams['axes.titlesize'] = 14
rcParams['axes.titleweight'] = 'bold'
rcParams['legend.fontsize'] = 12
rcParams['legend.frameon'] = False
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
sns.set_style("white")

# Load data
data = pd.read_csv(r'G:\OTHER_FOLDERS\USERS_DIFF\yogesh\MERRA_2\output\final_dataset_cropped.csv')
data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y %H:%M')
data_cleaned = data.dropna()

X = data_cleaned.drop(columns=['Date', 'Station', 'Moisture_soil'])
y = data_cleaned['Moisture_soil']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model and parameter grid
model_name = 'DecisionTreeRegressor'
model = DecisionTreeRegressor()
param_grid = {
    'criterion': ['squared_error', 'absolute_error'],
    'splitter': ['random'],
    'max_depth': [None, 10],
    'min_weight_fraction_leaf': [0.0],
    'max_features': [None, 'auto', 'sqrt'],
    'max_leaf_nodes': [None, 10],
    'min_impurity_decrease': [0.0],
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5,
                           n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')

# Fit and evaluate
print(f"Running GridSearchCV for {model_name}...")
grid_search.fit(X_scaled, y)
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_scaled)

mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)
mape = np.mean(np.abs((y - y_pred) / y)) * 100

print(f"Best Hyperparameters: {grid_search.best_params_}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R²: {r2}")
print(f"MAPE: {mape}%")

# Residual plot
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, y - y_pred, color='blue', edgecolor='k', label='Residuals')
plt.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero Line')
plt.xlabel('Predicted Values', fontweight='bold')
plt.ylabel('Residuals', fontweight='bold')
plt.title(f'{model_name} - Residual Plot', fontweight='bold')
plt.legend()
plt.grid(False)
plt.savefig('Residual_plot.png', dpi=1200)
plt.show()

# Actual vs Predicted plot
plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred, color='green', edgecolor='k', label='Predicted vs Actual')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='1:1 Line')
plt.xlabel('Actual Values', fontweight='bold')
plt.ylabel('Predicted Values', fontweight='bold')
plt.title(f'{model_name} - Actual vs. Predicted', fontweight='bold')
plt.legend()
plt.grid(False)
plt.savefig('Actual_vs_Predicted.png', dpi=1200)
plt.show()

# Learning curve
train_sizes, train_scores, validation_scores = learning_curve(best_model, X_scaled, y, cv=5, scoring='neg_mean_squared_error')

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, -np.mean(train_scores, axis=1), 'o-', label='Training Error', color='blue')
plt.plot(train_sizes, -np.mean(validation_scores, axis=1), 'o-', label='Validation Error', color='orange')
plt.xlabel('Training Examples', fontweight='bold')
plt.ylabel('Mean Squared Error', fontweight='bold')
plt.title(f'{model_name} - Learning Curve', fontweight='bold')
plt.legend()
plt.grid(False)
plt.savefig('Learning_Curve.png', dpi=1200)
plt.show()

# Validation curve
param_name = list(param_grid.keys())[0]
param_range = param_grid[param_name]

train_scores, validation_scores = validation_curve(
    best_model, X_scaled, y, param_name=param_name, param_range=param_range, cv=5,
    scoring='neg_mean_squared_error'
)

plt.figure(figsize=(8, 6))
plt.plot(param_range, -np.mean(train_scores, axis=1), 'o-', label='Training Error', color='blue')
plt.plot(param_range, -np.mean(validation_scores, axis=1), 'o-', label='Validation Error', color='orange')
plt.xlabel(param_name, fontweight='bold')
plt.ylabel('Mean Squared Error', fontweight='bold')
plt.title(f'{model_name} - Validation Curve', fontweight='bold')
plt.legend()
plt.grid(False)
plt.savefig('Validation_Curve.png', dpi=1200)
plt.show()

# Hyperparameter plots
for param, values in param_grid.items():
    if len(values) > 1:
        mean_scores = grid_search.cv_results_['mean_test_score'][:len(values)]
        plt.figure(figsize=(8, 6))
        plt.plot(values, mean_scores, 'o-', label=param, color='purple')
        plt.xlabel(param, fontweight='bold')
        plt.ylabel('Mean Test Score', fontweight='bold')
        plt.title(f'{model_name} - {param} Hyperparameter Plot', fontweight='bold')
        plt.legend()
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(f'{param}_Hyperparameter_Plot.png', dpi=1200)
        plt.show()
