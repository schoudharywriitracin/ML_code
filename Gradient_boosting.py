import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve
import seaborn as sns

# Set plot styles
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

# Load your dataset
data = pd.read_csv(r'G:\OTHER_FOLDERS\USERS_DIFF\yogesh\MERRA_2\output\final_dataset_cropped.csv')
data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y %H:%M')
data_cleaned = data.dropna()

X = data_cleaned.drop(columns=['Date', 'Station', 'Moisture_soil'])
y = data_cleaned['Moisture_soil']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Manual hyperparameter grid
param_grid = {
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 200],
    'max_depth': [3, 6],
    'subsample': [0.8, 1.0]
}

best_score = float('inf')
best_params = {}

# Manual Grid Search
for lr in param_grid['learning_rate']:
    for est in param_grid['n_estimators']:
        for md in param_grid['max_depth']:
            for subs in param_grid['subsample']:
                model = XGBRegressor(learning_rate=lr, n_estimators=est, max_depth=md,
                                     subsample=subs, verbosity=0, use_label_encoder=False)
                model.fit(X_scaled, y)
                y_pred = model.predict(X_scaled)
                mse = mean_squared_error(y, y_pred)

                print(f"Params: lr={lr}, est={est}, max_depth={md}, subsample={subs}, MSE={mse:.4f}")

                if mse < best_score:
                    best_score = mse
                    best_params = {'learning_rate': lr, 'n_estimators': est,
                                   'max_depth': md, 'subsample': subs}

# Best Model
model_name = 'XGBRegressor'
best_model = XGBRegressor(**best_params, verbosity=0, use_label_encoder=False)
best_model.fit(X_scaled, y)
y_pred = best_model.predict(X_scaled)

# Evaluation metrics
print("\nBest Parameters:", best_params)
print(f"MSE: {mean_squared_error(y, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y, y_pred)):.4f}")
print(f"MAE: {mean_absolute_error(y, y_pred):.4f}")
print(f"R²: {r2_score(y, y_pred):.4f}")
print(f"MAPE: {np.mean(np.abs((y - y_pred) / y)) * 100:.2f}%")

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

# Learning Curve
train_sizes, train_scores, validation_scores = learning_curve(
    best_model, X_scaled, y, cv=5, scoring='neg_mean_squared_error')

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
