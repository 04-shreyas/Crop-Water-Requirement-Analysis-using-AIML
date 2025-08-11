import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import joblib as jb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Load dataset
df = pd.read_csv("py_dataset33.csv")
df.dropna(inplace=True)

# Removing outliers
df = df[df["Water Requirement (mm/day)"] <= 3.9]

# Features and Target
X = df.drop("Water Requirement (mm/day)", axis=1)
y = df["Water Requirement (mm/day)"]

# Ordinal Encoding for categorical columns
categorical_cols = X.select_dtypes(include=["object"]).columns
oe = OrdinalEncoder()
X[categorical_cols] = oe.fit_transform(X[categorical_cols])

# Full model training split
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit CatBoost on full data
catboost_full = CatBoostRegressor(iterations=200, depth=6, learning_rate=0.1, verbose=0, random_state=42)
catboost_full.fit(X_train_full, y_train_full)
jb.dump(catboost_full, "catboost_model.pkl")

# Feature Importance from CatBoost
importances = catboost_full.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Top 6 features for learning curve/overfitting
top_features = feature_importance_df['Feature'].iloc[:6].tolist()
X_top = X[top_features]
X_train, X_test, y_train, y_test = train_test_split(X_top, y, test_size=0.2, random_state=42)

# Define all models
models = {
    "CatBoost": CatBoostRegressor(iterations=200, depth=6, learning_rate=0.1, verbose=0, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42),
    "Extra Trees": ExtraTreesRegressor(n_estimators=200, max_depth=12, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    results[name] = {
        "RMSE": round(rmse, 3),
        "MAE": round(mae, 3),
        "R²": round(r2, 4),
        "MAPE (%)": round(mape * 100, 2)
    }

    jb.dump(model, f"{name.lower().replace(' ', '_')}_model.pkl")

# Comparison Table
result_df = pd.DataFrame(results).T
print("\nModel Performance Comparison:\n")
print(result_df)

# Barplot of metrics
result_df_plot = result_df.drop(columns=["R²"]).reset_index().melt(id_vars="index")
plt.figure(figsize=(10, 6))
sns.barplot(data=result_df_plot, x="index", y="value", hue="variable", palette="Set2")
plt.title("Model Performance Metrics")
plt.xlabel("Model")
plt.ylabel("Score")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# Overfitting Check for CatBoost
catboost_model = models["CatBoost"]
y_train_pred = catboost_model.predict(X_train)
y_test_pred = catboost_model.predict(X_test)

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

print("\nOverfitting Check (CatBoost):")
print(f"R² Train: {r2_train:.4f}")
print(f"R² Test : {r2_test:.4f}")
print(f"RMSE Train: {rmse_train:.2f}")
print(f"RMSE Test : {rmse_test:.2f}")

# Learning Curve (CatBoost)
train_sizes, train_scores, test_scores = learning_curve(
    catboost_model, X_top, y, cv=5, scoring='r2', train_sizes=np.linspace(0.1, 1.0, 5)
)
train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, 'o-', label="Training R²", color="blue")
plt.plot(train_sizes, test_mean, 'o-', label="Validation R²", color="orange")
plt.fill_between(train_sizes, train_scores.min(axis=1), train_scores.max(axis=1), alpha=0.1, color="blue")
plt.fill_between(train_sizes, test_scores.min(axis=1), test_scores.max(axis=1), alpha=0.1, color="orange")
plt.title("Learning Curve: Train vs Validation R² (CatBoost)")
plt.xlabel("Training Set Size")
plt.ylabel("R² Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Actual vs Predicted
plt.figure(figsize=(7, 7))
plt.scatter(y_train, y_train_pred, color='green', alpha=0.6, label='Train')
plt.scatter(y_test, y_test_pred, color='red', alpha=0.6, label='Test')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted (Train & Test - CatBoost)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Residual Plot
residuals = y_test - y_test_pred
plt.figure(figsize=(8, 5))
sns.histplot(residuals, kde=True, color="purple")
plt.axvline(residuals.mean(), color='red', linestyle='--', label="Mean Residual")
plt.title("Distribution of Residuals (CatBoost)")
plt.xlabel("Residual (Actual - Predicted)")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
