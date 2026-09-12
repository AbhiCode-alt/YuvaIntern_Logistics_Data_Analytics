"""
Week 4 Task: Predictive Modeling and Optimization in Logistics Systems
Standalone implementation using the supplied Logistics_Delivery_Time_Dataset.csv
"""

import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load dataset
df = pd.read_csv("Logistics_Delivery_Time_Dataset.csv")

TARGET = "Delivery_Time_Hours"
X = df.drop(columns=[TARGET])
y = df[TARGET]

# 2. Identify feature types
categorical_features = ["Vehicle_Type", "Priority", "Day_of_Week"]
numeric_features = [c for c in X.columns if c not in categorical_features]

# 3. Preprocessing pipeline
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric_features),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_features)
])

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 5. Compare baseline and ensemble models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=250, min_samples_leaf=2, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

results = []

for name, model in models.items():
    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    r2 = r2_score(y_test, pred)

    results.append([name, mae, rmse, r2])

results_df = pd.DataFrame(
    results, columns=["Model", "MAE", "RMSE", "R2"]
).sort_values("RMSE")

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

# 6. Tune Random Forest using 5-fold CV
rf_pipe = Pipeline([
    ("prep", preprocessor),
    ("model", RandomForestRegressor(random_state=42, n_jobs=-1))
])

param_grid = {
    "model__n_estimators": [150, 250],
    "model__max_depth": [None, 12, 20],
    "model__min_samples_leaf": [1, 2, 4]
}

grid = GridSearchCV(
    rf_pipe,
    param_grid,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_
pred = best_model.predict(X_test)

print("\nBEST PARAMETERS")
print(grid.best_params_)

print("\nTUNED RANDOM FOREST PERFORMANCE")
print("MAE :", mean_absolute_error(y_test, pred))
print("RMSE:", mean_squared_error(y_test, pred) ** 0.5)
print("R2  :", r2_score(y_test, pred))

# 7. Independent 5-fold validation estimate
kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_rmse = -cross_val_score(
    best_model,
    X_train,
    y_train,
    cv=kf,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)

print("\n5-FOLD CV RMSE")
print("Mean:", cv_rmse.mean())
print("Std :", cv_rmse.std())

# 8. Example optimization scenario
scenario = pd.DataFrame({
    "Distance_km": [120, 80, 250],
    "Package_Weight_kg": [5, 8, 12],
    "Traffic_Index": [0.85, 0.70, 0.90],
    "Weather_Delay": [0, 0, 1],
    "Vehicle_Type": ["Van", "Van", "Truck"],
    "Priority": ["Standard", "Express", "Standard"],
    "Warehouse_Load": [0.90, 0.75, 0.85],
    "Day_of_Week": ["Mon", "Tue", "Wed"],
    "Driver_Experience_Years": [2, 5, 4]
})

scenario["Original_Predicted_Hours"] = best_model.predict(scenario)

optimized = scenario.drop(columns=["Original_Predicted_Hours"]).copy()
optimized["Warehouse_Load"] = (
    optimized["Warehouse_Load"] - 0.20
).clip(0.10, 1.0)
optimized["Traffic_Index"] = (
    optimized["Traffic_Index"] - 0.15
).clip(0.05, 1.0)

scenario["Optimized_Predicted_Hours"] = best_model.predict(optimized)
scenario["Estimated_Saving_Hours"] = (
    scenario["Original_Predicted_Hours"]
    - scenario["Optimized_Predicted_Hours"]
)

print("\nOPTIMIZATION SCENARIO")
print(scenario.to_string(index=False))

# Note:
# The dataset is synthetic and intended for academic demonstration.
# For real deployment, replace it with historical TMS/WMS/GPS/traffic data.
