"""
Week 1 Logistics Analytics - Code Illustration
Scenario: Demand forecasting + inventory analytics + routing preparation.

This file is a planning/illustration script. Replace column names and paths
with the actual dataset selected for implementation.
"""

import pandas as pd
import numpy as np

def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["Unit_Price"] = pd.to_numeric(df["Unit_Price"], errors="coerce")
    df = df.dropna(subset=["Order_Date", "Product_ID", "Quantity"])
    df = df[df["Quantity"] > 0].drop_duplicates()
    return df

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["day_of_week"] = df["Order_Date"].dt.dayofweek
    df["month"] = df["Order_Date"].dt.month
    df["year"] = df["Order_Date"].dt.year

    daily = (
        df.groupby(["Order_Date", "Product_ID"], as_index=False)["Quantity"]
          .sum()
          .sort_values(["Product_ID", "Order_Date"])
    )
    daily["lag_1"] = daily.groupby("Product_ID")["Quantity"].shift(1)
    daily["rolling_7"] = (
        daily.groupby("Product_ID")["Quantity"]
             .transform(lambda s: s.shift(1).rolling(7).mean())
    )
    return daily

def mae(actual, predicted):
    return np.mean(np.abs(np.asarray(actual) - np.asarray(predicted)))

def on_time_delivery_rate(on_time, total):
    return 100 * on_time / total if total else np.nan

def stockout_rate(stockouts, total_events):
    return 100 * stockouts / total_events if total_events else np.nan

# Example forecasting model
def train_random_forest(daily: pd.DataFrame):
    from sklearn.ensemble import RandomForestRegressor

    features = ["lag_1", "rolling_7", "day_of_week", "month"]
    model_data = daily.dropna(subset=features + ["Quantity"]).copy()

    cutoff = model_data["Order_Date"].quantile(0.8)
    train = model_data[model_data["Order_Date"] <= cutoff]
    valid = model_data[model_data["Order_Date"] > cutoff]

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    model.fit(train[features], train["Quantity"])
    predictions = model.predict(valid[features])

    return model, valid, predictions

# Example product segmentation
def segment_products(df: pd.DataFrame):
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    features = (
        df.groupby("Product_ID")
          .agg(
              total_units=("Quantity", "sum"),
              avg_price=("Unit_Price", "mean"),
              order_count=("Order_ID", "nunique")
          )
          .fillna(0)
    )

    X = StandardScaler().fit_transform(features)
    model = KMeans(n_clusters=4, random_state=42, n_init="auto")
    features["segment"] = model.fit_predict(X)
    return features

# Conceptual inventory rule
def reorder_flag(forecast_daily_demand, lead_time_days, demand_std,
                 z_value, current_inventory):
    lead_time_demand = forecast_daily_demand * lead_time_days
    safety_stock = z_value * demand_std * (lead_time_days ** 0.5)
    reorder_point = lead_time_demand + safety_stock
    return current_inventory <= reorder_point, reorder_point

if __name__ == "__main__":
    # Example:
    # df = load_and_clean("sales.csv")
    # daily = add_features(df)
    # model, valid, pred = train_random_forest(daily)
    # print("Validation MAE:", mae(valid["Quantity"], pred))
    print("Planning script ready. Supply the selected dataset and update column names.")
