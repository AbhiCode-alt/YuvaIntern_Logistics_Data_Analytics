import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("hypothetical_logistics_dataset.csv", parse_dates=["Date"])

# EDA
print(df.info())
print(df.describe())
print(df.isna().sum())
print(df.duplicated().sum())

# KPI calculations
on_time_rate = (df["Delivery_Status"] == "On Time").mean() * 100
print("On-time rate:", round(on_time_rate, 2), "%")
print("Average delivery time:", round(df["Delivery_Time_days"].mean(), 2))
print("Average total cost:", round(df["Total_Cost"].mean(), 2))

# Grouped analysis
mode_summary = df.groupby("Transport_Mode").agg(
    Shipments=("Shipment_ID","count"),
    Avg_Delivery_Days=("Delivery_Time_days","mean"),
    Avg_Delay_Days=("Delay_days","mean"),
    Avg_Total_Cost=("Total_Cost","mean"),
    Avg_Rating=("Customer_Rating","mean"),
    On_Time_Rate=("Delivery_Status",
                  lambda x: (x == "On Time").mean() * 100)
)

print(mode_summary.round(2))

# Visualizations
df["Date"] = pd.to_datetime(df["Date"])

plt.figure(figsize=(8,5))
df["Transport_Mode"].value_counts().plot(kind="bar")
plt.title("Shipment Volume by Transportation Mode")
plt.xlabel("Transport Mode")
plt.ylabel("Number of Shipments")
plt.tight_layout()
plt.show()

daily = df.groupby("Date")["Delivery_Time_days"].mean().sort_index()
daily.rolling(7, min_periods=1).mean().plot(figsize=(9,5))
plt.title("7-Day Rolling Average Delivery Time")
plt.xlabel("Date")
plt.ylabel("Delivery Time (days)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
plt.hist(df["Delivery_Time_days"], bins=30)
plt.title("Distribution of Delivery Time")
plt.xlabel("Delivery Time (days)")
plt.ylabel("Shipment Count")
plt.tight_layout()
plt.show()

df.boxplot(column="Total_Cost", by="Transport_Mode", figsize=(8,5))
plt.suptitle("")
plt.title("Total Logistics Cost by Transportation Mode")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
plt.scatter(df["Distance_km"], df["Total_Cost"], alpha=0.35)
plt.title("Distance vs Total Logistics Cost")
plt.xlabel("Distance (km)")
plt.ylabel("Total Cost")
plt.tight_layout()
plt.show()

corr = df.select_dtypes("number").corr()
plt.figure(figsize=(10,7))
plt.imshow(corr, aspect="auto")
plt.colorbar(label="Correlation")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

mode_summary["Avg_Delivery_Days"].plot(kind="bar", figsize=(8,5))
plt.title("Average Delivery Time by Transportation Mode")
plt.ylabel("Days")
plt.tight_layout()
plt.show()

df["Delivery_Status"].value_counts().plot(kind="pie", autopct="%1.1f%%", figsize=(6,6))
plt.title("On-Time vs Delayed Shipments")
plt.ylabel("")
plt.show()
