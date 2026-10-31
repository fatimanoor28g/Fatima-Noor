
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 1. Download the original tutorial dataset
data_url = (
    "https://raw.githubusercontent.com/jenfly/opsd/master/"
    "opsd_germany_daily.csv"
)

csv_file = Path(__file__).with_name("dataset.csv")

if not csv_file.exists():
    print("Downloading dataset...")
    df = pd.read_csv(data_url)
    df.to_csv(csv_file, index=False)
else:
    print("Loading existing dataset.csv...")
    df = pd.read_csv(csv_file)

print("\nFirst five rows:")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nColumn names:", df.columns.tolist())

# 2. Convert Date into datetime format
df["Date"] = pd.to_datetime(df["Date"])

# 3. Set Date as the index
df = df.set_index("Date")
df = df.sort_index()

print("\nDate range:")
print("Start:", df.index.min())
print("End:", df.index.max())

print("\nMissing values:")
print(df.isnull().sum())

# 4. Add Year, Month and Weekday columns
df["Year"] = df.index.year
df["Month"] = df.index.month
df["Weekday Name"] = df.index.day_name()

print("\nUpdated dataset:")
print(df.head())

# 5. Select data for 2017
print("\nJanuary 2017 data:")
print(df.loc["2017-01"])

# 6. Plot daily electricity consumption
sns.set_theme()

df["Consumption"].plot(
    figsize=(12, 5),
    linewidth=0.7,
    title="Daily Electricity Consumption in Germany"
)

plt.ylabel("Consumption (GWh)")
plt.xlabel("Date")
plt.tight_layout()
plt.show()

# 7. Plot Consumption, Solar and Wind
columns = ["Consumption", "Solar", "Wind"]

df[columns].plot(
    subplots=True,
    figsize=(12, 9),
    title=[
        "Daily Electricity Consumption",
        "Daily Solar Production",
        "Daily Wind Production"
    ]
)

plt.tight_layout()
plt.show()

# 8. Analyze seasonality by month
monthly_seasonality = df.groupby("Month")[
    ["Consumption", "Solar", "Wind"]
].mean()

print("\nAverage values by month:")
print(monthly_seasonality)

monthly_seasonality.plot(
    figsize=(12, 5),
    marker="o",
    title="Monthly Seasonality"
)

plt.ylabel("Average Daily Value (GWh)")
plt.tight_layout()
plt.show()

# 9. Weekly resampling
weekly_mean = df[
    ["Consumption", "Wind", "Solar", "Wind+Solar"]
].resample("W").mean()

print("\nWeekly mean:")
print(weekly_mean.head())

weekly_mean["Consumption"].plot(
    figsize=(12, 5),
    title="Weekly Average Electricity Consumption"
)

plt.ylabel("Consumption (GWh)")
plt.tight_layout()
plt.show()

# 10. Seven-day rolling mean
rolling_7d = df[
    ["Consumption", "Wind", "Solar", "Wind+Solar"]
].rolling(window=7).mean()

df["Consumption"].plot(
    figsize=(12, 5),
    alpha=0.4,
    label="Daily"
)

rolling_7d["Consumption"].plot(
    linewidth=2,
    label="7-Day Rolling Mean"
)

plt.title("Daily Consumption vs 7-Day Rolling Mean")
plt.ylabel("Consumption (GWh)")
plt.legend()
plt.tight_layout()
plt.show()

# 11. Long-term trend: 365-day rolling mean
rolling_365d = df[
    ["Consumption", "Wind", "Solar", "Wind+Solar"]
].rolling(window=365).mean()

rolling_365d[
    ["Consumption", "Wind", "Solar"]
].plot(
    figsize=(12, 6),
    title="Long-Term Electricity and Renewable Energy Trends"
)

plt.ylabel("365-Day Rolling Mean (GWh)")
plt.tight_layout()
plt.show()

print("\nTime Series Analysis completed successfully!")
