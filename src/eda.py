
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "Toronto Island Ferry Ticket Counts(1).csv"
)

OUTPUT_FILE = PROJECT_DIR / "data" / "ferry_analysis_2015_2025.csv"

# Load original data
df = pd.read_csv(INPUT_FILE)

# Convert columns to suitable types
df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
df["Sales Count"] = pd.to_numeric(df["Sales Count"], errors="coerce")
df["Redemption Count"] = pd.to_numeric(
    df["Redemption Count"], errors="coerce"
)

# Keep valid records within 2015-2025
df = df.dropna(
    subset=["Timestamp", "Sales Count", "Redemption Count"]
).copy()

df = df[
    (df["Timestamp"].dt.year >= 2015)
    & (df["Timestamp"].dt.year <= 2025)
].copy()

# Add time-based features
df["Year"] = df["Timestamp"].dt.year
df["Month"] = df["Timestamp"].dt.month
df["DayOfWeek"] = df["Timestamp"].dt.day_name()
df["Hour"] = df["Timestamp"].dt.hour
df["Date"] = df["Timestamp"].dt.date

# Activity proxy: sales plus redemptions
# This is NOT confirmed passenger count or capacity utilization.
df["Total Activity Load"] = (
    df["Sales Count"] + df["Redemption Count"]
)

# Save a separate analysis file
df.to_csv(OUTPUT_FILE, index=False)

print("=" * 50)
print("FERRY EDA PREPARATION")
print("=" * 50)

print(f"\nRecords for 2015-2025: {len(df):,}")
print(f"Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
print(f"Saved file: {OUTPUT_FILE.name}")

print("\n--- YEARLY SUMMARY ---")
yearly = df.groupby("Year").agg(
    Records=("Timestamp", "size"),
    Sales=("Sales Count", "sum"),
    Redemptions=("Redemption Count", "sum"),
    Average_Activity=("Total Activity Load", "mean")
)
print(yearly.to_string())

print("\n--- MONTHLY ACTIVITY SUMMARY ---")
monthly = df.groupby("Month").agg(
    Records=("Timestamp", "size"),
    Average_Activity=("Total Activity Load", "mean")
)
print(monthly.to_string())

print("\n--- DAY OF WEEK SUMMARY ---")
weekday = df.groupby("DayOfWeek").agg(
    Records=("Timestamp", "size"),
    Average_Activity=("Total Activity Load", "mean")
)
print(weekday.to_string())

print("\nEDA preparation completed successfully.")


import matplotlib.pyplot as plt

# Create a folder for generated charts
CHART_DIR = PROJECT_DIR / "charts"
CHART_DIR.mkdir(exist_ok=True)

# Monthly activity chart
monthly_activity = (
    df.groupby("Month")["Total Activity Load"].mean()
)

plt.figure(figsize=(10, 5))
monthly_activity.plot(kind="bar")
plt.title("Average Ferry Ticket Activity by Month (2015-2025)")
plt.xlabel("Month")
plt.ylabel("Average Activity per Recorded Interval")
plt.tight_layout()
plt.savefig(CHART_DIR / "monthly_activity.png", dpi=150)
plt.close()

# Yearly activity chart
yearly_activity = (
    df.groupby("Year")["Total Activity Load"].mean()
)

plt.figure(figsize=(10, 5))
yearly_activity.plot(kind="line", marker="o")
plt.title("Average Ferry Ticket Activity by Year (2015-2025)")
plt.xlabel("Year")
plt.ylabel("Average Activity per Recorded Interval")
plt.tight_layout()
plt.savefig(CHART_DIR / "yearly_activity.png", dpi=150)
plt.close()

print("\nCharts saved in:", CHART_DIR)
print(" - monthly_activity.png")
print(" - yearly_activity.png")


# Identify unusually high activity records
print("\n--- Unusual Activity Spike Analysis ---")

df["Total Activity Load"] = (
    df["Sales Count"] + df["Redemption Count"]
)

# Calculate the 99th percentile threshold
threshold = df["Total Activity Load"].quantile(0.99)

print(f"99th percentile threshold: {threshold:.2f}")

# Display the 20 highest activity records
high_activity = df[
    df["Total Activity Load"] >= threshold
].sort_values(
    "Total Activity Load", ascending=False
).head(20)

print("\nTop 20 highest activity records:")
print(
    high_activity[
        ["Timestamp", "Sales Count",
         "Redemption Count", "Total Activity Load"]
    ].to_string(index=False)
)

# Save results for further investigation
high_activity.to_csv(
    "data/high_activity_records.csv", index=False
)

print("\nSaved to data/high_activity_records.csv")

# Summarize high-activity records by year and month
print("\n--- High Activity by Year and Month ---")

high_activity["Timestamp"] = pd.to_datetime(
    high_activity["Timestamp"]
)

high_activity["Year"] = high_activity["Timestamp"].dt.year
high_activity["Month"] = high_activity["Timestamp"].dt.month

monthly_spikes = (
    high_activity.groupby(["Year", "Month"])
    .size()
    .reset_index(name="High_Activity_Record_Count")
    .sort_values(
        "High_Activity_Record_Count",
        ascending=False
    )
)

print(monthly_spikes.head(15).to_string(index=False))

monthly_spikes.to_csv(
    "data/high_activity_monthly_summary.csv",
    index=False
)

print("\nSaved to data/high_activity_monthly_summary.csv")
