
from pathlib import Path
import pandas as pd

# Locate the project and dataset
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data" / "Toronto Island Ferry Ticket Counts(1).csv"

print("=" * 55)
print("FERRY DATA QUALITY REPORT")
print("=" * 55)

# 1. Check that the dataset exists
if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}\n"
        "Check the CSV filename inside the data folder."
    )

# 2. Load the dataset
df = pd.read_csv(DATA_FILE)

# Clean accidental spaces in column names
df.columns = df.columns.str.strip()

print(f"\nTotal rows: {len(df):,}")
print(f"Total columns: {len(df.columns)}")
print(f"\nColumn names:\n{df.columns.tolist()}")

# 3. Check required columns
required_columns = [
    "_id",
    "Timestamp",
    "Sales Count",
    "Redemption Count"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Required columns missing: {missing_columns}")

# 4. Convert data types
df["Timestamp"] = pd.to_datetime(
    df["Timestamp"], errors="coerce"
)

for col in ["Sales Count", "Redemption Count"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("\n--- DATA TYPES ---")
print(df.dtypes)

# 5. Check date range
print("\n--- TIMESTAMP RANGE ---")
print("Earliest:", df["Timestamp"].min())
print("Latest:  ", df["Timestamp"].max())
print("Invalid timestamps:", df["Timestamp"].isna().sum())

# 6. Check missing values
print("\n--- MISSING VALUES ---")
print(df.isna().sum())

# 7. Check duplicates
print("\n--- DUPLICATE CHECKS ---")
print("Duplicate rows:", df.duplicated().sum())
print("Duplicate IDs:", df["_id"].duplicated().sum())
print("Duplicate timestamps:", df["Timestamp"].duplicated().sum())

# 8. Check invalid ticket counts
print("\n--- TICKET COUNT VALIDATION ---")
for col in ["Sales Count", "Redemption Count"]:
    print(f"{col} - negative values: {(df[col] < 0).sum()}")
    print(f"{col} - zero values: {(df[col] == 0).sum()}")

# 9. Check intervals between consecutive timestamps

# 9. Analyze timestamp gaps more accurately
timestamps = (
    df["Timestamp"]
    .dropna()
    .sort_values()
    .drop_duplicates()
)

intervals = timestamps.diff().dropna()
expected_interval = pd.Timedelta(minutes=15)

exact_15_min_gaps = (intervals == expected_interval).sum()
large_gaps = intervals[intervals > expected_interval]
short_gaps = intervals[intervals < expected_interval]

# Estimate missing slots when gaps are exact multiples of 15 minutes
missing_slots = 0
irregular_gaps = 0

for gap in large_gaps:
    number_of_intervals = gap / expected_interval

    if number_of_intervals.is_integer():
        missing_slots += int(number_of_intervals) - 1
    else:
        irregular_gaps += 1

print("\n--- DETAILED TIME GAP ANALYSIS ---")
print("Consecutive 15-minute intervals:", exact_15_min_gaps)
print("Gaps greater than 15 minutes:", len(large_gaps))
print("Estimated missing 15-minute slots:", missing_slots)
print("Irregular gaps not divisible by 15 minutes:", irregular_gaps)
print("Intervals shorter than 15 minutes:", len(short_gaps))

print("\n--- 10 LARGEST TIMESTAMP GAPS ---")

gap_table = pd.DataFrame({
    "Previous Timestamp": timestamps.shift(1).iloc[1:].values,
    "Next Timestamp": timestamps.iloc[1:].values,
    "Gap": intervals.values
})

print(
    gap_table.nlargest(10, "Gap").to_string(index=False)
)
