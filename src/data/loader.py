from pathlib import Path

import pandas as pd


TIMESTAMP_COLUMN = "Timestamp"
TARGET_COLUMNS = ("Sales Count", "Redemption Count")
REQUIRED_COLUMNS = (TIMESTAMP_COLUMN, *TARGET_COLUMNS)


def load_ticket_data(path: str | Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load ticket counts without filling unobserved intervals."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Ticket dataset not found: {path}")

    raw = pd.read_csv(path)
    raw.columns = raw.columns.map(lambda column: str(column).strip())
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in raw]
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    timestamps = pd.to_datetime(raw[TIMESTAMP_COLUMN], errors="coerce")
    counts = raw.loc[:, list(TARGET_COLUMNS)].apply(
        lambda column: pd.to_numeric(
            column.astype("string").str.replace(",", "", regex=False).str.strip(),
            errors="coerce",
        )
    )
    valid_timestamps = timestamps.notna()
    valid_counts = counts.notna().all(axis=1) & counts.ge(0).all(axis=1)
    valid = valid_timestamps & valid_counts
    valid_data = pd.DataFrame(
        {
            TIMESTAMP_COLUMN: timestamps.loc[valid],
            **{column: counts.loc[valid, column] for column in TARGET_COLUMNS},
        }
    )

    off_grid = valid_data[TIMESTAMP_COLUMN].dt.minute.mod(15).ne(0)
    off_grid |= valid_data[TIMESTAMP_COLUMN].dt.second.ne(0)
    if off_grid.any():
        raise ValueError(
            f"{int(off_grid.sum()):,} valid timestamps are not aligned to "
            "15-minute intervals."
        )

    duplicate_rows = int(valid_data[TIMESTAMP_COLUMN].duplicated(keep=False).sum())
    data = (
        valid_data.groupby(TIMESTAMP_COLUMN, as_index=False)[list(TARGET_COLUMNS)]
        .sum()
        .sort_values(TIMESTAMP_COLUMN)
        .reset_index(drop=True)
    )
    quality = {
        "rows_in_source": int(len(raw)),
        "invalid_timestamps": int((~valid_timestamps).sum()),
        "invalid_or_negative_counts": int((valid_timestamps & ~valid_counts).sum()),
        "duplicate_timestamp_rows": duplicate_rows,
        "duplicate_timestamps_aggregated": int(
            valid_data[TIMESTAMP_COLUMN].duplicated().sum()
        ),
        "observed_rows": int(len(data)),
        "zero_sales_rows": int(data["Sales Count"].eq(0).sum()),
        "zero_redemption_rows": int(data["Redemption Count"].eq(0).sum()),
    }
    return data, quality


def regularize_intervals(data: pd.DataFrame) -> pd.DataFrame:
    """Create the 15-minute time grid; absent observations remain NaN."""
    if data.empty:
        raise ValueError("No valid ticket observations are available.")

    indexed = data.set_index(TIMESTAMP_COLUMN).sort_index()
    full_index = pd.date_range(
        start=indexed.index.min(),
        end=indexed.index.max(),
        freq="15min",
        name=TIMESTAMP_COLUMN,
    )
    return indexed.reindex(full_index).rename_axis(TIMESTAMP_COLUMN)
