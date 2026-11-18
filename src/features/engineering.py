import numpy as np
import pandas as pd

from src.data.loader import TARGET_COLUMNS


LAGS = (1, 2, 4, 8, 12, 24, 96, 672)
ROLLING_WINDOWS = (4, 8, 96)


def build_features(
    grid: pd.DataFrame,
    horizon_steps: int,
) -> pd.DataFrame:
    """Build origin-time features using observed history and known calendar data."""
    if horizon_steps < 1:
        raise ValueError("Forecast horizon must be at least one interval.")

    features: dict[str, pd.Series] = {}
    for column in TARGET_COLUMNS:
        history = grid[column]
        observed_positions = pd.Series(
            np.where(history.notna(), np.arange(len(history)), np.nan),
            index=grid.index,
        ).ffill()
        features[f"{column}_last_observed"] = history.ffill()
        features[f"{column}_observation_age"] = (
            pd.Series(np.arange(len(history)), index=grid.index) - observed_positions
        )
        for lag in LAGS:
            features[f"{column}_lag_{lag}"] = history.shift(lag - 1)
        prior = history
        for window in ROLLING_WINDOWS:
            rolling = prior.rolling(window=window, min_periods=1)
            features[f"{column}_rolling_mean_{window}"] = rolling.mean()
            features[f"{column}_rolling_std_{window}"] = rolling.std(ddof=0)
            features[f"{column}_rolling_max_{window}"] = rolling.max()
            features[f"{column}_rolling_min_{window}"] = rolling.min()
            features[f"{column}_rolling_median_{window}"] = rolling.median()

    target_time = grid.index + pd.Timedelta(minutes=15 * horizon_steps)
    minute_of_day = target_time.hour * 60 + target_time.minute
    weekday = target_time.day_of_week
    features.update(
        {
            "hour_sin": pd.Series(
                np.sin(2 * np.pi * minute_of_day / 1440), index=grid.index
            ),
            "hour_cos": pd.Series(
                np.cos(2 * np.pi * minute_of_day / 1440), index=grid.index
            ),
            "weekday_sin": pd.Series(
                np.sin(2 * np.pi * weekday / 7), index=grid.index
            ),
            "weekday_cos": pd.Series(
                np.cos(2 * np.pi * weekday / 7), index=grid.index
            ),
            "day_of_month": pd.Series(target_time.day, index=grid.index),
            "month": pd.Series(target_time.month, index=grid.index),
            "is_weekend": pd.Series((weekday >= 5).astype(int), index=grid.index),
        }
    )
    result = pd.DataFrame(features, index=grid.index)
    result.index.name = grid.index.name
    return result


def make_supervised(
    grid: pd.DataFrame,
    target: str,
    horizon_steps: int,
) -> tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
    """Pair information available at each origin with its exact future target."""
    if target not in TARGET_COLUMNS:
        raise ValueError(f"Unsupported forecast target: {target}")
    features = build_features(grid, horizon_steps)
    labels = grid[target].shift(-horizon_steps)
    target_times = grid.index + pd.Timedelta(minutes=15 * horizon_steps)
    observed = labels.notna() & grid[target].notna()
    return (
        features.loc[observed],
        labels.loc[observed].astype(float),
        pd.DatetimeIndex(target_times[observed], name="Target Timestamp"),
    )
