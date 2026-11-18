from collections.abc import Iterator

import numpy as np
import pandas as pd


def chronological_boundaries(
    target_times: pd.DatetimeIndex,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between zero and one.")
    if not 0 < validation_fraction < 1 - train_fraction:
        raise ValueError(
            "validation_fraction must be positive and leave a test period."
        )
    ordered = target_times.sort_values()
    if len(ordered) < 20:
        raise ValueError("At least 20 supervised rows are required to split data.")
    return (
        ordered[int(len(ordered) * train_fraction)],
        ordered[int(len(ordered) * (train_fraction + validation_fraction))],
    )


def rolling_origin_windows(
    target_times: pd.DatetimeIndex,
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
    folds: int = 3,
) -> Iterator[tuple[pd.Timestamp, pd.Timestamp]]:
    """Yield expanding-window cutoffs and non-overlapping validation windows."""
    if folds < 1:
        raise ValueError("folds must be at least one.")
    window = (validation_end - validation_start) / folds
    for fold in range(folds):
        start = validation_start + window * fold
        end = validation_start + window * (fold + 1)
        if fold == folds - 1:
            end = validation_end
        yield start, end
