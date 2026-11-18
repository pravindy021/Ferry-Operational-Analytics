from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

from src.data.loader import TARGET_COLUMNS
from src.evaluation.metrics import regression_metrics
from src.evaluation.validation import chronological_boundaries, rolling_origin_windows
from src.features.engineering import make_supervised

HORIZONS = {1: "15 minutes", 2: "30 minutes", 4: "1 hour", 8: "2 hours"}
MODEL_NAMES = (
    "naive",
    "moving_average",
    "seasonal_naive_1d",
    "seasonal_naive_1w",
    "linear_regression",
    "hist_gradient_boosting",
)


def _new_estimator(name: str):
    if name == "linear_regression":
        return make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            LinearRegression(),
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            max_iter=80,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            learning_rate=0.08,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=42,
        )
    raise ValueError(f"Unknown fitted model: {name}")


def _baseline_prediction(name: str, features: pd.DataFrame, target: str) -> np.ndarray:
    last_value = features[f"{target}_last_observed"].to_numpy(dtype=float)
    if name == "naive":
        return last_value
    if name == "moving_average":
        return features[f"{target}_rolling_mean_4"].to_numpy(dtype=float)
    if name == "seasonal_naive_1d":
        seasonal = features[f"{target}_lag_96"].to_numpy(dtype=float)
    elif name == "seasonal_naive_1w":
        seasonal = features[f"{target}_lag_672"].to_numpy(dtype=float)
    else:
        raise ValueError(f"Unknown baseline model: {name}")
    return np.where(np.isfinite(seasonal), seasonal, last_value)


def _predict(
    name: str,
    estimator,
    features: pd.DataFrame,
    target: str,
) -> np.ndarray:
    if name in MODEL_NAMES[:4]:
        values = _baseline_prediction(name, features, target)
    else:
        values = estimator.predict(features)
    return np.maximum(np.asarray(values, dtype=float), 0.0)


def _fit_candidate(name: str, x_train: pd.DataFrame, y_train: pd.Series):
    if name in MODEL_NAMES[:4]:
        return None
    estimator = _new_estimator(name)
    estimator.fit(x_train, y_train)
    return estimator


def _finite_residuals(actual, predicted) -> np.ndarray:
    residuals = np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)
    return residuals[np.isfinite(residuals)]


def _interval_offsets(residuals: np.ndarray) -> dict[str, float]:
    if not len(residuals):
        raise ValueError("Validation residuals are required for prediction intervals.")
    return {
        "lower_80_offset": float(np.quantile(residuals, 0.10)),
        "upper_80_offset": float(np.quantile(residuals, 0.90)),
        "lower_95_offset": float(np.quantile(residuals, 0.025)),
        "upper_95_offset": float(np.quantile(residuals, 0.975)),
    }


def train_forecaster(
    grid: pd.DataFrame,
    rolling_folds: int = 3,
) -> dict:
    """Train direct horizon models and choose each using chronological validation."""
    if rolling_folds < 0:
        raise ValueError("rolling_folds cannot be negative.")
    trained: dict = {
        "schema_version": 1,
        "latest_observation": grid.dropna(how="all").index.max(),
        "horizons": HORIZONS.copy(),
        "targets": {},
        "evaluation": [],
        "quality": {
            "grid_intervals": int(len(grid)),
            "observed_sales_intervals": int(grid["Sales Count"].notna().sum()),
            "observed_redemption_intervals": int(
                grid["Redemption Count"].notna().sum()
            ),
            "missing_intervals": int(grid["Sales Count"].isna().sum()),
        },
        "validation_method": (
            "chronological 70/15/15 split; "
            f"{rolling_folds} expanding rolling-origin folds"
        ),
        "rolling_folds": rolling_folds,
    }

    for target in TARGET_COLUMNS:
        trained["targets"][target] = {}
        for horizon, horizon_label in HORIZONS.items():
            x, y, target_times = make_supervised(grid, target, horizon)
            train_end, validation_end = chronological_boundaries(target_times)
            train_mask = target_times < train_end
            validation_mask = (target_times >= train_end) & (
                target_times < validation_end
            )
            test_mask = target_times >= validation_end
            if not train_mask.any() or not validation_mask.any() or not test_mask.any():
                raise ValueError(
                    f"Insufficient chronological data for {target}, {horizon_label}."
                )

            x_train, y_train = x.loc[train_mask], y.loc[train_mask]
            x_validation, y_validation = x.loc[validation_mask], y.loc[validation_mask]
            validation_times = target_times[validation_mask]
            candidate_results: list[dict] = []
            candidates: dict[str, tuple[object, np.ndarray]] = {}
            candidate_offsets: dict[str, dict[str, float]] = {}

            for name in MODEL_NAMES:
                estimator = _fit_candidate(name, x_train, y_train)
                prediction = _predict(name, estimator, x_validation, target)
                candidate_metrics = regression_metrics(y_validation, prediction)
                candidates[name] = (estimator, prediction)
                candidate_offsets[name] = _interval_offsets(
                    _finite_residuals(y_validation, prediction)
                )
                candidate_results.append(
                    {"model": name, **candidate_metrics}
                )

            selected = min(candidate_results, key=lambda row: row["MAE"])
            selected_name = selected["model"]
            _, validation_prediction = candidates[selected_name]
            residuals = _finite_residuals(y_validation, validation_prediction)
            intervals = _interval_offsets(residuals)

            validation_rows = pd.DataFrame(
                {
                    "target_time": validation_times,
                    "actual": y_validation.to_numpy(),
                    "prediction": validation_prediction,
                }
            )
            rolling_results: list[dict] = []
            if rolling_folds:
                windows = list(
                    rolling_origin_windows(
                        validation_times,
                        validation_times.min(),
                        validation_times.max() + pd.Timedelta(nanoseconds=1),
                        folds=rolling_folds,
                    )
                )
                for fold_number, (fold_start, fold_end) in enumerate(windows, start=1):
                    fold_train = target_times < fold_start
                    fold_validation = (target_times >= fold_start) & (
                        target_times < fold_end
                    )
                    if not fold_train.any() or not fold_validation.any():
                        continue
                    for candidate_name in MODEL_NAMES:
                        fold_estimator = _fit_candidate(
                            candidate_name, x.loc[fold_train], y.loc[fold_train]
                        )
                        fold_prediction = _predict(
                            candidate_name,
                            fold_estimator,
                            x.loc[fold_validation],
                            target,
                        )
                        rolling_results.append(
                            {
                                "fold": fold_number,
                                "model": candidate_name,
                                "validation_start": str(fold_start),
                                "validation_end": str(fold_end),
                                **regression_metrics(
                                    y.loc[fold_validation], fold_prediction
                                ),
                            }
                        )

            refit_mask = train_mask | validation_mask
            final_estimator = _fit_candidate(
                selected_name, x.loc[refit_mask], y.loc[refit_mask]
            )
            candidate_estimators = {
                name: values[0] for name, values in candidates.items()
            }
            candidate_estimators[selected_name] = final_estimator
            test_prediction = _predict(
                selected_name, final_estimator, x.loc[test_mask], target
            )
            test_metrics = regression_metrics(y.loc[test_mask], test_prediction)
            test_rows = pd.DataFrame(
                {
                    "target_time": target_times[test_mask],
                    "actual": y.loc[test_mask].to_numpy(),
                    "prediction": test_prediction,
                    "residual": (
                        y.loc[test_mask].to_numpy() - test_prediction
                    ),
                }
            )

            trained["targets"][target][horizon] = {
                "model_name": selected_name,
                "estimator": final_estimator,
                "candidate_estimators": candidate_estimators,
                "feature_columns": list(x.columns),
                "interval_offsets": intervals,
                "candidate_interval_offsets": candidate_offsets,
                "validation_residuals": residuals,
                "validation_predictions": validation_rows,
                "test_predictions": test_rows,
                "candidate_metrics": candidate_results,
                "rolling_origin_metrics": rolling_results,
                "validation_start": str(train_end),
                "test_start": str(validation_end),
                "test_metrics": test_metrics,
            }
            for row in candidate_results:
                trained["evaluation"].append(
                    {
                        "target": target,
                        "horizon_steps": horizon,
                        "horizon": horizon_label,
                        "selected_model": selected_name,
                        "selected": row["model"] == selected_name,
                        "split": "validation",
                        **row,
                    }
                )
            trained["evaluation"].append(
                {
                    "target": target,
                    "horizon_steps": horizon,
                    "horizon": horizon_label,
                    "model": selected_name,
                    "selected_model": selected_name,
                    "selected": True,
                    "split": "test",
                    **test_metrics,
                }
            )

    trained["evaluation"] = pd.DataFrame(trained["evaluation"])
    return trained


def forecast_at(
    trained: dict,
    grid: pd.DataFrame,
    target: str,
    horizon_steps: int,
    origin: pd.Timestamp | None = None,
    model_name: str | None = None,
) -> dict:
    """Forecast one direct horizon from an observed timestamp."""
    if target not in trained["targets"]:
        raise ValueError(f"Unsupported forecast target: {target}")
    if horizon_steps not in HORIZONS:
        raise ValueError(f"Unsupported horizon: {horizon_steps} intervals.")
    if origin is None:
        origin = grid[target].dropna().index.max()
    origin = pd.Timestamp(origin)
    if origin not in grid.index or pd.isna(grid.at[origin, target]):
        raise ValueError(
            "Forecast origin must match an observed 15-minute timestamp."
        )

    fitted = trained["targets"][target][horizon_steps]
    model_training_cutoff = pd.Timestamp(fitted["test_start"])
    if origin < model_training_cutoff:
        raise ValueError(
            "Historical forecast origins must be on or after the model's "
            f"training cutoff ({model_training_cutoff})."
        )
    model_name = model_name or fitted["model_name"]
    if model_name not in fitted["candidate_estimators"]:
        raise ValueError(f"Unknown model for this forecast: {model_name}")
    feature_row = make_supervised_features_for_origin(grid, horizon_steps, origin)
    features = feature_row.reindex(columns=fitted["feature_columns"])
    prediction = float(
        _predict(
            model_name,
            fitted["candidate_estimators"][model_name],
            features,
            target,
        )[0]
    )
    offsets = fitted["candidate_interval_offsets"][model_name]
    return {
        "target": target,
        "origin": origin,
        "timestamp": origin + pd.Timedelta(minutes=15 * horizon_steps),
        "horizon_steps": horizon_steps,
        "horizon": HORIZONS[horizon_steps],
        "model": model_name,
        "prediction": prediction,
        "lower_80": max(0.0, prediction + offsets["lower_80_offset"]),
        "upper_80": max(0.0, prediction + offsets["upper_80_offset"]),
        "lower_95": max(0.0, prediction + offsets["lower_95_offset"]),
        "upper_95": max(0.0, prediction + offsets["upper_95_offset"]),
    }


def make_supervised_features_for_origin(
    grid: pd.DataFrame,
    horizon_steps: int,
    origin: pd.Timestamp,
) -> pd.DataFrame:
    from src.features.engineering import build_features

    return build_features(grid, horizon_steps).loc[[origin]]


def forecast_all_horizons(
    trained: dict,
    grid: pd.DataFrame,
    target: str,
    origin: pd.Timestamp | None = None,
    model_name: str | None = None,
) -> pd.DataFrame:
    results = [
        forecast_at(trained, grid, target, horizon, origin, model_name)
        for horizon in HORIZONS
    ]
    return pd.DataFrame(results)
