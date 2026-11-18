# Architecture

## Runtime flow

1. `src/data/loader.py` validates the source CSV, sorts observations, reports
   duplicate timestamps, and aggregates duplicate timestamp counts.
2. `regularize_intervals` creates a 15-minute index while leaving unobserved
   slots missing.
3. `src/features/engineering.py` builds lag, rolling, and calendar features
   from values at or before each forecast origin.
4. `src/models/forecasting.py` fits direct models for each target and horizon,
   selects by validation MAE, and reports final test results separately.
5. `src/models/artifacts.py` fingerprints the source CSV and caches trained
   estimators and evaluation results in `data/model_artifacts/`.
6. `src/web/server.py` serves JSON endpoints and the static dashboard. The
   server binds only to `127.0.0.1`.

## Main endpoints

| Endpoint | Purpose |
|---|---|
| `/api/overview` | Coverage, high-level metrics, and data-quality summary |
| `/api/history?days=14` | Recent observed intervals (1–365 days) |
| `/api/forecast?target=Sales%20Count&model=best` | Direct forecasts for all supported horizons |
| `/api/models` | Validation candidates, rolling-origin checks, and test scores |
| `/api/backtest?target=Sales%20Count&horizon=1` | Chronological test actuals and predictions |
| `/api/evaluation.csv` | Downloadable model comparison table |

Forecast origins must be observed 15-minute timestamps and must be at or after
the selected model's training cutoff. This prevents a historical interactive
forecast from silently using future training data.

## Preserved modules

The original `app.py`, `src/data_cleaning.py`, and `src/eda.py` remain in place.
The new browser dashboard is separate and does not replace the historical
analytics code.
