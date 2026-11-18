# Architecture

## Streamlit application

`app.py` is the deployment and local entry point. It provides the native
Streamlit navigation, historical analytics, forecast visualizations,
evaluation reports, and operational insights. The historical CSV uploader is
for analysis only; model forecasts use the committed Toronto Island Ferry
dataset so a different upload is never silently treated as forecast input.

The forecast data and fitted models are loaded through Streamlit's resource
cache. `src/models/artifacts.py` fingerprints the source CSV and reuses a
matching artifact. If that artifact is missing or stale, it invokes the
existing training pipeline and writes the replacement to
`data/model_artifacts/forecaster.joblib`.
Forecast outputs are cached by source digest and target, so app reruns do not
recompute every horizon.

## Forecasting flow

1. `src/data/loader.py` validates the source CSV, aggregates duplicate
   timestamps, reports data quality, and creates a 15-minute grid that keeps
   unobserved slots missing.
2. `src/features/engineering.py` builds lag, rolling, and calendar features
   using values available at or before each forecast origin.
3. `src/models/forecasting.py` fits direct models for sales and redemptions at
   1, 2, 4, and 8 interval horizons. It chooses each model by validation MAE,
   using chronological 70/15/15 train/validation/test partitions.
4. The selected estimator is evaluated on the held-out test partition.
   Horizon-specific 80% and 95% prediction intervals use empirical validation
   residual quantiles.
5. `src/prediction/demand_monitor.py` classifies forecasts against historical
   P50/P75/P90 values. These are relative activity descriptors, not vessel
   capacity, passenger, or safety thresholds.

## Deployment files

- Entry point: `app.py`
- Dependencies: `requirements.txt`
- Source dataset: `data/Toronto Island Ferry Ticket Counts(1).csv`
- Pre-trained model artifact: `data/model_artifacts/forecaster.joblib`

The dataset and matching model artifact are stored in the repository so Cloud
deployment can start without full-dataset training. Run
`python train_forecaster.py` to intentionally regenerate model artifacts.
