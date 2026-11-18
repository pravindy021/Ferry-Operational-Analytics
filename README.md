# Ferry Intelligence

Short-term ferry ticket-demand forecasting and predictive decision support for
Toronto Island Ferry operations. The Streamlit application combines historical
analytics with the project's existing forecasting pipeline.

## Dashboard features

- **Overview:** historical sales and redemptions, latest available observation,
  recent activity, and operational KPIs.
- **Demand Forecasting:** separate direct forecasts for sales and redemptions at
  15-minute, 30-minute, 1-hour, and 2-hour horizons, with held-out actual-vs-
  predicted charts.
- **Prediction Intervals:** horizon-specific 80% and 95% validation-residual
  bounds and point forecasts.
- **Model Performance:** MAE, RMSE, MAPE, candidate-model comparisons,
  chronological validation results, and rolling-origin metrics.
- **Operational Insights:** relative demand-spike flags, historical demand
  patterns, and decision-support notes.
- Historical trends, anomalies, data-quality reporting, CSV upload for
  historical analysis, and downloadable reports.

Forecasting uses the existing direct-horizon feature engineering and candidate
models (naive, moving-average, seasonal-naive, linear regression, and histogram
gradient boosting). Models are selected by chronological validation MAE using
the existing 70/15/15 train/validation/test split. Missing time intervals remain
missing; they are not interpolated or converted to zero. Forecasting is enabled
for the committed Toronto Island Ferry dataset. Uploaded CSVs are for
historical analysis and are not silently used as model input.

## Run locally

Use Python 3.13 to match the included, pre-trained model artifact. From the
repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The app loads the tracked dataset and cached model at
`data/model_artifacts/forecaster.joblib`. Streamlit caches loaded data and
models between reruns and caches generated forecasts. If the artifact is
absent or does not match the source dataset, the app uses the existing training
pipeline once and saves a replacement artifact. To generate or refresh it
explicitly, run:

```powershell
python train_forecaster.py
```

The source data is `data/Toronto Island Ferry Ticket Counts(1).csv`. Forecast
artifacts are stored in `data/model_artifacts/`; the trained forecaster is
included in the repository so Streamlit Community Cloud does not need to
retrain on startup. Generated evaluation and metadata files remain ignored.

## Deploy to Streamlit Community Cloud

1. Push this repository and its changes to GitHub. Include
   `data/model_artifacts/forecaster.joblib`; it is a required deployment asset.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) and
   choose **Create app**.
3. Select `pravindy021/Ferry-Operational-Analytics`, the branch you pushed
   (usually `main`), and set **Main file path** to `app.py`.
4. In the app's advanced settings, select **Python 3.13**, then deploy.
   Community Cloud installs dependencies from `requirements.txt`.
5. After deployment, open the app and confirm the Overview and Demand
   Forecasting pages show the committed source data and model results.

No secrets or external services are required. The source CSV and matching
pre-trained model artifact must both be present in the deployed GitHub branch.

## Forecast interpretation

Prediction intervals are empirical, horizon-specific ranges based on
validation residuals; they are not confidence scores or guaranteed coverage.
Demand levels use historical P50/P75/P90 thresholds, not vessel capacity or
safety limits. The dataset does not contain sailing schedules, passenger
counts, vessel capacity, weather, or service disruptions. Therefore, the app
provides monitoring and readiness signals only; it does not recommend exact
departures, ferry capacity, or passenger loads. Ticket sales and redemptions
are separate events and must not be interpreted as unique passengers.

## Validation

Run the focused forecasting tests:

```powershell
python -m unittest discover -s tests -v
```

See [the architecture overview](docs/ARCHITECTURE.md),
[forecasting methodology](docs/FORECASTING_METHODOLOGY.md), and
[data dictionary](docs/DATA_DICTIONARY.md).
