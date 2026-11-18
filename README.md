# Ferry Intelligence

Short-term ticket-demand forecasting and decision support for Toronto Island
Ferry operations. This extends the existing historical analytics project with
a reusable forecasting pipeline and a small, framework-neutral browser
dashboard. The new dashboard uses a Python standard-library HTTP server and
plain HTML, CSS, and JavaScript; it does not use Streamlit.

## What it provides

- Independent forecasts for **Sales Count** and **Redemption Count** at 15,
  30, 60, and 120 minutes.
- Naive, moving-average, daily/weekly seasonal-naive, linear-regression, and
  histogram-gradient-boosting candidates.
- A chronological 70/15/15 train/validation/test strategy, expanding
  rolling-origin validation, and horizon-wise MAE, RMSE, MAPE, and WAPE.
- Validation-selected models, untouched test reporting, and validation
  residual-based 80% and 95% prediction intervals.
- Historical activity charts, model comparisons, a held-out actual-vs-predicted
  view, demand-level indicators, and downloadable evaluation results.
- Explicit data-quality reporting. Missing intervals remain missing; they are
  not interpolated or converted to zero.

## Run locally

From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train_forecaster.py
python run_dashboard.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The server binds to
loopback only. If a matching saved model artifact already exists, training
reuses it. Pass `--force` to `train_forecaster.py` to retrain, or use
`--rolling-folds 0` to disable rolling-origin checks. Changing that option
causes the artifact to be regenerated on the next run.

The source dataset expected by default is
`data/Toronto Island Ferry Ticket Counts(1).csv`. Forecasting artifacts and
generated evaluation files are stored under the ignored
`data/model_artifacts/` directory and are not committed.

## Validation

Run the focused test suite:

```powershell
python -m unittest discover -s tests -v
```

## Forecast interpretation

The dashboard forecasts ticket sales and ticket redemptions separately.
Prediction intervals are empirical, horizon-specific intervals based on
validation residuals; they are not model confidence scores or guaranteed
coverage. Demand levels use historical P50/P75/P90 thresholds, not vessel
capacity or safety limits. The latest source observation is shown in the
dashboard so users can judge data freshness.

The dataset does not contain sailing schedules, passenger counts, vessel
capacity, onboard occupancy, weather, or service disruptions. Therefore, the
system provides monitoring and readiness signals only; it does not recommend
exact departures, ferry capacity, or passenger loads. Missing time slots may
reflect operating schedules or collection gaps; their causes are not known.

## Existing project

The original historical analytics, preprocessing, and EDA files are retained.
The predictive dashboard is a separate interface so the existing application
does not need to be rewritten or used to access forecasting.

See [the architecture overview](docs/ARCHITECTURE.md),
[forecasting methodology](docs/FORECASTING_METHODOLOGY.md), and
[data dictionary](docs/DATA_DICTIONARY.md).
