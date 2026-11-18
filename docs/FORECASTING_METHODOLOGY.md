# Forecasting methodology

## Data preparation

The loader requires `Timestamp`, `Sales Count`, and `Redemption Count`.
Unparseable timestamps, unparseable counts, and negative counts are excluded
and counted in the quality report. Timestamps must align to 15-minute
boundaries. Duplicate timestamps are reported and their ticket counts are
summed. Zero is retained as a genuine observed count. The regularized grid
represents missing intervals with missing values, never with interpolated
counts or assumed zeroes.

## Forecast definition and features

Each model is direct: a forecast origin is an observed interval end, and each
target model predicts the count exactly 1, 2, 4, or 8 grid steps later (15
minutes, 30 minutes, 1 hour, or 2 hours). Missing future labels are excluded
from fitting and evaluation. Historical features include lags at 1, 2, 4, 8,
12, 24, 96, and 672 intervals; rolling mean, standard deviation, maximum,
minimum, and median; and cyclical hour/weekday plus calendar features. Feature
values at an origin never use observations after that origin.

## Models and selection

Candidates are naive, four-interval moving average, daily seasonal naive,
weekly seasonal naive, linear regression, and histogram gradient boosting.
Seasonal-naive forecasts fall back to the last observed value when the
corresponding seasonal observation is absent. The supervised rows are sorted
chronologically and partitioned into 70% training, 15% validation, and 15%
test data. Validation MAE chooses the model independently for each target and
horizon. The selected estimator is refit on training plus validation data;
the later test partition is used only for final reporting.

The selected model also receives expanding rolling-origin checks over the
validation period. The default is three folds; use
`python train_forecaster.py --rolling-folds 0` to disable those supplemental
checks.

## Metrics and intervals

MAE and RMSE are measured in tickets per interval. MAPE excludes zero actual
values; WAPE is the sum of absolute error divided by the sum of absolute actual
counts and is undefined when that denominator is zero. sMAPE is also computed
internally. The dashboard displays MAE, RMSE, MAPE, and WAPE without converting
them into unsupported accuracy percentages.

For each model and horizon, the 80% and 95% intervals are formed by adding
empirical validation-residual quantiles to the point forecast and clipping
lower bounds at zero. These are historical residual-based prediction
intervals; distribution shift can affect their future coverage. They are not
confidence scores or formal guarantees.

## Demand monitoring

Forecast labels use historical P50, P75, and P90 for the selected count:
below P50 is Low, P50–P75 is Moderate, P75–P90 is High, and P90 or above is
Very High / a historical-spike warning. These are relative activity
descriptors. They are not occupancy, capacity, safety, or departure thresholds.
