import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import load_ticket_data, regularize_intervals
from src.evaluation.metrics import regression_metrics
from src.features.engineering import build_features, make_supervised
from src.models.forecasting import forecast_at
from src.models.forecasting import train_forecaster


class DataPipelineTests(unittest.TestCase):
    def test_missing_intervals_remain_missing_and_zero_stays_zero(self):
        source = pd.DataFrame(
            {
                "Timestamp": ["2026-01-01 00:00", "2026-01-01 00:30"],
                "Sales Count": [0, 4],
                "Redemption Count": [1, 0],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tickets.csv"
            source.to_csv(path, index=False)
            data, quality = load_ticket_data(path)
        grid = regularize_intervals(data)
        self.assertEqual(len(grid), 3)
        self.assertEqual(grid.iloc[0]["Sales Count"], 0)
        self.assertTrue(pd.isna(grid.iloc[1]["Sales Count"]))
        self.assertEqual(quality["zero_sales_rows"], 1)

    def test_duplicate_timestamps_are_aggregated_and_reported(self):
        source = pd.DataFrame(
            {
                "Timestamp": ["2026-01-01 00:00", "2026-01-01 00:00"],
                "Sales Count": [2, 3],
                "Redemption Count": [1, 4],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tickets.csv"
            source.to_csv(path, index=False)
            data, quality = load_ticket_data(path)
        self.assertEqual(len(data), 1)
        self.assertEqual(data.iloc[0]["Sales Count"], 5)
        self.assertEqual(quality["duplicate_timestamps_aggregated"], 1)

    def test_unaligned_timestamp_is_rejected(self):
        source = pd.DataFrame(
            {
                "Timestamp": ["2026-01-01 00:07"],
                "Sales Count": [2],
                "Redemption Count": [1],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tickets.csv"
            source.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "not aligned"):
                load_ticket_data(path)


class ForecastFeatureTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("2026-01-01", periods=30, freq="15min")
        self.grid = pd.DataFrame(
            {
                "Sales Count": np.arange(30, dtype=float),
                "Redemption Count": np.arange(30, dtype=float) * 2,
            },
            index=self.index,
        )
        self.grid.index.name = "Timestamp"

    def test_lags_use_only_values_at_or_before_forecast_origin(self):
        features = build_features(self.grid, horizon_steps=1)
        self.assertEqual(features.iloc[5]["Sales Count_lag_1"], 5)
        self.assertEqual(features.iloc[5]["Sales Count_lag_2"], 4)
        changed_future = self.grid.copy()
        changed_future.iloc[6:, 0] = 10000
        changed_features = build_features(changed_future, horizon_steps=1)
        pd.testing.assert_series_equal(features.iloc[:6, 0], changed_features.iloc[:6, 0])

    def test_direct_horizon_targets_exact_future_observations(self):
        x, y, target_times = make_supervised(
            self.grid, "Sales Count", horizon_steps=4
        )
        self.assertEqual(y.iloc[5], self.grid.iloc[9]["Sales Count"])
        self.assertEqual(target_times[5], self.index[9])
        self.assertEqual(x.index[5], self.index[5])

    def test_forecast_requires_observed_origin(self):
        model = {
            "targets": {
                "Sales Count": {
                    1: {
                        "model_name": "naive",
                        "test_start": self.index[0],
                        "estimator": None,
                        "candidate_estimators": {
                            name: None
                            for name in (
                                "naive",
                                "moving_average",
                                "seasonal_naive_1d",
                                "seasonal_naive_1w",
                                "linear_regression",
                                "hist_gradient_boosting",
                            )
                        },
                        "feature_columns": list(build_features(self.grid, 1).columns),
                        "candidate_interval_offsets": {
                            name: {
                                "lower_80_offset": -1,
                                "upper_80_offset": 1,
                                "lower_95_offset": -2,
                                "upper_95_offset": 2,
                            }
                            for name in (
                                "naive",
                                "moving_average",
                                "seasonal_naive_1d",
                                "seasonal_naive_1w",
                                "linear_regression",
                                "hist_gradient_boosting",
                            )
                        },
                        "interval_offsets": {
                            "lower_80_offset": -1,
                            "upper_80_offset": 1,
                            "lower_95_offset": -2,
                            "upper_95_offset": 2,
                        },
                    }
                }
            }
        }
        self.assertEqual(
            forecast_at(model, self.grid, "Sales Count", 1)["prediction"], 29
        )
        with self.assertRaisesRegex(ValueError, "observed"):
            forecast_at(
                model,
                self.grid.assign(**{"Sales Count": self.grid["Sales Count"].mask(
                    self.grid.index == self.index[10]
                )}),
                "Sales Count",
                1,
                self.index[10],
            )

    def test_metrics_handle_zero_observations(self):
        metrics = regression_metrics([0, 2, 4], [1, 2, 5])
        self.assertAlmostEqual(metrics["MAE"], 2 / 3)
        self.assertAlmostEqual(metrics["WAPE"], 2 / 6 * 100)
        self.assertAlmostEqual(metrics["sMAPE"], (200 + 0 + 2 / 9 * 100) / 3)

    def test_training_selects_on_validation_and_reports_held_out_test(self):
        index = pd.date_range("2026-01-01", periods=1200, freq="15min", name="Timestamp")
        sequence = np.arange(len(index), dtype=float)
        grid = pd.DataFrame(
            {
                "Sales Count": 4 + sequence % 11,
                "Redemption Count": 2 + sequence % 7,
            },
            index=index,
        )
        artifact = train_forecaster(grid, rolling_folds=1)
        selected = artifact["targets"]["Sales Count"][1]
        self.assertIn(selected["model_name"], [row["model"] for row in selected["candidate_metrics"]])
        self.assertGreater(len(selected["test_predictions"]), 0)
        self.assertGreaterEqual(selected["test_metrics"]["MAE"], 0)
        self.assertEqual(len(selected["rolling_origin_metrics"]), 6)
        forecast = forecast_at(artifact, grid, "Sales Count", 1)
        self.assertGreaterEqual(forecast["lower_95"], 0)
        self.assertLessEqual(forecast["lower_95"], forecast["upper_95"])
        with self.assertRaisesRegex(ValueError, "training cutoff"):
            forecast_at(artifact, grid, "Sales Count", 1, origin=index[0])


if __name__ == "__main__":
    unittest.main()
