from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pandas as pd

from src.data.loader import TARGET_COLUMNS, load_ticket_data, regularize_intervals
from src.models.artifacts import train_or_load
from src.models.forecasting import HORIZONS, forecast_at
from src.prediction.demand_monitor import classify_forecast

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_DIR / "data" / "Toronto Island Ferry Ticket Counts(1).csv"
STATIC_DIR = PROJECT_DIR / "dashboard"


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


class DashboardState:
    def __init__(self) -> None:
        if not DATASET_PATH.is_file():
            raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
        data, self.quality = load_ticket_data(DATASET_PATH)
        self.grid = regularize_intervals(data)
        artifact_path = (
            PROJECT_DIR / "data" / "model_artifacts" / "forecaster.joblib"
        )
        self.artifact, _ = train_or_load(
            DATASET_PATH, artifact_path, rolling_folds=3
        )
        self.data = data

    def overview(self) -> dict:
        valid = self.data
        latest = valid.iloc[-1]
        activity = valid["Sales Count"] + valid["Redemption Count"]
        return {
            "source": DATASET_PATH.name,
            "first_observation": valid["Timestamp"].min(),
            "latest_observation": latest["Timestamp"],
            "observations": len(valid),
            "grid_intervals": len(self.grid),
            "missing_intervals": int(self.grid["Sales Count"].isna().sum()),
            "sales_total": valid["Sales Count"].sum(),
            "redemptions_total": valid["Redemption Count"].sum(),
            "sales_average": valid["Sales Count"].mean(),
            "redemptions_average": valid["Redemption Count"].mean(),
            "peak_activity": activity.max(),
            "latest_sales": latest["Sales Count"],
            "latest_redemptions": latest["Redemption Count"],
            "latest_activity": latest["Sales Count"] + latest["Redemption Count"],
            "quality": {**self.quality, **self.artifact["quality"]},
        }

    def forecast(
        self,
        target: str,
        origin: pd.Timestamp | None,
        model_name: str | None,
    ) -> list[dict]:
        if target not in TARGET_COLUMNS:
            raise ValueError(f"Target must be one of {', '.join(TARGET_COLUMNS)}.")
        if model_name == "best":
            model_name = None
        output = []
        for horizon in HORIZONS:
            row = forecast_at(
                self.artifact,
                self.grid,
                target,
                horizon,
                origin=origin,
                model_name=model_name,
            )
            forecast_origin = row["origin"]
            frame = classify_forecast(
                pd.DataFrame([row]), self.grid.loc[:forecast_origin, target].dropna()
            )
            record = frame.iloc[0].to_dict()
            record["model_options"] = [
                candidate["model"]
                for candidate in self.artifact["targets"][target][horizon][
                    "candidate_metrics"
                ]
            ]
            output.append(record)
        return output

    def model_report(self) -> dict:
        rows = []
        latest_training_cutoff = None
        for target in TARGET_COLUMNS:
            for horizon, label in HORIZONS.items():
                item = self.artifact["targets"][target][horizon]
                cutoff = pd.Timestamp(item["test_start"])
                if latest_training_cutoff is None or cutoff > latest_training_cutoff:
                    latest_training_cutoff = cutoff
                rows.append(
                    {
                        "target": target,
                        "horizon_steps": horizon,
                        "horizon": label,
                        "selected_model": item["model_name"],
                        "validation": item["candidate_metrics"],
                        "rolling_origin": item["rolling_origin_metrics"],
                        "test": item["test_metrics"],
                    }
                )
        return {
            "validation_method": self.artifact["validation_method"],
            "minimum_forecast_origin": latest_training_cutoff,
            "models": rows,
        }

    def history(self, days: int) -> list[dict]:
        if not 1 <= days <= 365:
            raise ValueError("History range must be between 1 and 365 days.")
        latest = self.grid.index.max()
        start = latest - pd.Timedelta(days=days)
        sample = self.grid.loc[start:latest]
        step = max(1, (len(sample) + 999) // 1000)
        sample = sample.iloc[::step]
        result = sample.reset_index()
        result["Activity"] = result["Sales Count"] + result["Redemption Count"]
        return result.to_dict(orient="records")

    def evaluation_csv(self) -> str:
        return self.artifact["evaluation"].to_csv(index=False)

    def backtest(self, target: str, horizon: int) -> list[dict]:
        if target not in TARGET_COLUMNS:
            raise ValueError(f"Target must be one of {', '.join(TARGET_COLUMNS)}.")
        if horizon not in HORIZONS:
            raise ValueError("Horizon must be 1, 2, 4, or 8 intervals.")
        rows = self.artifact["targets"][target][horizon]["test_predictions"]
        return rows.to_dict(orient="records")


class FerryDashboardHandler(BaseHTTPRequestHandler):
    state: DashboardState

    def do_HEAD(self) -> None:
        route = urlsplit(self.path)
        if route.path in ("/", "/dashboard.css", "/dashboard.js"):
            name = "index.html" if route.path == "/" else route.path.lstrip("/")
            content_type = (
                "text/html; charset=utf-8"
                if name == "index.html"
                else "text/css; charset=utf-8"
                if name.endswith(".css")
                else "text/javascript; charset=utf-8"
            )
            payload = (STATIC_DIR / name).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
        else:
            self.send_error(404, "Route not found.")

    def do_GET(self) -> None:
        route = urlsplit(self.path)
        if route.path == "/":
            self._send_file("index.html", "text/html; charset=utf-8")
        elif route.path == "/dashboard.css":
            self._send_file("dashboard.css", "text/css; charset=utf-8")
        elif route.path == "/dashboard.js":
            self._send_file("dashboard.js", "text/javascript; charset=utf-8")
        elif route.path.startswith("/api/"):
            self._handle_api(route.path, parse_qs(route.query))
        else:
            self.send_error(404, "Route not found.")

    def _handle_api(self, path: str, query: dict[str, list[str]]) -> None:
        try:
            if path == "/api/overview":
                self._send_json(self.state.overview())
            elif path == "/api/forecast":
                target = self._query_value(query, "target", TARGET_COLUMNS[0])
                raw_origin = self._query_value(query, "origin", "")
                origin = pd.Timestamp(raw_origin) if raw_origin else None
                if origin is not None and pd.isna(origin):
                    raise ValueError("Forecast origin is not a valid date and time.")
                model = self._query_value(query, "model", "best")
                self._send_json(
                    {
                        "forecasts": self.state.forecast(target, origin, model),
                        "target": target,
                    }
                )
            elif path == "/api/models":
                self._send_json(self.state.model_report())
            elif path == "/api/backtest":
                target = self._query_value(query, "target", TARGET_COLUMNS[0])
                horizon = int(self._query_value(query, "horizon", "1"))
                self._send_json(
                    {
                        "target": target,
                        "horizon_steps": horizon,
                        "records": self.state.backtest(target, horizon),
                    }
                )
            elif path == "/api/history":
                days = int(self._query_value(query, "days", "14"))
                self._send_json({"records": self.state.history(days)})
            elif path == "/api/evaluation.csv":
                payload = self.state.evaluation_csv().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header(
                    "Content-Disposition",
                    'attachment; filename="ferry_model_evaluation.csv"',
                )
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            else:
                self._send_json({"error": "API route not found."}, status=404)
        except (ValueError, TypeError, KeyError) as error:
            self._send_json({"error": str(error)}, status=400)

    @staticmethod
    def _query_value(query: dict[str, list[str]], key: str, default: str) -> str:
        values = query.get(key, [default])
        return values[0]

    def _send_json(self, value, status: int = 200) -> None:
        payload = json.dumps(_json_value(value), allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_file(self, name: str, content_type: str) -> None:
        payload = (STATIC_DIR / name).read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}")


def serve(port: int = 8000) -> None:
    state = DashboardState()
    handler = type(
        "ConfiguredFerryDashboardHandler",
        (FerryDashboardHandler,),
        {"state": state},
    )
    host = "127.0.0.1"
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Ferry Intelligence dashboard: http://{host}:{port}")
    print(f"Latest observed data: {state.artifact['latest_observation']}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Ferry Intelligence dashboard.")
    finally:
        server.server_close()
