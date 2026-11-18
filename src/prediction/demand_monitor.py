import pandas as pd


def demand_level(value: float, thresholds: dict[str, float]) -> str:
    if value < thresholds["moderate"]:
        return "Low"
    if value < thresholds["high"]:
        return "Moderate"
    if value < thresholds["very_high"]:
        return "High"
    return "Very High"


def operational_note(level: str, is_spike: bool) -> str:
    if is_spike:
        return "Review operational readiness and monitor for an unusual demand spike."
    if level == "Very High":
        return "Prepare staff and monitor upcoming demand."
    if level == "High":
        return "Monitor demand and review staffing readiness."
    return "Continue routine monitoring."


def classify_forecast(
    forecasts: pd.DataFrame,
    historical_values: pd.Series,
) -> pd.DataFrame:
    """Classify forecasts against historical percentiles, not vessel capacity."""
    if historical_values.empty:
        raise ValueError("Historical observations are required for thresholds.")
    thresholds = historical_values.quantile([0.50, 0.75, 0.90]).to_dict()
    threshold_map = {
        "moderate": float(thresholds[0.50]),
        "high": float(thresholds[0.75]),
        "very_high": float(thresholds[0.90]),
    }
    result = forecasts.copy()
    result["Demand Level"] = result["prediction"].map(
        lambda value: demand_level(float(value), threshold_map)
    )
    result["Spike Warning"] = result["prediction"] >= threshold_map["very_high"]
    result["Operational Note"] = result.apply(
        lambda row: operational_note(
            row["Demand Level"], bool(row["Spike Warning"])
        ),
        axis=1,
    )
    result.attrs["historical_thresholds"] = threshold_map
    return result
