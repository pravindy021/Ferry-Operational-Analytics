import numpy as np


def regression_metrics(actual, predicted) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[valid]
    predicted = predicted[valid]
    if not len(actual):
        return {"MAE": float("nan"), "RMSE": float("nan"), "MAPE": float("nan"), "WAPE": float("nan"), "sMAPE": float("nan")}

    errors = actual - predicted
    absolute_errors = np.abs(errors)
    nonzero = actual != 0
    mape = (
        float(np.mean(absolute_errors[nonzero] / np.abs(actual[nonzero])) * 100)
        if nonzero.any()
        else float("nan")
    )
    denominator = np.abs(actual).sum()
    wape = float(absolute_errors.sum() / denominator * 100) if denominator else float("nan")
    smape_denominator = np.abs(actual) + np.abs(predicted)
    smape_terms = np.divide(
        2 * absolute_errors,
        smape_denominator,
        out=np.zeros_like(absolute_errors),
        where=smape_denominator != 0,
    )
    return {
        "MAE": float(absolute_errors.mean()),
        "RMSE": float(np.sqrt(np.mean(np.square(errors)))),
        "MAPE": mape,
        "WAPE": wape,
        "sMAPE": float(smape_terms.mean() * 100),
    }
