"""Past-only features and expanding-window evaluation for daily demand."""

import io

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import archive, figure, regression, save_report


def features(frame):
    frame = frame.sort_values("dteday").copy()
    dates = pd.to_datetime(frame.dteday)
    result = pd.DataFrame(index=frame.index)
    result["weekday"] = dates.dt.day_of_week
    result["month_sin"] = np.sin(2 * np.pi * dates.dt.day_of_year / 365.25)
    result["month_cos"] = np.cos(2 * np.pi * dates.dt.day_of_year / 365.25)
    result["trend"] = (dates - pd.Timestamp("2011-01-01")).dt.days
    result["holiday"] = frame.holiday.astype(int)
    for lag in [1, 7, 14, 28]:
        result[f"lag_{lag}"] = frame.cnt.shift(lag)
    for window in [7, 28]:
        result[f"mean_{window}"] = frame.cnt.shift(1).rolling(window).mean()
    return result


def train(root):
    with archive(root) as source:
        frame = (
            pd.read_csv(io.BytesIO(source.read("day.csv")))
            .sort_values("dteday")
            .reset_index(drop=True)
        )
    X = features(frame).dropna()
    y = frame.loc[X.index, "cnt"]
    cut = len(X) - 120
    candidates = {
        "scaled_ridge": make_pipeline(StandardScaler(), Ridge(alpha=10)),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=180,
            max_leaf_nodes=15,
            l2_regularization=10,
            learning_rate=0.05,
            random_state=42,
        ),
    }
    folds = list(TimeSeriesSplit(n_splits=4, test_size=60).split(X.iloc[:cut]))
    cv, boundaries = {}, []
    for fit, valid in folds:
        boundaries.append(
            {
                "train_end": frame.loc[X.index[fit[-1]], "dteday"],
                "validation_start": frame.loc[X.index[valid[0]], "dteday"],
                "validation_end": frame.loc[X.index[valid[-1]], "dteday"],
            }
        )
    for name, model in candidates.items():
        scores = []
        for fit, valid in folds:
            model.fit(X.iloc[fit], y.iloc[fit])
            scores.append(regression(y.iloc[valid], model.predict(X.iloc[valid]))["mae"])
        cv[name] = {"fold_mae": scores, "mean_mae": float(np.mean(scores))}
    seasonal_scores = [regression(y.iloc[valid], X.iloc[valid].lag_7)["mae"] for _, valid in folds]
    cv["seasonal_naive"] = {
        "fold_mae": seasonal_scores,
        "mean_mae": float(np.mean(seasonal_scores)),
    }
    selected = min(cv, key=lambda name: cv[name]["mean_mae"])
    results = {"seasonal_naive": regression(y.iloc[cut:], X.iloc[cut:].lag_7)}
    for name, model in candidates.items():
        model.fit(X.iloc[:cut], y.iloc[:cut])
        results[name] = regression(y.iloc[cut:], model.predict(X.iloc[cut:]))
    prediction = (
        X.iloc[cut:].lag_7.to_numpy()
        if selected == "seasonal_naive"
        else candidates[selected].predict(X.iloc[cut:])
    )
    plt.figure(figsize=(10, 4))
    dates = pd.to_datetime(frame.loc[X.index[cut:], "dteday"])
    plt.plot(dates, y.iloc[cut:], label="Observed", color="#0f172a")
    plt.plot(dates, prediction, label=selected, color="#2563eb", alpha=0.85)
    plt.ylabel("Daily rentals")
    plt.legend()
    plt.title("One day ahead: final 120 days")
    figure(root, "evaluation.png")
    result = save_report(
        root,
        {
            "raw_rows": len(frame),
            "usable_rows": len(X),
            "split": {"train": cut, "test": 120, "test_start": frame.loc[X.index[cut], "dteday"]},
            "selected": selected,
            "cross_validation": cv,
            "backtest_windows": boundaries,
            "test": results,
        },
    )
    joblib.dump(
        {"model": candidates.get(selected), "selected": selected, "features": X.columns.tolist()},
        root / "artifacts/model.joblib",
    )
    return result


def predict(root, payload):
    history = np.asarray(payload["previous_28_days"], dtype=float)
    if history.shape != (28,) or not np.isfinite(history).all() or (history < 0).any():
        raise ValueError(
            "previous_28_days must contain 28 finite, nonnegative counts, oldest first"
        )
    date = pd.Timestamp(payload["date"])
    frame = pd.DataFrame(
        {
            "dteday": pd.date_range(date - pd.Timedelta(days=28), periods=29),
            "cnt": np.r_[history, np.nan],
            "holiday": [0] * 28 + [int(payload.get("holiday", False))],
        }
    )
    X = features(frame).iloc[[-1]]
    artifact = joblib.load(root / "artifacts/model.joblib")
    value = (
        X.lag_7.iloc[0]
        if artifact["selected"] == "seasonal_naive"
        else artifact["model"].predict(X[artifact["features"]])[0]
    )
    return {
        "date": str(date.date()),
        "predicted_rentals": max(0.0, float(value)),
        "horizon_days": 1,
    }
