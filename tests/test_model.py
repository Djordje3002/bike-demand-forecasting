import numpy as np
import pandas as pd
import pytest

from bike_forecast.model import features, predict


def history():
    return pd.DataFrame(
        {
            "dteday": pd.date_range("2011-01-01", periods=70),
            "cnt": np.arange(70, dtype=float),
            "holiday": 0,
            "casual": 10000,
            "registered": 10000,
            "temp": 10000,
        }
    )


def test_future_targets_cannot_change_past_features():
    frame = history()
    before = features(frame)
    frame.loc[40:, "cnt"] = 999999
    after = features(frame)
    pd.testing.assert_frame_equal(before.loc[:40], after.loc[:40])


def test_rolls_end_yesterday_and_seasonal_lag_is_seven_days():
    result = features(history())
    assert result.loc[40, "mean_7"] == np.mean(np.arange(33, 40))
    assert result.loc[40, "lag_7"] == 33
    assert result.loc[40, "lag_1"] == 39


def test_leaky_same_day_columns_are_absent():
    assert not {"cnt", "casual", "registered", "temp"} & set(features(history()).columns)


@pytest.mark.parametrize("counts", [[1] * 27, [-1] * 28, [float("nan")] * 28])
def test_invalid_history_rejected(tmp_path, counts):
    with pytest.raises(ValueError):
        predict(tmp_path, {"date": "2013-01-01", "previous_28_days": counts})
