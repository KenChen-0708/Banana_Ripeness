import pytest

from src.forecasting import forecast_ripeness_progress


def test_unripe_forecast_uses_baseline_room_temperature_range():
    forecast = forecast_ripeness_progress("unripe", confidence=0.91, blemish_ratio=0.02)

    assert (forecast.days_min, forecast.days_max) == (3, 5)
    assert forecast.headline == "Estimated to become ripe in 3–5 days"
    assert not forecast.manual_review_advised


def test_cool_storage_extends_unripe_forecast_and_ethylene_reduces_it():
    cool = forecast_ripeness_progress("unripe", confidence=0.9, blemish_ratio=0.02, temperature_c=12.0)
    ethylene = forecast_ripeness_progress("unripe", confidence=0.9, blemish_ratio=0.02, near_ethylene_fruit=True)

    assert (cool.days_min, cool.days_max) == (4, 7)
    assert (ethylene.days_min, ethylene.days_max) == (2, 4)


def test_blemished_ripe_fruit_has_shorter_consumption_window():
    forecast = forecast_ripeness_progress("ripe", confidence=0.8, blemish_ratio=0.20, temperature_c=22.0)

    assert (forecast.days_min, forecast.days_max) == (0, 2)
    assert forecast.headline == "Best consumed within 0–2 days"


def test_rotten_or_heavily_blemished_fruit_requires_immediate_review():
    forecast = forecast_ripeness_progress("rotten", confidence=0.95, blemish_ratio=0.01)
    heavily_blemished = forecast_ripeness_progress("ripe", confidence=0.95, blemish_ratio=0.30)

    assert forecast.days_max == 0
    assert heavily_blemished.days_max == 0
    assert forecast.manual_review_advised


def test_forecast_validates_inputs():
    with pytest.raises(ValueError, match="Unsupported ripeness"):
        forecast_ripeness_progress("green", confidence=0.8, blemish_ratio=0.1)
    with pytest.raises(ValueError, match="Confidence"):
        forecast_ripeness_progress("ripe", confidence=1.1, blemish_ratio=0.1)