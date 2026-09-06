from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RipenessForecast:
    """A transparent shelf-life estimate derived from an image assessment and storage inputs."""

    predicted_class: str
    days_min: int
    days_max: int
    headline: str
    recommendation: str
    storage_effect: str
    manual_review_advised: bool


def forecast_ripeness_progress(
    predicted_class: str,
    confidence: float,
    blemish_ratio: float,
    temperature_c: float = 22.0,
    near_ethylene_fruit: bool = False,
) -> RipenessForecast:
    """Estimate remaining ripening or consumption time using documented, conservative rules.

    This is not a trained time-series model. The classifier determines the current
    stage from image features; this function adjusts stage-based shelf-life ranges
    for surface blemishes, storage temperature, and nearby ethylene-producing fruit.
    """
    stage = predicted_class.strip().lower()
    if stage not in {"unripe", "ripe", "overripe", "rotten"}:
        raise ValueError(f"Unsupported ripeness class: {predicted_class!r}.")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("Confidence must be between 0 and 1.")
    if not 0.0 <= blemish_ratio <= 1.0:
        raise ValueError("Blemish ratio must be between 0 and 1.")
    if not 0.0 <= temperature_c <= 45.0:
        raise ValueError("Storage temperature must be between 0 and 45 °C.")

    if stage == "rotten" or blemish_ratio >= 0.30:
        return RipenessForecast(
            predicted_class=stage,
            days_min=0,
            days_max=0,
            headline="Immediate quality review",
            recommendation="Do not store for later consumption. Discard or compost if decay is confirmed.",
            storage_effect="Surface decay or heavy blemishing overrides the normal shelf-life estimate.",
            manual_review_advised=True,
        )

    base_ranges = {"unripe": (3, 5), "ripe": (1, 3), "overripe": (0, 1)}
    days_min, days_max = base_ranges[stage]
    adjustments: list[str] = []

    if temperature_c < 18.0:
        days_min += 1
        days_max += 2
        adjustments.append("Cool storage slows ripening")
    elif temperature_c >= 28.0:
        days_min = max(0, days_min - 1)
        days_max = max(0, days_max - 1)
        adjustments.append("Warm storage accelerates ripening")
    else:
        adjustments.append("Moderate room temperature uses the baseline estimate")

    if near_ethylene_fruit:
        days_min = max(0, days_min - 1)
        days_max = max(0, days_max - 1)
        adjustments.append("Nearby ethylene-producing fruit accelerates ripening")

    if blemish_ratio >= 0.15:
        days_min = max(0, days_min - 1)
        days_max = max(0, days_max - 1)
        adjustments.append("Moderate surface blemishing shortens the estimate")

    if stage == "unripe":
        headline = f"Estimated to become ripe in {days_min}–{days_max} days"
        recommendation = "Keep at room temperature until yellow, unless you need to slow ripening."
    elif stage == "ripe":
        headline = f"Best consumed within {days_min}–{days_max} days"
        recommendation = "Consume soon; refrigerate to slow further ripening after the desired stage is reached."
    else:
        headline = f"Use within {days_min}–{days_max} days"
        recommendation = "Use immediately in cooking or smoothies, or discard if odour, mould, or soft decay is present."

    return RipenessForecast(
        predicted_class=stage,
        days_min=days_min,
        days_max=days_max,
        headline=headline,
        recommendation=recommendation,
        storage_effect=". ".join(adjustments) + ".",
        manual_review_advised=confidence < 0.60,
    )