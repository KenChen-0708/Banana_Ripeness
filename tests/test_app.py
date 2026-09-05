from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest

from app import ensure_explanation, render_explanation
from src.types import PredictionRecord


def test_dashboard_starts_without_exceptions():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run(timeout=60)
    assert not app.exception
    assert any("Assessment workspace" in markdown.value for markdown in app.markdown)
    assert app.sidebar.radio
    app.sidebar.radio[0].set_value("Video Analysis").run(timeout=60)
    assert not app.exception
    assert any("Near-real-time processing" in markdown.value for markdown in app.markdown)


def test_legacy_prediction_explanation_is_upgraded(monkeypatch):
    image = np.full((80, 120, 3), 240, dtype=np.uint8)
    image[25:55, 20:100] = (225, 195, 35)
    record = PredictionRecord(
        filename="legacy.jpg",
        predicted_class="ripe",
        confidence=0.9,
        probabilities={"ripe": 0.9},
        algorithm="Legacy model",
        processing_time_ms=1.0,
        explanation={"fruit_mask": np.ones((80, 120), dtype=np.uint8) * 255},
    )

    class Column:
        def image(self, *args, **kwargs):
            return None

        def metric(self, *args, **kwargs):
            return None

    class Expander:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr("app.st.columns", lambda count: [Column() for _ in range(count)])
    monkeypatch.setattr("app.st.expander", lambda *args, **kwargs: Expander())
    monkeypatch.setattr("app.st.image", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.st.write", lambda *args, **kwargs: None)

    ensure_explanation(image, record)
    assert "enhanced_rgb" in record.explanation
    assert "detection_overlay" in record.explanation
    assert "blemish_ratio" in record.explanation
    assert "fruit_area_ratio" in record.explanation
