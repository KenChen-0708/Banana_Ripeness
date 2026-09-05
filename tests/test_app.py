from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_starts_without_exceptions():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run(timeout=60)
    assert not app.exception
    assert any("Assessment workspace" in markdown.value for markdown in app.markdown)
    assert app.sidebar.radio
    assert app.sidebar.radio[0].options == ["Assessment", "Live Assessment", "Model evaluation", "Model training"]


def test_live_assessment_page_exposes_capture_and_realtime_options():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run(timeout=60)
    app.sidebar.radio[0].set_value("Live Assessment").run(timeout=60)

    assert not app.exception
    assert any("Live assessment" in markdown.value for markdown in app.markdown)
    assert [tab.label for tab in app.tabs] == ["Image Capture", "Real-Time Assessment"]
