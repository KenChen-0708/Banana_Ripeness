from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_starts_without_exceptions():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run(timeout=60)
    assert not app.exception
    assert any("Banana Ripeness" in title.value for title in app.title)
    assert app.sidebar.radio
