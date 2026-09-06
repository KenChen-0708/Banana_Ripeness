from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_starts_without_exceptions():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run(timeout=60)
    assert not app.exception
    assert any("Assessment studio" in markdown.value for markdown in app.markdown)
    assert app.sidebar.radio


def test_assessment_page_defines_uploaded_image_preview():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    source = app_path.read_text(encoding="utf-8")

    assert "def render_uploaded_image_preview" in source
    assert "render_uploaded_image_preview(images)" in source
