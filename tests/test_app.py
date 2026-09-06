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
    assert "render_uploaded_image_preview(images, image_metadata)" in source


def test_assessment_preview_and_crop_use_reference_card_controls():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")

    assert "Filename" in source
    assert "Dimensions" in source
    assert "File size" in source
    assert "Crop Banana Before Assessment <small>(Optional)</small>" in source
    assert "Manually crop banana" in source
    assert 'st.button("Run Assessment  →"' in source


def test_assessment_settings_follow_preview_and_crop_controls():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")

    assert source.index("render_uploaded_image_preview(images, image_metadata)") < source.index("Optional benchmark, calibration, and forecast settings")
    assert source.index("render_manual_crop_control(filename, original_image)") < source.index("Optional benchmark, calibration, and forecast settings")


def test_manual_crop_does_not_render_a_duplicate_original_image_preview():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    manual_crop_section = source[source.index("def render_manual_crop_control") : source.index("def render_uploaded_image_preview")]

    assert 'caption="Original image"' not in manual_crop_section
    assert "st_cropper(" in manual_crop_section
    assert 'caption="Cropped preview"' in manual_crop_section
