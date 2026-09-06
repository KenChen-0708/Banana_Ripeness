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
    assert "render_uploaded_image_preview(images, image_metadata, crop_states, selected_filename)" in source


def test_assessment_preview_and_crop_use_reference_card_controls():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")

    assert "Filename" in source
    assert "Dimensions" in source
    assert "File size" in source
    assert "Crop Banana Before Assessment <small>(Optional)</small>" in source
    assert "Manually crop banana" in source
    assert 'st.button("Run Assessment  →"' in source
    assert source.count("Crop Banana Before Assessment <small>(Optional)</small>") == 1


def test_assessment_settings_follow_preview_and_crop_controls():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")

    assert source.index("render_uploaded_image_preview(images, image_metadata, crop_states, selected_filename)") < source.index("Optional benchmark, calibration, and forecast settings")
    assert source.index("render_manual_crop_control(images, crop_states)") < source.index("Optional benchmark, calibration, and forecast settings")


def test_manual_crop_does_not_render_a_duplicate_original_image_preview():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    manual_crop_section = source[source.index("def render_manual_crop_control") : source.index("def render_uploaded_image_preview")]

    assert "st_cropper(" in manual_crop_section
    assert 'st.markdown("**Cropped Preview**")' in manual_crop_section
    assert "st.image(pending_crop, width=400)" in manual_crop_section
    assert "resize_for_crop_editor(original_image)" in manual_crop_section
    assert "should_resize_image=False" in manual_crop_section


def test_assessment_crop_supports_per_image_selection_and_mixed_batch_inputs():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")

    crop_section = source[source.index("def render_manual_crop_control") : source.index("def render_uploaded_image_preview")]
    assert 'if mode == "Use original image":' in crop_section
    assert "if len(filenames) > 1:" in crop_section
    assert crop_section.index('if mode == "Use original image":') < crop_section.index('"Select image to edit"')
    assert "assessment_crop_states" in source
    assert "analysis_images_for_assessment(images, crop_states)" in source
    assert "for filename, image in analysis_images.items()" in source
    assert "st.image(pending_crop, width=400)" in source


def test_inspection_selection_reuses_matching_saved_assessment_results():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")

    assert "def assessment_input_key" in source
    assert 'st.session_state.get("assessment_result")' in source
    assert 'saved_result["input_key"] != input_key' in source
    assert 'all_records = saved_result["records"]' in source
    assert "Deep-Dive Pipeline Explorer" in source
