from pathlib import Path

from streamlit.testing.v1 import AppTest

from app import parse_training_progress


def test_dashboard_starts_without_exceptions():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run(timeout=60)
    assert not app.exception
    assert any("Banana Ripeness" in markdown.value for markdown in app.markdown)
    assert app.sidebar.radio


def test_assessment_page_defines_uploaded_image_preview():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    source = app_path.read_text(encoding="utf-8")

    assert "def render_uploaded_image_preview" in source
    assert "render_uploaded_image_preview(images, image_metadata, crop_states, selected_filename)" in source


def test_assessment_accepts_uploads_and_camera_photos_through_one_workflow():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")

    assert '["📁 Upload Images", "📷 Take Photo"]' in source
    assert 'key="assessment_camera_capture"' in source
    assert 'camera_filename = "Camera Capture.jpg"' in source
    assert "images[camera_filename] = decode_image(camera_bytes)" in source
    assert "st.session_state.assessment_selected_image = camera_filename" in source
    assert source.index("images[camera_filename] = decode_image(camera_bytes)") < source.index("analysis_images_for_assessment(images, crop_states)")


def test_assessment_preview_and_crop_use_reference_card_controls():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")

    assert "Filename" in source
    assert "Dimensions" in source
    assert "File size" in source
    assert "Crop Banana Before Assessment <small>(Optional)</small>" in source
    assert "Manually crop banana" in source
    assert 'st.button("Run Assessment' in source
    assert source.count("Crop Banana Before Assessment <small>(Optional)</small>") == 1


def test_assessment_settings_follow_preview_and_crop_controls():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")

    assert source.index("render_uploaded_image_preview(images, image_metadata, crop_states, selected_filename)") < source.index("Optional Benchmark, Calibration, and Forecast Settings")
    assert source.index("render_manual_crop_control(images, crop_states)") < source.index("Optional Benchmark, Calibration, and Forecast Settings")


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
    assert "st.image(gallery_thumbnail(image), width=\"stretch\")" in source


def test_inspection_selection_reuses_matching_saved_assessment_results():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")

    assert "def assessment_input_key" in source
    assert 'st.session_state.get("assessment_result")' in source
    assert 'saved_result["input_key"] != input_key' in source
    assert 'all_records = saved_result["records"]' in source
    assert "Deep-Dive Pipeline Explorer" in source


def test_model_loader_cache_is_invalidated_when_artifacts_are_retrained():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")

    assert "def load_models(model_dir: str, artifact_token" in source
    assert "path.stat().st_mtime_ns" in source
    assert "load_models(str(model_dir), artifact_token)" in source


def test_training_progress_records_are_parsed_and_clamped():
    assert parse_training_progress('__TRAINING_PROGRESS__ {"progress": 0.42, "message": "Fitting"}') == (0.42, "Fitting")
    assert parse_training_progress('__TRAINING_PROGRESS__ {"progress": 2, "message": "Done"}') == (1.0, "Done")
    assert parse_training_progress("ordinary training output") is None


def test_continuous_camera_supports_all_or_one_selected_algorithm():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")

    assert '"Live Camera Algorithm"' in source
    assert 'st.tabs(["🎥 Uploaded Video", "📹 Live Camera"])' in source
    assert "📸 Camera Snapshot" not in source
    assert '["All 4 Algorithms"] + list(algorithm_options)' in source
    assert 'key="live_algorithm_selection"' in source
    assert 'selected_keys = list(models) if selection == "All 4 Algorithms"' in source
    assert "selected_models = {key: models[key] for key in selected_keys}" in source
    assert "SampledFrameAssessor(selected_models, interval_seconds=live_interval)" in source
    assert 'live_key = "-".join(selected_keys)' in source
