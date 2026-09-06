import numpy as np
import pytest

from src.cropping import (
    CropBox,
    CropValidationError,
    analysis_images_for_assessment,
    confirm_crop_state,
    crop_box_from_display,
    crop_box_to_display,
    crop_from_component_box,
    crop_image,
    image_crop_key,
    reconcile_crop_states,
    resize_for_crop_editor,
    reset_crop_state,
    validate_crop_box,
)
from src.live_assessment import assess_image
from src.types import PredictionRecord


@pytest.fixture
def source_image() -> np.ndarray:
    return np.arange(120 * 160 * 3, dtype=np.uint8).reshape(120, 160, 3)


def test_crop_uses_correct_coordinates_and_dimensions(source_image):
    cropped = crop_image(source_image, CropBox(left=20, top=30, width=80, height=60))

    assert cropped.shape == (60, 80, 3)
    assert np.array_equal(cropped, source_image[30:90, 20:100])


def test_crop_does_not_change_original_and_reset_can_use_original(source_image):
    original = source_image.copy()
    cropped = crop_image(source_image, CropBox(left=20, top=30, width=80, height=60))
    cropped[0, 0] = 0

    assert np.array_equal(source_image, original)
    reset_analysis_image = source_image
    assert np.array_equal(reset_analysis_image, original)


@pytest.mark.parametrize(
    ("box", "message"),
    [
        (CropBox(left=0, top=0, width=0, height=60), "greater than zero"),
        (CropBox(left=0, top=0, width=20, height=60), "too small"),
        (CropBox(left=100, top=70, width=80, height=60), "inside the original"),
        (CropBox(left=-1, top=0, width=60, height=60), "must not be negative"),
    ],
)
def test_invalid_crops_are_rejected(source_image, box, message):
    with pytest.raises(CropValidationError, match=message):
        crop_image(source_image, box)


def test_component_crop_box_is_converted_to_validated_crop(source_image):
    cropped = crop_from_component_box(source_image, {"left": 20, "top": 30, "width": 80, "height": 60})

    assert np.array_equal(cropped, source_image[30:90, 20:100])


def test_crop_editor_display_is_bounded_without_changing_original_image():
    original = np.full((1200, 1600, 3), 100, dtype=np.uint8)
    display = resize_for_crop_editor(original, maximum_dimension=420)

    assert display.shape == (315, 420, 3)
    assert original.shape == (1200, 1600, 3)


def test_display_crop_coordinates_map_back_to_full_resolution_crop():
    original = np.full((1200, 1600, 3), 100, dtype=np.uint8)
    display = resize_for_crop_editor(original, maximum_dimension=400)
    original_box = CropBox(left=160, top=120, width=960, height=600)

    display_coords = crop_box_to_display(original_box, original, display)
    mapped_box = crop_box_from_display(
        {"left": display_coords[0], "top": display_coords[2], "width": display_coords[1] - display_coords[0], "height": display_coords[3] - display_coords[2]},
        original,
        display,
    )

    assert mapped_box == original_box
    assert crop_image(original, mapped_box).shape == (600, 960, 3)


def test_confirmed_crop_is_passed_to_every_existing_assessment_pipeline(source_image):
    analysis_image = crop_image(source_image, CropBox(left=20, top=30, width=80, height=60))

    class RecordingPipeline:
        def __init__(self, name):
            self.display_name = name
            self.images = []

        def predict(self, image, filename):
            self.images.append(image)
            return PredictionRecord(filename, "ripe", 0.9, {"ripe": 0.9}, self.display_name, 1.0)

    pipelines = [RecordingPipeline(f"Pipeline {index}") for index in range(4)]
    models = {str(index): (pipeline, {}) for index, pipeline in enumerate(pipelines)}
    records = assess_image(models, analysis_image, "cropped.jpg")

    assert len(records) == 4
    assert all(pipeline.images == [analysis_image] for pipeline in pipelines)
    assert all(pipeline.images[0].shape == (60, 80, 3) for pipeline in pipelines)


def test_all_four_real_pipeline_classes_predict_after_cropping():
    from src.constants import CLASS_NAMES
    from src.pipelines import PIPELINE_CLASSES

    images, labels = [], []
    for label, colour in zip(CLASS_NAMES, ((50, 180, 50), (230, 210, 40), (160, 110, 35), (65, 45, 30)), strict=True):
        for brightness_shift in (0, 10):
            image = np.full((120, 160, 3), 245, dtype=np.uint8)
            shifted_colour = tuple(min(255, value + brightness_shift) for value in colour)
            image[20:100, 20:140] = shifted_colour
            images.append(crop_image(image, CropBox(left=20, top=20, width=120, height=80)))
            labels.append(label)

    for pipeline_class in PIPELINE_CLASSES.values():
        pipeline = pipeline_class().fit(images, labels)
        record = pipeline.predict(images[0], "cropped.jpg")
        assert record.predicted_class in CLASS_NAMES


def test_multiple_images_keep_independent_confirmed_crop_states(source_image):
    second = np.full_like(source_image, 200)
    states = confirm_crop_state({}, "first.jpg", source_image, CropBox(20, 30, 80, 60))
    states = confirm_crop_state(states, "second.jpg", second, CropBox(40, 20, 70, 70))

    first_state = states[image_crop_key("first.jpg", source_image)]
    second_state = states[image_crop_key("second.jpg", second)]
    assert first_state["crop_box"] == CropBox(20, 30, 80, 60)
    assert second_state["crop_box"] == CropBox(40, 20, 70, 70)
    assert first_state["confirmed_crop"].shape == (60, 80, 3)
    assert second_state["confirmed_crop"].shape == (70, 70, 3)


def test_reset_only_removes_selected_image_crop_state(source_image):
    second = np.full_like(source_image, 200)
    states = confirm_crop_state({}, "first.jpg", source_image, CropBox(20, 30, 80, 60))
    states = confirm_crop_state(states, "second.jpg", second, CropBox(40, 20, 70, 70))
    reset = reset_crop_state(states, "first.jpg", source_image)

    assert image_crop_key("first.jpg", source_image) not in reset
    assert image_crop_key("second.jpg", second) in reset


def test_analysis_images_mix_confirmed_crops_and_original_images_without_mutation(source_image):
    second = np.full_like(source_image, 200)
    original_first, original_second = source_image.copy(), second.copy()
    images = {"first.jpg": source_image, "second.jpg": second}
    states = confirm_crop_state({}, "first.jpg", source_image, CropBox(20, 30, 80, 60))
    analysis_images = analysis_images_for_assessment(images, states)

    assert analysis_images["first.jpg"].shape == (60, 80, 3)
    assert analysis_images["second.jpg"] is second
    assert np.array_equal(source_image, original_first)
    assert np.array_equal(second, original_second)


def test_reconcile_crop_states_removes_images_no_longer_uploaded(source_image):
    second = np.full_like(source_image, 200)
    states = confirm_crop_state({}, "first.jpg", source_image, CropBox(20, 30, 80, 60))
    states = confirm_crop_state(states, "second.jpg", second, CropBox(40, 20, 70, 70))

    active_states = reconcile_crop_states(states, {"second.jpg": second})
    assert image_crop_key("first.jpg", source_image) not in active_states
    assert image_crop_key("second.jpg", second) in active_states


def test_all_pipelines_receive_mixed_cropped_and_original_batch_images(source_image):
    second = np.full_like(source_image, 200)
    images = {"cropped.jpg": source_image, "original.jpg": second}
    states = confirm_crop_state({}, "cropped.jpg", source_image, CropBox(20, 30, 80, 60))
    analysis_images = analysis_images_for_assessment(images, states)

    class RecordingPipeline:
        def __init__(self):
            self.display_name = "Recording"
            self.received = []

        def predict(self, image, filename):
            self.received.append((filename, image.shape))
            return PredictionRecord(filename, "ripe", 0.9, {"ripe": 0.9}, self.display_name, 1.0)

    pipelines = [RecordingPipeline() for _ in range(4)]
    for pipeline in pipelines:
        for filename, image in analysis_images.items():
            pipeline.predict(image, filename)
        assert pipeline.received == [("cropped.jpg", (60, 80, 3)), ("original.jpg", source_image.shape)]


def test_all_four_real_pipelines_predict_a_mixed_cropped_and_original_batch():
    from src.constants import CLASS_NAMES
    from src.pipelines import PIPELINE_CLASSES

    training_images, labels = [], []
    for label, colour in zip(CLASS_NAMES, ((50, 180, 50), (230, 210, 40), (160, 110, 35), (65, 45, 30)), strict=True):
        for brightness_shift in (0, 10):
            image = np.full((120, 160, 3), 245, dtype=np.uint8)
            image[20:100, 20:140] = tuple(min(255, channel + brightness_shift) for channel in colour)
            training_images.append(image)
            labels.append(label)

    batch_images = {"cropped.jpg": training_images[0], "original.jpg": training_images[1]}
    states = confirm_crop_state({}, "cropped.jpg", training_images[0], CropBox(20, 20, 120, 80))
    mixed_analysis_images = analysis_images_for_assessment(batch_images, states)

    assert mixed_analysis_images["cropped.jpg"].shape == (80, 120, 3)
    assert mixed_analysis_images["original.jpg"] is training_images[1]
    for pipeline_class in PIPELINE_CLASSES.values():
        pipeline = pipeline_class().fit(training_images, labels)
        records = [pipeline.predict(image, filename) for filename, image in mixed_analysis_images.items()]
        assert len(records) == 2
        assert all(record.predicted_class in CLASS_NAMES for record in records)