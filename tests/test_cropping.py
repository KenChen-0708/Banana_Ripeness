import numpy as np
import pytest

from src.cropping import CropBox, CropValidationError, crop_from_component_box, crop_image, validate_crop_box
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