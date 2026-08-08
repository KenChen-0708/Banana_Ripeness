from __future__ import annotations

from .base import BaseBananaPipeline
from .glcm_rf import GLCMMorphologyRandomForestPipeline
from .hsv_rf import HSVHistogramRandomForestPipeline
from .hsv_svm import HSVSegmentationSVMPipeline
from .lab_knn import LabKMeansKNNPipeline

PIPELINE_CLASSES: dict[str, type[BaseBananaPipeline]] = {
    HSVSegmentationSVMPipeline.key: HSVSegmentationSVMPipeline,
    LabKMeansKNNPipeline.key: LabKMeansKNNPipeline,
    HSVHistogramRandomForestPipeline.key: HSVHistogramRandomForestPipeline,
    GLCMMorphologyRandomForestPipeline.key: GLCMMorphologyRandomForestPipeline,
}


def create_pipeline(key: str) -> BaseBananaPipeline:
    try:
        return PIPELINE_CLASSES[key]()
    except KeyError as exc:
        raise ValueError(f"Unknown pipeline '{key}'. Available: {', '.join(PIPELINE_CLASSES)}") from exc

