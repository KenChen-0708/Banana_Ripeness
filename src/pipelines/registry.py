from __future__ import annotations

from .base import BaseBananaPipeline
from .member_a_hsv_svm import HSVSegmentationSVMPipeline
from .member_b_lab_knn import LabKMeansKNNPipeline
from .member_c_hsv_rf import HSVHistogramRandomForestPipeline
from .member_d_glcm_rf import GLCMMorphologyRandomForestPipeline

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
