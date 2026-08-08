from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from .constants import CLASS_NAMES
from .pipelines import BaseBananaPipeline

MANIFEST_SCHEMA_VERSION = 1


class IncompatibleModelError(ValueError):
    pass


def build_manifest(
    pipeline: BaseBananaPipeline,
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "pipeline_key": pipeline.key,
        "algorithm": pipeline.display_name,
        "feature_version": pipeline.feature_version,
        "image_size": list(pipeline.image_size),
        "class_order": list(CLASS_NAMES),
        "parameters": _json_safe(pipeline.parameters()),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def validate_manifest(manifest: dict[str, Any], pipeline: BaseBananaPipeline | None = None) -> None:
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise IncompatibleModelError("Unsupported model manifest schema.")
    if tuple(manifest.get("class_order", [])) != CLASS_NAMES:
        raise IncompatibleModelError("Model class order is incompatible with this application.")
    if pipeline is not None:
        if manifest.get("pipeline_key") != pipeline.key:
            raise IncompatibleModelError("The manifest belongs to a different pipeline.")
        if manifest.get("feature_version") != pipeline.feature_version:
            raise IncompatibleModelError("The model feature version is incompatible.")
        if tuple(manifest.get("image_size", [])) != pipeline.image_size:
            raise IncompatibleModelError("The model image size is incompatible.")


def save_bundle(pipeline: BaseBananaPipeline, manifest: dict[str, Any], model_path: str | Path) -> None:
    validate_manifest(manifest, pipeline)
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "manifest": manifest}, path)
    path.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_bundle(model_path: str | Path) -> tuple[BaseBananaPipeline, dict[str, Any]]:
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or "pipeline" not in bundle or "manifest" not in bundle:
        raise IncompatibleModelError("Model file is not a versioned banana-pipeline bundle.")
    pipeline = bundle["pipeline"]
    manifest = bundle["manifest"]
    if not isinstance(pipeline, BaseBananaPipeline):
        raise IncompatibleModelError("Bundle does not contain a banana pipeline.")
    validate_manifest(manifest, pipeline)
    return pipeline, manifest


def discover_bundles(model_dir: str | Path) -> dict[str, tuple[BaseBananaPipeline, dict[str, Any]]]:
    models: dict[str, tuple[BaseBananaPipeline, dict[str, Any]]] = {}
    for path in sorted(Path(model_dir).glob("*.joblib")):
        pipeline, manifest = load_bundle(path)
        models[pipeline.key] = (pipeline, manifest)
    return models


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return repr(value)

