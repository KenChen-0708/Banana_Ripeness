from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .constants import CLASS_NAMES, SUPPORTED_EXTENSIONS


@dataclass(frozen=True, slots=True)
class DatasetItem:
    path: Path
    label: str
    split: str


def load_split(dataset_root: str | Path, split: str, limit_per_class: int | None = None) -> list[DatasetItem]:
    root = Path(dataset_root) / split
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset split not found: {root}")
    items: list[DatasetItem] = []
    for label in CLASS_NAMES:
        class_dir = root / label
        if not class_dir.is_dir():
            raise ValueError(f"Missing class folder: {class_dir}")
        files = sorted(path for path in class_dir.iterdir() if path.suffix.lower() in SUPPORTED_EXTENSIONS)
        if limit_per_class is not None:
            files = files[:limit_per_class]
        items.extend(DatasetItem(path=path, label=label, split=split) for path in files)
    return items


def class_counts(items: list[DatasetItem]) -> dict[str, int]:
    counts = Counter(item.label for item in items)
    return {label: counts.get(label, 0) for label in CLASS_NAMES}


def source_key(path: Path) -> str:
    stem = path.stem.split("_jpg.rf.")[0]
    return stem.replace("-Copy", "").replace("-copy", "")


def assert_no_source_leakage(splits: dict[str, list[DatasetItem]]) -> None:
    keys = {split: {source_key(item.path) for item in items} for split, items in splits.items()}
    names = list(keys)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = keys[left] & keys[right]
            if overlap:
                examples = ", ".join(sorted(overlap)[:3])
                raise ValueError(f"Source leakage between {left} and {right}: {examples}")

