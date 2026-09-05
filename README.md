# Banana Ripeness Assessment System

A Streamlit application that compares four classical image-processing pipelines for banana ripeness classification and surface-quality assessment.

## Features

- Four labels: `unripe`, `ripe`, `overripe`, and `rotten`.
- Four pipelines: HSV segmentation + SVM, L\*a\*b\* K-Means + KNN, HSV histogram + Random Forest, and morphology/GLCM + Random Forest.
- Single-image comparison, batch assessment, model evaluation dashboard, CSV export, and PDF export.
- Bounded video ingestion with configurable frame sampling, live annotated predictions, temporal trend charts, and frame-level CSV/PDF exports.
- Median denoising, CLAHE contrast enhancement, fruit segmentation, contour/bounding-box detection, aspect-ratio-preserving rectification, and optional physical calibration.
- Surface-quality measurements including fruit-area, green/yellow coverage, and blemish-to-total-surface ratios.
- Reproducible model manifests, immutable train/validation/test splits, leakage checks, and cached feature extraction.

## Set-up

```powershell
py -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If `py` is unavailable, create the environment with any Python 3.11 or 3.12 installation.

## Train and evaluate

```powershell
.venv\Scripts\python.exe scripts\train_compare.py --dataset fruit_dataset
```

Useful development options:

```powershell
# Fast smoke run with a balanced subset
.venv\Scripts\python.exe scripts\train_compare.py --dataset fruit_dataset --limit-per-class 30

# Train selected pipelines and reuse feature caches
.venv\Scripts\python.exe scripts\train_compare.py --dataset fruit_dataset --pipelines hsv_rf hsv_svm
```

The command writes versioned model bundles and evaluation data under `artifacts/`. The held-out test split is evaluated only after each pipeline is fitted and validated.
Feature caches are invalidated automatically when a pipeline's feature version or image geometry changes.

## Run the dashboard

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

## Test

```powershell
.venv\Scripts\python.exe -m pytest
```

## Main project structure

- `app.py` - Streamlit interface.
- `src/pipelines/` - common pipeline contract and four implementations.
- `scripts/train_compare.py` - training, validation, testing, manifests, and ranking.
- `scripts/update_report.py` - creates the completed report from the supplied Word document and measured results.
- `tests/` - deterministic feature, validation, export, and model-store tests.

