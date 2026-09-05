# Banana Ripeness Assessment System

A Streamlit application that compares four classical image-processing pipelines for banana ripeness classification and surface-quality assessment.

## Features

- Four labels: `unripe`, `ripe`, `overripe`, and `rotten`.
- Four pipelines: HSV segmentation + SVM, L\*a\*b\* K-Means + KNN, HSV histogram + Random Forest, and morphology/GLCM + Random Forest.
- Single-image comparison, batch assessment, live camera assessment, model evaluation dashboard, CSV export, and PDF export.
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

## Run the dashboard

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

## Live Assessment

- **Image Capture** uses the browser camera to take one image. Review the preview, then choose **Confirm assessment** to run the same existing processing and trained-model flow used for uploaded files. Retake by capturing a new image.
- **Real-Time Assessment** uses a browser WebRTC camera stream. Select a 1–5 second assessment interval, then use the camera panel's **Start**/**Stop** controls. Only sampled frames are assessed so the four existing pipelines remain responsive.
- Browser camera access normally requires `localhost` during development or HTTPS after deployment. The app assumes one clearly visible banana and does not perform object detection.

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

