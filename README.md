# Banana Ripeness Assessment System

A Streamlit application that compares four classical image-processing pipelines for banana ripeness classification and surface-quality assessment.

## Features

- **Four Ripeness Stages**: `unripe`, `ripe`, `overripe`, and `rotten`.
- **Four Classical Image Processing Pipelines**:
  1. *HSV Histogram + Random Forest*
  2. *L\*a\*b\* K-Means Color Quantisation + KNN*
  3. *HSV Segmentation + Brown Spot Ratio + SVM*
  4. *Morphology + GLCM Texture + Random Forest*
- **Multimodal Ingestion**:
  - Single & batch image upload.
  - 1-click representative benchmark sample presets.
  - Video stream ingestion (`.mp4`, `.avi`) for conveyor belt monitoring with temporal progression chart.
  - Live webcam / camera snapshot classification.
- **Preprocessing & Filter Lab**:
  - Interactive noise reduction: Gaussian, Median, and Bilateral edge-preserving smoothing.
  - Contrast enhancement: Linear min-max contrast stretching and CLAHE (Adaptive Histogram Equalization).
  - Morphological filtering: Erosion, Dilation, Opening, and Closing.
  - Live blemish sensitivity and HSV threshold tuning.
- **Spatial Calibration & Geometric Morphometry**:
  - Real-world pixel-to-metric ($\text{px}/\text{cm}$) scaling.
  - Bounding box, rotated rectangle, centroid, length ($\text{cm}$), width ($\text{cm}$), surface area ($\text{cm}^2$), and blemish area ($\text{cm}^2$).
  - Automated commercial quality grading: *Grade A (Premium)*, *Grade B (Standard)*, *Grade C (Processing)*, and *Reject/Rotten*.
- **Comprehensive Reporting & Evaluation**:
  - Multi-page automated PDF audit report with executive summary and detailed morphometry table.
  - CSV exports for image assessments and video timeline data.
  - Benchmark leaderboard, confusion matrices, class-level metrics, and reproducibility manifests.

## Set-up

```powershell
py -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If `py` is unavailable, create the environment with any Python 3.11, 3.12, or 3.14 installation.

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

## Test

```powershell
.venv\Scripts\python.exe -m pytest -v
```

## Main project structure

- `app.py` - Streamlit Studio interface (Assessment Studio, Video & Camera, Filter Lab, Evaluation, Training).
- `src/preprocessing.py` - Classical filtering (Gaussian, Median, Bilateral, CLAHE), morphology, blemish segmentation, and spatial calibration.
- `src/video.py` - Video stream and conveyor frame analysis module.
- `src/pipelines/` - Base pipeline contract and 4 classical feature extraction models.
- `src/exports.py` - Multi-page PDF audit and CSV export generator.
- `scripts/train_compare.py` - Pipeline training, validation, testing, manifests, and ranking.
- `tests/` - Comprehensive test suite (16 passing unit and integration tests).

