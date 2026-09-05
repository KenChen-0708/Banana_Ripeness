from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.constants import (
    CLASS_NAMES,
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_PIXELS_PER_CM,
    PROJECT_ROOT,
    QUALITY_GRADES,
    SAMPLE_PRESETS,
)
from src.exports import records_csv, records_dataframe, records_pdf
from src.model_store import IncompatibleModelError, discover_bundles
from src.pipelines import PIPELINE_CLASSES
from src.preprocessing import (
    ImageValidationError,
    calibrate_fruit_geometry,
    decode_image,
    determine_quality_grade,
    draw_annotated_geometry,
    mask_overlay,
)

st.set_page_config(page_title="BananaVision Studio", page_icon="🍌", layout="wide")
PIPELINE_ACCENTS = ["#F4B942", "#3BA272", "#5470C6", "#9A60B4"]


@st.cache_resource
def load_models(model_dir: str):
    return discover_bundles(model_dir)


@st.cache_data
def load_evaluation(path: str):
    evaluation_path = Path(path)
    return json.loads(evaluation_path.read_text(encoding="utf-8")) if evaluation_path.exists() else None


def model_state():
    try:
        return load_models(str(DEFAULT_ARTIFACT_DIR / "models"))
    except (OSError, IncompatibleModelError, ValueError) as exc:
        st.error(f"Models could not be loaded: {exc}")
        return {}


def hero(title: str, subtitle: str, eyebrow: str) -> None:
    st.markdown(f"### {eyebrow.upper()}\n# {title}\n{subtitle}")
    st.divider()


def confidence_label(value: float) -> str:
    return "High confidence" if value >= 0.8 else "Moderate confidence" if value >= 0.6 else "Manual review advised"


def assessment_page(models, recommended_key: str) -> None:
    st.markdown("### 🍌 Assessment Studio")
    st.markdown("Upload an image or take a photo to instantly compare predictions across all four algorithms.")
    
    col1, col2 = st.columns(2)
    with col1:
        upload = st.file_uploader("Upload Banana Image", type=["jpg", "png", "jpeg"])
    with col2:
        cam = st.camera_input("Or Take a Photo")
        
    source = upload or cam
    if not source:
        st.info("Awaiting image input...")
        return
        
    try:
        image = decode_image(source.getvalue())
    except Exception as e:
        st.error(f"Invalid image: {e}")
        return
        
    st.image(image, caption="Input Image", width=300)
    st.divider()
    
    st.subheader("Algorithm Comparison (Mode A)")
    cols = st.columns(len(models))
    
    for i, (key, (pipeline, _)) in enumerate(models.items()):
        record = pipeline.predict(image, "input.jpg")
        with cols[i]:
            st.markdown(f"**{pipeline.display_name}**")
            st.metric("Prediction", record.predicted_class.title())
            st.caption(f"Confidence: {record.confidence:.1%}")
            if key == recommended_key:
                st.success("✨ Recommended Model")

def confusion_figure(matrix, title: str):
    figure, axis = plt.subplots(figsize=(6.2, 4.8))
    # We remove custom background colors here to support dark mode gracefully
    values = np.asarray(matrix)
    image = axis.imshow(values, cmap="YlGn", aspect="auto")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            colour = "white" if values[row, column] > values.max() * 0.55 else "black"
            axis.text(column, row, str(values[row, column]), ha="center", va="center", color=colour, fontweight="bold")
    axis.set_xticks(range(len(CLASS_NAMES)), [x.title() for x in CLASS_NAMES], rotation=30, ha="right")
    axis.set_yticks(range(len(CLASS_NAMES)), [x.title() for x in CLASS_NAMES])
    axis.set(xlabel="Predicted class", ylabel="Actual class", title=title)
    colour_bar = figure.colorbar(image, ax=axis, shrink=0.8, label="Images")
    figure.tight_layout()
    return figure


def evaluation_page(summary, models) -> None:
    st.markdown("### 📊 Model Evaluation")
    if not summary:
        st.warning("Run `scripts/train_compare.py` to generate evaluation results.")
        return

    st.subheader("Benchmark Leaderboard")
    rows = []
    for rank, item in enumerate(summary["results"], 1):
        test = item["test_metrics"]
        rows.append({
            "Rank": rank,
            "Algorithm": item["algorithm"],
            "Test Accuracy": f"{test['accuracy']:.2%}",
            "Macro-F1": f"{test['macro_f1']:.2%}"
        })
    import pandas as pd
    st.table(pd.DataFrame(rows).set_index("Rank"))

    st.subheader("Confusion Matrix Explorer")
    selected = st.selectbox("Select Pipeline", [item["algorithm"] for item in summary["results"]])
    item = next(result for result in summary["results"] if result["algorithm"] == selected)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.pyplot(confusion_figure(item["test_metrics"]["confusion_matrix"], f"{selected}"), width="stretch")
    with col2:
        report = pd.DataFrame(item["test_metrics"]["classification_report"]).T.loc[list(CLASS_NAMES), ["precision", "recall", "f1-score"]]
        st.dataframe(report.style.format("{:.3f}"), width="stretch")

def training_page(models) -> None:
    hero("Model training & reproducibility", "Configure training runs, feature caching, and reproduce held-out benchmarks.", "Pipeline training")
    options = {PIPELINE_CLASSES[key]().display_name: key for key in PIPELINE_CLASSES}
    with st.form("training_configuration"):
        selected = st.multiselect("Pipelines", list(options), default=list(options))
        dataset = st.text_input("Dataset directory", value="fruit_dataset")
        first, second = st.columns(2)
        limited = first.checkbox("Use balanced development subset", value=False)
        limit = first.number_input("Images per class / split", 5, 5000, 30, disabled=not limited)
        use_cache = second.checkbox("Reuse validated feature cache", value=True)
        second.caption("Disable cache after changing feature extraction logic.")
        submitted = st.form_submit_button("Validate configuration", width="stretch")

    if not selected:
        st.warning("Select at least one pipeline.")
        return

    command = [sys.executable, "scripts/train_compare.py", "--dataset", dataset, "--pipelines", *[options[name] for name in selected]]
    if limited:
        command.extend(["--limit-per-class", str(limit)])
    if not use_cache:
        command.append("--no-cache")

    configuration = {
        "dataset": dataset,
        "pipelines": [options[name] for name in selected],
        "balanced_limit_per_class": int(limit) if limited else None,
        "reuse_feature_cache": use_cache,
        "evaluation_protocol": "Train / validation / held-out test",
        "ranking_metric": "Test macro-F1, then accuracy",
    }
    st.subheader("Experiment preview")
    cols = st.columns(4)
    for column, (label, value) in zip(cols, [("Pipelines", len(selected)), ("Classes", len(CLASS_NAMES)), ("Image size", "224 × 224"), ("Selection metric", "Macro-F1")], strict=True):
        column.metric(label, value)
    st.code(subprocess.list2cmdline(command), language="powershell")
    st.download_button("Export configuration JSON", json.dumps(configuration, indent=2), "training_configuration.json", "application/json")
    st.caption("Training progress and evaluation artefacts are written under `artifacts/`.")

    if submitted:
        st.success("Configuration is valid and ready to run.") if Path(dataset).exists() else st.error(f"Dataset directory does not exist: {dataset}")

    if st.button("Start model training", type="primary", width="stretch"):
        if not Path(dataset).exists():
            st.error(f"Dataset directory does not exist: {dataset}")
        else:
            with st.status("Training and evaluating selected pipelines...", expanded=True) as status:
                result = subprocess.run(command, cwd=Path(__file__).parent, capture_output=True, text=True)
                st.code((result.stdout + "\n" + result.stderr)[-12000:])
                if result.returncode == 0:
                    status.update(label="Training complete", state="complete")
                    load_models.clear()
                    load_evaluation.clear()
                    st.success("Models and evaluation artefacts were updated. Refresh to load the new bundles.")
                else:
                    status.update(label="Training failed", state="error")


def main() -> None:
    models = model_state()
    summary = load_evaluation(str(DEFAULT_ARTIFACT_DIR / "evaluation_summary.json"))
    st.sidebar.markdown("## 🍌 BananaVision Studio")
    st.sidebar.caption("Classical Image Processing & Fruit Ripeness Assessment")

    page = st.sidebar.radio(
        "Navigation",
        [
            "🍌 Assessment Studio",
            "📊 Model Evaluation",
            "⚙️ Model Training",
        ],
    )

    if not models:
        st.warning("No trained model bundles were found. Open Model training to configure the first run.")
        if page == "⚙️ Model Training":
            training_page(models)
        return

    recommended_key = summary.get("recommended_pipeline") if summary else next(iter(models))
    if recommended_key not in models:
        recommended_key = next(iter(models))
    st.sidebar.success(f"Recommended Pipeline:\n\n**{models[recommended_key][0].display_name}**")

    if page == "🍌 Assessment Studio":
        assessment_page(models, recommended_key)
    elif page == "📊 Model Evaluation":
        evaluation_page(summary, models)
    else:
        training_page(models)


if __name__ == "__main__":
    main()
