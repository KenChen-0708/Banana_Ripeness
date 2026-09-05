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


def assessment_summary(records, recommended_name: str) -> None:
    frame = records_dataframe(records)
    recommended = frame[frame["algorithm"] == recommended_name]
    agreement = frame.groupby("filename")["predicted_class"].nunique().eq(1).mean()
    mean_blemish = frame["blemish_percent"].mean() if "blemish_percent" in frame.columns else 0.0

    st.subheader("Completed all-pipeline consensus summary")
    cols = st.columns(5)
    cols[0].metric("Images assessed", frame["filename"].nunique())
    cols[1].metric("Pipeline runs", len(frame))
    cols[2].metric("Cross-model agreement", f"{agreement:.0%}")
    cols[3].metric("Mean confidence", f"{frame['confidence'].mean():.1%}")
    cols[4].metric("Avg blemish surface", f"{mean_blemish:.1f}%")

    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.caption(f"Maturity stage distribution ({recommended_name})")
        st.bar_chart(recommended["predicted_class"].value_counts().reindex(CLASS_NAMES, fill_value=0), color="#3BA272")
    with col_chart2:
        st.caption("Commercial quality grade breakdown")
        grade_order = ["Grade A (Premium)", "Grade B (Standard)", "Grade C (Processing)", "Reject / Rotten"]
        st.bar_chart(recommended["quality_grade"].value_counts().reindex(grade_order, fill_value=0), color="#F4B942")

    a, b, _ = st.columns([1, 1, 2])
    a.download_button("Export all-pipeline CSV", records_csv(records), "all_pipeline_assessment.csv", "text/csv", width="stretch")
    b.download_button("Export audit PDF report", records_pdf(records), "all_pipeline_assessment.pdf", "application/pdf", width="stretch")


def render_explanation(image: np.ndarray, record, px_cm: float) -> None:
    explanation = record.explanation
    geometry = explanation.get("geometry") or calibrate_fruit_geometry(image, pixels_per_cm=px_cm)
    annotated = draw_annotated_geometry(image, geometry)

    cols = st.columns(3)
    cols[0].image(annotated, caption="Calibrated contour & bounding box", width="stretch")
    cols[1].image(explanation["fruit_mask"], caption="Detected fruit region", clamp=True, width="stretch")
    cols[2].image(mask_overlay(image, explanation["brown_mask"]), caption="Blemish & dark lesion overlay", width="stretch")

    gcols = st.columns(5)
    gcols[0].metric("Physical length", f"{geometry.get('physical_length_cm', 0.0):.1f} cm")
    gcols[1].metric("Max width", f"{geometry.get('physical_width_cm', 0.0):.1f} cm")
    gcols[2].metric("Surface area", f"{geometry.get('physical_area_cm2', 0.0):.1f} cm²")
    gcols[3].metric("Blemish area", f"{geometry.get('blemish_area_cm2', 0.0):.1f} cm² ({geometry.get('blemish_ratio', 0.0):.1%})")
    gcols[4].metric("Quality grade", explanation.get("quality_grade", "Grade A (Premium)"))


def render_pipeline_tab(pipeline_name: str, pipeline_records, images: dict[str, np.ndarray], accent: str, px_cm: float) -> None:
    frame = records_dataframe(pipeline_records)
    metrics = st.columns(4)
    metrics[0].metric("Predictions", len(frame))
    metrics[1].metric("Average confidence", f"{frame['confidence'].mean():.1%}")
    metrics[2].metric("Average processing time", f"{frame['processing_time_ms'].mean():.1f} ms")
    metrics[3].metric("Review flags", int((frame["confidence"] < 0.6).sum()))

    filename = st.selectbox("Inspect image", frame["filename"].tolist(), key=f"inspect_{pipeline_name}")
    record = next(item for item in pipeline_records if item.filename == filename)
    grade = record.explanation.get("quality_grade", "Grade A (Premium)")

    st.markdown(
        f"**{filename}**<br>"
        f"Ripeness: **{record.predicted_class.title()}** | Commercial Grade: **{grade}** · "
        f"{record.confidence:.1%} · {record.processing_time_ms:.1f} ms · {confidence_label(record.confidence)}",
        unsafe_allow_html=True,
    )

    probabilities = pd.DataFrame(
        {"Class": [x.title() for x in CLASS_NAMES], "Probability": [record.probabilities.get(x, 0) for x in CLASS_NAMES]}
    ).set_index("Class")

    chart_col, visual_col = st.columns([0.75, 1.75])
    with chart_col:
        st.caption("Class probability profile")
        st.bar_chart(probabilities, color=accent)
        st.caption(f"Green ratio: {record.explanation.get('green_ratio', 0):.1%} | Yellow ratio: {record.explanation.get('yellow_ratio', 0):.1%}")
    with visual_col:
        render_explanation(images[filename], record, px_cm)

    st.subheader("Pipeline result ledger")
    display_cols = ["filename", "predicted_class", "quality_grade", "confidence", "blemish_percent", "length_cm", "width_cm", "processing_time_ms"]
    display = frame[[c for c in display_cols if c in frame.columns]]
    st.dataframe(
        display.style.format(
            {
                "confidence": "{:.1%}",
                "blemish_percent": "{:.1f}%",
                "length_cm": "{:.1f} cm",
                "width_cm": "{:.1f} cm",
                "processing_time_ms": "{:.1f}",
            }
        ),
        hide_index=True,
        width="stretch",
    )

    a, b, _ = st.columns([1, 1, 2])
    slug = pipeline_name.lower().replace(" ", "_").replace("+", "plus").replace("*", "")
    a.download_button("Export pipeline CSV", records_csv(pipeline_records), f"{slug}_results.csv", "text/csv", key=f"csv_{slug}", width="stretch")
    b.download_button(
        "Export detail PDF",
        records_pdf(pipeline_records, f"{pipeline_name} - Detailed Assessment Report"),
        f"{slug}_report.pdf",
        "application/pdf",
        key=f"pdf_{slug}",
        width="stretch",
    )


def assessment_page(models, recommended_key: str) -> None:
    hero("Assessment studio", "Upload images, take a camera snapshot, or select presets. All four pipelines evaluate ripeness & geometry simultaneously.", "Surface quality & maturity")

    top_l, top_r = st.columns([2, 1])
    with top_l:
        preset_choice = st.selectbox("Quick-load benchmark preset samples", ["-- Choose a preset or upload your own files --"] + list(SAMPLE_PRESETS.keys()))
        uploads = st.file_uploader("Or drop custom banana images here", type=["jpg", "jpeg", "png"], accept_multiple_files=True, help="Start with one image, then add more at any time.", key="assessment_uploads")
        cam_image = st.camera_input("Or take a camera snapshot")
    with top_r:
        px_cm = st.slider("Spatial calibration (pixels / cm)", min_value=1.0, max_value=40.0, value=DEFAULT_PIXELS_PER_CM, step=0.5, help="Standardizes pixel dimensions to physical measurements in cm.")
        st.caption("Commercial quality rules: Grade A (<5% spots), Grade B (5-15%), Grade C (15-30%), Reject (>30% or Rotten).")

    images, errors = {}, []
    if preset_choice and preset_choice in SAMPLE_PRESETS:
        preset_path = PROJECT_ROOT / SAMPLE_PRESETS[preset_choice]
        if preset_path.exists():
            try:
                images[preset_choice] = decode_image(preset_path)
            except ImageValidationError as exc:
                errors.append(f"{preset_choice}: {exc}")

    if uploads:
        for upload in uploads:
            try:
                images[upload.name] = decode_image(upload.getvalue())
            except ImageValidationError as exc:
                errors.append(f"{upload.name}: {exc}")

    if cam_image:
        try:
            images["camera_snapshot.jpg"] = decode_image(cam_image.getvalue())
        except ImageValidationError as exc:
            errors.append(f"camera_snapshot.jpg: {exc}")

    if errors:
        st.warning("Some files were skipped:\n\n" + "\n\n".join(errors))

    if not images:
        st.info("Upload at least one JPG or PNG image, take a camera snapshot, or choose a preset sample above to begin assessment.")
        return

    all_records, completed = [], 0
    total = len(images) * len(models)
    progress = st.progress(0, text="Running all image-processing pipelines...")
    for pipeline, _ in models.values():
        for filename, image in images.items():
            record = pipeline.predict(image, filename)
            record.explanation["geometry"] = calibrate_fruit_geometry(image, mask=record.explanation["fruit_mask"], pixels_per_cm=px_cm)
            record.explanation["quality_grade"] = determine_quality_grade(record.predicted_class, record.explanation["geometry"]["blemish_ratio"])
            all_records.append(record)
            completed += 1
            progress.progress(completed / total, text=f"Completed {completed} of {total} assessments")
    progress.empty()

    assessment_summary(all_records, models[recommended_key][0].display_name)
    st.divider()

    tabs = st.tabs([pipeline.display_name for pipeline, _ in models.values()])
    for tab, (pipeline, _), accent in zip(tabs, models.values(), PIPELINE_ACCENTS, strict=True):
        with tab:
            render_pipeline_tab(pipeline.display_name, [r for r in all_records if r.algorithm == pipeline.display_name], images, accent, px_cm)


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
    hero("Model evaluation & leaderboard", "Compare held-out performance, macro-F1, and inspect class-level error matrices.", "Mode A: comparative study")
    if not summary:
        st.warning("Run `scripts/train_compare.py` to generate evaluation results.")
        return
    rows = []
    for rank, item in enumerate(summary["results"], 1):
        validation, test = item["validation_metrics"], item["test_metrics"]
        rows.append(
            {
                "Rank": rank,
                "Algorithm": item["algorithm"],
                "Validation accuracy": validation["accuracy"],
                "Test accuracy": test["accuracy"],
                "Macro-F1": test["macro_f1"],
                "Weighted-F1": test["weighted_f1"],
                "Generalisation gap": validation["accuracy"] - test["accuracy"],
                "Training time (s)": item.get("elapsed_seconds", 0),
            }
        )
    frame = pd.DataFrame(rows)
    best, totals = frame.iloc[0], {split: sum(counts.values()) for split, counts in summary.get("split_counts", {}).items()}
    metrics = st.columns(4)
    metrics[0].metric("Recommended pipeline", best["Algorithm"])
    metrics[1].metric("Best test accuracy", f"{best['Test accuracy']:.2%}")
    metrics[2].metric("Best macro-F1", f"{best['Macro-F1']:.2%}")
    metrics[3].metric("Held-out test images", totals.get("test", "N/A"))

    st.subheader("Benchmark leaderboard")
    percent_cols = ["Validation accuracy", "Test accuracy", "Macro-F1", "Weighted-F1", "Generalisation gap"]
    st.dataframe(frame.style.format({x: "{:.2%}" for x in percent_cols}), hide_index=True, width="stretch")
    st.bar_chart(frame.set_index("Algorithm")[["Test accuracy", "Macro-F1", "Weighted-F1"]], color=["#3BA272", "#F4B942", "#5470C6"])

    st.subheader("Dataset balance and experimental protocol")
    split_frame = pd.DataFrame(summary.get("split_counts", {})).reindex(CLASS_NAMES).fillna(0).astype(int)
    split_frame.index = [x.title() for x in split_frame.index]
    left, right = st.columns([1.35, 1])
    left.bar_chart(split_frame, color=["#28745A", "#E5A93D", "#5470C6"])
    right.markdown(
        f"**Ranking rule:** {summary.get('ranking_rule', 'Not recorded')}  \n"
        f"**Dataset:** `{summary.get('dataset', 'Not recorded')}`  \n"
        f"**Protocol:** fixed train, validation and held-out test splits; macro-F1 limits dominance by larger classes."
    )

    st.subheader("Pipeline diagnostic explorer")
    selected = st.selectbox("Pipeline", [item["algorithm"] for item in summary["results"]])
    item = next(result for result in summary["results"] if result["algorithm"] == selected)
    report = pd.DataFrame(item["test_metrics"]["classification_report"]).T.loc[list(CLASS_NAMES), ["precision", "recall", "f1-score", "support"]]
    plot_col, class_col = st.columns([1.05, 1])
    with plot_col:
        st.pyplot(confusion_figure(item["test_metrics"]["confusion_matrix"], f"{selected}\nHeld-out test confusion matrix"), width="stretch")
    with class_col:
        st.caption("Class-level performance")
        st.dataframe(report.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1-score": "{:.3f}", "support": "{:.0f}"}), width="stretch")
        st.bar_chart(report[["precision", "recall", "f1-score"]], color=["#5470C6", "#F4B942", "#3BA272"])

    if item["pipeline_key"] in models:
        with st.expander("Reproducibility manifest and estimator parameters"):
            manifest = models[item["pipeline_key"]][1]
            st.json({key: manifest.get(key) for key in ["pipeline_key", "feature_version", "image_size", "class_order", "trained_at_utc", "parameters"]})


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
