from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from time import monotonic

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.constants import CLASS_NAMES, DEFAULT_ARTIFACT_DIR
from src.exports import records_csv, records_dataframe, records_pdf
from src.live_assessment import SampledFrameAssessor, assess_image
from src.model_store import IncompatibleModelError, discover_bundles
from src.pipelines import PIPELINE_CLASSES
from src.preprocessing import ImageValidationError, decode_image, mask_overlay, resize_rgb

st.set_page_config(page_title="BananaVision Studio", page_icon="🍌", layout="wide")
PIPELINE_ACCENTS = ["#F4B942", "#3BA272", "#5470C6", "#9A60B4"]


def apply_theme() -> None:
    st.markdown("""
    <style>
    :root {--forest:#173b2d;--forest-2:#245b45;--leaf:#3f7d5c;--sage:#d2e2ce;
      --sage-light:#e8f0df;--sage-mid:#bdd3b8;--gold:#c69214;--cream:#f2e7c9}
    html,body,[data-testid="stAppViewContainer"],.stApp {
      color:var(--forest);background:#fff}
    [data-testid="stHeader"] {background:rgba(255,255,255,.92)}
    [data-testid="stSidebar"] {background:linear-gradient(180deg,#173b2d,#245b45)}
    [data-testid="stSidebar"] * {color:#e8f0df}
    h1,h2,h3,h4,h5,h6,p,label,[data-testid="stCaptionContainer"] {color:var(--forest);letter-spacing:-.012em}
    h1,h2,h3 {letter-spacing:-.025em}
    div[data-testid="stMetric"] {background:linear-gradient(135deg,#d2e2ce,#e7dfbd);color:var(--forest);
      border:1px solid #9ebc9a;border-radius:16px;padding:16px;box-shadow:0 8px 24px rgba(24,74,53,.12)}
    div[data-testid="stMetric"] * {color:var(--forest)}
    .hero {padding:24px;border-radius:22px;color:#e8f0df;background:linear-gradient(115deg,#123b2d,#28745a 62%,#b88612);margin-bottom:20px}
    .hero h1,.hero p,.hero .eyebrow {color:#e8f0df}
    .hero p {opacity:.88;margin:.25rem 0 0}.eyebrow{text-transform:uppercase;letter-spacing:.14em;font-size:.72rem;font-weight:700;opacity:.78}
    .result-card {color:var(--forest);background:linear-gradient(120deg,#dce9d5,#eee2bd);border:1px solid #a5c0a1;border-left:5px solid #3f7d5c;border-radius:14px;padding:16px;margin:8px 0}
    .stTabs [data-baseweb="tab-list"] {gap:8px;background:#bdd3b8;border-radius:14px;padding:8px}
    .stTabs [data-baseweb="tab"] {color:var(--forest);background:#dce9d5;border-radius:10px;padding:10px 16px}
    .stTabs [aria-selected="true"] {color:#e8f0df;background:#245b45!important}
    .stTabs [aria-selected="true"] * {color:#e8f0df!important}
    [data-testid="stFileUploaderDropzone"],form,[data-testid="stExpander"],
    [data-testid="stDataFrame"],div[data-baseweb="select"]>div,
    [data-baseweb="input"]>div,[data-baseweb="textarea"]>div {
      color:var(--forest);background:#dce9d5!important;border-color:#9ebc9a!important;border-radius:14px}
    [data-testid="stFileUploaderDropzone"] {padding:20px!important}
    form {padding:20px!important}
    [data-testid="stExpander"] details>summary {padding:14px 16px!important}
    [data-testid="stFileUploaderDropzone"] * ,[data-testid="stExpander"] * {color:var(--forest)}
    input,textarea,[role="listbox"],[role="option"] {color:var(--forest)!important;background:#e7dfbd!important}
    [data-baseweb="popover"],[data-baseweb="menu"] {color:var(--forest)!important;background:#e7dfbd!important}
    [data-baseweb="popover"] * ,[data-baseweb="menu"] * {color:var(--forest)!important}
    button[kind="secondary"],button[kind="primary"],.stDownloadButton button {
      color:#fff!important;background:#245b45!important;border:1px solid #3f7d5c!important;border-radius:10px!important}
    button[kind="secondary"] *,button[kind="primary"] *,.stDownloadButton button * {
      color:#fff!important;fill:#fff!important}
    button[kind="secondary"]:hover,button[kind="primary"]:hover,.stDownloadButton button:hover {background:#b88612!important;border-color:#8c670e!important}
    [data-testid="stAlert"] {color:var(--forest);background:#e7dfbd;border:1px solid #c69214;padding:16px;border-radius:12px}
    [data-testid="stAlert"] * {color:var(--forest)}
    [data-testid="stProgressBar"]>div>div {background:#c69214}
    hr {border-color:#9ebc9a}
    [data-testid="stVegaLiteChart"],[data-testid="stArrowVegaLiteChart"],
    [data-testid="stImage"],.stPlotlyChart {background:#dce9d5;border:1px solid #a5c0a1;border-radius:14px;padding:12px}
    [data-testid="stDataFrame"] {overflow:hidden;box-shadow:0 8px 20px rgba(23,59,45,.08)}
    [data-testid="stSidebar"] [role="radiogroup"] label {border-radius:10px;padding:.45rem .6rem;margin:.18rem 0}
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {background:#e7dfbd}
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) * {color:var(--forest)!important}
    [data-testid="stTooltipContent"] {color:#e8f0df!important;background:#173b2d!important}
    </style>""", unsafe_allow_html=True)


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
    st.markdown(f'<div class="hero"><div class="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def confidence_label(value: float) -> str:
    return "High confidence" if value >= .8 else "Moderate confidence" if value >= .6 else "Manual review advised"


def assessment_summary(records, recommended_name: str, key_prefix: str = "assessment") -> None:
    frame = records_dataframe(records)
    recommended = frame[frame["algorithm"] == recommended_name]
    agreement = frame.groupby("filename")["predicted_class"].nunique().eq(1).mean()
    st.subheader("Completed all-pipeline summary")
    cols = st.columns(5)
    cols[0].metric("Images assessed", frame["filename"].nunique())
    cols[1].metric("Pipeline runs", len(frame))
    cols[2].metric("Cross-model agreement", f"{agreement:.0%}")
    cols[3].metric("Mean confidence", f"{frame['confidence'].mean():.1%}")
    cols[4].metric("Review flags", int((frame["confidence"] < .6).sum()))
    st.caption(f"Grade distribution · {recommended_name}")
    st.bar_chart(recommended["predicted_class"].value_counts().reindex(CLASS_NAMES, fill_value=0), color="#3BA272")
    a, b, _ = st.columns([1, 1, 2])
    a.download_button("Export all-pipeline CSV", records_csv(records), "all_pipeline_assessment.csv", "text/csv", key=f"{key_prefix}_all_csv", use_container_width=True)
    b.download_button("Export summary PDF", records_pdf(records), "all_pipeline_assessment.pdf", "application/pdf", key=f"{key_prefix}_summary_pdf", use_container_width=True)


def render_explanation(image: np.ndarray, record) -> None:
    explanation = record.explanation
    original = resize_rgb(image, (224, 224))
    cols = st.columns(3)
    cols[0].image(original, caption="Standardised RGB input", use_container_width=True)
    cols[1].image(explanation["fruit_mask"], caption="Detected fruit region", clamp=True, use_container_width=True)
    cols[2].image(mask_overlay(original, explanation["brown_mask"]), caption="Blemish / dark-region overlay", use_container_width=True)
    ratio_values = [
        ("Green surface ratio", explanation.get("green_ratio", 0)),
        ("Yellow surface ratio", explanation.get("yellow_ratio", 0)),
    ]
    if "brown_spot_ratio" in explanation:
        ratio_values.append(("Brown spot ratio", explanation["brown_spot_ratio"]))
    if "dark_ratio" in explanation:
        ratio_values.append(("Dark-region ratio", explanation["dark_ratio"]))
    ratios = st.columns(len(ratio_values))
    for column, (label, value) in zip(ratios, ratio_values, strict=True):
        column.metric(label, f"{value:.1%}")


def render_pipeline_tab(pipeline_name: str, pipeline_records, images: dict[str, np.ndarray], accent: str, key_prefix: str = "assessment") -> None:
    frame = records_dataframe(pipeline_records)
    metrics = st.columns(4)
    metrics[0].metric("Predictions", len(frame))
    metrics[1].metric("Average confidence", f"{frame['confidence'].mean():.1%}")
    metrics[2].metric("Average processing time", f"{frame['processing_time_ms'].mean():.1f} ms")
    metrics[3].metric("Review flags", int((frame["confidence"] < .6).sum()))
    filename = st.selectbox("Inspect image", frame["filename"].tolist(), key=f"{key_prefix}_inspect_{pipeline_name}")
    record = next(item for item in pipeline_records if item.filename == filename)
    st.markdown(f'<div class="result-card" style="border-left-color:{accent}"><b>{filename}</b><br>Prediction: <b>{record.predicted_class.title()}</b> · {record.confidence:.1%} · {record.processing_time_ms:.1f} ms · {confidence_label(record.confidence)}</div>', unsafe_allow_html=True)
    probabilities = pd.DataFrame({"Class": [x.title() for x in CLASS_NAMES], "Probability": [record.probabilities.get(x, 0) for x in CLASS_NAMES]}).set_index("Class")
    chart_col, visual_col = st.columns([.75, 1.75])
    with chart_col:
        st.caption("Class probability profile")
        st.bar_chart(probabilities, color=accent)
    with visual_col:
        render_explanation(images[filename], record)
    st.subheader("Pipeline result ledger")
    display = frame[["filename", "predicted_class", "confidence", "processing_time_ms"]]
    st.dataframe(display.style.format({"confidence": "{:.1%}", "processing_time_ms": "{:.1f}"}), hide_index=True, use_container_width=True)
    a, b, _ = st.columns([1, 1, 2])
    slug = pipeline_name.lower().replace(" ", "_").replace("+", "plus").replace("*", "")
    a.download_button("Export pipeline CSV", records_csv(pipeline_records), f"{slug}_results.csv", "text/csv", key=f"{key_prefix}_csv_{slug}", use_container_width=True)
    b.download_button("Export detail PDF", records_pdf(pipeline_records, f"{pipeline_name} - Detailed Assessment Report"), f"{slug}_report.pdf", "application/pdf", key=f"{key_prefix}_pdf_{slug}", use_container_width=True)


def render_assessment_results(records, images: dict[str, np.ndarray], models, recommended_key: str, key_prefix: str) -> None:
    """Render existing all-pipeline result UI for any image source."""
    if not records:
        return
    assessment_summary(records, models[recommended_key][0].display_name, key_prefix)
    st.divider()
    tabs = st.tabs([pipeline.display_name for pipeline, _ in models.values()])
    for tab, (pipeline, _), accent in zip(tabs, models.values(), PIPELINE_ACCENTS, strict=True):
        with tab:
            pipeline_records = [record for record in records if record.algorithm == pipeline.display_name]
            render_pipeline_tab(pipeline.display_name, pipeline_records, images, accent, key_prefix)


def assessment_page(models, recommended_key: str) -> None:
    hero("Assessment workspace", "Upload one image or a batch. All four pipelines run automatically.", "Image assessment")
    uploads = st.file_uploader("Drop banana images here", type=["jpg", "jpeg", "png"], accept_multiple_files=True, help="Start with one image, then add more at any time.", key="assessment_uploads")
    if not uploads:
        st.info("Upload at least one JPG or PNG image to begin the four-pipeline comparison.")
        return
    images, errors = {}, []
    for upload in uploads:
        try:
            images[upload.name] = decode_image(upload.getvalue())
        except ImageValidationError as exc:
            errors.append(f"{upload.name}: {exc}")
    if errors:
        st.warning("Some files were skipped:\n\n" + "\n\n".join(errors))
    if not images:
        return
    all_records, completed = [], 0
    total = len(images) * len(models)
    progress = st.progress(0, text="Running all image-processing pipelines...")
    for filename, image in images.items():
        image_records = assess_image(models, image, filename)
        all_records.extend(image_records)
        completed += len(image_records)
        progress.progress(completed / total, text=f"Completed {completed} of {total} assessments")
    progress.empty()
    render_assessment_results(all_records, images, models, recommended_key, "upload_assessment")


def _captured_image_state() -> tuple[bytes | None, object | None]:
    """Return the latest camera bytes and clear a stale confirmed result after a retake."""
    captured = st.session_state.get("live_capture_input")
    captured_bytes = captured.getvalue() if captured is not None else None
    if captured_bytes != st.session_state.get("live_capture_confirmed_bytes"):
        st.session_state.pop("live_capture_records", None)
    return captured_bytes, captured


def image_capture_page(models, recommended_key: str) -> None:
    st.subheader("Image Capture")
    st.caption("Capture one clearly visible banana, review the preview, then confirm the assessment.")
    captured = st.camera_input("Capture a banana image", key="live_capture_input")
    captured_bytes, captured = _captured_image_state()
    if captured is None:
        st.info("Allow camera access, centre one banana in the frame, and take a picture.")
        return

    try:
        image = decode_image(captured_bytes)
    except ImageValidationError as exc:
        st.error(f"Captured image could not be used: {exc}")
        return

    st.image(image, caption="Captured image preview", use_container_width=True)
    first, second, _ = st.columns([1, 1, 2])
    if first.button("Confirm assessment", type="primary", key="confirm_live_capture", use_container_width=True):
        with st.spinner("Running existing image-processing pipelines..."):
            st.session_state["live_capture_records"] = assess_image(models, image, "captured_banana.jpg")
            st.session_state["live_capture_confirmed_bytes"] = captured_bytes
    second.caption("To retake, use the camera control above to capture a new image.")

    records = st.session_state.get("live_capture_records", [])
    if records and st.session_state.get("live_capture_confirmed_bytes") == captured_bytes:
        render_assessment_results(records, {"captured_banana.jpg": image}, models, recommended_key, "capture")


def _live_assessor(models, interval_seconds: float) -> SampledFrameAssessor:
    current = st.session_state.get("live_frame_assessor")
    if current is None or current["interval"] != interval_seconds:
        st.session_state["live_frame_assessor"] = {
            "interval": interval_seconds,
            "assessor": SampledFrameAssessor(models, interval_seconds),
        }
    return st.session_state["live_frame_assessor"]["assessor"]


def real_time_assessment_page(models, recommended_key: str) -> None:
    st.subheader("Real-Time Assessment")
    st.caption("Start the camera below. The system samples one frame at a time to keep all four existing pipelines responsive.")
    interval_seconds = st.slider("Assessment interval (seconds)", 1.0, 5.0, 2.0, 0.5, key="live_interval")
    assessor = _live_assessor(models, interval_seconds)
    try:
        from streamlit_webrtc import WebRtcMode, webrtc_streamer
    except ImportError:
        st.error("Live video support is unavailable. Install the project dependencies and restart Streamlit.")
        st.code(".venv\\Scripts\\python.exe -m pip install -r requirements.txt", language="powershell")
        return

    def video_frame_callback(frame):
        try:
            bgr = frame.to_ndarray(format="bgr24")
            assessor.submit(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        except Exception as exc:
            assessor.record_error(f"Camera frame could not be processed: {exc}")
        return frame

    try:
        context = webrtc_streamer(
            key="banana_live_assessment",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
    except Exception as exc:
        st.error(f"The live camera could not be started: {exc}")
        return

    if context.state.playing:
        st.success("Live camera is running. Use the Stop control in the camera panel to end the stream.")
    else:
        st.info("Use Start in the camera panel and allow browser camera permission.")

    @st.fragment(run_every="1s")
    def render_live_snapshot() -> None:
        snapshot = assessor.snapshot()
        if snapshot.error:
            st.error(snapshot.error)
        elif snapshot.records and snapshot.image is not None:
            age = max(0.0, monotonic() - (snapshot.captured_at or 0.0))
            st.caption(f"Latest sampled frame: {age:.1f} seconds ago")
            render_assessment_results(
                list(snapshot.records),
                {"live_camera_frame": snapshot.image},
                models,
                recommended_key,
                "live",
            )
        elif context.state.playing:
            st.info("Waiting for the first sampled camera frame...")

    render_live_snapshot()


def live_assessment_page(models, recommended_key: str) -> None:
    hero("Live assessment", "Capture one banana image or assess sampled frames from a live camera feed.", "Camera assessment")
    capture_tab, live_tab = st.tabs(["Image Capture", "Real-Time Assessment"])
    with capture_tab:
        image_capture_page(models, recommended_key)
    with live_tab:
        real_time_assessment_page(models, recommended_key)


def confusion_figure(matrix, title: str):
    figure, axis = plt.subplots(figsize=(6.2, 4.8))
    figure.patch.set_facecolor("#DCE9D5")
    axis.set_facecolor("#DCE9D5")
    values = np.asarray(matrix)
    image = axis.imshow(values, cmap="YlGn", aspect="auto")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            colour = "#F2E7C9" if values[row, column] > values.max() * .55 else "#173B2D"
            axis.text(column, row, str(values[row, column]), ha="center", va="center", color=colour, fontweight="bold")
    axis.set_xticks(range(len(CLASS_NAMES)), [x.title() for x in CLASS_NAMES], rotation=30, ha="right")
    axis.set_yticks(range(len(CLASS_NAMES)), [x.title() for x in CLASS_NAMES])
    axis.set(xlabel="Predicted class", ylabel="Actual class", title=title)
    axis.tick_params(colors="#173B2D")
    axis.xaxis.label.set_color("#173B2D")
    axis.yaxis.label.set_color("#173B2D")
    axis.title.set_color("#173B2D")
    colour_bar = figure.colorbar(image, ax=axis, shrink=.8, label="Images")
    colour_bar.ax.tick_params(colors="#173B2D")
    colour_bar.set_label("Images", color="#173B2D")
    figure.tight_layout()
    return figure


def evaluation_page(summary, models) -> None:
    hero("Model evaluation", "Compare held-out performance and inspect class-level errors.", "Performance analysis")
    if not summary:
        st.warning("Run `scripts/train_compare.py` to generate evaluation results.")
        return
    rows = []
    for rank, item in enumerate(summary["results"], 1):
        validation, test = item["validation_metrics"], item["test_metrics"]
        rows.append({"Rank": rank, "Algorithm": item["algorithm"], "Validation accuracy": validation["accuracy"], "Test accuracy": test["accuracy"], "Macro-F1": test["macro_f1"], "Weighted-F1": test["weighted_f1"], "Generalisation gap": validation["accuracy"] - test["accuracy"], "Training time (s)": item.get("elapsed_seconds", 0)})
    frame = pd.DataFrame(rows)
    best, totals = frame.iloc[0], {split: sum(counts.values()) for split, counts in summary.get("split_counts", {}).items()}
    metrics = st.columns(4)
    metrics[0].metric("Recommended pipeline", best["Algorithm"])
    metrics[1].metric("Best test accuracy", f"{best['Test accuracy']:.2%}")
    metrics[2].metric("Best macro-F1", f"{best['Macro-F1']:.2%}")
    metrics[3].metric("Held-out test images", totals.get("test", "N/A"))
    st.subheader("Benchmark leaderboard")
    percent_cols = ["Validation accuracy", "Test accuracy", "Macro-F1", "Weighted-F1", "Generalisation gap"]
    st.dataframe(frame.style.format({x: "{:.2%}" for x in percent_cols}), hide_index=True, use_container_width=True)
    st.bar_chart(frame.set_index("Algorithm")[["Test accuracy", "Macro-F1", "Weighted-F1"]], color=["#3BA272", "#F4B942", "#5470C6"])
    st.subheader("Dataset balance and experimental design")
    split_frame = pd.DataFrame(summary.get("split_counts", {})).reindex(CLASS_NAMES).fillna(0).astype(int)
    split_frame.index = [x.title() for x in split_frame.index]
    left, right = st.columns([1.35, 1])
    left.bar_chart(split_frame, color=["#28745A", "#E5A93D", "#5470C6"])
    right.markdown(f"**Ranking rule:** {summary.get('ranking_rule', 'Not recorded')}  \n**Dataset:** `{summary.get('dataset', 'Not recorded')}`  \n**Protocol:** fixed train, validation and held-out test splits; macro-F1 limits dominance by larger classes.")
    st.subheader("Pipeline diagnostic explorer")
    selected = st.selectbox("Pipeline", [item["algorithm"] for item in summary["results"]])
    item = next(result for result in summary["results"] if result["algorithm"] == selected)
    report = pd.DataFrame(item["test_metrics"]["classification_report"]).T.loc[list(CLASS_NAMES), ["precision", "recall", "f1-score", "support"]]
    plot_col, class_col = st.columns([1.05, 1])
    with plot_col:
        st.pyplot(confusion_figure(item["test_metrics"]["confusion_matrix"], f"{selected}\nHeld-out test confusion matrix"), use_container_width=True)
    with class_col:
        st.caption("Class-level performance")
        st.dataframe(report.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1-score": "{:.3f}", "support": "{:.0f}"}), use_container_width=True)
        st.bar_chart(report[["precision", "recall", "f1-score"]], color=["#5470C6", "#F4B942", "#3BA272"])
    if item["pipeline_key"] in models:
        with st.expander("Reproducibility manifest and estimator parameters"):
            manifest = models[item["pipeline_key"]][1]
            st.json({key: manifest.get(key) for key in ["pipeline_key", "feature_version", "image_size", "class_order", "trained_at_utc", "parameters"]})


def training_page(models) -> None:
    hero("Model training", "Choose pipelines and configure a reproducible training run.", "Training configuration")
    options = {
        PIPELINE_CLASSES[key]().display_name: key
        for key in PIPELINE_CLASSES
    }
    with st.form("training_configuration"):
        selected = st.multiselect("Pipelines", list(options), default=list(options))
        dataset = st.text_input("Dataset directory", value="fruit_dataset")
        first, second = st.columns(2)
        limited = first.checkbox("Use balanced development subset", value=False)
        limit = first.number_input("Images per class / split", 5, 5000, 30, disabled=not limited)
        use_cache = second.checkbox("Reuse validated feature cache", value=True)
        second.caption("Disable cache after changing feature extraction logic.")
        submitted = st.form_submit_button("Validate configuration", use_container_width=True)
    if not selected:
        st.warning("Select at least one pipeline.")
        return
    command = [sys.executable, "scripts/train_compare.py", "--dataset", dataset, "--pipelines", *[options[name] for name in selected]]
    if limited:
        command.extend(["--limit-per-class", str(limit)])
    if not use_cache:
        command.append("--no-cache")
    configuration = {"dataset": dataset, "pipelines": [options[name] for name in selected], "balanced_limit_per_class": int(limit) if limited else None, "reuse_feature_cache": use_cache, "evaluation_protocol": "Train / validation / held-out test", "ranking_metric": "Test macro-F1, then accuracy"}
    st.subheader("Experiment preview")
    cols = st.columns(4)
    for column, (label, value) in zip(cols, [("Pipelines", len(selected)), ("Classes", len(CLASS_NAMES)), ("Image size", "224 × 224"), ("Selection metric", "Macro-F1")], strict=True):
        column.metric(label, value)
    st.code(subprocess.list2cmdline(command), language="powershell")
    st.download_button("Export configuration JSON", json.dumps(configuration, indent=2), "training_configuration.json", "application/json")
    st.caption("Training can be computationally intensive. Progress and evaluation artefacts are written under `artifacts/`.")
    if submitted:
        st.success("Configuration is valid and ready to run.") if Path(dataset).exists() else st.error(f"Dataset directory does not exist: {dataset}")
    if st.button("Start model training", type="primary", use_container_width=True):
        if not Path(dataset).exists():
            st.error(f"Dataset directory does not exist: {dataset}")
        else:
            with st.status("Training and evaluating selected pipelines...", expanded=True) as status:
                result = subprocess.run(command, cwd=Path(__file__).parent, capture_output=True, text=True)
                st.code((result.stdout + "\n" + result.stderr)[-12000:])
                if result.returncode == 0:
                    status.update(label="Training complete", state="complete")
                    load_models.clear(); load_evaluation.clear()
                    st.success("Models and evaluation artefacts were updated. Refresh to load the new bundles.")
                else:
                    status.update(label="Training failed", state="error")


def main() -> None:
    apply_theme()
    models = model_state()
    summary = load_evaluation(str(DEFAULT_ARTIFACT_DIR / "evaluation_summary.json"))
    st.sidebar.markdown("## 🍌 BananaVision")
    page = st.sidebar.radio("Navigation", ["Assessment", "Live Assessment", "Model evaluation", "Model training"])
    if not models:
        st.warning("No trained model bundles were found. Open Model training to configure the first run.")
        if page == "Model training": training_page(models)
        return
    recommended_key = summary.get("recommended_pipeline") if summary else next(iter(models))
    if recommended_key not in models: recommended_key = next(iter(models))
    st.sidebar.success(f"Recommended\n\n{models[recommended_key][0].display_name}")
    if page == "Assessment": assessment_page(models, recommended_key)
    elif page == "Live Assessment": live_assessment_page(models, recommended_key)
    elif page == "Model evaluation": evaluation_page(summary, models)
    else: training_page(models)


if __name__ == "__main__":
    main()
