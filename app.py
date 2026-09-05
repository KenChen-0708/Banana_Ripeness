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
    apply_bilateral_filter,
    apply_clahe,
    apply_contrast_stretching,
    apply_gaussian_filter,
    apply_median_filter,
    apply_morphology,
    banana_mask,
    brown_spot_ratio,
    calibrate_fruit_geometry,
    colour_masks,
    decode_image,
    determine_quality_grade,
    draw_annotated_geometry,
    mask_overlay,
    resize_rgb,
)
from src.video import process_video

st.set_page_config(page_title="BananaVision Studio", page_icon="🍌", layout="wide")
PIPELINE_ACCENTS = ["#F4B942", "#3BA272", "#5470C6", "#9A60B4"]


def apply_theme() -> None:
    return
    st.markdown(
        """
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
    </style>""",
        unsafe_allow_html=True,
    )


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
    st.markdown(
        f'<div class="hero"><div class="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


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

    # Physical Geometry KPI Cards
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
        f'<div class="result-card" style="border-left-color:{accent}">'
        f"<b>{filename}</b><br>"
        f"Ripeness: <b>{record.predicted_class.title()}</b> | Commercial Grade: <b>{grade}</b> · "
        f"{record.confidence:.1%} · {record.processing_time_ms:.1f} ms · {confidence_label(record.confidence)}</div>",
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
    hero("Assessment studio", "Upload images or select presets. All four pipelines evaluate ripeness & geometry simultaneously.", "Surface quality & maturity")

    top_l, top_r = st.columns([2, 1])
    with top_l:
        preset_choice = st.selectbox("Quick-load benchmark preset samples", ["-- Choose a preset or upload your own files --"] + list(SAMPLE_PRESETS.keys()))
        uploads = st.file_uploader("Or drop custom banana images here", type=["jpg", "jpeg", "png"], accept_multiple_files=True, help="Start with one image, then add more at any time.", key="assessment_uploads")
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

    if errors:
        st.warning("Some files were skipped:\n\n" + "\n\n".join(errors))

    if not images:
        st.info("Upload at least one JPG or PNG image or choose a preset sample above to begin assessment.")
        return

    all_records, completed = [], 0
    total = len(images) * len(models)
    progress = st.progress(0, text="Running all image-processing pipelines...")
    for pipeline, _ in models.values():
        for filename, image in images.items():
            record = pipeline.predict(image, filename)
            # update explanation geometry with current calibration scale
            record.explanation["geometry"] = calibrate_fruit_geometry(image, mask=record.explanation["fruit_mask"], pixels_per_cm=px_cm)
            record.explanation["quality_grade"] = determine_quality_grade(record.predicted_class, record.explanation["geometry"]["blemish_ratio"])
            all_records.append(record)
            completed += 1
            progress.progress(completed / total, text=f"Completed {completed} of {total} assessments")
    progress.empty()

    with st.expander("📊 Advanced Consensus Summary & Data Exports", expanded=False):
        assessment_summary(all_records, models[recommended_key][0].display_name)
    st.divider()

    with st.expander("🔍 Deep-Dive Pipeline Explorer (Masks & Overlays)", expanded=False):
        tabs = st.tabs([pipeline.display_name for pipeline, _ in models.values()])
        for tab, (pipeline, _), accent in zip(tabs, models.values(), PIPELINE_ACCENTS, strict=True):
            with tab:
                render_pipeline_tab(pipeline.display_name, [r for r in all_records if r.algorithm == pipeline.display_name], images, accent, px_cm)


def video_analysis_page(models, recommended_key: str) -> None:
    hero("Video stream & live camera inspection", "Conveyor belt video frame-by-frame ripeness tracking and live webcam capture.", "Extra efforts: video processing")

    pipeline, _ = models[recommended_key]
    st.markdown(f"**Active grading pipeline:** `{pipeline.display_name}`")

    v_tab1, v_tab2 = st.tabs(["🎥 Video file analysis (conveyor belt)", "📸 Live camera snapshot"])

    with v_tab1:
        v_col1, v_col2 = st.columns([1.5, 1])
        with v_col1:
            video_file = st.file_uploader("Upload conveyor video stream", type=["mp4", "avi", "mov", "mkv"])
        with v_col2:
            sample_fps = st.slider("Analysis sample rate (FPS)", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
            v_px_cm = st.slider("Video spatial scale (px / cm)", min_value=1.0, max_value=40.0, value=DEFAULT_PIXELS_PER_CM, step=0.5, key="v_scale")

        if video_file is not None:
            if st.button("Process video stream", type="primary", width="stretch"):
                with st.spinner("Analyzing video frames and measuring blemish progression..."):
                    video_bytes = video_file.getvalue()
                    res = process_video(video_bytes, pipeline, sample_fps=sample_fps, pixels_per_cm=v_px_cm)
                    st.session_state["video_result"] = res

            if "video_result" in st.session_state:
                res = st.session_state["video_result"]
                st.success(f"Analyzed {res.frames_analyzed} frames across {res.duration_sec:.1f}s video duration ({res.fps:.1f} source FPS).")

                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                kpi1.metric("Analyzed frames", res.frames_analyzed)
                kpi2.metric("Mean confidence", f"{res.mean_confidence:.1%}")
                kpi3.metric("Avg blemish surface", f"{res.mean_blemish_ratio:.1%}")
                dominant = max(res.class_counts, key=res.class_counts.get) if res.class_counts else "N/A"
                kpi4.metric("Dominant ripeness", dominant.title())

                st.subheader("Temporal ripeness progression chart")
                if not res.timeline_df.empty:
                    chart_df = res.timeline_df.set_index("timestamp_sec")[["blemish_percent", "confidence"]]
                    st.line_chart(chart_df, color=["#E5A93D", "#3BA272"])

                st.subheader("Frame visual inspector")
                if res.sampled_frames:
                    frame_indices = [f"Frame {idx+1} ({t:.2f}s)" for idx, (t, _, _, _) in enumerate(res.sampled_frames)]
                    chosen_frame_name = st.select_slider("Scrub frames", options=frame_indices)
                    chosen_idx = frame_indices.index(chosen_frame_name)
                    t_sec, annotated_frame, rec, geom = res.sampled_frames[chosen_idx]

                    f_left, f_right = st.columns([1.4, 1])
                    with f_left:
                        st.image(annotated_frame, caption=f"Timestamp: {t_sec:.2f}s | Ripeness: {rec.predicted_class.title()} | {geom.get('quality_grade')}", width="stretch")
                    with f_right:
                        st.markdown(f"**Predicted Maturity:** `{rec.predicted_class.title()}`")
                        st.markdown(f"**Quality Grade:** `{geom.get('quality_grade')}`")
                        st.markdown(f"**Confidence:** `{rec.confidence:.1%}`")
                        st.markdown(f"**Blemish Area:** `{geom.get('blemish_area_cm2', 0.0):.1f} cm²` ({geom.get('blemish_ratio', 0.0):.1%})")
                        st.markdown(f"**Physical Dimensions:** `{geom.get('physical_length_cm', 0.0):.1f}cm × {geom.get('physical_width_cm', 0.0):.1f}cm`")

                st.subheader("Complete frame ledger")
                st.dataframe(res.timeline_df, hide_index=True, width="stretch")
                st.download_button("Export video analysis CSV", res.timeline_df.to_csv(index=False).encode("utf-8"), "video_timeline_analysis.csv", "text/csv")
        else:
            st.info("Upload a video clip (e.g., MP4 conveyor recording) to execute real-time frame-by-frame analysis.")

    with v_tab2:
        st.caption("Capture an instant snapshot directly from your webcam or mobile device camera.")
        cam_image = st.camera_input("Take a photo of a banana")
        if cam_image is not None:
            cam_arr = decode_image(cam_image.getvalue())
            record = pipeline.predict(cam_arr, filename="camera_snapshot.jpg")
            geometry = calibrate_fruit_geometry(cam_arr, pixels_per_cm=DEFAULT_PIXELS_PER_CM)
            grade = determine_quality_grade(record.predicted_class, geometry["blemish_ratio"])
            annotated = draw_annotated_geometry(cam_arr, geometry)

            st.markdown(
                f'<div class="result-card"><b>Camera snapshot assessment</b><br>'
                f"Ripeness: <b>{record.predicted_class.title()}</b> | Grade: <b>{grade}</b> · "
                f"Confidence: {record.confidence:.1%} · Processing: {record.processing_time_ms:.1f} ms</div>",
                unsafe_allow_html=True,
            )
            c1, c2, c3 = st.columns(3)
            c1.image(annotated, caption="Calibrated contour & bounding box", width="stretch")
            c2.image(record.explanation["fruit_mask"], caption="Segmented mask", clamp=True, width="stretch")
            c3.image(mask_overlay(cam_arr, record.explanation["brown_mask"]), caption="Blemish overlay", width="stretch")


def filter_lab_page(models) -> None:
    hero("Preprocessing & filter lab", "Step-by-step classical image filtering, contrast stretching, morphology, and live blemish sensitivity tuning.", "Core image processing")

    preset_name = st.selectbox("Select sample for filter experimentation", list(SAMPLE_PRESETS.keys()))
    image_path = PROJECT_ROOT / SAMPLE_PRESETS[preset_name]
    raw_image = decode_image(image_path)

    st.subheader("1. Noise filtering & smoothing comparison")
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        g_k = st.slider("Gaussian kernel size", 3, 15, 5, step=2)
        g_sigma = st.slider("Gaussian sigma", 0.5, 5.0, 1.5, step=0.5)
        gaussian_img = apply_gaussian_filter(raw_image, kernel_size=g_k, sigma=g_sigma)
        st.image(gaussian_img, caption=f"Gaussian Filter (k={g_k}, σ={g_sigma})", width="stretch")
    with f_col2:
        m_k = st.slider("Median filter size", 3, 15, 5, step=2)
        median_img = apply_median_filter(raw_image, kernel_size=m_k)
        st.image(median_img, caption=f"Median Filter (k={m_k}) - Impulse noise removal", width="stretch")
    with f_col3:
        b_d = st.slider("Bilateral diameter", 3, 15, 9, step=2)
        bilateral_img = apply_bilateral_filter(raw_image, d=b_d)
        st.image(bilateral_img, caption=f"Bilateral Filter (d={b_d}) - Edge-preserving", width="stretch")

    st.subheader("2. Contrast enhancement & histogram equalization")
    c_col1, c_col2, c_col3 = st.columns(3)
    with c_col1:
        st.image(raw_image, caption="Original RGB input", width="stretch")
    with c_col2:
        stretched_img = apply_contrast_stretching(raw_image)
        st.image(stretched_img, caption="Linear Min-Max Contrast Stretching", width="stretch")
    with c_col3:
        clahe_clip = st.slider("CLAHE clip limit", 1.0, 6.0, 2.5, step=0.5)
        clahe_img = apply_clahe(raw_image, clip_limit=clahe_clip)
        st.image(clahe_img, caption=f"CLAHE (Clip={clahe_clip}) - Adaptive Equalization", width="stretch")

    st.subheader("3. Morphological operations & binary mask clean-up")
    m_cols = st.columns(4)
    base_mask = banana_mask(raw_image)
    with m_cols[0]:
        st.image(base_mask, caption="Raw segmented mask", clamp=True, width="stretch")
    with m_cols[1]:
        eroded = apply_morphology(base_mask, op="erode", kernel_size=5)
        st.image(eroded, caption="Erosion (Removes noise fringes)", clamp=True, width="stretch")
    with m_cols[2]:
        dilated = apply_morphology(base_mask, op="dilate", kernel_size=5)
        st.image(dilated, caption="Dilation (Expands boundaries)", clamp=True, width="stretch")
    with m_cols[3]:
        closed = apply_morphology(base_mask, op="close", kernel_size=7, iterations=2)
        st.image(closed, caption="Closing (Fills internal holes)", clamp=True, width="stretch")

    st.subheader("4. Interactive blemish sensitivity & HSV threshold tuning")
    t_col1, t_col2 = st.columns([1, 2])
    with t_col1:
        brown_h_low = st.slider("Brown Hue Min", 0, 30, 3)
        brown_h_high = st.slider("Brown Hue Max", 10, 45, 24)
        dark_v_thresh = st.slider("Darkness Threshold (V-channel)", 20, 100, 65)
    with t_col2:
        custom_masks = colour_masks(
            raw_image,
            mask=closed,
            brown_h_range=(brown_h_low, brown_h_high),
            dark_v_thresh=dark_v_thresh,
        )
        custom_spot_ratio = brown_spot_ratio(
            raw_image,
            mask=closed,
            brown_h_range=(brown_h_low, brown_h_high),
            dark_v_thresh=dark_v_thresh,
        )
        grade = determine_quality_grade("ripe", custom_spot_ratio)
        ov_col1, ov_col2 = st.columns(2)
        with ov_col1:
            blemish_mask = np.bitwise_or(custom_masks["brown"], custom_masks["dark"])
            st.image(mask_overlay(raw_image, blemish_mask), caption=f"Dynamic defect overlay (Spots: {custom_spot_ratio:.1%})", width="stretch")
        with ov_col2:
            st.metric("Tuned blemish surface ratio", f"{custom_spot_ratio:.1%}")
            st.metric("Computed commercial grade", grade)
            st.caption("Adjusting HSV ranges modifies sensitivity for different lighting conditions and fruit cultivars.")


def confusion_figure(matrix, title: str):
    figure, axis = plt.subplots(figsize=(6.2, 4.8))
    figure.patch.set_facecolor("#DCE9D5")
    axis.set_facecolor("#DCE9D5")
    values = np.asarray(matrix)
    image = axis.imshow(values, cmap="YlGn", aspect="auto")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            colour = "#F2E7C9" if values[row, column] > values.max() * 0.55 else "#173B2D"
            axis.text(column, row, str(values[row, column]), ha="center", va="center", color=colour, fontweight="bold")
    axis.set_xticks(range(len(CLASS_NAMES)), [x.title() for x in CLASS_NAMES], rotation=30, ha="right")
    axis.set_yticks(range(len(CLASS_NAMES)), [x.title() for x in CLASS_NAMES])
    axis.set(xlabel="Predicted class", ylabel="Actual class", title=title)
    axis.tick_params(colors="#173B2D")
    axis.xaxis.label.set_color("#173B2D")
    axis.yaxis.label.set_color("#173B2D")
    axis.title.set_color("#173B2D")
    colour_bar = figure.colorbar(image, ax=axis, shrink=0.8, label="Images")
    colour_bar.ax.tick_params(colors="#173B2D")
    colour_bar.set_label("Images", color="#173B2D")
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
    apply_theme()
    models = model_state()
    summary = load_evaluation(str(DEFAULT_ARTIFACT_DIR / "evaluation_summary.json"))
    st.sidebar.markdown("## 🍌 BananaVision Studio")
    st.sidebar.caption("Classical Image Processing & Fruit Ripeness Assessment")

    page = st.sidebar.radio(
        "Navigation",
        [
            "🍌 Assessment Studio",
            "🎥 Video & Live Camera Stream",
            "🔬 Preprocessing & Filter Lab",
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
    elif page == "🎥 Video & Live Camera Stream":
        video_analysis_page(models, recommended_key)
    elif page == "🔬 Preprocessing & Filter Lab":
        filter_lab_page(models)
    elif page == "📊 Model Evaluation":
        evaluation_page(summary, models)
    else:
        training_page(models)


if __name__ == "__main__":
    main()
