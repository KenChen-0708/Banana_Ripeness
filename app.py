from __future__ import annotations

import base64
import json
import subprocess
import sys
from html import escape
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.ticker import PercentFormatter

from src.constants import CLASS_NAMES, DEFAULT_ARTIFACT_DIR
from src.exports import records_csv, records_dataframe, records_pdf
from src.model_store import IncompatibleModelError, discover_bundles
from src.pipelines import PIPELINE_CLASSES
from src.preprocessing import ImageValidationError, analyse_object, decode_image, mask_overlay, resize_rgb
from src.video import VideoValidationError, annotate_frame, sample_video

st.set_page_config(
    page_title="Banana Ripeness Assessment",
    page_icon="🍌",
    layout="wide",
    initial_sidebar_state="expanded",
)
PIPELINE_ACCENTS = ["#F4B942", "#3BA272", "#5470C6", "#9A60B4"]
ASSET_DIR = Path(__file__).parent / "assets"
PIPELINE_DETAILS = {
    "hsv_rf": {
        "features": "Fruit-masked HSV colour histograms (16 bins × 3 channels)",
        "processing": "Segmentation, masking, resize and HSV conversion",
        "classifier": "Random Forest (200 trees)",
    },
    "lab_knn": {
        "features": "L*a*b* K-means colour-cluster features",
        "processing": "Resize, L*a*b* conversion and K-means clustering",
        "classifier": "Distance-weighted KNN (k=7)",
    },
    "hsv_svm": {
        "features": "HSV colour ratios, statistics and brown-spot ratio",
        "processing": "Banana segmentation, morphology and colour masking",
        "classifier": "Standardised support vector machine",
    },
    "glcm_rf": {
        "features": "Fruit-masked GLCM texture and morphology measurements",
        "processing": "Banana masking with morphological open/close operations",
        "classifier": "Random Forest (250 trees)",
    },
}


@st.cache_data
def image_data_uri(path: str) -> str:
    suffix = Path(path).suffix.removeprefix(".") or "png"
    payload = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:image/{suffix};base64,{payload}"


def apply_theme(dark_mode: bool = True) -> None:
    if dark_mode:
        palette = {
            "canvas": "#071114", "canvas_soft": "#0b1719", "surface": "#0e1d1e",
            "surface_2": "#112324", "line": "rgba(148, 189, 175, .13)",
            "text": "#f5f7f7", "muted": "#aeb8c5", "green": "#7de58e",
            "green_soft": "rgba(92, 205, 119, .14)", "sidebar": "#091316",
        }
    else:
        palette = {
            "canvas": "#f4f8f5", "canvas_soft": "#eef5f0", "surface": "#ffffff",
            "surface_2": "#eaf3ec", "line": "rgba(27, 75, 57, .13)",
            "text": "#102019", "muted": "#627069", "green": "#2fae58",
            "green_soft": "rgba(47, 174, 88, .13)", "sidebar": "#e8f2eb",
        }
    css = """
    <style>
    :root {--canvas:__CANVAS__;--canvas-soft:__CANVAS_SOFT__;--surface:__SURFACE__;
      --surface-2:__SURFACE_2__;--line:__LINE__;--text:__TEXT__;--muted:__MUTED__;
      --green:__GREEN__;--green-soft:__GREEN_SOFT__;--sidebar:__SIDEBAR__}
    html,body,[data-testid="stAppViewContainer"],.stApp {color:var(--text);background:
      radial-gradient(circle at 78% 15%,rgba(82,196,109,.065),transparent 24%),
      radial-gradient(circle at 53% 62%,rgba(52,132,79,.035),transparent 26%),var(--canvas)}
    html,body,.stApp {font-family:Inter,"Segoe UI",Arial,sans-serif}
    [data-testid="stAppViewContainer"]>.main {margin-left:0}
    .main .block-container,[data-testid="stMainBlockContainer"] {
      max-width:1180px;padding:1rem 2.25rem 3rem!important}
    [data-testid="stHeader"] {background:transparent;height:0}
    [data-testid="stToolbar"],[data-testid="stDecoration"],#MainMenu,footer {display:none!important}
    [data-testid="stSidebar"] {background:var(--sidebar);border-right:1px solid var(--line);width:297px!important}
    [data-testid="stSidebar"]>div:first-child {width:330px!important;padding:1.4rem 1rem;zoom:.9}
    [data-testid="stSidebar"] * {color:var(--text)}
    [data-testid="stSidebarCollapseButton"] {display:none}
    h1,h2,h3,h4,h5,h6,p,label,[data-testid="stCaptionContainer"] {color:var(--text);letter-spacing:-.012em}
    h1,h2,h3 {letter-spacing:-.035em}
    .material-symbols-rounded {font-family:"Material Symbols Rounded","Material Symbols Outlined";font-weight:normal;font-style:normal;line-height:1;letter-spacing:normal;text-transform:none;white-space:nowrap;word-wrap:normal;direction:ltr;font-feature-settings:"liga";font-variation-settings:"FILL" 0,"wght" 400,"GRAD" 0,"opsz" 24}
    .brand {display:flex;align-items:center;gap:14px;padding:5px 8px 22px;border-bottom:1px solid var(--line);margin-bottom:14px}
    .brand-mark {display:grid;place-items:center;width:48px;height:48px;min-width:48px;border-radius:14px;background:var(--green-soft);color:var(--green);font-size:29px}
    .brand-title {font-size:24px;line-height:1.25;font-weight:760;letter-spacing:-.025em}
    [data-testid="stSidebar"] [role="radiogroup"] {gap:8px}
    [data-testid="stSidebar"] [role="radiogroup"] label {border-radius:14px;padding:.78rem .9rem;margin:0;min-height:54px;transition:.2s ease}
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {background:rgba(91,190,115,.08)}
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {background:linear-gradient(100deg,rgba(67,145,91,.30),rgba(69,130,86,.17));box-shadow:inset 0 0 0 1px rgba(125,229,142,.04)}
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {color:#9bf0a4!important}
    [data-testid="stSidebar"] [role="radiogroup"] p {display:flex;align-items:center;gap:11px;font-size:17px!important;font-weight:540}
    [data-testid="stSidebar"] [role="radiogroup"] p:before {font-family:"Material Symbols Rounded","Material Symbols Outlined";font-size:21px;font-weight:normal;line-height:1;color:var(--muted);font-feature-settings:"liga";font-variation-settings:"FILL" 0,"wght" 400,"GRAD" 0,"opsz" 24}
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input[value="0"]) p:before {content:"image_search"}
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input[value="1"]) p:before {content:"analytics"}
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input[value="2"]) p:before {content:"model_training"}
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input[value="3"]) p:before {content:"videocam"}
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p:before {color:var(--green)}
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] {display:none}
    .sidebar-foot {position:fixed;bottom:24px;left:30px;width:260px;color:var(--muted);font-size:12px;line-height:1.5}
    .mode-row {margin-top:-8px}
    .st-key-dark_mode {position:fixed!important;top:16px!important;right:18px!important;left:auto!important;width:max-content!important;z-index:1000}
    .st-key-dark_mode div[data-testid="stToggle"] {position:static!important;width:max-content!important;display:flex!important;justify-content:flex-end;align-items:center}
    .st-key-dark_mode label {display:flex!important;flex-direction:row!important;align-items:center!important;gap:7px;white-space:nowrap;transform:scale(.88)!important;transform-origin:right center}
    div[data-testid="stToggle"] p {font-size:14px;font-weight:550;color:var(--text)}
    div[data-testid="stToggle"] [data-baseweb="toggle"],
    div[data-testid="stToggle"] [role="switch"] {background-color:var(--green)!important;border-color:var(--green)!important}
    .landing-hero {position:relative;min-height:235px;padding:16px 0 8px;overflow:hidden}
    .hero-copy {position:relative;z-index:2;width:62%;padding-top:10px}
    .eyebrow {color:var(--green);text-transform:uppercase;letter-spacing:.27em;font-size:13px;font-weight:800;margin-bottom:11px}
    .landing-hero h1 {font-size:clamp(2.8rem,4.2vw,4.15rem);line-height:1.02;margin:0;max-width:680px;font-weight:790;color:var(--text)}
    .landing-hero h1 span {display:block;color:var(--green);margin-top:4px}
    .landing-hero .subtitle {position:absolute;left:0;top:205px;font-size:16px;color:var(--muted);margin:0;max-width:570px;line-height:1.5}
    .banana-hero {position:absolute;z-index:1;right:1.5%;top:4px;width:35%;max-width:385px;filter:drop-shadow(0 16px 20px rgba(0,0,0,.30))}
    .scribble {position:absolute;z-index:3;right:29%;top:35px;width:128px;font-family:"Segoe Print","Comic Sans MS",cursive;font-size:19px;line-height:1.38;color:#eef4f0;transform:rotate(-4deg);text-align:center}
    .scribble:after {content:"";display:block;width:70px;height:10px;border-top:3px solid var(--green);border-radius:50%;margin:8px auto;transform:rotate(-10deg)}
    .benefits {display:grid;grid-template-columns:repeat(3,minmax(0,1fr));margin:0 auto 16px;padding:9px 12px;gap:0;max-width:1040px;background:rgba(14,29,30,.45);border:1px solid rgba(148,189,175,.07);border-radius:15px}
    .benefit {display:flex;align-items:center;gap:13px;padding:7px 16px;border-right:1px solid var(--line)}
    .benefit:first-child {padding-left:0}.benefit:last-child {border-right:0}
    .benefit-icon {display:grid;place-items:center;width:46px;height:46px;min-width:46px;border-radius:50%;background:var(--green-soft);font-size:20px;color:var(--green)}
    .benefit strong {display:block;color:var(--text);font-size:15px;margin-bottom:3px}.benefit span {color:var(--muted);font-size:13px;white-space:nowrap}
    div[data-testid="stMetric"] {background:var(--surface);color:var(--text);border:1px solid var(--line);border-radius:16px;padding:16px;box-shadow:0 8px 24px rgba(0,0,0,.12)}
    div[data-testid="stMetric"] * {color:var(--text)}
    .hero {padding:28px;border-radius:20px;color:var(--text);background:linear-gradient(115deg,var(--surface),var(--surface-2));border:1px solid var(--line);margin:20px 0}
    .hero h1,.hero p,.hero .eyebrow {color:var(--text)}
    .hero p {color:var(--muted);margin:.35rem 0 0}.hero .eyebrow {color:var(--green)}
    .result-card {color:var(--text);background:var(--surface);border:1px solid var(--line);border-left:5px solid var(--green);border-radius:14px;padding:16px;margin:8px 0}
    .stTabs [data-baseweb="tab-list"] {gap:8px;background:var(--surface);border-radius:14px;padding:8px}
    .stTabs [data-baseweb="tab"] {color:var(--muted);background:transparent;border-radius:10px;padding:10px 16px}
    .stTabs [aria-selected="true"] {color:#071114;background:var(--green)!important}
    .stTabs [aria-selected="true"] * {color:#071114!important}
    [data-testid="stFileUploaderDropzone"],form,[data-testid="stExpander"],
    [data-testid="stDataFrame"],div[data-baseweb="select"]>div,
    [data-baseweb="input"]>div,[data-baseweb="textarea"]>div {
      color:var(--text);background:var(--surface)!important;border-color:var(--line)!important;border-radius:14px}
    [data-testid="stFileUploader"] {max-width:820px;margin:12px auto 0;background:var(--surface);border:1px solid var(--line);border-radius:16px;padding:14px 18px 17px}
    [data-testid="stFileUploader"]>label {display:none}
    [data-testid="stFileUploaderDropzone"] {padding:8px 18px!important;background:linear-gradient(145deg,rgba(11,24,29,.84),rgba(16,27,31,.7))!important;border:1.5px dashed rgba(172,186,194,.7)!important;border-radius:13px!important;min-height:132px;display:flex!important;flex-direction:column!important;align-items:center!important;justify-content:center!important;gap:5px!important}
    [data-testid="stFileUploaderDropzone"]:before {content:"Drag & drop a banana image here\\A or";white-space:pre-line;display:block;padding-top:31px;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23d9dc55' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M7 18h10a4 4 0 0 0 .7-7.94A6 6 0 0 0 6.26 8.1 5 5 0 0 0 7 18Z'/%3E%3Cpath d='m9 13 3-3 3 3M12 10v8'/%3E%3C/svg%3E");background-repeat:no-repeat;background-position:center top;background-size:29px 29px;color:var(--muted);font-size:13px;line-height:1.35;text-align:center}
    [data-testid="stFileUploaderDropzone"]>span {display:flex!important;justify-content:center!important;width:100%!important;text-align:center!important}
    [data-testid="stFileUploaderDropzoneInstructions"] {display:none!important}
    [data-testid="stFileUploaderDropzone"]:after {content:"JPG, PNG, JPEG (Max. 10MB)";display:block;color:var(--muted);font-size:12px;line-height:1.35;text-align:center}
    [data-testid="stFileUploaderDropzone"] small {font-size:14px!important;color:var(--muted)!important}
    [data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"] {display:inline-flex!important;align-items:center!important;justify-content:center!important;gap:8px!important;background:#f2d33b!important;color:#172015!important;border:0!important;border-radius:8px!important;padding:.48rem 1.35rem!important;font-weight:750!important;font-size:14px!important;min-width:165px;min-height:36px}
    [data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"] [data-testid="stIconMaterial"],
    [data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"] [data-testid="stMarkdownContainer"] {display:none!important}
    [data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"]:before {content:"image";font-family:"Material Symbols Rounded";font-size:18px!important;font-weight:500;color:#172015!important;line-height:1}
    [data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"]:after {content:"Browse Image";font-size:14px!important;color:#172015!important;line-height:1}
    [data-testid="stFileUploaderFile"] button[data-testid="stBaseButton-minimal"] {min-width:auto!important;width:34px!important;height:34px!important;padding:6px!important;background:transparent!important;border:0!important;color:var(--muted)!important}
    [data-testid="stFileUploaderFile"] button[data-testid="stBaseButton-minimal"]:before,
    [data-testid="stFileUploaderFile"] button[data-testid="stBaseButton-minimal"]:after {content:none!important}
    .st-key-video_upload [data-testid="stFileUploaderDropzone"]:before {content:"Drag & drop a banana video here\\A or"}
    .st-key-video_upload [data-testid="stFileUploaderDropzone"]:after {content:"MP4, AVI, MOV, MKV (Max. 200MB)"}
    .st-key-video_upload [data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"]:after {content:"Browse Video"}
    .result-section-title {margin:26px 0 12px}.result-section-title h2{font-size:24px;margin:0}.result-section-title p{color:var(--muted);margin:4px 0 0}
    .prediction-card {min-height:230px;background:var(--surface);border:1px solid var(--line);border-radius:16px;padding:22px}
    .prediction-kicker {font-size:12px;text-transform:uppercase;letter-spacing:.16em;color:var(--muted);font-weight:750}
    .prediction-label {font-size:42px;line-height:1.05;color:var(--green);font-weight:800;margin:14px 0 4px}
    .prediction-confidence {font-size:22px;color:var(--text);font-weight:700}.prediction-note {font-size:14px;color:var(--muted);line-height:1.5;margin-top:18px}
    .details-card {margin:16px 0 8px;background:var(--surface);border:1px solid var(--line);border-radius:16px;padding:18px 20px}
    .details-card h3 {font-size:18px;margin:0 0 13px}.details-grid {display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
    .detail-label {display:block;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px}.detail-value{color:var(--text);font-size:14px;line-height:1.45}
    [data-testid="stProgress"] [data-testid="stProgressBar"]>div>div {background:var(--green)!important}
    [data-testid="stVerticalBlockBorderWrapper"] {border-color:var(--line)!important;background:var(--surface);border-radius:16px!important}
    form {padding:20px!important}
    [data-testid="stExpander"] details>summary {padding:14px 16px!important}
    [data-testid="stFileUploaderDropzone"] * ,[data-testid="stExpander"] * {color:var(--text)}
    input,textarea,[role="listbox"],[role="option"] {color:var(--text)!important;background:var(--surface-2)!important}
    [data-baseweb="popover"],[data-baseweb="menu"] {color:var(--text)!important;background:var(--surface-2)!important}
    [data-baseweb="popover"] * ,[data-baseweb="menu"] * {color:var(--text)!important}
    button[kind="secondary"],button[kind="primary"],.stDownloadButton button {
      color:#071114!important;background:var(--green)!important;border:1px solid var(--green)!important;border-radius:10px!important}
    button[kind="secondary"] *,button[kind="primary"] *,.stDownloadButton button * {
      color:#071114!important;fill:#071114!important}
    button[kind="secondary"]:hover,button[kind="primary"]:hover,.stDownloadButton button:hover {filter:brightness(1.08)}
    [data-testid="stAlert"] {color:var(--text);background:var(--surface);border:1px solid var(--line);padding:16px;border-radius:12px}
    [data-testid="stAlert"] * {color:var(--text)}
    [data-testid="stProgressBar"]>div>div {background:var(--green)}
    hr {border-color:var(--line)}
    [data-testid="stVegaLiteChart"],[data-testid="stArrowVegaLiteChart"],
    [data-testid="stImage"],.stPlotlyChart {background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:12px}
    [data-testid="stDataFrame"] {overflow:hidden;box-shadow:0 8px 20px rgba(0,0,0,.08)}
    [data-testid="stTooltipContent"] {color:#f5f7f7!important;background:#0f2420!important}
    @media(max-width:900px){
      [data-testid="stSidebar"] {width:270px!important}[data-testid="stSidebar"]>div:first-child{width:270px!important;zoom:1}
      .main .block-container,[data-testid="stMainBlockContainer"]{padding:1rem!important}.landing-hero{min-height:465px}.hero-copy{width:100%}.landing-hero .subtitle{position:static;margin-top:14px}.banana-hero{width:68%;top:225px;right:-5%}
      .scribble{display:none}.benefits{grid-template-columns:1fr}.benefit{border-right:0;border-bottom:1px solid var(--line);padding:14px 0}.details-grid{grid-template-columns:1fr}
    }
    </style>"""
    for name, value in palette.items():
        css = css.replace(f"__{name.upper()}__", value)
    st.markdown(css, unsafe_allow_html=True)


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


def landing_hero() -> None:
    banana_uri = image_data_uri(str(ASSET_DIR / "banana-hero.png"))
    st.markdown(
        f"""
        <section class="landing-hero" data-page="Assessment workspace">
          <div class="hero-copy">
            <div class="eyebrow">AI Powered</div>
            <h1>Banana Ripeness <span>Assessment</span></h1>
            <p class="subtitle">Upload a banana image to assess its ripeness level instantly.</p>
          </div>
          <div class="scribble">Right<br>Ripeness,<br>Better Taste,<br>Less Waste</div>
          <img class="banana-hero" src="{banana_uri}" alt="A bunch of ripe and unripe bananas">
        </section>
        <section class="benefits" aria-label="Assessment benefits">
          <div class="benefit"><div class="benefit-icon"><span class="material-symbols-rounded">bolt</span></div><div><strong>Fast</strong><span>Get results in seconds</span></div></div>
          <div class="benefit"><div class="benefit-icon"><span class="material-symbols-rounded">analytics</span></div><div><strong>Accurate</strong><span>Trained and evaluated on banana data</span></div></div>
          <div class="benefit"><div class="benefit-icon"><span class="material-symbols-rounded">eco</span></div><div><strong>Banana-Focused</strong><span>Designed for ripeness classification</span></div></div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def confidence_label(value: float) -> str:
    return "High confidence" if value >= .8 else "Moderate confidence" if value >= .6 else "Manual review advised"


def ripeness_recommendation(predicted_class: str) -> str:
    return {
        "unripe": "Allow it to ripen at room temperature before eating.",
        "ripe": "Best for immediate consumption.",
        "overripe": "Ideal for baking, smoothies, or freezing.",
        "rotten": "Discard if spoiled, leaking, or mouldy.",
    }.get(predicted_class, "Review the image and prediction details below.")


def calibrated_measurements(explanation: dict, known_length_cm: float | None) -> dict[str, float]:
    if not known_length_cm or known_length_cm <= 0:
        return {}
    pixels_per_cm = float(explanation.get("major_dimension_px", 0)) / known_length_cm
    if pixels_per_cm <= 0:
        return {}
    return {
        "pixels_per_cm": pixels_per_cm,
        "surface_area_cm2": float(explanation.get("fruit_area_px", 0)) / pixels_per_cm**2,
        "perimeter_cm": float(explanation.get("perimeter_px", 0)) / pixels_per_cm,
    }


def ensure_explanation(image: np.ndarray, record) -> dict:
    """Upgrade prediction evidence produced by a stale in-memory model object."""
    required_fields = {
        "enhanced_rgb",
        "rectified_rgb",
        "fruit_mask",
        "brown_mask",
        "detection_overlay",
        "blemish_ratio",
        "fruit_area_ratio",
    }
    if not required_fields.issubset(record.explanation):
        record.explanation = analyse_object(image)
    return record.explanation


def featured_assessment(images, records, recommended_key: str, models, known_length_cm: float | None) -> None:
    pipeline, manifest = models[recommended_key]
    filename = next(iter(images))
    record = next(
        item for item in records
        if item.filename == filename and item.algorithm == pipeline.display_name
    )
    st.markdown(
        '<div class="result-section-title"><h2>Assessment Result</h2>'
        f'<p>Primary result from {pipeline.display_name}</p></div>',
        unsafe_allow_html=True,
    )
    calibrated = calibrated_measurements(record.explanation, known_length_cm)
    quality = st.columns(4)
    quality[0].metric("Blemished surface", f"{record.explanation.get('blemish_ratio', 0):.1%}")
    quality[1].metric("Detected fruit area", f"{record.explanation.get('fruit_area_ratio', 0):.1%}")
    quality[2].metric("Contour perimeter", f"{record.explanation.get('perimeter_px', 0):.1f} px")
    quality[3].metric(
        "Calibrated surface area",
        f"{calibrated['surface_area_cm2']:.1f} cm²" if calibrated else "Add reference length",
    )
    image_column, prediction_column, probability_column = st.columns([1, 1, 1.2], gap="medium")
    with image_column:
        with st.container(border=True):
            st.caption("Uploaded image")
            st.image(images[filename], caption=filename, width="stretch")
    with prediction_column:
        st.markdown(
            f"""
            <div class="prediction-card">
              <div class="prediction-kicker">Predicted ripeness</div>
              <div class="prediction-label">{record.predicted_class.title()}</div>
              <div class="prediction-confidence">{record.confidence:.1%} confidence</div>
              <div class="prediction-note">{ripeness_recommendation(record.predicted_class)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with probability_column:
        with st.container(border=True):
            st.caption("Ripeness probability")
            for class_name in ("unripe", "ripe", "overripe", "rotten"):
                value = float(record.probabilities.get(class_name, 0))
                st.progress(value, text=f"{class_name.title()} · {value:.1%}")

    details = PIPELINE_DETAILS.get(recommended_key, {})
    st.markdown(
        f"""
        <section class="details-card">
          <h3>Assessment Details</h3>
          <div class="details-grid">
            <div><span class="detail-label">Feature extraction</span><span class="detail-value">{details.get('features', manifest.get('feature_version', 'Recorded in model manifest'))}</span></div>
            <div><span class="detail-label">Image processing</span><span class="detail-value">{details.get('processing', 'Recorded in pipeline implementation')}</span></div>
            <div><span class="detail-label">Classification model</span><span class="detail-value">{details.get('classifier', pipeline.display_name)}</span></div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def assessment_summary(records, recommended_name: str) -> None:
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
    a.download_button("Export all-pipeline CSV", records_csv(records), "all_pipeline_assessment.csv", "text/csv", width="stretch")
    b.download_button("Export summary PDF", records_pdf(records), "all_pipeline_assessment.pdf", "application/pdf", width="stretch")


def render_explanation(image: np.ndarray, record, known_length_cm: float | None = None) -> None:
    explanation = ensure_explanation(image, record)
    original = resize_rgb(image, (224, 224))
    cols = st.columns(4)
    cols[0].image(original, caption="Standardised RGB input", width="stretch")
    cols[1].image(explanation["enhanced_rgb"], caption="Median + CLAHE enhancement", width="stretch")
    cols[2].image(explanation["detection_overlay"], caption="Contour and bounding box", width="stretch")
    cols[3].image(mask_overlay(original, explanation["brown_mask"]), caption="Blemish / dark-region overlay", width="stretch")
    ratios = st.columns(4)
    ratios[0].metric("Green surface ratio", f"{explanation.get('green_ratio', 0):.1%}")
    ratios[1].metric("Yellow surface ratio", f"{explanation.get('yellow_ratio', 0):.1%}")
    ratios[2].metric("Blemish ratio", f"{explanation.get('blemish_ratio', 0):.1%}")
    calibrated = calibrated_measurements(explanation, known_length_cm)
    ratios[3].metric("Calibrated perimeter", f"{calibrated['perimeter_cm']:.1f} cm" if calibrated else "Not calibrated")
    with st.expander("Spatial calibration and rectification evidence"):
        st.image(explanation["rectified_rgb"], caption="Aspect-ratio-preserving object rectification", width=280)
        x, y, width, height = explanation.get("bounding_box", (0, 0, 0, 0))
        st.write(f"Bounding box: x={x}, y={y}, width={width}px, height={height}px")
        if calibrated:
            st.write(
                f"Scale: {calibrated['pixels_per_cm']:.2f} px/cm · "
                f"Estimated projected surface area: {calibrated['surface_area_cm2']:.2f} cm²"
            )


def probability_figure(probabilities: dict[str, float], accent: str):
    """Create a compact bounded chart that cannot overflow a Streamlit column."""
    labels = [class_name.title() for class_name in CLASS_NAMES]
    values = [float(probabilities.get(class_name, 0.0)) for class_name in CLASS_NAMES]
    figure, axis = plt.subplots(figsize=(4.5, 2.75))
    figure.patch.set_alpha(0)
    axis.set_facecolor("#0B1718")
    bars = axis.barh(labels, values, color=accent, height=0.58)
    axis.invert_yaxis()
    axis.set_xlim(0, 1)
    axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.set_xlabel("Probability", color="#D8E1DC", labelpad=7)
    axis.tick_params(axis="both", colors="#D8E1DC", labelsize=9, length=0)
    axis.grid(axis="x", color="#6F817A", alpha=0.28, linewidth=0.8)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_visible(False)
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            min(value + 0.025, 0.94),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1%}",
            va="center",
            ha="left",
            color="#F4F7F5",
            fontsize=9,
            fontweight="bold",
        )
    figure.tight_layout(pad=1.1)
    return figure


def render_pipeline_tab(pipeline_name: str, pipeline_records, images: dict[str, np.ndarray], accent: str, known_length_cm: float | None) -> None:
    frame = records_dataframe(pipeline_records)
    metrics = st.columns(4)
    metrics[0].metric("Predictions", len(frame))
    metrics[1].metric("Average confidence", f"{frame['confidence'].mean():.1%}")
    metrics[2].metric("Average processing time", f"{frame['processing_time_ms'].mean():.1f} ms")
    metrics[3].metric("Review flags", int((frame["confidence"] < .6).sum()))
    filename = st.selectbox("Inspect image", frame["filename"].tolist(), key=f"inspect_{pipeline_name}")
    record = next(item for item in pipeline_records if item.filename == filename)
    safe_filename = escape(filename)
    st.markdown(f'<div class="result-card" style="border-left-color:{accent}"><b>{safe_filename}</b><br>Prediction: <b>{record.predicted_class.title()}</b> · {record.confidence:.1%} · {record.processing_time_ms:.1f} ms · {confidence_label(record.confidence)}</div>', unsafe_allow_html=True)
    chart_col, visual_col = st.columns([1, 1.75], gap="large")
    with chart_col:
        st.caption("Class probability profile")
        figure = probability_figure(record.probabilities, accent)
        st.pyplot(figure, width="stretch")
        plt.close(figure)
    with visual_col:
        render_explanation(images[filename], record, known_length_cm)
    st.subheader("Pipeline result ledger")
    display = frame[["filename", "predicted_class", "confidence", "processing_time_ms"]]
    st.dataframe(display.style.format({"confidence": "{:.1%}", "processing_time_ms": "{:.1f}"}), hide_index=True, width="stretch")
    a, b, _ = st.columns([1, 1, 2])
    slug = pipeline_name.lower().replace(" ", "_").replace("+", "plus").replace("*", "")
    a.download_button("Export pipeline CSV", records_csv(pipeline_records), f"{slug}_results.csv", "text/csv", key=f"csv_{slug}", width="stretch")
    b.download_button("Export detail PDF", records_pdf(pipeline_records, f"{pipeline_name} - Detailed Assessment Report"), f"{slug}_report.pdf", "application/pdf", key=f"pdf_{slug}", width="stretch")


def assessment_page(models, recommended_key: str, show_landing: bool = True) -> None:
    if show_landing:
        landing_hero()
    else:
        hero("Try the assessment", "Upload one image or a batch. All four pipelines run automatically.", "Image assessment")
    with st.expander("Optional physical measurement calibration"):
        st.caption("Enter the known longest dimension of the photographed banana or bunch. The detected major axis is used to derive pixels per centimetre.")
        known_length = st.number_input("Known reference length (cm)", min_value=0.0, max_value=200.0, value=0.0, step=0.5)
    known_length_cm = known_length if known_length > 0 else None
    uploads = st.file_uploader(
        "Upload banana images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="JPG, JPEG or PNG up to 10 MB each.",
        key="assessment_uploads_home" if show_landing else "assessment_uploads_try",
    )
    if not uploads:
        return
    images, errors = {}, []
    for upload in uploads:
        try:
            display_name = upload.name
            duplicate = 2
            while display_name in images:
                path = Path(upload.name)
                display_name = f"{path.stem} ({duplicate}){path.suffix}"
                duplicate += 1
            images[display_name] = decode_image(upload.getvalue())
        except ImageValidationError as exc:
            errors.append(f"{upload.name}: {exc}")
    if errors:
        st.warning("Some files were skipped:\n\n" + "\n\n".join(errors))
    if not images:
        return
    all_records, completed = [], 0
    total = len(images) * len(models)
    progress = st.progress(0, text="Running all image-processing pipelines...")
    for pipeline, _ in models.values():
        for filename, image in images.items():
            record = pipeline.predict(image, filename)
            ensure_explanation(image, record)
            all_records.append(record)
            completed += 1
            progress.progress(completed / total, text=f"Completed {completed} of {total} assessments")
    progress.empty()
    featured_assessment(images, all_records, recommended_key, models, known_length_cm)
    st.divider()
    assessment_summary(all_records, models[recommended_key][0].display_name)
    st.divider()
    tabs = st.tabs([pipeline.display_name for pipeline, _ in models.values()])
    for tab, (pipeline, _), accent in zip(tabs, models.values(), PIPELINE_ACCENTS, strict=True):
        with tab:
            render_pipeline_tab(pipeline.display_name, [r for r in all_records if r.algorithm == pipeline.display_name], images, accent, known_length_cm)


def confusion_figure(matrix, title: str):
    figure, axis = plt.subplots(figsize=(6.2, 4.8))
    figure.patch.set_facecolor("#0E1D1E")
    axis.set_facecolor("#0E1D1E")
    values = np.asarray(matrix)
    image = axis.imshow(values, cmap="YlGn", aspect="auto")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            colour = "#071315" if values[row, column] > values.max() * .55 else "#F4F7F5"
            axis.text(column, row, str(values[row, column]), ha="center", va="center", color=colour, fontweight="bold")
    axis.set_xticks(range(len(CLASS_NAMES)), [x.title() for x in CLASS_NAMES], rotation=30, ha="right")
    axis.set_yticks(range(len(CLASS_NAMES)), [x.title() for x in CLASS_NAMES])
    axis.set(xlabel="Predicted class", ylabel="Actual class", title=title)
    axis.tick_params(colors="#D8E1DC")
    axis.xaxis.label.set_color("#D8E1DC")
    axis.yaxis.label.set_color("#D8E1DC")
    axis.title.set_color("#F4F7F5")
    colour_bar = figure.colorbar(image, ax=axis, shrink=.8, label="Images")
    colour_bar.ax.tick_params(colors="#D8E1DC")
    colour_bar.set_label("Images", color="#D8E1DC")
    figure.tight_layout()
    return figure


def video_analysis_page(models, recommended_key: str) -> None:
    hero(
        "Video analysis",
        "Process a video frame-by-frame and inspect ripeness, confidence, and surface-quality changes over time.",
        "Near-real-time processing",
    )
    uploaded = st.file_uploader(
        "Upload a banana video",
        type=["mp4", "avi", "mov", "mkv"],
        accept_multiple_files=False,
        help="MP4, AVI, MOV or MKV up to 200 MB.",
        key="video_upload",
    )
    if uploaded is None:
        st.info("Upload a short video containing a visible banana or bunch. Stable lighting and a simple background give the most reliable masks.")
        return

    payload = uploaded.getvalue()
    preview_column, settings_column = st.columns([0.9, 1.35], gap="large")
    with preview_column:
        st.video(payload, width=520)
    with settings_column:
        pipeline_names = {pipeline.display_name: key for key, (pipeline, _) in models.items()}
        recommended_name = models[recommended_key][0].display_name
        selected_name = st.selectbox(
            "Processing pipeline",
            list(pipeline_names),
            index=list(pipeline_names).index(recommended_name),
        )
        sample_rate = st.slider("Analysis rate (frames per second)", 0.5, 10.0, 2.0, 0.5)
        max_frames = st.slider("Maximum analysed frames", 10, 500, 180, 10)
        st.caption("The analysis is capped at the first five minutes and the selected frame limit for predictable performance.")
        analyse = st.button("Analyse video", type="primary", width="stretch")
    if not analyse:
        return

    try:
        with st.spinner("Decoding and sampling video frames..."):
            sampled = sample_video(
                payload,
                Path(uploaded.name).suffix.lower(),
                sample_fps=sample_rate,
                max_frames=max_frames,
            )
    except VideoValidationError as exc:
        st.error(str(exc))
        return

    pipeline = models[pipeline_names[selected_name]][0]
    records = []
    st.subheader("Live frame analysis")
    _, live_column, _ = st.columns([1, 2.1, 1])
    live_frame = live_column.empty()
    progress = st.progress(0.0, text="Starting frame analysis...")
    started = perf_counter()
    preview_interval = max(1, len(sampled.frames) // 30)
    scalar_explanation_keys = {
        "blemish_ratio",
        "fruit_area_ratio",
        "fruit_area_px",
        "contour_area_px",
        "perimeter_px",
        "major_dimension_px",
        "minor_dimension_px",
        "green_ratio",
        "yellow_ratio",
    }
    for position, (frame, frame_index, timestamp) in enumerate(
        zip(sampled.frames, sampled.frame_indices, sampled.timestamps_seconds, strict=True),
        start=1,
    ):
        label = f"{uploaded.name} · frame {frame_index} · {timestamp:.2f}s"
        record = pipeline.predict(frame, label)
        ensure_explanation(frame, record)
        if position == 1 or position == len(sampled.frames) or position % preview_interval == 0:
            live_frame.image(
                annotate_frame(
                    frame,
                    record.predicted_class,
                    record.confidence,
                    float(record.explanation.get("blemish_ratio", 0)),
                ),
                caption=f"Live analysis · frame {frame_index} · {timestamp:.2f} seconds",
                width="stretch",
            )
        record.explanation = {
            key: value for key, value in record.explanation.items()
            if key in scalar_explanation_keys
        }
        records.append(record)
        progress.progress(position / len(sampled.frames), text=f"Analysed {position} of {len(sampled.frames)} sampled frames")
    elapsed = perf_counter() - started
    progress.empty()

    frame = records_dataframe(records)
    frame.insert(1, "frame_index", sampled.frame_indices)
    frame.insert(2, "timestamp_seconds", sampled.timestamps_seconds)
    # Build measurement columns explicitly. This remains compatible with
    # PredictionRecord instances retained by Streamlit from an older schema.
    frame["blemish_ratio"] = [
        float(record.explanation.get("blemish_ratio", 0.0)) for record in records
    ]
    frame["fruit_area_ratio"] = [
        float(record.explanation.get("fruit_area_ratio", 0.0)) for record in records
    ]
    dominant_class = str(frame["predicted_class"].mode().iloc[0])
    throughput = len(frame) / max(elapsed, 1e-9)
    metrics = st.columns(5)
    metrics[0].metric("Source duration", f"{sampled.duration_seconds:.1f} s")
    metrics[1].metric("Frames analysed", len(frame))
    metrics[2].metric("Dominant class", dominant_class.title())
    metrics[3].metric("Mean confidence", f"{frame['confidence'].mean():.1%}")
    metrics[4].metric("Processing throughput", f"{throughput:.1f} fps")

    st.subheader("Temporal analysis")
    probability_columns = [f"probability_{label}" for label in CLASS_NAMES]
    probability_timeline = frame.set_index("timestamp_seconds")[probability_columns]
    probability_timeline.columns = [label.title() for label in CLASS_NAMES]
    st.caption("Class probabilities across sampled video time")
    st.line_chart(probability_timeline, color=["#72DF8A", "#F4B942", "#D98B39", "#C65353"])
    quality_timeline = frame.set_index("timestamp_seconds")[["blemish_ratio", "fruit_area_ratio"]]
    quality_timeline.columns = ["Blemished surface", "Detected fruit area"]
    st.caption("Surface-quality and detection stability")
    st.line_chart(quality_timeline, color=["#C65353", "#5470C6"])

    st.subheader("Frame result ledger")
    display_columns = [
        "frame_index", "timestamp_seconds", "predicted_class", "confidence",
        "blemish_ratio", "fruit_area_ratio", "processing_time_ms",
    ]
    st.dataframe(
        frame[display_columns].style.format(
            {
                "timestamp_seconds": "{:.2f}",
                "confidence": "{:.1%}",
                "blemish_ratio": "{:.1%}",
                "fruit_area_ratio": "{:.1%}",
                "processing_time_ms": "{:.1f}",
            }
        ),
        hide_index=True,
        width="stretch",
    )
    first, second, _ = st.columns([1, 1, 2])
    first.download_button(
        "Export frame CSV",
        frame.to_csv(index=False).encode("utf-8"),
        "video_frame_analysis.csv",
        "text/csv",
        width="stretch",
    )
    second.download_button(
        "Export video PDF",
        records_pdf(records, f"{selected_name} - Video Analysis Report"),
        "video_analysis_report.pdf",
        "application/pdf",
        width="stretch",
    )


def evaluation_page(summary, models) -> None:
    hero("Model evaluation", "Compare held-out performance and inspect class-level errors.", "Performance analysis")
    if not summary:
        st.warning("Run `scripts/train_compare.py` to generate evaluation results.")
        return
    selected = st.selectbox("Evaluation pipeline", [item["algorithm"] for item in summary["results"]])
    item = next(result for result in summary["results"] if result["algorithm"] == selected)
    test_metrics = item["test_metrics"]
    macro = test_metrics["classification_report"]["macro avg"]
    st.caption(f"Held-out test performance · {sum(summary['split_counts']['test'].values()):,} images")
    metrics = st.columns(4)
    metrics[0].metric("Accuracy", f"{test_metrics['accuracy']:.2%}")
    metrics[1].metric("Precision", f"{macro['precision']:.2%}")
    metrics[2].metric("Recall", f"{macro['recall']:.2%}")
    metrics[3].metric("F1-Score", f"{macro['f1-score']:.2%}")

    st.subheader("Confusion matrix")
    report = pd.DataFrame(test_metrics["classification_report"]).T.loc[list(CLASS_NAMES), ["precision", "recall", "f1-score", "support"]]
    plot_col, class_col = st.columns([1.05, 1])
    with plot_col:
        st.pyplot(confusion_figure(test_metrics["confusion_matrix"], f"{selected}\nHeld-out test set"), width="stretch")
    with class_col:
        st.caption("Class-level performance")
        st.dataframe(report.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1-score": "{:.3f}", "support": "{:.0f}"}), width="stretch")
        st.bar_chart(report[["precision", "recall", "f1-score"]], color=["#5470C6", "#F4B942", "#72DF8A"])

    rows = []
    for rank, result_item in enumerate(summary["results"], 1):
        validation, test = result_item["validation_metrics"], result_item["test_metrics"]
        rows.append({"Rank": rank, "Algorithm": result_item["algorithm"], "Validation accuracy": validation["accuracy"], "Test accuracy": test["accuracy"], "Macro-F1": test["macro_f1"], "Weighted-F1": test["weighted_f1"], "Generalisation gap": validation["accuracy"] - test["accuracy"], "Training time (s)": result_item.get("elapsed_seconds", 0)})
    frame = pd.DataFrame(rows)
    st.subheader("Benchmark leaderboard")
    percent_cols = ["Validation accuracy", "Test accuracy", "Macro-F1", "Weighted-F1", "Generalisation gap"]
    st.dataframe(frame.style.format({x: "{:.2%}" for x in percent_cols}), hide_index=True, width="stretch")
    st.bar_chart(frame.set_index("Algorithm")[["Test accuracy", "Macro-F1", "Weighted-F1"]], color=["#3BA272", "#F4B942", "#5470C6"])
    st.subheader("Dataset balance and experimental design")
    split_frame = pd.DataFrame(summary.get("split_counts", {})).reindex(CLASS_NAMES).fillna(0).astype(int)
    split_frame.index = [x.title() for x in split_frame.index]
    left, right = st.columns([1.35, 1])
    left.bar_chart(split_frame, color=["#28745A", "#E5A93D", "#5470C6"])
    right.markdown(f"**Ranking rule:** {summary.get('ranking_rule', 'Not recorded')}  \n**Dataset:** `{summary.get('dataset', 'Not recorded')}`  \n**Protocol:** fixed train, validation and held-out test splits; macro-F1 limits dominance by larger classes.")
    if item["pipeline_key"] in models:
        with st.expander("Reproducibility manifest and estimator parameters"):
            manifest = models[item["pipeline_key"]][1]
            st.json({key: manifest.get(key) for key in ["pipeline_key", "feature_version", "image_size", "class_order", "trained_at_utc", "parameters"]})


def training_page(models, summary=None) -> None:
    hero("Model training", "Review the experiment design and retrain the image-processing pipelines.", "Academic workflow")
    options = {
        PIPELINE_CLASSES[key]().display_name: key
        for key in PIPELINE_CLASSES
    }
    split_counts = {
        split: sum(counts.values())
        for split, counts in (summary or {}).get("split_counts", {}).items()
    }
    total_images = sum(split_counts.values())
    best_result = (summary or {}).get("results", [None])[0]
    status = "Trained" if models else "Not trained"
    overview = st.columns(4)
    overview[0].metric("Dataset images", f"{total_images:,}" if total_images else "Not available")
    overview[1].metric("Train / validation / test", f"{split_counts.get('train', 0):,} / {split_counts.get('valid', 0):,} / {split_counts.get('test', 0):,}" if split_counts else "Not available")
    overview[2].metric("Training status", status)
    overview[3].metric("Best test accuracy", f"{best_result['test_metrics']['accuracy']:.2%}" if best_result else "Not available")

    st.subheader("Experiment design")
    dataset_name = Path((summary or {}).get("dataset", "fruit_dataset")).name
    st.caption(f"Dataset: {dataset_name} · Fixed train, validation and held-out test splits · Four ripeness classes")
    technique_rows = []
    for name, key in options.items():
        details = PIPELINE_DETAILS[key]
        result = next((entry for entry in (summary or {}).get("results", []) if entry["pipeline_key"] == key), None)
        technique_rows.append({
            "Pipeline": name,
            "Feature extraction": details["features"],
            "Image processing": details["processing"],
            "Classifier": details["classifier"],
            "Test accuracy": result["test_metrics"]["accuracy"] if result else None,
        })
    technique_frame = pd.DataFrame(technique_rows)
    st.dataframe(
        technique_frame.style.format({"Test accuracy": lambda value: f"{value:.2%}" if pd.notna(value) else "Not trained"}),
        hide_index=True,
        width="stretch",
    )
    if best_result:
        st.success(
            f"Current recommended model: {best_result['algorithm']} · "
            f"Test macro-F1 {best_result['test_metrics']['macro_f1']:.2%} · "
            f"Training completed in {best_result.get('elapsed_seconds', 0):.1f} seconds."
        )

    with st.expander("Advanced training controls"):
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
    configuration = {"dataset": dataset, "pipelines": [options[name] for name in selected], "balanced_limit_per_class": int(limit) if limited else None, "reuse_feature_cache": use_cache, "evaluation_protocol": "Train / validation / held-out test", "ranking_metric": "Test macro-F1, then accuracy"}
    with st.expander("Run configuration preview"):
        st.code(subprocess.list2cmdline(command), language="powershell")
        st.download_button("Export configuration JSON", json.dumps(configuration, indent=2), "training_configuration.json", "application/json")
        st.caption("Training can be computationally intensive. Progress and evaluation artefacts are written under `artifacts/`.")
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
                    load_models.clear(); load_evaluation.clear()
                    st.success("Models and evaluation artefacts were updated. Refresh to load the new bundles.")
                else:
                    status.update(label="Training failed", state="error")


def main() -> None:
    dark_mode = st.session_state.get("dark_mode", True)
    apply_theme(dark_mode)
    models = model_state()
    summary = load_evaluation(str(DEFAULT_ARTIFACT_DIR / "evaluation_summary.json"))
    st.sidebar.markdown(
        '<div class="brand"><span class="brand-mark material-symbols-rounded">eco</span><div class="brand-title">Banana Ripeness<br>Assessment</div></div>',
        unsafe_allow_html=True,
    )
    page = st.sidebar.radio(
        "Navigation",
        [
            "Assessment",
            "Model Evaluation",
            "Model Training",
            "Video Analysis",
        ],
    )
    st.sidebar.markdown('<div class="sidebar-foot">AI-assisted ripeness insight<br>Version 1.0</div>', unsafe_allow_html=True)

    _, mode_column = st.columns([5.4, 1.1])
    with mode_column:
        st.toggle("Dark Mode", value=True, key="dark_mode")

    if not models:
        st.warning("No trained model bundles were found. Open Model training to configure the first run.")
        if "Model Training" in page:
            training_page(models, summary)
        return
    recommended_key = summary.get("recommended_pipeline") if summary else next(iter(models))
    if recommended_key not in models: recommended_key = next(iter(models))
    if page == "Assessment":
        assessment_page(models, recommended_key, show_landing=True)
    elif page == "Model Evaluation":
        evaluation_page(summary, models)
    elif page == "Model Training":
        training_page(models, summary)
    else:
        video_analysis_page(models, recommended_key)


if __name__ == "__main__":
    main()
