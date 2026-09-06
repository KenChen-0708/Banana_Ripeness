from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_cropper import st_cropper

from src.constants import (
    CLASS_NAMES,
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_PIXELS_PER_CM,
    PROJECT_ROOT,
    QUALITY_GRADES,
    SAMPLE_PRESETS,
)
from src.cropping import (
    CropBox,
    CropValidationError,
    analysis_images_for_assessment,
    confirm_crop_state,
    crop_box_from_display,
    crop_box_to_display,
    crop_from_component_box,
    image_crop_key,
    reconcile_crop_states,
    resize_for_crop_editor,
    reset_crop_state,
)
from src.exports import records_csv, records_dataframe, records_pdf
from src.forecasting import forecast_ripeness_progress
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

st.set_page_config(
    page_title="Banana Ripeness Assessment",
    page_icon="🍌",
    layout="wide",
    initial_sidebar_state="expanded",
)
PIPELINE_ACCENTS = ["#F4B942", "#3BA272", "#5470C6", "#9A60B4"]
ASSET_DIR = Path(__file__).parent / "assets"


@st.cache_data
def image_data_uri(path: str) -> str:
    suffix = Path(path).suffix.removeprefix(".") or "png"
    payload = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:image/{suffix};base64,{payload}"


def apply_theme() -> None:
    st.markdown(
        """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Strict Black Background */
    .stApp {
        background-color: #000000;
    }
    
    [data-testid="stSidebar"] {
        background-color: #0a0a0a;
        border-right: 1px solid #222;
    }
    
    /* Elegant Navigation Bar (Sidebar Radio Buttons) */
    [data-testid="stSidebar"] [role="radiogroup"] label {
        border-radius: 8px;
        padding: 10px 14px;
        margin: 4px 0;
        transition: background-color 0.2s ease;
        cursor: pointer;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background-color: #1a1a1a;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        background-color: #222222;
        border-left: 3px solid #ffffff;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) * {
        color: #ffffff !important;
        font-weight: 600;
    }
    
    /* Clean Metrics */
    div[data-testid="stMetric"] {
        background-color: #0a0a0a;
        border: 1px solid #222;
        border-radius: 12px;
        padding: 16px;
    }
    
    /* Clean Buttons */
    .stButton>button {
        border-radius: 6px;
        border: 1px solid #333;
        background-color: #111;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        border-color: #666;
        background-color: #222;
        color: white;
    }
    
    /* Expanders */
    [data-testid="stExpander"] {
        background-color: #0a0a0a;
        border: 1px solid #222;
        border-radius: 8px;
    }
    

    /* Modern Header and Usage Banner */
    .hero {
        background: #0d1117;
        border: 1px solid #21262d;
        border-left: 4px solid #58a6ff;
        border-radius: 10px;
        padding: 18px 24px;
        margin-bottom: 20px;
    }
    .hero h1 {
        font-size: 1.8rem;
        margin: 0;
        color: #f0f6fc;
    }
    .hero p {
        color: #8b949e;
        margin: 6px 0 0 0;
        font-size: 0.95rem;
    }
    .hero .eyebrow {
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.72rem;
        font-weight: 700;
        color: #58a6ff;
    }
    .guide-box {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 20px;
    }
    .guide-box h4 {
        margin: 0 0 6px 0;
        color: #58a6ff;
        font-size: 1rem;
    }
    .section-desc {
        color: #8b949e;
        font-size: 0.88rem;
        margin-bottom: 12px;
        line-height: 1.4;
    }

    .result-card {
        background-color: #0a0a0a;
        border: 1px solid #222;
        border-radius: 10px;
        padding: 16px;
        margin: 10px 0;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


def apply_reference_theme(dark_mode: bool = True) -> None:
    """Apply the agreed banana-themed presentation without changing app logic."""
    palette = (
        {
            "canvas": "#071114", "surface": "#0e1d1e", "surface_2": "#112324",
            "line": "rgba(148,189,175,.13)", "text": "#f5f7f7", "muted": "#aeb8c5",
            "green": "#7de58e", "green_soft": "rgba(92,205,119,.14)", "sidebar": "#091316",
        }
        if dark_mode
        else {
            "canvas": "#f4f8f5", "surface": "#ffffff", "surface_2": "#eaf3ec",
            "line": "rgba(27,75,57,.13)", "text": "#102019", "muted": "#627069",
            "green": "#2fae58", "green_soft": "rgba(47,174,88,.13)", "sidebar": "#e8f2eb",
        }
    )
    css = """
    <style>
    :root {--canvas:__CANVAS__;--surface:__SURFACE__;--surface-2:__SURFACE_2__;
      --line:__LINE__;--text:__TEXT__;--muted:__MUTED__;--green:__GREEN__;
      --green-soft:__GREEN_SOFT__;--sidebar:__SIDEBAR__}
    html,body,[data-testid="stAppViewContainer"],.stApp {color:var(--text);background:
      radial-gradient(circle at 78% 15%,rgba(82,196,109,.065),transparent 24%),
      radial-gradient(circle at 53% 62%,rgba(52,132,79,.035),transparent 26%),var(--canvas)!important}
    html,body,.stApp,[class*="css"] {font-family:Inter,"Segoe UI",Arial,sans-serif}
    [data-testid="stAppViewContainer"]>.main {margin-left:0}
    .main .block-container,[data-testid="stMainBlockContainer"] {max-width:1180px;padding:1rem 2.25rem 3rem!important}
    [data-testid="stHeader"] {background:transparent!important;height:0}
    [data-testid="stToolbar"],[data-testid="stDecoration"],#MainMenu,footer {display:none!important}
    [data-testid="stSidebar"] {background:var(--sidebar)!important;border-right:1px solid var(--line)!important;width:297px!important}
    [data-testid="stSidebar"]>div:first-child {width:330px!important;padding:1.4rem 1rem;zoom:.9}
    [data-testid="stSidebar"] * {color:var(--text)}
    [data-testid="stSidebarCollapseButton"] {display:none}
    h1,h2,h3,h4,h5,h6,p,label,[data-testid="stCaptionContainer"] {color:var(--text);letter-spacing:-.012em}
    h1,h2,h3 {letter-spacing:-.035em}
    .material-symbols-rounded {font-family:"Material Symbols Rounded","Material Symbols Outlined";font-weight:normal;font-style:normal;line-height:1;letter-spacing:normal;text-transform:none;white-space:nowrap;word-wrap:normal;direction:ltr;font-feature-settings:"liga";font-variation-settings:"FILL" 0,"wght" 400,"GRAD" 0,"opsz" 24}
    .brand {display:flex;align-items:center;gap:14px;padding:5px 8px 22px;border-bottom:1px solid var(--line);margin-bottom:14px}
    .brand-mark {display:grid;place-items:center;width:48px;height:48px;min-width:48px;border-radius:14px;background:var(--green-soft);color:var(--green)!important;font-size:29px}
    .brand-title {font-size:24px;line-height:1.25;font-weight:760;letter-spacing:-.025em}
    [data-testid="stSidebar"] [role="radiogroup"] {gap:8px}
    [data-testid="stSidebar"] [role="radiogroup"] label {border:0!important;border-radius:14px!important;padding:.78rem .9rem!important;margin:0!important;min-height:54px;transition:.2s ease;background:transparent!important}
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {background:rgba(91,190,115,.08)!important}
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {background:linear-gradient(100deg,rgba(67,145,91,.30),rgba(69,130,86,.17))!important;border-left:0!important;box-shadow:inset 0 0 0 1px rgba(125,229,142,.04)}
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {color:#9bf0a4!important}
    [data-testid="stSidebar"] [role="radiogroup"] p {display:flex;align-items:center;gap:11px;font-size:17px!important;font-weight:540}
    [data-testid="stSidebar"] [role="radiogroup"] p:before {font-family:"Material Symbols Rounded","Material Symbols Outlined";font-size:21px;font-weight:normal;line-height:1;color:var(--muted);font-feature-settings:"liga";font-variation-settings:"FILL" 0,"wght" 400,"GRAD" 0,"opsz" 24}
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input[value="0"]) p:before {content:"image_search"}
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input[value="1"]) p:before {content:"videocam"}
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input[value="2"]) p:before {content:"tune"}
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input[value="3"]) p:before {content:"analytics"}
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input[value="4"]) p:before {content:"model_training"}
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p:before {color:var(--green)}
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] {display:none}
    .sidebar-foot {position:fixed;bottom:24px;left:30px;width:260px;color:var(--muted);font-size:12px;line-height:1.5}
    .st-key-dark_mode {position:relative!important;width:100%!important;z-index:2!important;display:flex!important;justify-content:flex-end!important;min-height:34px!important;margin-top:42px!important;pointer-events:auto!important}
    .st-key-dark_mode div[data-testid="stToggle"] {width:max-content!important;display:flex!important;justify-content:flex-end;align-items:center}
    .st-key-dark_mode label {display:flex!important;flex-direction:row!important;align-items:center!important;gap:7px;white-space:nowrap;cursor:pointer!important;padding:6px!important;pointer-events:auto!important}
    div[data-testid="stToggle"] p {font-size:14px;font-weight:550;color:var(--text)}
    div[data-testid="stToggle"] [data-baseweb="toggle"],div[data-testid="stToggle"] [role="switch"] {background-color:var(--green)!important;border-color:var(--green)!important}
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
    div[data-testid="stMetric"] {background:var(--surface)!important;color:var(--text);border:1px solid var(--line)!important;border-radius:16px!important;padding:16px;box-shadow:0 8px 24px rgba(0,0,0,.12)}
    div[data-testid="stMetric"] * {color:var(--text)}
    .hero {padding:28px!important;border-radius:20px!important;color:var(--text);background:linear-gradient(115deg,var(--surface),var(--surface-2))!important;border:1px solid var(--line)!important;border-left:1px solid var(--line)!important;margin:20px 0!important}
    .hero h1,.hero p,.hero .eyebrow {color:var(--text)}.hero p {color:var(--muted);margin:.35rem 0 0}.hero .eyebrow {color:var(--green)}
    .guide-box,.result-card,[data-testid="stExpander"],form,[data-testid="stDataFrame"],[data-testid="stVerticalBlockBorderWrapper"] {color:var(--text);background:var(--surface)!important;border:1px solid var(--line)!important;border-radius:14px!important}
    .guide-box h4 {color:var(--green)}.section-desc {display:block;color:var(--muted);margin-bottom:0!important;padding-bottom:24px!important}
    .stTabs [data-baseweb="tab-list"] {gap:8px;background:var(--surface);border-radius:14px;padding:8px}.stTabs [data-baseweb="tab"] {color:var(--muted);background:transparent;border-radius:10px;padding:10px 16px}.stTabs [aria-selected="true"] {color:#071114;background:var(--green)!important}.stTabs [aria-selected="true"] * {color:#071114!important}
    div[data-baseweb="select"]>div,[data-baseweb="input"]>div,[data-baseweb="textarea"]>div {color:var(--text);background:var(--surface)!important;border-color:var(--line)!important;border-radius:14px}
    .stMultiSelect [data-tag] {background:var(--green)!important;color:#071114!important}
    .stMultiSelect [data-tag] span,
    .stMultiSelect [data-tag] button,
    .stMultiSelect [data-tag] svg {color:#071114!important;fill:#071114!important;font-weight:600!important}
    .stMultiSelect [data-tag] svg path {stroke:#071114!important}
    .st-key-assessment_uploads [data-testid="stFileUploader"] {max-width:820px;margin:12px auto 0;background:var(--surface);border:1px solid var(--line);border-radius:16px;padding:14px 18px 17px}
    .st-key-assessment_uploads [data-testid="stFileUploader"]>label {display:none}
    .st-key-assessment_uploads [data-testid="stFileUploaderDropzone"] {padding:8px 18px!important;background:linear-gradient(145deg,rgba(11,24,29,.84),rgba(16,27,31,.7))!important;border:1.5px dashed rgba(172,186,194,.7)!important;border-radius:13px!important;min-height:132px;display:flex!important;flex-direction:column!important;align-items:center!important;justify-content:center!important;gap:5px!important}
    .st-key-assessment_uploads [data-testid="stFileUploaderDropzone"]:before {content:"Drag & drop a banana image here\\A or";white-space:pre-line;display:block;padding-top:31px;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23d9dc55' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M7 18h10a4 4 0 0 0 .7-7.94A6 6 0 0 0 6.26 8.1 5 5 0 0 0 7 18Z'/%3E%3Cpath d='m9 13 3-3 3 3M12 10v8'/%3E%3C/svg%3E");background-repeat:no-repeat;background-position:center top;background-size:29px 29px;color:var(--muted);font-size:13px;line-height:1.35;text-align:center}
    .st-key-assessment_uploads [data-testid="stFileUploaderDropzone"]>span {display:flex!important;justify-content:center!important;width:100%!important;text-align:center!important}.st-key-assessment_uploads [data-testid="stFileUploaderDropzoneInstructions"] {display:none!important}
    .st-key-assessment_uploads [data-testid="stFileUploaderDropzone"]:after {content:"JPG, PNG, JPEG (Max. 10MB)";display:block;color:var(--muted);font-size:12px;line-height:1.35;text-align:center}
    .st-key-assessment_uploads [data-testid="stFileUploaderDropzone"] button:has([data-testid="stMarkdownContainer"]) {display:inline-flex!important;align-items:center!important;justify-content:center!important;gap:8px!important;background:#f2d33b!important;color:#172015!important;border:0!important;border-radius:8px!important;padding:.48rem 1.35rem!important;font-weight:750!important;font-size:14px!important;min-width:165px;min-height:36px}
    .st-key-assessment_uploads [data-testid="stFileUploaderDropzone"] button:has([data-testid="stMarkdownContainer"]) [data-testid="stIconMaterial"],.st-key-assessment_uploads [data-testid="stFileUploaderDropzone"] button:has([data-testid="stMarkdownContainer"]) [data-testid="stMarkdownContainer"] {display:none!important}
    .st-key-assessment_uploads [data-testid="stFileUploaderDropzone"] button:has([data-testid="stMarkdownContainer"]):before {content:"image";font-family:"Material Symbols Rounded";font-size:18px!important;font-weight:500;color:#172015!important;line-height:1}.st-key-assessment_uploads [data-testid="stFileUploaderDropzone"] button:has([data-testid="stMarkdownContainer"]):after {content:"Browse Image";font-size:14px!important;color:#172015!important;line-height:1}
    [data-testid="stFileUploaderDropzone"] {background:var(--surface)!important;border-color:var(--line)!important}[data-testid="stFileUploaderDropzone"] * {color:var(--text)}
    [data-testid="stVerticalBlockBorderWrapper"] {background:linear-gradient(145deg,rgba(17,35,36,.82),rgba(10,25,27,.8));border-color:var(--line)!important;border-radius:16px!important}
    .preview-card-title,.crop-card-title {font-size:1.08rem;font-weight:760;color:var(--text);margin:0 0 .15rem}.preview-card-title .material-symbols-rounded,.crop-card-title .material-symbols-rounded {color:var(--green);vertical-align:-4px;margin-right:8px}
    .preview-card-subtitle,.crop-card-subtitle {font-size:.84rem;color:var(--muted);margin:0 0 1rem}
    .preview-meta {display:grid;gap:13px;padding:8px 4px}.preview-meta-row {display:grid;grid-template-columns:24px 110px 1fr;gap:8px;align-items:center;font-size:.9rem}.preview-meta-row .material-symbols-rounded {color:var(--muted);font-size:19px}.preview-meta-label {color:var(--muted)}.preview-meta-value {font-weight:700;color:var(--text);word-break:break-word}
    [class*="st-key-assessment_crop_mode_"] [role="radiogroup"] {display:grid!important;grid-template-columns:1fr 1fr;gap:12px!important}[class*="st-key-assessment_crop_mode_"] [role="radiogroup"] label {min-height:72px;border:1px solid var(--line)!important;border-radius:12px!important;padding:16px!important;background:rgba(5,17,18,.22)!important}[class*="st-key-assessment_crop_mode_"] [role="radiogroup"] label:has(input:checked) {border-color:var(--green)!important;background:rgba(92,205,119,.08)!important;box-shadow:inset 0 0 0 1px rgba(125,229,142,.16)}[class*="st-key-assessment_crop_mode_"] [role="radiogroup"] p {font-weight:700!important}
    .gallery-status {float:right;border:1px solid var(--line);border-radius:999px;padding:2px 8px;font-size:.72rem;font-weight:750;color:var(--muted)}.gallery-status.cropped {color:#f6df58;border-color:rgba(246,223,88,.38);background:rgba(246,211,59,.12)}.gallery-status.selected {color:#071114;border-color:var(--green);background:var(--green)}
    .st-key-run_assessment_button button {min-height:48px!important;font-size:1rem!important;font-weight:800!important}
    [data-testid="stProgressBar"]>div>div {background:var(--green)!important}
    button[kind="secondary"],button[kind="primary"],.stDownloadButton button {color:#071114!important;background:var(--green)!important;border:1px solid var(--green)!important;border-radius:10px!important}
    button[kind="secondary"] *,button[kind="primary"] *,.stDownloadButton button * {color:#071114!important;fill:#071114!important}
    input,textarea,[role="listbox"],[role="option"],[data-baseweb="popover"],[data-baseweb="menu"] {color:var(--text)!important;background:var(--surface-2)!important}
    [data-testid="stAlert"] {color:var(--text);background:var(--surface);border:1px solid var(--line);padding:16px;border-radius:12px}[data-testid="stAlert"] * {color:var(--text)}
    hr {border-color:var(--line)}[data-testid="stDataFrame"] {overflow:hidden;box-shadow:0 8px 20px rgba(0,0,0,.08)}
    @media(max-width:900px){[data-testid="stSidebar"] {width:270px!important}[data-testid="stSidebar"]>div:first-child{width:270px!important;zoom:1}.main .block-container,[data-testid="stMainBlockContainer"]{padding:1rem!important}.landing-hero{min-height:465px}.hero-copy{width:100%}.landing-hero .subtitle{position:static;margin-top:14px}.banana-hero{width:68%;top:225px;right:-5%}.scribble{display:none}.benefits{grid-template-columns:1fr}.benefit{border-right:0;border-bottom:1px solid var(--line);padding:14px 0}}
    </style>
    """
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



def section_guide(title: str, rubric_context: str, how_to_use: str) -> None:
    st.markdown(
        f"""<div class="guide-box">
            <h4>📘 {title}</h4>
            <div class="section-desc"><b>🎯 Academic Purpose:</b> {rubric_context}</div>
            <div class="section-desc"><b>💡 How to Test:</b> {how_to_use}</div>
        </div>""",
        unsafe_allow_html=True,
    )

def hero(title: str, subtitle: str, eyebrow: str) -> None:
    st.markdown(
        f'<div class="hero"><div class="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def landing_hero() -> None:
    banana_uri = image_data_uri(str(ASSET_DIR / "banana-hero.png"))
    st.markdown(
        f"""
        <section class="landing-hero" data-page="Assessment studio">
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


def render_ripeness_forecast(record, temperature_c: float, near_ethylene_fruit: bool) -> None:
    """Render a transparent shelf-life estimate for the recommended pipeline result."""
    blemish_ratio = float(record.explanation.get("geometry", {}).get("blemish_ratio", record.explanation.get("blemish_ratio", 0.0)))
    forecast = forecast_ripeness_progress(
        record.predicted_class,
        confidence=record.confidence,
        blemish_ratio=blemish_ratio,
        temperature_c=temperature_c,
        near_ethylene_fruit=near_ethylene_fruit,
    )
    st.markdown("### ⏳ Ripeness Progress Forecast")
    left, middle, right = st.columns(3)
    left.metric("Current stage", record.predicted_class.title())
    middle.metric("Estimated window", f"{forecast.days_min}–{forecast.days_max} days")
    right.metric("Detected blemishes", f"{blemish_ratio:.1%}")
    st.info(f"**{forecast.headline}.** {forecast.recommendation}")
    st.caption(f"Storage adjustment: {forecast.storage_effect}")
    st.caption("Estimate only: this rule-based forecast uses the current image classification, visible blemishes, and selected storage conditions; it is not a trained days-to-ripeness model.")
    if forecast.manual_review_advised:
        st.warning("Manual review advised because the predicted condition indicates decay or the classification confidence is below 60%.")


def _format_file_size(size_bytes: int | None) -> str:
    if size_bytes is None:
        return "Not available"
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.0f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def assessment_input_key(analysis_images: dict[str, np.ndarray], pixels_per_cm: float) -> tuple[tuple[str, ...], float]:
    """Identify the exact images and calibration used to produce saved assessment records."""
    image_keys = tuple(image_crop_key(filename, image) for filename, image in analysis_images.items())
    return image_keys, float(pixels_per_cm)


def render_manual_crop_control(images: dict[str, np.ndarray], crop_states: dict[str, dict]) -> tuple[str, dict[str, dict]]:
    """Edit one selected image while preserving independent crop state for every upload."""
    filenames = list(images)
    selected_filename = st.session_state.get("assessment_selected_image", filenames[0])
    if selected_filename not in images:
        selected_filename = filenames[0]
    with st.container(border=True):
        st.markdown('<p class="crop-card-title"><span class="material-symbols-rounded">crop</span>Crop Banana Before Assessment <small>(Optional)</small></p>', unsafe_allow_html=True)
        st.markdown('<p class="crop-card-subtitle">If automatic segmentation is not accurate, you can manually crop the banana region.</p>', unsafe_allow_html=True)
        mode = st.radio(
            "Crop Banana Before Assessment",
            ("Use original image", "Manually crop banana"),
            horizontal=True,
            key="assessment_crop_mode",
            label_visibility="collapsed",
            help="Cropping only changes the input image. Existing banana segmentation and all four model pipelines still run normally.",
        )
        if mode == "Use original image":
            return selected_filename, crop_states

        if len(filenames) > 1:
            selected_filename = st.selectbox(
                "Select image to edit",
                filenames,
                key="assessment_selected_image",
                help="Select an uploaded image to crop. Each image keeps its own crop state.",
            )
        else:
            selected_filename = filenames[0]
        original_image = images[selected_filename]
        source_key = image_crop_key(selected_filename, original_image)
        state = crop_states.get(source_key)

        default_coords = None
        editor_image = resize_for_crop_editor(original_image)
        if state:
            box = state["crop_box"]
            default_coords = crop_box_to_display(box, original_image, editor_image)

        editor_col, preview_col, action_col = st.columns([1.35, 1, 0.8])
        with editor_col:
            st.markdown("**Original Image**")
            crop_result = st_cropper(
                Image.fromarray(editor_image),
                realtime_update=True,
                default_coords=default_coords,
                box_color="#7DE58E",
                aspect_ratio=None,
                return_type="both",
                key=f"assessment_cropper_{source_key}",
                should_resize_image=False,
            )
            _, component_box = crop_result
            st.caption("Drag to select the crop area")
        try:
            pending_box = crop_box_from_display(component_box, original_image, editor_image)
            pending_crop = crop_from_component_box(
                original_image,
                {"left": pending_box.left, "top": pending_box.top, "width": pending_box.width, "height": pending_box.height},
            )
        except CropValidationError as exc:
            pending_crop, pending_box = None, None
            preview_col.error(f"Invalid crop: {exc}")
        with preview_col:
            st.markdown("**Cropped Preview**")
            if pending_crop is not None:
                st.image(pending_crop, width=400)
                st.caption(f"Preview: {pending_crop.shape[1]} × {pending_crop.shape[0]} px")
                st.caption("Actual analysis uses the full-resolution crop.")
        with action_col:
            st.markdown("&nbsp;", unsafe_allow_html=True)
            if st.button("✓  Confirm Crop", type="primary", disabled=pending_crop is None, key=f"confirm_assessment_crop_{source_key}", width="stretch"):
                crop_states = confirm_crop_state(crop_states, selected_filename, original_image, pending_box)
                st.session_state.assessment_crop_states = crop_states
                st.rerun()
            if st.button("↻  Reset Crop", key=f"reset_assessment_crop_{source_key}", width="stretch"):
                crop_states = reset_crop_state(crop_states, selected_filename, original_image)
                st.session_state.assessment_crop_states = crop_states
                st.rerun()
            st.info("Only the selected image will be cropped. Other images remain unchanged.")
        if source_key in crop_states:
            st.success("Crop confirmed. This image will use its full-resolution crop during assessment.")
    return selected_filename, crop_states


def render_uploaded_image_preview(images: dict[str, np.ndarray], image_metadata: dict[str, int | None], crop_states: dict[str, dict], selected_filename: str) -> None:
    """Show decoded originals before optional cropping or assessment processing begins."""
    if len(images) == 1:
        filename, image = next(iter(images.items()))
        with st.container(border=True):
            st.markdown('<p class="preview-card-title"><span class="material-symbols-rounded">image</span>Uploaded Image Preview</p>', unsafe_allow_html=True)
            st.markdown('<p class="preview-card-subtitle">This is the original image you uploaded.</p>', unsafe_allow_html=True)
            image_col, metadata_col = st.columns([1, 1.75])
            with image_col:
                st.image(image, caption="Original image", width="stretch")
            with metadata_col:
                height, width = image.shape[:2]
                st.markdown(
                    f'''<div class="preview-meta">
                    <div class="preview-meta-row"><span class="material-symbols-rounded">description</span><span class="preview-meta-label">Filename</span><span class="preview-meta-value">{filename}</span></div>
                    <div class="preview-meta-row"><span class="material-symbols-rounded">photo_size_select_large</span><span class="preview-meta-label">Dimensions</span><span class="preview-meta-value">{width} × {height} px</span></div>
                    <div class="preview-meta-row"><span class="material-symbols-rounded">data_object</span><span class="preview-meta-label">File size</span><span class="preview-meta-value">{_format_file_size(image_metadata.get(filename))}</span></div>
                    </div>''',
                    unsafe_allow_html=True,
                )
        return

    with st.container(border=True):
        st.markdown('<p class="preview-card-title"><span class="material-symbols-rounded">collections</span>Uploaded Images Preview</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="preview-card-subtitle">Select an image below to preview or crop. You can crop each of the {len(images)} images individually.</p>', unsafe_allow_html=True)
        columns = st.columns(min(4, len(images)))
        for index, (filename, image) in enumerate(images.items()):
            with columns[index % len(columns)]:
                status = "Cropped" if image_crop_key(filename, image) in crop_states else "Original"
                status_class = "cropped" if status == "Cropped" else ""
                if filename == selected_filename:
                    status, status_class = "Selected", "selected"
                st.image(image, width="stretch")
                st.markdown(f'<span class="preview-meta-value">{filename}</span><span class="gallery-status {status_class}">{status}</span>', unsafe_allow_html=True)


def render_pipeline_tab(pipeline_name: str, pipeline_records, images: dict[str, np.ndarray], accent: str, px_cm: float) -> None:
    frame = records_dataframe(pipeline_records)
    
    if len(frame) > 1:
        filename = st.selectbox("Select image to inspect", frame["filename"].tolist(), index=len(frame)-1, key=f"inspect_{pipeline_name}")
    else:
        filename = frame["filename"].iloc[0]

    record = next(item for item in pipeline_records if item.filename == filename)
    image = images[filename]
    explanation = record.explanation
    geometry = explanation.get("geometry") or calibrate_fruit_geometry(image, pixels_per_cm=px_cm)
    annotated = draw_annotated_geometry(image, geometry)
    grade = explanation.get("quality_grade", "Grade A (Premium)")

    # 1. Three neatly aligned images across 3 equal columns
    st.markdown("##### 🔬 Computer Vision Diagnostic Stages")
    img_c1, img_c2, img_c3 = st.columns(3)
    with img_c1:
        st.image(annotated, caption="1. Spatial Calibration (MinAreaRect)", width="stretch")
    with img_c2:
        st.image(explanation["fruit_mask"], caption="2. Binary Fruit Mask (Otsu)", clamp=True, width="stretch")
    with img_c3:
        st.image(mask_overlay(image, explanation["brown_mask"]), caption="3. Necrotic Blemish Mask", width="stretch")

    # 2. Balanced Metrics and Probability Chart
    stat_c1, stat_c2 = st.columns([1.3, 1])
    with stat_c1:
        st.markdown("##### 📐 Physical Dimensions & Quality Metrics")
        m_row1 = st.columns(2)
        m_row1[0].metric("Ripeness Stage", record.predicted_class.title(), f"{record.confidence:.1%} confidence")
        m_row1[1].metric("Commercial Export Grade", grade, f"{geometry.get('blemish_ratio', 0.0):.1%} blemish")

        m_row2 = st.columns(3)
        m_row2[0].metric("Length", f"{geometry.get('physical_length_cm', 0.0):.1f} cm")
        m_row2[1].metric("Max Width", f"{geometry.get('physical_width_cm', 0.0):.1f} cm")
        m_row2[2].metric("Surface Area", f"{geometry.get('physical_area_cm2', 0.0):.1f} cm²")

    with stat_c2:
        st.markdown("##### 📊 Class Probability Profile")
        probabilities = pd.DataFrame(
            {"Class": [x.title() for x in CLASS_NAMES], "Probability": [record.probabilities.get(x, 0) for x in CLASS_NAMES]}
        ).set_index("Class")
        st.bar_chart(probabilities, color=accent, height=170)

    # 3. Data Ledger Expander
    with st.expander("📋 View Data Table Ledger & Export Reports", expanded=False):
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
    landing_hero()
    uploads = st.file_uploader(
        "Upload banana images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Start with one image, then add more at any time.",
        key="assessment_uploads",
    )
    preset_choice = st.selectbox(
        "Quick-load benchmark preset samples",
        ["-- Choose a preset or upload your own files --"] + list(SAMPLE_PRESETS.keys()),
        help="Optional: load one representative project image instead of uploading a file.",
    )

    images, image_metadata, errors = {}, {}, []
    if preset_choice and preset_choice in SAMPLE_PRESETS:
        preset_path = PROJECT_ROOT / SAMPLE_PRESETS[preset_choice]
        if preset_path.exists():
            try:
                images[preset_choice] = decode_image(preset_path)
                image_metadata[preset_choice] = preset_path.stat().st_size
            except ImageValidationError as exc:
                errors.append(f"{preset_choice}: {exc}")

    if uploads:
        for upload in uploads:
            try:
                images[upload.name] = decode_image(upload.getvalue())
                image_metadata[upload.name] = upload.size
            except ImageValidationError as exc:
                errors.append(f"{upload.name}: {exc}")

    if errors:
        st.warning("Some files were skipped:\n\n" + "\n\n".join(errors))

    if not images:
        return

    crop_states = reconcile_crop_states(st.session_state.get("assessment_crop_states", {}), images)
    st.session_state.assessment_crop_states = crop_states
    selected_filename = st.session_state.get("assessment_selected_image", next(iter(images)))
    if selected_filename not in images:
        selected_filename = next(iter(images))
    render_uploaded_image_preview(images, image_metadata, crop_states, selected_filename)
    selected_filename, crop_states = render_manual_crop_control(images, crop_states)
    st.session_state.assessment_crop_states = crop_states
    analysis_images = analysis_images_for_assessment(images, crop_states)

    with st.expander("Optional benchmark, calibration, and forecast settings"):
        px_cm = st.slider(
            "Spatial calibration (pixels / cm)",
            min_value=1.0,
            max_value=40.0,
            value=DEFAULT_PIXELS_PER_CM,
            step=0.5,
            help="Standardizes pixel dimensions to physical measurements in cm.",
        )
        st.caption("Commercial quality rules: Grade A (<5% spots), Grade B (5-15%), Grade C (15-30%), Reject (>30% or Rotten).")
        st.markdown("##### Ripeness forecast storage conditions")
        storage_l, storage_r = st.columns(2)
        with storage_l:
            storage_temperature_c = st.slider(
                "Storage temperature (°C)", min_value=0.0, max_value=40.0, value=22.0, step=1.0,
                help="Cool storage slows progression; warm storage speeds it up.",
            )
        with storage_r:
            near_ethylene_fruit = st.checkbox(
                "Stored near apples, avocados, or other ethylene-producing fruit",
                help="Ethylene exposure can accelerate banana ripening.",
            )

    action_col, _ = st.columns([1, 3])
    with action_col:
        run_assessment = st.button("Run Assessment  →", type="primary", key="run_assessment_button", width="stretch")
    input_key = assessment_input_key(analysis_images, px_cm)
    saved_result = st.session_state.get("assessment_result")
    if saved_result and saved_result["input_key"] != input_key:
        st.session_state.pop("assessment_result", None)
        saved_result = None

    if run_assessment:
        all_records, completed = [], 0
        total = len(analysis_images) * len(models)
        progress = st.progress(0, text="Running all image-processing pipelines...")
        for pipeline, _ in models.values():
            for filename, image in analysis_images.items():
                record = pipeline.predict(image, filename)
                # update explanation geometry with current calibration scale
                record.explanation["geometry"] = calibrate_fruit_geometry(image, mask=record.explanation["fruit_mask"], pixels_per_cm=px_cm)
                record.explanation["quality_grade"] = determine_quality_grade(record.predicted_class, record.explanation["geometry"]["blemish_ratio"])
                all_records.append(record)
                completed += 1
                progress.progress(completed / total, text=f"Completed {completed} of {total} assessments")
        progress.empty()
        st.session_state.assessment_result = {
            "input_key": input_key,
            "records": all_records,
            "analysis_images": analysis_images,
        }
    elif saved_result:
        all_records = saved_result["records"]
        analysis_images = saved_result["analysis_images"]
    else:
        return

    # --- Instant Side-by-Side Model Comparison ---
    latest_img_name = selected_filename
    st.markdown(f"### 🍌 Instant Algorithm Comparison (Mode A) — `{latest_img_name}`")
    
    # 4 balanced horizontal prediction cards spanning the screen evenly
    m_cols = st.columns(len(models))
    for idx, (m_key, (p_obj, _)) in enumerate(models.items()):
        rec_match = next((r for r in all_records if r.algorithm == p_obj.display_name and r.filename == latest_img_name), None)
        with m_cols[idx]:
            is_best = (m_key == recommended_key)
            tag = "⭐ Recommended Model" if is_best else f"Member {chr(65+idx)}"
            accent = PIPELINE_ACCENTS[idx % len(PIPELINE_ACCENTS)]
            pred_text = rec_match.predicted_class.title() if rec_match else "N/A"
            conf_text = f"{rec_match.confidence:.1%}" if rec_match else "N/A"
            speed_text = f"{rec_match.processing_time_ms:.1f} ms" if rec_match else "N/A"
            
            st.markdown(
                f"""<div style="background:#111827; border:1px solid #1e293b; border-top:3px solid {accent}; border-radius:10px; padding:16px; text-align:center; height:210px; box-sizing:border-box; display:flex; flex-direction:column; align-items:center;">
                    <div style="font-size:0.75rem; font-weight:700; color:{accent}; text-transform:uppercase;">{tag}</div>
                    <div style="font-size:0.85rem; color:#94a3b8; margin-bottom:8px; min-height:48px; display:flex; align-items:flex-start; justify-content:center;">{p_obj.display_name}</div>
                    <div style="font-size:1.5rem; font-weight:800; color:#ffffff; margin:4px 0;">{pred_text}</div>
                    <div style="font-size:0.82rem; color:#cbd5e1;">Confidence: <b>{conf_text}</b></div>
                    <div style="font-size:0.78rem; color:#64748b; margin-top:auto;">Inference: {speed_text}</div>
                </div>""",
                unsafe_allow_html=True
            )

    recommended_pipeline = models[recommended_key][0]
    recommended_record = next(
        record
        for record in all_records
        if record.algorithm == recommended_pipeline.display_name and record.filename == latest_img_name
    )
    render_ripeness_forecast(recommended_record, storage_temperature_c, near_ethylene_fruit)

    st.divider()

    # Expanders with expanded=True on the Deep Dive Explorer
    with st.expander("🔍 Deep-Dive Pipeline Explorer (Inspect Masks, Contours & Probabilities)", expanded=True):
        st.caption("Select any pipeline tab below to inspect its intermediate computer vision stages, color distributions, and blemish detection.")
        tabs = st.tabs([pipeline.display_name for pipeline, _ in models.values()])
        for tab, (pipeline, _), accent in zip(tabs, models.values(), PIPELINE_ACCENTS, strict=True):
            with tab:
                render_pipeline_tab(pipeline.display_name, [r for r in all_records if r.algorithm == pipeline.display_name], analysis_images, accent, px_cm)

    with st.expander("📊 Advanced Consensus Summary & Data Exports (Batch Auditing)", expanded=False):
        st.caption("Aggregated statistical ledger and commercial grade distribution across all evaluated images.")
        assessment_summary(all_records, models[recommended_key][0].display_name)



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
    hero("Preprocessing & filter lab", "Interactive experimentation with classical filters, histogram equalization, morphology, and colour slicing.", "Core image processing")
    preset_name = st.selectbox("Select sample for filter experimentation", list(SAMPLE_PRESETS.keys()))
    image_path = PROJECT_ROOT / SAMPLE_PRESETS[preset_name]
    raw_image = decode_image(image_path)

    st.subheader("1. Spatial Smoothing & Noise Suppression")
    st.markdown("""<div class="section-desc"><b>Why this matters:</b> Packhouse cameras introduce high-frequency sensor noise. We compare Gaussian vs. Median vs. Bilateral filtering. Notice how the <b>Bilateral Filter</b> smooths the peel surface while keeping the boundary edges of brown necrotic spots sharp!</div>""", unsafe_allow_html=True)
    f_control_col1, f_control_col2, f_control_col3 = st.columns(3)
    with f_control_col1:
        g_k = st.slider("Gaussian kernel size", 3, 15, 5, step=2)
        g_sigma = st.slider("Gaussian sigma", 0.5, 5.0, 1.5, step=0.5)
    with f_control_col2:
        m_k = st.slider("Median filter size", 3, 15, 5, step=2)
    with f_control_col3:
        b_d = st.slider("Bilateral diameter", 3, 15, 9, step=2)

    gaussian_img = apply_gaussian_filter(raw_image, kernel_size=g_k, sigma=g_sigma)
    median_img = apply_median_filter(raw_image, kernel_size=m_k)
    bilateral_img = apply_bilateral_filter(raw_image, d=b_d)
    f_image_col1, f_image_col2, f_image_col3 = st.columns(3)
    f_image_col1.image(gaussian_img, caption=f"Gaussian Filter (k={g_k}, σ={g_sigma})", width="stretch")
    f_image_col2.image(median_img, caption=f"Median Filter (k={m_k}) - Impulse noise removal", width="stretch")
    f_image_col3.image(bilateral_img, caption=f"Bilateral Filter (d={b_d}) - Edge-preserving", width="stretch")

    st.subheader("2. Illumination Invariance & Contrast Enhancement")
    st.markdown("""<div class="section-desc"><b>Why this matters:</b> Conveyor lighting can be uneven. Linear stretching standardizes dynamic range, while <b>CLAHE (Contrast Limited Adaptive Histogram Equalization)</b> amplifies subtle green-yellow hue boundaries without over-amplifying background noise.</div>""", unsafe_allow_html=True)
    _, _, clahe_control_col = st.columns(3)
    with clahe_control_col:
        clahe_clip = st.slider("CLAHE clip limit", 1.0, 6.0, 2.5, step=0.5)

    stretched_img = apply_contrast_stretching(raw_image)
    clahe_img = apply_clahe(raw_image, clip_limit=clahe_clip)
    c_image_col1, c_image_col2, c_image_col3 = st.columns(3)
    c_image_col1.image(raw_image, caption="Original RGB input", width="stretch")
    c_image_col2.image(stretched_img, caption="Linear Min-Max Contrast Stretching", width="stretch")
    c_image_col3.image(clahe_img, caption=f"CLAHE (Clip={clahe_clip}) - Adaptive Equalization", width="stretch")

    st.subheader("3. Mathematical Morphology & Binary Mask Refinement")
    st.markdown("""<div class="section-desc"><b>Why this matters:</b> Initial color thresholding produces ragged edges and pinhole voids due to specular reflections. <b>Erosion</b> removes isolated dust pixels, <b>Dilation</b> expands boundaries, and <b>Closing</b> seals internal glare voids to create a solid banana mask.</div>""", unsafe_allow_html=True)
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

    st.subheader("4. Parametric HSV Color Slicing (Member A Blemish Detection)")
    st.markdown("""<div class="section-desc"><b>Why this matters:</b> Member A's pipeline relies on extracting necrotic spots via HSV color thresholding. Tune the sliders below to calibrate the exact Hue range and Darkness (Value) threshold to isolate brown spots under different illumination conditions.</div>""", unsafe_allow_html=True)
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


def sync_theme_preference() -> None:
    """Keep the selected theme when the browser page is refreshed."""
    st.query_params["theme"] = "dark" if st.session_state.dark_mode else "light"


def main() -> None:
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = st.query_params.get("theme", "dark") != "light"

    dark_mode = st.toggle(
        "Dark Mode",
        key="dark_mode",
        on_change=sync_theme_preference,
    )
    apply_reference_theme(dark_mode)
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
            "Video Analysis",
            "Preprocessing Lab",
            "Model Evaluation",
            "Model Training",
        ],
    )
    st.sidebar.markdown(
        '<div class="sidebar-foot">AI-assisted ripeness insight<br>Version 1.0</div>',
        unsafe_allow_html=True,
    )

    if not models:
        st.warning("No trained model bundles were found. Open Model training to configure the first run.")
        if page == "Model Training":
            training_page(models)
        return

    recommended_key = summary.get("recommended_pipeline") if summary else next(iter(models))
    if recommended_key not in models:
        recommended_key = next(iter(models))
    if page == "Assessment":
        assessment_page(models, recommended_key)
    elif page == "Video Analysis":
        video_analysis_page(models, recommended_key)
    elif page == "Preprocessing Lab":
        filter_lab_page(models)
    elif page == "Model Evaluation":
        evaluation_page(summary, models)
    else:
        training_page(models)


if __name__ == "__main__":
    main()
