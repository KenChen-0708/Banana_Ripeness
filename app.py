from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.constants import CLASS_NAMES, DEFAULT_ARTIFACT_DIR
from src.exports import records_csv, records_dataframe, records_pdf
from src.model_store import IncompatibleModelError, discover_bundles
from src.preprocessing import ImageValidationError, decode_image, mask_overlay, resize_rgb

st.set_page_config(page_title="Banana Quality Assessment", page_icon="🍌", layout="wide")


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


def prediction_table(records):
    frame = records_dataframe(records)
    columns = ["algorithm", "predicted_class", "confidence", "brown_spot_ratio", "processing_time_ms"]
    st.dataframe(
        frame[columns].style.format(
            {"confidence": "{:.1%}", "brown_spot_ratio": "{:.1%}", "processing_time_ms": "{:.1f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )


def render_visual_explanation(image, record):
    explanation = record.explanation
    left, centre, right = st.columns(3)
    left.image(image, caption="Standardised RGB input", use_container_width=True)
    centre.image(explanation["fruit_mask"], caption="Detected banana foreground", clamp=True, use_container_width=True)
    right.image(
        mask_overlay(image, explanation["brown_mask"]),
        caption=f"Brown/dark surface regions ({record.brown_spot_ratio:.1%})",
        use_container_width=True,
    )


def single_page(models, recommended_key):
    st.header("Single-image assessment")
    uploaded = st.file_uploader("Upload a banana image", type=["jpg", "jpeg", "png"], key="single")
    if uploaded is None:
        st.info("Upload a JPG or PNG image to compare the four algorithms.")
        return
    try:
        image = decode_image(uploaded.getvalue())
    except ImageValidationError as exc:
        st.error(str(exc))
        return

    records = [pipeline.predict(image, uploaded.name) for pipeline, _ in models.values()]
    prediction_table(records)
    recommended = next((record for record in records if models[recommended_key][0].display_name == record.algorithm), records[0])
    st.success(
        f"Recommended result: **{recommended.predicted_class.title()}** "
        f"({recommended.confidence:.1%} confidence) using {recommended.algorithm}."
    )
    render_visual_explanation(resize_rgb(image, (224, 224)), recommended)


def batch_page(models, recommended_key):
    st.header("Batch assessment")
    options = {pipeline.display_name: key for key, (pipeline, _) in models.items()}
    default_name = models[recommended_key][0].display_name if recommended_key in models else next(iter(options))
    selected_name = st.selectbox("Assessment algorithm", list(options), index=list(options).index(default_name))
    uploads = st.file_uploader(
        "Upload multiple banana images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="batch",
    )
    if not uploads:
        st.info("Select multiple JPG or PNG files for batch grading.")
        return
    pipeline = models[options[selected_name]][0]
    records = []
    errors = []
    with st.spinner("Assessing images..."):
        for upload in uploads:
            try:
                records.append(pipeline.predict(decode_image(upload.getvalue()), upload.name))
            except ImageValidationError as exc:
                errors.append(f"{upload.name}: {exc}")
    if errors:
        st.warning("\n".join(errors))
    if not records:
        return
    frame = records_dataframe(records)
    first, second, third = st.columns(3)
    first.metric("Images assessed", len(records))
    second.metric("Average confidence", f"{frame['confidence'].mean():.1%}")
    third.metric("Average blemish ratio", f"{frame['brown_spot_ratio'].mean():.1%}")
    st.bar_chart(frame["predicted_class"].value_counts().reindex(CLASS_NAMES, fill_value=0))
    prediction_table(records)
    low_confidence = frame[frame["confidence"] < 0.60]
    if not low_confidence.empty:
        st.warning(f"{len(low_confidence)} result(s) are below 60% confidence and should be reviewed manually.")
    csv_col, pdf_col = st.columns(2)
    csv_col.download_button("Download CSV", records_csv(records), "banana_assessment.csv", "text/csv")
    pdf_col.download_button("Download PDF", records_pdf(records), "banana_assessment.pdf", "application/pdf")


def evaluation_page(summary):
    st.header("Model evaluation")
    if not summary:
        st.warning("Run `scripts/train_compare.py` to generate evaluation results.")
        return
    rows = []
    for rank, item in enumerate(summary["results"], start=1):
        rows.append(
            {
                "Rank": rank,
                "Algorithm": item["algorithm"],
                "Validation accuracy": item["validation_metrics"]["accuracy"],
                "Test accuracy": item["test_metrics"]["accuracy"],
                "Test macro-F1": item["test_metrics"]["macro_f1"],
                "Test weighted-F1": item["test_metrics"]["weighted_f1"],
            }
        )
    frame = pd.DataFrame(rows)
    st.dataframe(
        frame.style.format({column: "{:.2%}" for column in frame.columns if column not in ("Rank", "Algorithm")}),
        hide_index=True,
        use_container_width=True,
    )
    st.bar_chart(frame.set_index("Algorithm")[["Test accuracy", "Test macro-F1"]])
    selected = st.selectbox("Confusion matrix", [item["algorithm"] for item in summary["results"]])
    item = next(result for result in summary["results"] if result["algorithm"] == selected)
    matrix = pd.DataFrame(item["test_metrics"]["confusion_matrix"], index=CLASS_NAMES, columns=CLASS_NAMES)
    matrix.index.name = "Actual / Predicted"
    st.dataframe(matrix, use_container_width=True)
    report = pd.DataFrame(item["test_metrics"]["classification_report"]).T
    st.subheader("Class-level performance")
    st.dataframe(report.style.format("{:.3f}"), use_container_width=True)


def about_page():
    st.header("System flow and interpretation")
    st.markdown(
        """
        1. The uploaded image is validated, orientation-corrected, and converted to RGB.
        2. A reproducible foreground mask separates the banana from most background pixels.
        3. Each pipeline extracts its own colour, cluster, histogram, or texture features.
        4. Four supervised classifiers independently predict the ripeness class.
        5. The dashboard compares the outputs and highlights the model with the highest held-out macro-F1.
        6. Brown and dark pixels are quantified relative to the detected banana surface.

        The result is decision support, not a food-safety guarantee. Low-confidence images and unusual acquisition conditions should be reviewed manually.
        """
    )


def main():
    st.title("🍌 Banana Ripeness & Surface Quality Assessment")
    st.caption("Classical image processing • four-model comparison • reproducible evaluation")
    models = model_state()
    summary = load_evaluation(str(DEFAULT_ARTIFACT_DIR / "evaluation_summary.json"))
    if not models:
        st.warning(
            "No trained model bundles were found. Run `python scripts/train_compare.py --dataset fruit_dataset`, then refresh this page."
        )
        return
    recommended_key = summary.get("recommended_pipeline") if summary else next(iter(models))
    if recommended_key not in models:
        recommended_key = next(iter(models))
    page = st.sidebar.radio("Navigation", ["Single image", "Batch assessment", "Model evaluation", "System flow"])
    st.sidebar.success(f"Recommended model: {models[recommended_key][0].display_name}")
    if page == "Single image":
        single_page(models, recommended_key)
    elif page == "Batch assessment":
        batch_page(models, recommended_key)
    elif page == "Model evaluation":
        evaluation_page(summary)
    else:
        about_page()


if __name__ == "__main__":
    main()

