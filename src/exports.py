from __future__ import annotations

from io import BytesIO
from typing import Iterable

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .types import PredictionRecord


def records_dataframe(records: Iterable[PredictionRecord]) -> pd.DataFrame:
    rows = []
    for record in records:
        data = record.to_dict()
        exp = record.explanation or {}
        geom = exp.get("geometry", {})
        data["blemish_percent"] = (exp.get("blemish_ratio", 0.0)) * 100.0
        data["quality_grade"] = exp.get("quality_grade", "Grade A (Premium)")
        data["length_cm"] = geom.get("physical_length_cm", 0.0)
        data["width_cm"] = geom.get("physical_width_cm", 0.0)
        data["area_cm2"] = geom.get("physical_area_cm2", 0.0)
        rows.append(data)
    return pd.DataFrame(rows)


def records_csv(records: Iterable[PredictionRecord]) -> bytes:
    return records_dataframe(records).to_csv(index=False).encode("utf-8")


def records_pdf(records: Iterable[PredictionRecord], title: str = "Banana Ripeness Assessment Report") -> bytes:
    """Create a readable management summary followed by auditable result details."""
    frame = records_dataframe(records)
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=12 * mm,
        leftMargin=12 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
    )
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]), Spacer(1, 3 * mm)]
    story.append(
        Paragraph(
            "Automated comparison of classical image-processing pipelines for banana "
            "ripeness, surface-quality grading, and geometric morphometry.",
            styles["BodyText"],
        )
    )
    story.append(Spacer(1, 5 * mm))
    if frame.empty:
        story.append(Paragraph("No prediction records were supplied.", styles["BodyText"]))
    else:
        total_images = int(frame["filename"].nunique())
        total_pipelines = int(frame["algorithm"].nunique())
        low_confidence = int((frame["confidence"] < 0.60).sum())
        mean_blemish = frame["blemish_percent"].mean() if "blemish_percent" in frame.columns else 0.0
        summary_data = [
            ["Images", "Pipelines", "Predictions", "Avg Confidence", "Avg Blemish %", "Avg Time", "Review Flags"],
            [
                str(total_images),
                str(total_pipelines),
                str(len(frame)),
                f"{frame['confidence'].mean():.1%}",
                f"{mean_blemish:.1f}%",
                f"{frame['processing_time_ms'].mean():.1f} ms",
                str(low_confidence),
            ],
        ]
        summary_table = Table(summary_data, colWidths=[30 * mm, 30 * mm, 30 * mm, 35 * mm, 35 * mm, 30 * mm, 30 * mm])
        summary_table.setStyle(_table_style(header_colour="#164E3B", font_size=8))
        story.extend([summary_table, Spacer(1, 5 * mm)])

        aggregate = (
            frame.groupby("algorithm", sort=False)
            .agg(
                predictions=("filename", "count"),
                average_confidence=("confidence", "mean"),
                average_blemish=("blemish_percent", "mean"),
                average_time_ms=("processing_time_ms", "mean"),
            )
            .reset_index()
        )
        aggregate["average_confidence"] = aggregate["average_confidence"].map(lambda value: f"{value:.2%}")
        aggregate["average_blemish"] = aggregate["average_blemish"].map(lambda value: f"{value:.1f}%")
        aggregate["average_time_ms"] = aggregate["average_time_ms"].map(lambda value: f"{value:.1f}")
        aggregate.columns = ["Pipeline", "Predictions", "Avg Confidence", "Avg Blemish", "Avg Time (ms)"]
        aggregate_table = Table(
            [list(aggregate.columns)] + aggregate.astype(str).values.tolist(),
            repeatRows=1,
            colWidths=[80 * mm, 30 * mm, 35 * mm, 35 * mm, 35 * mm],
        )
        aggregate_table.setStyle(_table_style(header_colour="#28745A", font_size=8))
        story.extend(
            [
                Paragraph("All-pipeline summary", styles["Heading2"]),
                aggregate_table,
                Spacer(1, 4 * mm),
                Paragraph(
                    "Results below 60% confidence are flagged for manual review. "
                    "Commercial grading considers both ripening stage and surface blemish percentage.",
                    styles["BodyText"],
                ),
                PageBreak(),
                Paragraph("Detailed predictions & Calibrated Morphometry", styles["Heading2"]),
                Spacer(1, 2 * mm),
            ]
        )
        columns = ["filename", "algorithm", "predicted_class", "quality_grade", "confidence", "blemish_percent", "length_cm", "width_cm"]
        display = frame[[c for c in columns if c in frame.columns]].copy()
        display["confidence"] = display["confidence"].map(lambda value: f"{float(value):.1%}")
        if "blemish_percent" in display.columns:
            display["blemish_percent"] = display["blemish_percent"].map(lambda value: f"{float(value):.1f}%")
        if "length_cm" in display.columns:
            display["length_cm"] = display["length_cm"].map(lambda value: f"{float(value):.1f}cm")
        if "width_cm" in display.columns:
            display["width_cm"] = display["width_cm"].map(lambda value: f"{float(value):.1f}cm")
        
        display.columns = ["File", "Algorithm", "Class", "Grade", "Confidence", "Blemish", "Length", "Width"]
        data = [list(display.columns)] + display.astype(str).values.tolist()
        table = Table(data, repeatRows=1, colWidths=[45 * mm, 55 * mm, 25 * mm, 35 * mm, 22 * mm, 20 * mm, 20 * mm, 20 * mm])
        table.setStyle(_table_style(header_colour="#24543A", font_size=7))
        story.append(table)
    document.build(story)
    return buffer.getvalue()


def _table_style(header_colour: str, font_size: int) -> TableStyle:
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_colour)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), font_size),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#B8C7BE")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F7F3")]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]
    )
