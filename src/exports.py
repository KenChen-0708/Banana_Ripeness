from __future__ import annotations

from io import BytesIO
from typing import Iterable

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .types import PredictionRecord


def records_dataframe(records: Iterable[PredictionRecord]) -> pd.DataFrame:
    return pd.DataFrame(record.to_dict() for record in records)


def records_csv(records: Iterable[PredictionRecord]) -> bytes:
    return records_dataframe(records).to_csv(index=False).encode("utf-8")


def records_pdf(records: Iterable[PredictionRecord], title: str = "Banana Ripeness Assessment Summary") -> bytes:
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
    story = [Paragraph(title, styles["Title"]), Spacer(1, 6 * mm)]
    if frame.empty:
        story.append(Paragraph("No prediction records were supplied.", styles["BodyText"]))
    else:
        columns = ["filename", "algorithm", "predicted_class", "confidence", "processing_time_ms"]
        display = frame[columns].copy()
        display["confidence"] = display["confidence"].map(lambda value: f"{float(value):.2%}")
        display["processing_time_ms"] = display["processing_time_ms"].map(lambda value: f"{float(value):.1f}")
        data = [list(display.columns)] + display.astype(str).values.tolist()
        table = Table(data, repeatRows=1, colWidths=[60 * mm, 70 * mm, 35 * mm, 30 * mm, 34 * mm])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#24543A")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#B8C7BE")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F7F3")]),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(table)
    document.build(story)
    return buffer.getvalue()

