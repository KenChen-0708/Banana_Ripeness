from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit.components.v1 as components
from PIL import Image


_polygon_component = components.declare_component(
    "banana_polygon_cropper",
    path=str(Path(__file__).parent / "frontend"),
)


def polygon_cropper(
    image: np.ndarray,
    initial_points: list[tuple[float, float]] | None = None,
    key: str | None = None,
) -> dict:
    """Render an interactive point-and-drag polygon editor for an RGB image."""
    stream = BytesIO()
    Image.fromarray(image).save(stream, format="PNG")
    image_data = "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")
    points = [list(point) for point in (initial_points or [])]
    return _polygon_component(
        image_data=image_data,
        width=int(image.shape[1]),
        height=int(image.shape[0]),
        initial_points=points,
        key=key,
        default={"points": points},
    )
