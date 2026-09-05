from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from .constants import CLASS_NAMES


def probability_figure(probabilities: dict[str, float], accent: str):
    """Create a compact bounded chart that cannot overflow a dashboard column."""
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
        axis.text(min(value + 0.025, 0.94), bar.get_y() + bar.get_height() / 2, f"{value:.1%}", va="center", ha="left", color="#F4F7F5", fontsize=9, fontweight="bold")
    figure.tight_layout(pad=1.1)
    return figure


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
    axis.set_xticks(range(len(CLASS_NAMES)), [value.title() for value in CLASS_NAMES], rotation=30, ha="right")
    axis.set_yticks(range(len(CLASS_NAMES)), [value.title() for value in CLASS_NAMES])
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
