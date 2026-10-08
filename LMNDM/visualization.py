"""Notebook-friendly visualization of per-nodule metastatic scores."""

from __future__ import annotations

import math
import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import SimpleITK as sitk

from LMNDM.detection.preprocessing import read_image, resample_patch


DISPLAY_FIELDS = (
    "Primary tumor",
    "density_type",
    "Label",
    "metastatic_score",
    "predicted_class",
)
CLASS_COLORS = {
    "Metastatic": "tab:red",
    "Indeterminate": "tab:orange",
    "Benign": "tab:cyan",
}
SLICE_NOTE = (
    "The displayed images are cropped axial slices through the centers of "
    "the target nodules."
)


def _display_value(value):
    return f"{value:.4f}" if isinstance(value, float) else value


def show_nodules(
    image,
    nodules,
    *,
    columns: int = 2,
    field_of_view_mm: float = 60.0,
    pixel_mm: float = 0.5,
    window=(-1000.0, 400.0),
    box_scale: float = 1.5,
    note: str | None = SLICE_NOTE,
):
    """Show one axial panel per nodule, titled with ``DISPLAY_FIELDS`` only.

    Each nodule is a mapping with physical ``x``, ``y``, ``z``, ``width`` and
    ``height`` in millimetres plus the display fields.
    """
    if columns <= 0 or field_of_view_mm <= 0 or pixel_mm <= 0 or box_scale <= 0:
        raise ValueError("columns, field of view, pixel size and box scale must be positive")

    image = read_image(image)
    size = int(round(field_of_view_mm / pixel_mm))
    half = size * pixel_mm / 2.0
    rows = max(math.ceil(len(nodules) / columns), 1)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(6.0 * columns, 7.3 * rows + 0.6),
        squeeze=False,
        layout="constrained",
    )
    axes = axes.ravel()
    for axis, nodule in zip(axes, nodules):
        x, y, z = (float(nodule[key]) for key in ("x", "y", "z"))
        # One axial plane through the nodule center, resampled to patient axes.
        plane = resample_patch(
            image,
            (x - half, y - half, z),
            (pixel_mm, pixel_mm, 1.0),
            (size, size, 1),
            default_value=window[0],
        )
        axis.imshow(
            sitk.GetArrayViewFromImage(plane)[0],
            cmap="gray",
            vmin=window[0],
            vmax=window[1],
            extent=(-half, half, half, -half),
        )
        width = float(nodule["width"]) * box_scale
        height = float(nodule["height"]) * box_scale
        color = CLASS_COLORS[nodule["predicted_class"]]
        axis.add_patch(
            Rectangle(
                (-width / 2.0, -height / 2.0),
                width,
                height,
                fill=False,
                edgecolor=color,
                linewidth=2,
            )
        )
        axis.set_title(
            "\n".join(
                textwrap.fill(f"{field}: {_display_value(nodule[field])}", width=40)
                for field in DISPLAY_FIELDS
            ),
            fontsize=16,
            loc="left",
        )
        axis.axis("off")
    for axis in axes[len(nodules) :]:
        axis.axis("off")
    if note:
        figure.suptitle(note, x=0.0, ha="left", fontsize=16)
    return figure
