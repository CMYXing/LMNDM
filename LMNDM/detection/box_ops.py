"""NumPy implementation of the legacy 3-D box decoder and NMS."""

from __future__ import annotations

import numpy as np


def _as_boxes(boxes):
    boxes = np.asarray(boxes, dtype=np.float64)
    if boxes.size == 0:
        return np.empty((0, 7), dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 7:
        raise ValueError("boxes must have shape (N, 7)")
    return boxes


def _overlap_against(box, others, denominator: str):
    box_min = box[1:4] - box[4:7] / 2.0
    box_max = box[1:4] + box[4:7] / 2.0
    other_min = others[:, 1:4] - others[:, 4:7] / 2.0
    other_max = others[:, 1:4] + others[:, 4:7] / 2.0
    intersection_size = np.maximum(
        np.minimum(box_max, other_max) - np.maximum(box_min, other_min),
        0.0,
    )
    intersection = np.prod(intersection_size, axis=1)
    box_volume = float(np.prod(box[4:7]))
    other_volume = np.prod(others[:, 4:7], axis=1)
    if denominator == "union":
        base = box_volume + other_volume - intersection
    elif denominator == "min":
        base = np.minimum(box_volume, other_volume)
    else:
        raise ValueError(f"unknown overlap denominator: {denominator}")
    return np.divide(
        intersection,
        base,
        out=np.zeros_like(intersection),
        where=base > 0,
    )


def nonmaximum_suppression(
    boxes,
    *,
    option: int,
    overlap_threshold: float,
    denominator: str = "union",
    sort_by: str = "probability",
):
    boxes = _as_boxes(boxes)
    if len(boxes) == 0 or option == -1:
        return boxes
    if option not in (0, 1):
        raise ValueError("NMS option must be -1, 0, or 1")

    if sort_by == "probability":
        sort_values = boxes[:, 0]
    elif sort_by == "volume":
        sort_values = np.prod(boxes[:, 4:7], axis=1)
    else:
        raise ValueError("sort_by must be probability or volume")

    remaining = np.argsort(-sort_values, kind="stable")
    keep = []
    while len(remaining):
        selected = int(remaining[0])
        keep.append(selected)
        if len(remaining) == 1:
            break
        candidates = remaining[1:]
        overlaps = _overlap_against(boxes[selected], boxes[candidates], denominator)
        if option == 0:
            suppress = overlaps > 0.0
        else:
            suppress = overlaps >= float(overlap_threshold)
        remaining = candidates[~suppress]

    selected_boxes = boxes[np.asarray(keep, dtype=np.int64)]
    return selected_boxes[np.argsort(-selected_boxes[:, 0], kind="stable")]


def decode_detection_boxes(
    positive_outputs,
    positive_indices,
    image_origin,
    image_spacing,
    down_ratio,
    anchor_sizes,
):
    outputs = np.asarray(positive_outputs, dtype=np.float32)
    indices = np.asarray(positive_indices, dtype=np.int32)
    anchors = np.asarray(anchor_sizes, dtype=np.float64)
    if outputs.ndim != 2 or outputs.shape[1] != 7:
        raise ValueError("positive_outputs must have shape (N, 7)")
    if indices.shape != (len(outputs), 4):
        raise ValueError("positive_indices must have shape (N, 4)")

    spacing = np.asarray(image_spacing, dtype=np.float64) * int(down_ratio)
    start = np.asarray(image_origin, dtype=np.float64) + spacing / 2.0
    grid_xyz = indices[:, (2, 1, 0)].astype(np.float64)
    anchor_index = indices[:, 3]
    anchor_size = anchors[anchor_index]
    anchor_center = start + grid_xyz * spacing

    boxes = np.empty((len(outputs), 7), dtype=np.float64)
    boxes[:, 0] = outputs[:, 0]
    boxes[:, 1:4] = outputs[:, 1:4] * anchor_size + anchor_center
    boxes[:, 4:7] = np.exp(outputs[:, 4:7]) * anchor_size
    return boxes
