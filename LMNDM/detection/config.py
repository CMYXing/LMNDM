"""Configuration parser for the released cascade detector."""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path


def _triple(text: str, cast):
    values = tuple(cast(token.strip()) for token in text.split(","))
    if len(values) != 3:
        raise ValueError(f"expected three comma-separated values, got {text!r}")
    return values


@dataclass(frozen=True)
class DetectionConfig:
    margin_mm: tuple[float, float, float]
    patch_size: tuple[int, int, int]
    cascade_thresholds: tuple[float, ...]
    nms_option: int
    nms_overlap_threshold: float
    nest_nms_option: int
    nest_nms_overlap_threshold: float
    lung_mask_reduction: bool
    lung_dilation_mm: float
    min_size_mm: float
    max_size_mm: float
    fp_reduction: bool
    fp_threshold: float
    fp_max_candidates: int
    fp_weight: float

    @classmethod
    def from_ini(cls, path, patch_size=None):
        path = Path(path)
        parser = configparser.ConfigParser()
        if not parser.read(path, encoding="utf-8"):
            raise FileNotFoundError(path)

        levels = parser.getint("Runtime", "levels")
        thresholds = tuple(
            parser.getfloat(f"Cascade_L{level}", "positive_threshold")
            for level in range(levels)
        )
        configured_patch_size = _triple(
            parser.get("Inference", "patch_size", fallback="160,160,128"),
            int,
        )
        if patch_size is not None:
            configured_patch_size = tuple(int(value) for value in patch_size)
            if len(configured_patch_size) != 3:
                raise ValueError("patch_size must contain three integers")
        if any(value <= 0 or value % 16 != 0 for value in configured_patch_size):
            raise ValueError("each patch dimension must be a positive multiple of 16")

        dilation_spacing = _triple(
            parser.get("Lung_FP_Reduction", "dilation_spacing"),
            float,
        )
        dilation_radius = parser.getfloat("Lung_FP_Reduction", "dilation_radius")
        dilation_mm = max(dilation_spacing) * dilation_radius

        return cls(
            margin_mm=_triple(parser.get("Runtime", "margin_mm"), float),
            patch_size=configured_patch_size,
            cascade_thresholds=thresholds,
            nms_option=parser.getint("NMS", "option"),
            nms_overlap_threshold=parser.getfloat("NMS", "overlap_threshold"),
            nest_nms_option=parser.getint("NMS_NEST", "option"),
            nest_nms_overlap_threshold=parser.getfloat(
                "NMS_NEST",
                "overlap_threshold",
            ),
            lung_mask_reduction=parser.getboolean("Lung_FP_Reduction", "enable"),
            lung_dilation_mm=dilation_mm,
            min_size_mm=parser.getfloat("Filter", "box_size_min_threshold_mm"),
            max_size_mm=parser.getfloat("Filter", "box_size_max_threshold_mm"),
            fp_reduction=parser.getboolean("Runtime", "classification", fallback=False),
            fp_threshold=parser.getfloat(
                "Classification",
                "positive_threshold",
                fallback=0.0,
            ),
            fp_max_candidates=parser.getint(
                "Classification",
                "max_cls_num",
                fallback=-1,
            ),
            fp_weight=parser.getfloat(
                "Runtime",
                "classification_weight",
                fallback=0.5,
            ),
        )
