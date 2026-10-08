"""End-to-end cascade detector."""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import torch

from LMNDM.detection.box_ops import (
    decode_detection_boxes,
    nonmaximum_suppression,
)
from LMNDM.detection.config import DetectionConfig
from LMNDM.checkpoints import load_checkpoint
from LMNDM.detection.preprocessing import (
    automatic_lung_mask,
    dilate_mask,
    image_world_bounds,
    mask_world_bounds,
    normalized_array_from_config,
    plan_patches,
    points_inside_mask,
    read_image,
    resample_patch,
)
from LMNDM.network.FPNVNet import DetectionNet
from LMNDM.runtime import resolve_device


class CascadeDetector:
    def __init__(
        self,
        model_dir,
        device="auto",
        *,
        fp_model_dir=None,
        patch_size=None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.device = resolve_device(device)
        self.config = DetectionConfig.from_ini(
            self.model_dir / "config.ini",
            patch_size=patch_size,
        )
        self.networks = []
        self.model_config = None
        self._load_networks()

        self.fp_reducer = None
        if fp_model_dir is not None and self.config.fp_reduction:
            from LMNDM.fp_reduction import FalsePositiveReducer

            self.fp_reducer = FalsePositiveReducer(fp_model_dir, device=self.device)

    def _load_networks(self) -> None:
        model_paths = sorted(self.model_dir.glob("level*.pth"))
        if len(model_paths) != len(self.config.cascade_thresholds):
            raise ValueError(
                f"expected {len(self.config.cascade_thresholds)} detector levels, "
                f"found {len(model_paths)}"
            )

        for model_path in model_paths:
            checkpoint = load_checkpoint(model_path)
            network = DetectionNet(
                int(checkpoint["in_channels"]),
                len(checkpoint["aspect_ratios"]),
                checkpoint["down_ratios"],
            )
            network.load_state_dict(checkpoint["state_dict"], strict=True)
            network.to(self.device).eval()
            self.networks.append(network)
            if self.model_config is None:
                self.model_config = checkpoint
            else:
                for key in ("spacing", "down_ratios", "scales", "aspect_ratios"):
                    if not np.array_equal(self.model_config[key], checkpoint[key]):
                        raise ValueError(f"detector levels disagree on {key}")

    @property
    def in_channels(self) -> int:
        return int(self.model_config["in_channels"])

    def _infer_patch(self, inputs, origin):
        level_outputs = []
        with torch.inference_mode():
            for network in self.networks:
                outputs = network(inputs)
                level_outputs.append(
                    [output.detach().cpu().numpy() for output in outputs]
                )
        return self._cascade_boxes(level_outputs, origin)

    def _cascade_boxes(self, level_outputs, origin):
        average_probabilities = []
        for level, outputs in enumerate(level_outputs):
            threshold = self.config.cascade_thresholds[level]
            for head_index, output in enumerate(outputs):
                logits = np.clip(output[0, ..., 0], -80.0, 80.0)
                probability = 1.0 / (1.0 + np.exp(-logits))
                if level == 0:
                    probability[probability <= threshold] = 0.0
                    average_probabilities.append(probability)
                else:
                    probability[average_probabilities[head_index] == 0.0] = 0.0
                    probability = (
                        probability + average_probabilities[head_index] * level
                    ) / (level + 1)
                    probability[probability <= threshold] = 0.0
                    average_probabilities[head_index] = probability

        boxes = []
        final_outputs = level_outputs[-1]
        aspect_ratios = np.asarray(self.model_config["aspect_ratios"])
        for head_index, output in enumerate(final_outputs):
            positive_indices = np.argwhere(average_probabilities[head_index] > 0.0)
            if len(positive_indices) == 0:
                continue
            positive_outputs = output[
                0,
                positive_indices[:, 0],
                positive_indices[:, 1],
                positive_indices[:, 2],
                positive_indices[:, 3],
                :,
            ].copy()
            positive_outputs[:, 0] = average_probabilities[head_index][
                tuple(positive_indices.T)
            ]
            anchor_sizes = (
                int(self.model_config["scales"][head_index]) * aspect_ratios
            )
            boxes.append(
                decode_detection_boxes(
                    positive_outputs,
                    positive_indices,
                    origin,
                    self.model_config["spacing"],
                    int(self.model_config["down_ratios"][head_index]),
                    anchor_sizes,
                )
            )
        return np.concatenate(boxes, axis=0) if boxes else np.empty((0, 7))

    def detect(
        self,
        images,
        lung_mask=None,
        *,
        mask_mode: str = "auto",
        apply_fp_reduction: bool = True,
        verbose: bool = True,
    ):
        if not isinstance(images, (list, tuple)):
            images = [images]
        if len(images) < self.in_channels:
            raise ValueError(
                f"detector requires {self.in_channels} input channel(s), "
                f"received {len(images)}"
            )
        images = [read_image(image) for image in images[: self.in_channels]]
        if mask_mode not in {"auto", "provided", "none"}:
            raise ValueError("mask_mode must be auto, provided, or none")
        if lung_mask is not None:
            search_mask = read_image(lung_mask)
        elif mask_mode == "provided":
            raise ValueError("mask_mode='provided' requires a lung mask")
        elif mask_mode == "auto":
            search_mask = automatic_lung_mask(images[0])
        else:
            search_mask = None

        if search_mask is None:
            world_min, world_max = image_world_bounds(images[0])
            if verbose:
                print("Lung region: full CT volume")
        else:
            world_min, world_max = mask_world_bounds(search_mask)
            if verbose:
                source = "provided mask" if lung_mask is not None else "automatic estimate"
                print(f"Lung region: {source}")
        spacing = np.asarray(self.model_config["spacing"], dtype=np.float64)
        normalizers = self.model_config["crop_normalizers"]
        all_boxes = []

        patches = list(
            plan_patches(
                world_min,
                world_max,
                spacing,
                self.config.margin_mm,
                self.config.patch_size,
            )
        )
        for origin, size, patch_index, total in patches:
            started = time.time()
            arrays = []
            for channel, image in enumerate(images):
                crop = resample_patch(
                    image,
                    origin,
                    spacing,
                    size,
                    linear=True,
                    default_value=float(normalizers[channel].get("mean", -600.0)),
                )
                arrays.append(
                    normalized_array_from_config(
                        crop,
                        normalizers[channel],
                    )
                )
            tensor = torch.from_numpy(np.stack(arrays, axis=0))
            tensor = tensor.unsqueeze(0).to(self.device, dtype=torch.float32)
            boxes = self._infer_patch(tensor, origin)
            if len(boxes):
                all_boxes.append(boxes)
            if verbose:
                print(
                    f"Patch {patch_index}/{total}: "
                    f"{time.time() - started:.2f}s, {len(boxes)} candidates"
                )

        boxes = (
            np.concatenate(all_boxes, axis=0)
            if all_boxes
            else np.empty((0, 7), dtype=np.float64)
        )
        boxes = nonmaximum_suppression(
            boxes,
            option=self.config.nms_option,
            overlap_threshold=self.config.nms_overlap_threshold,
            denominator="union",
            sort_by="probability",
        )
        if self.config.nest_nms_option != -1:
            boxes = nonmaximum_suppression(
                boxes,
                # Historical nest-NMS options select the sort key. The
                # suppression rule itself is always overlap-over-min-box.
                option=1,
                overlap_threshold=self.config.nest_nms_overlap_threshold,
                denominator="min",
                sort_by=(
                    "volume"
                    if self.config.nest_nms_option == 1
                    else "probability"
                ),
            )

        if len(boxes) and self.config.lung_mask_reduction and search_mask is not None:
            dilated_mask = dilate_mask(search_mask, self.config.lung_dilation_mm)
            boxes = boxes[points_inside_mask(boxes[:, 1:4], dilated_mask)]

        if (
            len(boxes)
            and self.config.min_size_mm > 0
            and self.config.min_size_mm < self.config.max_size_mm
        ):
            mean_xy_size = (boxes[:, 4] + boxes[:, 5]) / 2.0
            boxes = boxes[
                (mean_xy_size >= self.config.min_size_mm)
                & (mean_xy_size <= self.config.max_size_mm)
            ]

        if apply_fp_reduction and self.fp_reducer is not None and len(boxes):
            boxes = self.fp_reducer.reduce(
                images[0],
                boxes,
                detector_weight=1.0 - self.config.fp_weight,
                threshold=self.config.fp_threshold,
                max_candidates=self.config.fp_max_candidates,
            )
        return boxes
