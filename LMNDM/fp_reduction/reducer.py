"""Pure-PyTorch bagging classifier for detector false-positive reduction."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import torch

from LMNDM.checkpoints import load_checkpoint, strip_module_prefix
from LMNDM.detection.preprocessing import (
    centered_patch,
    normalized_array_from_config,
)
from LMNDM.runtime import resolve_device


class FalsePositiveReducer:
    def __init__(self, model_dir, device="auto", batch_size: int = 8) -> None:
        self.model_dir = Path(model_dir)
        self.device = resolve_device(device)
        self.batch_size = max(int(batch_size), 1)
        self.models = []
        self.spacing = None
        self.crop_size = None
        self.normalizer = None
        self._load_models()

    def _load_models(self) -> None:
        model_paths = sorted((self.model_dir / "basic").glob("*.pth"))
        if len(model_paths) != 3:
            raise FileNotFoundError(
                "expected exactly three BasicNet false-positive checkpoints in "
                f"{self.model_dir / 'basic'}, found {len(model_paths)}"
            )

        for model_path in model_paths:
            checkpoint = load_checkpoint(model_path)
            net_name = checkpoint["net"]
            if net_name != "BasicNet":
                raise ValueError(
                    f"{model_path} declares {net_name!r}; only BasicNet is supported"
                )
            module = importlib.import_module(f"LMNDM.network.{net_name}")
            network = module.ClassificationNet(
                int(checkpoint["input_channel"]),
                int(checkpoint["num_classes"]),
                checkpoint["crop_size"],
            )
            network.load_state_dict(
                strip_module_prefix(checkpoint["state_dict"]),
                strict=True,
            )
            network.to(self.device).eval()
            self.models.append(network)

            spacing = tuple(float(value) for value in checkpoint["spacing"])
            crop_size = tuple(int(value) for value in checkpoint["crop_size"])
            normalizer = checkpoint["crop_normalizers"][0]
            if self.spacing is None:
                self.spacing = spacing
                self.crop_size = crop_size
                self.normalizer = normalizer
            elif (
                spacing != self.spacing
                or crop_size != self.crop_size
                or normalizer != self.normalizer
            ):
                raise ValueError("false-positive ensemble preprocessing is inconsistent")

    def _crop_tensor(self, image, boxes):
        arrays = []
        for box in boxes:
            crop = centered_patch(
                image,
                center=box[1:4],
                spacing=self.spacing,
                size=self.crop_size,
                default_value=float(self.normalizer.get("mean", -600.0)),
            )
            array = normalized_array_from_config(crop, self.normalizer)
            arrays.append(np.transpose(array, (2, 1, 0)))
        array = np.stack(arrays, axis=0)[:, np.newaxis]
        return torch.from_numpy(np.ascontiguousarray(array))

    def positive_probabilities(self, image, boxes):
        boxes = np.asarray(boxes, dtype=np.float64)
        if len(boxes) == 0:
            return np.empty((0,), dtype=np.float64)

        inputs = self._crop_tensor(image, boxes)
        ensemble_probabilities = np.zeros(len(boxes), dtype=np.float64)
        with torch.inference_mode():
            for network in self.models:
                model_probabilities = []
                for start in range(0, len(inputs), self.batch_size):
                    batch = inputs[start : start + self.batch_size].to(
                        self.device,
                        dtype=torch.float32,
                    )
                    prediction = network(batch)
                    model_probabilities.append(prediction[:, 1].detach().cpu().numpy())
                ensemble_probabilities += np.concatenate(model_probabilities)
        ensemble_probabilities /= len(self.models)
        return ensemble_probabilities

    def reduce(
        self,
        image,
        boxes,
        *,
        detector_weight: float = 0.5,
        threshold: float = 0.58,
        max_candidates: int = 100,
    ):
        boxes = np.asarray(boxes, dtype=np.float64)
        if len(boxes) == 0:
            return np.empty((0, 7), dtype=np.float64)
        if max_candidates > 0:
            boxes = boxes[:max_candidates]

        classification_probability = self.positive_probabilities(image, boxes)
        combined = (
            boxes[:, 0] * float(detector_weight)
            + classification_probability * (1.0 - float(detector_weight))
        )
        output = boxes.copy()
        output[:, 0] = combined
        output = output[combined > float(threshold)]
        if len(output):
            output = output[np.argsort(-output[:, 0], kind="stable")]
        return output
