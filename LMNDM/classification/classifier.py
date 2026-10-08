"""Pure-PyTorch metastatic scoring of pulmonary nodules."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from LMNDM.checkpoints import load_checkpoint, strip_module_prefix
from LMNDM.detection.preprocessing import (
    centered_patch,
    normalized_array_from_config,
    read_image,
)
from LMNDM.network.BResidualNet import ClassificationNet
from LMNDM.runtime import resolve_device
from LMNDM.scoring import metastatic_scores, score_class


class MetastasisClassifier:
    def __init__(
        self,
        checkpoint_path,
        label_index_path,
        *,
        device="auto",
        batch_size: int = 8,
    ) -> None:
        self.device = resolve_device(device)
        self.batch_size = max(int(batch_size), 1)
        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint.get("net") != "BResidualNet":
            raise ValueError("metastasis checkpoint must use BResidualNet")

        self.network = ClassificationNet(
            int(checkpoint["input_channel"]),
            int(checkpoint["num_classes"]),
            checkpoint["crop_size"],
        )
        self.network.load_state_dict(
            strip_module_prefix(checkpoint["state_dict"]),
            strict=True,
        )
        self.network.to(self.device).eval()
        self.spacing = tuple(float(value) for value in checkpoint["spacing"])
        self.crop_size = tuple(int(value) for value in checkpoint["crop_size"])
        self.normalizer = checkpoint["crop_normalizers"][0]
        self.labels = self._read_labels(label_index_path)
        self.metastatic_index = next(
            (
                index
                for index, label in self.labels.items()
                if label == "metastatic"
            ),
            1,
        )

    @staticmethod
    def _read_labels(path):
        with Path(path).open("r", newline="", encoding="utf-8") as stream:
            labels = {
                int(row["index"]): row["label_name"]
                for row in csv.DictReader(stream)
            }
        if not labels:
            raise ValueError(f"label index is empty: {path}")
        return labels

    def _input_tensor(self, image, centers):
        arrays = []
        for center in centers:
            crop = centered_patch(
                image,
                center=center,
                spacing=self.spacing,
                size=self.crop_size,
                default_value=float(self.normalizer.get("mean", -600.0)),
            )
            array = normalized_array_from_config(crop, self.normalizer)
            arrays.append(np.transpose(array, (2, 1, 0)))
        return torch.from_numpy(
            np.ascontiguousarray(np.stack(arrays, axis=0)[:, np.newaxis])
        )

    def _probabilities(self, image, centers):
        inputs = self._input_tensor(image, centers)
        probabilities = []
        with torch.inference_mode():
            for start in range(0, len(inputs), self.batch_size):
                batch = inputs[start : start + self.batch_size].to(
                    self.device,
                    dtype=torch.float32,
                )
                probabilities.append(self.network(batch).detach().cpu().numpy())
        return np.concatenate(probabilities, axis=0)

    def _scores(self, image, centers):
        outputs = self._probabilities(image, centers)[:, self.metastatic_index]
        return [
            {
                "metastatic_score": float(score),
                "predicted_class": score_class(score),
            }
            for score in metastatic_scores(outputs)
        ]

    def predict_centers(self, image, centers):
        """Score the nodules of one case given by physical (x, y, z) centers.

        All centers passed in one call form one case for the score adjustment.
        """
        image = read_image(image)
        centers = np.asarray(centers, dtype=np.float64)
        if centers.size == 0:
            return []
        if centers.ndim != 2 or centers.shape[1] != 3:
            raise ValueError("centers must have shape (N, 3)")

        return self._scores(image, centers)

    def predict(self, image, boxes):
        """Return one result dictionary for each detector box."""
        image = read_image(image)
        boxes = np.asarray(boxes, dtype=np.float64)
        if boxes.size == 0:
            return []
        if boxes.ndim != 2 or boxes.shape[1] != 7:
            raise ValueError("boxes must have shape (N, 7)")

        results = []
        for box, score in zip(boxes, self._scores(image, boxes[:, 1:4])):
            results.append(
                {
                    "detection_probability": float(box[0]),
                    **score,
                    "x": float(box[1]),
                    "y": float(box[2]),
                    "z": float(box[3]),
                    "width": float(box[4]),
                    "height": float(box[5]),
                    "depth": float(box[6]),
                }
            )
        return results
