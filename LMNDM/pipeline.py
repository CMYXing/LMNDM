"""Unified detection and lung-metastasis classification pipeline."""

from __future__ import annotations

import csv
from pathlib import Path

from LMNDM.classification import MetastasisClassifier
from LMNDM.detection import CascadeDetector
from LMNDM.paths import default_model_root
from LMNDM.runtime import configure_threads, resolve_device


RESULT_FIELDS = [
    "detection_probability",
    "metastatic_score",
    "predicted_class",
    "x",
    "y",
    "z",
    "width",
    "height",
    "depth",
]


class LMNDMPipeline:
    def __init__(
        self,
        model_root=None,
        *,
        device="auto",
        patch_size=None,
        num_threads=None,
        classifier_batch_size: int = 8,
    ) -> None:
        self.model_root = Path(model_root or default_model_root())
        self.device = resolve_device(device)
        self.num_threads = configure_threads(num_threads)
        self.detector = CascadeDetector(
            self.model_root / "detection",
            device=self.device,
            fp_model_dir=self.model_root / "false_positive_reduction",
            patch_size=patch_size,
        )
        self.classifier = MetastasisClassifier(
            self.model_root / "metastasis_classification" / "model.pth",
            self.model_root / "metastasis_classification" / "label_index.csv",
            device=self.device,
            batch_size=classifier_batch_size,
        )

    def predict(
        self,
        image,
        *,
        lung_mask=None,
        mask_mode: str = "auto",
        verbose: bool = True,
    ):
        boxes = self.detector.detect(
            image,
            lung_mask,
            mask_mode=mask_mode,
            apply_fp_reduction=True,
            verbose=verbose,
        )
        return self.classifier.predict(image, boxes)


def save_results(results, output_path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=RESULT_FIELDS,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(results)
    return output_path
