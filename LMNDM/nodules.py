"""Read nodule CSV files, score their nodules case by case, and save results."""

from __future__ import annotations

import csv
from pathlib import Path, PureWindowsPath


def read_nodules(csv_path):
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as stream:
        return [
            {key.strip(): value.strip() for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]


def resolve_image(path_text, image_root):
    path = Path(path_text)
    if path.is_file():
        return path
    # Accept Windows or POSIX paths recorded on another machine.
    parts = PureWindowsPath(path_text).parts
    candidate = Path(image_root).joinpath(*parts[-2:])
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"cannot find {path_text} under {image_root}")


def group_cases(nodules, image_root):
    """Group nodules by CT image; nodules sharing an image form one case."""
    cases = {}
    for nodule in nodules:
        image_path = resolve_image(nodule["image_path"], image_root)
        cases.setdefault(image_path, []).append(nodule)
    return cases


def score_case(classifier, image, nodules):
    """Add ``metastatic_score`` and ``predicted_class`` to one case's nodules."""
    centers = [
        [float(nodule[key]) for key in ("x", "y", "z")]
        for nodule in nodules
    ]
    for nodule, result in zip(nodules, classifier.predict_centers(image, centers)):
        nodule.update(result)
    return nodules


def save_nodules(nodules, output_path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for nodule in nodules for key in nodule))
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(nodules)
    return output_path
