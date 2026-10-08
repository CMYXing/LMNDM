"""Case-level metastatic score and three-class grading."""

from __future__ import annotations

import numpy as np


BASE_WEIGHT = 0.7
INDETERMINATE_THRESHOLD = 0.5893
METASTATIC_THRESHOLD = 0.7937


def metastatic_scores(probabilities) -> np.ndarray:
    """Adjust the nodule outputs of one case toward the case maximum.

    With ``m`` the highest output in the case and ``w = BASE_WEIGHT * m``, each
    nodule scores ``(1 - w) * p + w * m``. The highest-scoring nodule and a
    solitary nodule keep their original value.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.size == 0:
        return probabilities
    case_max = float(probabilities.max())
    weight = BASE_WEIGHT * case_max
    return (1.0 - weight) * probabilities + weight * case_max


def score_class(score: float) -> str:
    if score >= METASTATIC_THRESHOLD:
        return "Metastatic"
    if score >= INDETERMINATE_THRESHOLD:
        return "Indeterminate"
    return "Benign"
