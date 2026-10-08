"""Command-line metastatic scoring of nodules listed in a CSV file."""

from __future__ import annotations

import argparse
from pathlib import Path

from LMNDM.classification import MetastasisClassifier
from LMNDM.nodules import group_cases, read_nodules, save_nodules, score_case
from LMNDM.paths import default_model_root
from LMNDM.runtime import configure_threads, resolve_device


def build_parser():
    parser = argparse.ArgumentParser(
        prog="lmndm",
        description=(
            "Compute the case-level metastatic score and class of each "
            "nodule listed in a CSV file."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--nodules",
        required=True,
        help="nodule CSV with image_path, x, y, and z columns",
    )
    parser.add_argument(
        "--image-root",
        help="directory searched for images not found at image_path; "
        "defaults to the CSV directory",
    )
    parser.add_argument(
        "--models",
        default=str(default_model_root()),
        help="root containing the metastasis classification model",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, mps, cuda, or cuda:<index>",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="maximum PyTorch CPU threads",
    )
    parser.add_argument(
        "--output",
        default="output/metastatic_scores.csv",
        help="output CSV path",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    nodules = read_nodules(args.nodules)
    cases = group_cases(nodules, args.image_root or Path(args.nodules).parent)

    device = resolve_device(args.device)
    num_threads = configure_threads(args.threads)
    model_dir = Path(args.models) / "metastasis_classification"
    classifier = MetastasisClassifier(
        model_dir / "model.pth",
        model_dir / "label_index.csv",
        device=device,
    )
    print(f"Device: {device}; CPU threads: {num_threads}")
    for image_path, case_nodules in cases.items():
        score_case(classifier, image_path, case_nodules)
    output = save_nodules(nodules, args.output)
    print(f"Saved {len(nodules)} nodules from {len(cases)} cases to {output}")


if __name__ == "__main__":
    main()
