# LMNDM model bundle

This directory contains the PyTorch checkpoint required for LMNDM metastatic
scoring, together with its class mapping and `manifest.json`.

- `metastasis_classification/`: BResidualNet checkpoint and class mapping.

The `.pth` file is a CPU-portable PyTorch checkpoint and can be read with
restricted loading (`torch.load(..., weights_only=True)`). Keep this directory
intact and pass its path to `lmndm --models`.

This bundle contains no CT volumes, annotations, patient mappings, or
inference outputs. The weights are licensed under CC BY-NC 4.0; see the
included `LICENSE-MODELS-DATA` file.
