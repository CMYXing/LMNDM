# LMNDM

LMNDM is the inference code accompanying our study of lung-metastasis
classification of pulmonary nodules. For each nodule listed in a CSV file, it
computes a case-level metastatic score and grades the nodule as Benign,
Indeterminate, or Metastatic.

This repository includes the inference source code, the pretrained metastasis
classification model, and four de-identified example chest CT volumes with a
nodule table. The CT volumes and the checkpoint are stored with Git LFS.

## Installation

```bash
git clone https://github.com/CMYXing/LMNDM.git
cd LMNDM
git lfs pull
conda create -n lmndm python=3.11 pip -y
conda activate lmndm
python -m pip install -e ".[notebook]"
```

Python 3.10–3.12 is supported on Linux and macOS.

## Metastatic score

The classification network assigns each nodule an output `p_i`. Nodules that
share a CT image form one case. With `m` the highest output in the case and
`λ = 0.7 m`, the metastatic score of each nodule is

```text
s_i = (1 - λ) p_i + λ m
```

The highest-scoring nodule and a solitary nodule keep their original value.
Each nodule is then graded by its score:

| Class         | Metastatic score     |
|---------------|----------------------|
| Benign        | s < 0.5893           |
| Indeterminate | 0.5893 ≤ s < 0.7937  |
| Metastatic    | s ≥ 0.7937           |

## Model assets

```text
models/
  manifest.json
  metastasis_classification/
    label_index.csv
    model.pth
```

The command-line interface and notebook use this `models/` directory by
default. You can select another bundle with `--models` or `LMNDM_MODEL_DIR`.
The checkpoint is compatible with restricted PyTorch loading
(`weights_only=True`).

## Scoring nodules

```bash
lmndm \
  --nodules data/example/nodules.csv \
  --device auto \
  --threads 4 \
  --output output/metastatic_scores.csv
```

The nodule CSV needs `image_path`, `x`, `y`, and `z` columns, with nodule
centers in physical millimetres. Include every nodule of a case, because the
score of each nodule depends on the other nodules sharing its CT image. An
`image_path` that does not exist as written is looked up as
`<image root>/<case folder>/<file name>`; the image root is the CSV directory
unless `--image-root` is given.

The output CSV repeats the input columns and adds `metastatic_score` and
`predicted_class`.

The same workflow is available from Python:

```python
from LMNDM import (
    MetastasisClassifier,
    group_cases,
    read_nodules,
    save_nodules,
    score_case,
)

classifier = MetastasisClassifier(
    "models/metastasis_classification/model.pth",
    "models/metastasis_classification/label_index.csv",
    device="auto",
)
nodules = read_nodules("data/example/nodules.csv")
for image_path, case_nodules in group_cases(nodules, "data/example").items():
    score_case(classifier, image_path, case_nodules)
save_nodules(nodules, "output/metastatic_scores.csv")
```

## Visualization

The notebook at `examples/metastatic_scoring.ipynb` scores the nodules listed
in a CSV file. It shows each nodule on a cropped axial slice through its center,
labelled with the primary tumor, density type, reference label, metastatic
score, and predicted class.

The CSV needs these columns: `image_path`, `x`, `y`, `z`, `width`, `height`,
`Primary tumor`, `density_type`, and `Label`. Coordinates and sizes are in
physical millimetres.

Set `LMNDM_MODEL_DIR`, `LMNDM_NODULE_CSV`, and `LMNDM_IMAGE_ROOT` to choose the
model directory, nodule CSV, and CT directory.

By default the notebook uses the four example cases and
`data/example/nodules.csv`. Run `git lfs pull` after cloning if the CT volumes
or the checkpoint appear as small pointer files.

## Repository contents

```text
LMNDM/          inference package
data/example/   four de-identified example cases and nodule CSV
examples/       metastatic scoring notebook
models/         pretrained metastasis classification model
```

Source code is licensed under Apache-2.0. The included model weights and
example data use CC BY-NC 4.0; see `LICENSE-MODELS-DATA`.

LMNDM is research software, not a medical device, and must not be used as the
sole basis for diagnosis or treatment.
