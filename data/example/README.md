# De-identified example data

This directory contains four de-identified chest CT volumes and a nodule table
for the scoring notebook in `examples/metastatic_scoring.ipynb`:

```text
case_0001/std_1.0_11.nii.gz
case_0002/std_1.0_9.nii.gz
case_0003/std_1.0_10.nii.gz
case_0004/std_1.0_402.nii.gz
nodules.csv
```

`nodules.csv` lists one annotated nodule per row. `image_path` is relative to
this directory, and `x`, `y`, `z`, `width`, `height`, and `depth` are physical
coordinates and sizes in millimetres. `Primary tumor`, `density_type`, and
`Label` (reference class, Benign or Metastatic) are shown next to the
metastatic score and predicted class.

Case folders use non-identifying names. The NIfTI headers carry no free-text
fields or extensions; only the geometry needed for physical-coordinate
processing is retained. The volumes are stored with Git LFS.

These data are intended for software demonstration and research
reproducibility, not clinical evaluation.

License: CC BY-NC 4.0. See `../../LICENSE-MODELS-DATA`.
