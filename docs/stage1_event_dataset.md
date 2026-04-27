# Stage-1 Event Dataset (JEPA)

This project now includes a stage-1 dataset path compatible with vjepa2-style training loops.

## Added modules

- `src/datasets/event_dataset.py`
- `src/datasets/transforms.py`
- `src/datasets/data_manager.py`
- `src/masks/presets.py`

## Expected input

Preprocessed HDF5 files with at least:

- `/voxels` with shape `(N, C, H, W)` where `N` is window count.
  (`scripts/preprocess/*` の既定では `split_polarity=True` のため通常 `C=2*t_bins`。)

All preprocess outputs created by:

- `scripts/preprocess/preprocess_dsec.py`
- `scripts/preprocess/preprocess_1mpx.py`
- `scripts/preprocess/preprocess_eventscape.py`

are supported.

## Dataset behavior

- Returns `(buffer, label, clip_indices)` like vjepa2 `VideoDataset`.
- `buffer` is a list of clips.
- Each clip is transformed to `[C, T, H, W]`.
- `clip_indices` is a list of sampled window-index arrays (used by `MaskCollator`).

This supports:

- single-window loading (`dataset_fpcs=1`)
- multi-window pseudo-video loading (`dataset_fpcs>1`)

## Transform behavior

`EventVideoTransform` applies:

- random resized crop (same crop across all frames in the clip)
- horizontal flip (50%)

## Suggested mask presets

```python
from src.masks.presets import STAGE1_EVENT_MASKS, STAGE1_IMAGE_MASKS
```

- `STAGE1_EVENT_MASKS`:
  - `num_blocks=8`, `spatial_scale=(0.15, 0.15)`, `aspect_ratio=(0.75, 1.5)`
  - `num_blocks=2`, `spatial_scale=(0.7, 0.7)`, `aspect_ratio=(0.75, 1.5)`
- `STAGE1_IMAGE_MASKS`:
  - `num_blocks=10`, `spatial_scale=(0.15, 0.15)`, `aspect_ratio=(0.75, 1.5)`
