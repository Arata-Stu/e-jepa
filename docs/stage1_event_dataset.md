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
  (`scripts/preprocess/*` の既定では `representation=voxel_grid` かつ `split_polarity=True` のため通常 `C=2*t_bins`。`representation=event_image` では `C=3` の RGB イベント画像。)

All preprocess outputs created by:

- `scripts/preprocess/preprocess_dsec.py`
- `scripts/preprocess/preprocess_1mpx.py`
- `scripts/preprocess/preprocess_eventscape.py`
- `scripts/preprocess/preprocess_m3ed.py`

are supported.

When preprocessing with `representation=event_image`, each script can also emit a companion `.mp4`
beside the HDF5 by enabling `save_mp4=true` / `--save_mp4`. The HDF5 still stores `(N, 3, H, W)`
RGB event images, and the MP4 stores the same windows as standard video frames. If `mp4_fps` is
omitted, the exporter infers FPS from timestamp spacing between generated windows/anchors.
When 20-second H5 chunk split is enabled, companion MP4s are split alongside the H5 files using
the same `_partXXXX.mp4` suffix convention.

Hydra presets for this mode are available as:

- `dataset=eventscape_event_image`
- `dataset=dsec_semantic_event_image`
- `dataset=m3ed_semantic_event_image`
- `dataset=1mpx_event_image`

## V-JEPA2.1 Local Config

A minimal local video-only V-JEPA2.1 scratch config is available at:

- `tmp/vjepa2/configs/train_2_1/local/event_video_vitb16_256px_16f_scratch.yaml`

Typical workflow:

1. Build a `.npy` manifest containing absolute paths to event-video `.mp4` files.
2. Override `data.datasets[0]` and `folder` in the config or on the command line.
3. Launch `tmp/vjepa2/app/main.py` on one or more GPUs.

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

## VJEPA2 Interop

`tmp/vjepa2/src/datasets/video_dataset.py` treats `.jpg/.png/.jpeg` as still images and sends all
other paths such as `.mp4` through `decord.VideoReader`, so exported event-image MP4s can be used
as regular video inputs for `vjepa2_1`.

For label-free retraining / pretraining style runs, the simplest manifest is either:

- a `.npy` list of MP4 paths, because `VideoDataset` assigns dummy label `0` for `.npy` entries
- or a shell-generated `.csv` with lines like `/abs/path/video.mp4 0`

Classification-style fine-tuning later would still need a labeled `.csv`.
