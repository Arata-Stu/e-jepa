# Downstream Training (Semantic / Depth)

## Entry point

- `scripts/downstream/run_downstream.py`
- Config root: `scripts/downstream/conf`

## Supported tasks

- `task=dsec_semantic`
- `task=m3ed_semantic`
- `task=m3ed_raw_semantic`
- `task=m3ed_depth`

## Modes

- Linear probe (default): `model.freeze_encoder=true`
- Fine-tune encoder: `model.freeze_encoder=false`

## Example commands

### DSEC semantic (linear probe)

```bash
python3 scripts/downstream/run_downstream.py \
  task=dsec_semantic \
  folder=outputs/downstream_dsec_sem \
  task.train_roots=[/data/DSEC_voxels_seg_train] \
  task.val_roots=[/data/DSEC_voxels_seg_val] \
  model.checkpoint_path=/path/to/stage1/latest.pth.tar \
  model.freeze_encoder=true
```

### M3ED semantic (fine-tune)

```bash
python3 scripts/downstream/run_downstream.py \
  task=m3ed_semantic \
  folder=outputs/downstream_m3ed_sem \
  task.train_roots=[/data/m3ed_voxels_semantic/train] \
  task.val_roots=[/data/m3ed_voxels_semantic/val] \
  model.checkpoint_path=/path/to/stage1/latest.pth.tar \
  model.freeze_encoder=false
```

### M3ED raw semantic (linear probe)

Downloaded M3ED can be used without voxel preprocessing. The loader pairs each
`<sequence>_data.h5` with the sibling `<sequence>_semantics.h5`, generates the
event voxel representation on demand, and reads semantic labels from
`_semantics.h5:/predictions`.

The `m3ed_raw_semantic` preset uses the F3 standard semantic split by default:

- train: `confs/segmentation/m3ed_train.yml`
- val/eval: `confs/segmentation/m3ed_test.yml`

Its `task.train_sequence_ranges` / `task.val_sequence_ranges` entries follow
F3's range convention: `[start_fraction, step, stop_fraction]`.

```bash
python3 scripts/downstream/run_downstream.py \
  task=m3ed_raw_semantic \
  model=vit_tiny_m3ed_raw_linear_probe \
  folder=outputs/downstream_m3ed_raw_sem_stopgrad \
  task.train_roots=[/data/M3ED] \
  task.val_roots=[/data/M3ED] \
  model.checkpoint_path=/path/to/stage1/latest.pth.tar \
  model.freeze_encoder=true
```

With the default preset, `train_roots` and `val_roots` may both point to the
same downloaded M3ED root because the F3 split is applied inside the dataset.
If you override `task.*_sequence_ranges` or disable the split, keep train and
validation samples disjoint; otherwise the same semantic frames can appear in
both sets.

For the stopgrad/EMA vs SIGReg JEPA checkpoints trained on raw M3ED, there is a
comparison launcher:

```bash
M3ED_ROOT=/data/M3ED \
  ./scripts/experiments/downstream_m3ed_raw_semantic_sigreg_comparison.sh
```

You can pass extra Hydra overrides to the launcher, for example:

```bash
M3ED_ROOT=/data/M3ED \
  ./scripts/experiments/downstream_m3ed_raw_semantic_sigreg_comparison.sh \
  task.train_sequence_include=[car_urban_day_penno_big_loop,car_urban_day_city_hall] \
  task.val_sequence_include=[car_urban_day_ucity_small_loop]
```

### M3ED depth (linear probe)

```bash
python3 scripts/downstream/run_downstream.py \
  task=m3ed_depth \
  folder=outputs/downstream_m3ed_depth \
  task.train_roots=[/data/m3ed_voxels_depth/train] \
  task.val_roots=[/data/m3ed_voxels_depth/val] \
  model.checkpoint_path=/path/to/stage1/latest.pth.tar \
  model.freeze_encoder=true
```

## Validation visualization

After training, you can load the saved downstream checkpoint and render validation examples as PNGs:

```bash
python3 scripts/downstream/visualize_val.py \
  --run-dir outputs/downstream_dsec_sem/2026-05-21/12-34-56 \
  --checkpoint-tag best \
  --num-samples 12
```

Useful options:

- `--checkpoint /path/to/custom_downstream.pth.tar`
- `--sample-indices 0 10 42`
- `--split train` (default is `val`)
- `--output-dir /path/to/output_dir`

Outputs:

- `sample_*.png`: per-sample activity / target / prediction visualization
- `contact_sheet.png`: grid view of all rendered samples
- `manifest.csv`: sample index, file path, window index, and per-sample metrics

The script reads `<run-dir>/params-downstream-resolved.yaml`, so you do not need to repeat the original Hydra overrides.

To make an MP4 from semantic segmentation results, use contiguous samples:

```bash
python3 scripts/downstream/visualize_val.py \
  --run-dir outputs/downstream_dsec_sem/2026-05-21/12-34-56 \
  --checkpoint-tag best \
  --split val \
  --sample-mode contiguous \
  --start-index 0 \
  --num-samples 240 \
  --write-video \
  --video-fps 12 \
  --video-width 1280
```

Video-related options:

- `--write-video`: write `<output-dir>/val_visualizations_best.mp4`
- `--video-path /path/to/result.mp4`: custom MP4 path; this also enables video export
- `--video-fps 12`: MP4 frame rate
- `--video-width 1280`: resize rendered visualization frames before encoding; `0` keeps original size
- `--sample-mode contiguous --start-index N --sample-stride K`: useful for smooth sequence videos

## Notes

- Input is voxel clips sampled around each anchor window (`task.clip_num_frames`, `task.clip_frame_stride`).
- Preprocessed dense tasks read labels from embedded datasets stored in each preprocessed H5.
- `task=m3ed_raw_semantic` is the exception: it reads downloaded M3ED `*_data.h5` events and sibling `*_semantics.h5` labels directly.
- DSEC semantic requires preprocessing with `dataset.sync_segmentation=true`; using `dataset.window_mode=image_middle` is the intended path for label-aligned windows.
- DSEC official `test` split does not ship semantic labels, so downstream evaluation needs a user-created validation split carved out from labeled `train`.
- Splitting after preprocessing is fine if you move/copy whole H5 files into separate `train` / `val` roots. If you rewrite H5 contents, preserve embedded label datasets and alignment metadata such as `embedded_segmentation`, `embedded_semantics`, `embedded_depth`, `segmentation_available`, and `window_index`.
- Preprocessed M3ED semantic/depth labels are read only from embedded labels stored in each preprocessed H5.
- If an old H5 depended on `source_file`, `segmentation_dir`, or other external label metadata, re-preprocess it with the current pipeline before downstream training.
- If semantic class count is unknown, set `task.num_classes=0` and it will be inferred from sampled labels.
- Logs:
  - CSV: `<folder>/downstream_log.csv`
  - TensorBoard: `<folder>/tensorboard`
- Checkpoints:
  - latest: `<folder>/latest_downstream.pth.tar`
  - best: `<folder>/best_downstream.pth.tar`
