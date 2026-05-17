# Downstream Training (Semantic / Depth)

## Entry point

- `scripts/downstream/run_downstream.py`
- Config root: `scripts/downstream/conf`

## Supported tasks

- `task=dsec_semantic`
- `task=m3ed_semantic`
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

## Notes

- Input is voxel clips sampled around each anchor window (`task.clip_num_frames`, `task.clip_frame_stride`).
- DSEC semantic labels are read from `segmentation_relpath` / `segmentation_dir` metadata in each preprocessed H5.
- DSEC semantic requires preprocessing with `dataset.sync_segmentation=true`; using `dataset.window_mode=image_middle` is the intended path for label-aligned windows.
- DSEC official `test` split does not ship semantic labels, so downstream evaluation needs a user-created validation split carved out from labeled `train`.
- Splitting after preprocessing is fine if you move/copy whole H5 files into separate `train` / `val` roots. If you rewrite H5 contents, preserve segmentation datasets/attrs such as `embedded_segmentation`, `segmentation_available`, `segmentation_relpath`, and `segmentation_dir`.
- M3ED semantic/depth labels are read from the raw `source_file` referenced by each preprocessed H5.
- If semantic class count is unknown, set `task.num_classes=0` and it will be inferred from sampled labels.
- Logs:
  - CSV: `<folder>/downstream_log.csv`
  - TensorBoard: `<folder>/tensorboard`
- Checkpoints:
  - latest: `<folder>/latest_downstream.pth.tar`
  - best: `<folder>/best_downstream.pth.tar`
