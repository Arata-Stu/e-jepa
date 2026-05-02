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
  task.train_roots=[/data/DSEC_voxels_seg/train] \
  task.val_roots=[/data/DSEC_voxels_seg/val] \
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
- M3ED semantic/depth labels are read from the raw `source_file` referenced by each preprocessed H5.
- If semantic class count is unknown, set `task.num_classes=0` and it will be inferred from sampled labels.
- Logs:
  - CSV: `<folder>/downstream_log.csv`
  - TensorBoard: `<folder>/tensorboard`
- Checkpoints:
  - latest: `<folder>/latest_downstream.pth.tar`
  - best: `<folder>/best_downstream.pth.tar`

