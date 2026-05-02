# MAE Pretraining (Hydra)

## Entry point

- `scripts/mae/run_mae.py`
- Config root: `scripts/mae/conf`

## Objective

- Randomly mask a fraction of patch tokens (`model.mask_ratio`, default `0.75`).
- Reconstruct only masked patches in voxel space with:
  - `model.loss_type=l2` (MSE) or
  - `model.loss_type=l1` (MAE)
- Keep augmentation and dataset setup aligned with JEPA config for ablation.

## Single GPU example

```bash
python3 scripts/mae/run_mae.py \
  folder=outputs/stage1_mae_dsec \
  data.datasets=[/data/DSEC_voxels/train] \
  data.dataset_fpcs=[16] \
  data.batch_size=8
```

## Multi GPU example (`torchrun`)

```bash
torchrun --nproc_per_node=4 scripts/mae/run_mae.py \
  folder=outputs/stage1_mae_4gpu \
  data.datasets=[/data/DSEC_voxels/train,/data/1mpx_voxels/train] \
  data.dataset_fpcs=[16,16] \
  data.datasets_weights=[0.7,0.3] \
  data.batch_size=8 \
  optimization.ipe=300
```

## Notes

- Checkpoint:
  - latest: `<folder>/latest_mae.pth.tar`
  - periodic: `<folder>/e{epoch}_mae.pth.tar`
- For downstream compatibility, MAE checkpoints include an `encoder` key.
- Resume options are the same style as JEPA:
  - `meta.load_checkpoint=true`
  - `meta.read_checkpoint=/path/to/latest_mae.pth.tar`
  - or `meta.auto_resume_latest=true`
- Current implementation is video-branch only (`model.img_temporal_dim_size` must be `null`).

## Downstream with MAE encoder

Use the existing downstream trainer and point `model.checkpoint_path` to MAE checkpoint:

```bash
python3 scripts/downstream/run_downstream.py \
  task=dsec_semantic \
  task.train_roots=[/data/DSEC_voxels_seg/train] \
  task.val_roots=[/data/DSEC_voxels_seg/val] \
  model.checkpoint_path=/path/to/stage1_mae/latest_mae.pth.tar \
  model.checkpoint_key=encoder \
  model.freeze_encoder=true
```
