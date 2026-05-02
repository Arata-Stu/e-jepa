# Stage-1 JEPA Training (Hydra)

## Entry point

- `scripts/train/run_train.py`
- Config root: `scripts/train/conf`

## Setup

```bash
python3 -m pip install -r requirements.txt
```

## Single GPU example

```bash
python3 scripts/train/run_train.py \
  folder=outputs/stage1_dsec_single \
  data.datasets=[/data/DSEC_voxels/train] \
  data.dataset_fpcs=[8] \
  data.batch_size=8
```

## Multi GPU example (`torchrun`)

```bash
torchrun --nproc_per_node=4 scripts/train/run_train.py \
  folder=outputs/stage1_dsec_4gpu \
  data.datasets=[/data/DSEC_voxels/train,/data/1mpx_voxels/train] \
  data.dataset_fpcs=[8,8] \
  data.datasets_weights=[0.7,0.3] \
  data.batch_size=8 \
  optimization.ipe=300
```

## Multi GPU with image-branch split (event H5 single-frame)

```bash
torchrun --nproc_per_node=4 scripts/train/run_train.py \
  folder=outputs/stage1_dsec_imgsplit \
  data.datasets=[/data/DSEC_voxels/train] \
  data.dataset_fpcs=[8] \
  data.batch_size=8 \
  img_data=event_h5_single_frame \
  img_data.datasets=[/data/1mpx_voxels/train] \
  img_data.batch_size=16 \
  img_data.rank_ratio=0.25 \
  img_mask=stage1_image
```

## Notes

- Core logic follows vjepa2.1-style training with:
  - stop-grad target encoder
  - EMA target update
  - predictor/context JEPA loss
  - DDP-aware dataloader and checkpointing
- TensorBoard logs are written to `<folder>/tensorboard` by rank 0.
- A resolved config snapshot is written to `<folder>/params-train-resolved.yaml`.
- Default mask is `mask=stage1_event` (8-block + 2-block).  
  You can switch to image-style mask with `mask=stage1_image`.
- `model.in_chans` must match voxel channels (`C` in `/voxels`).
  - default preprocess (`t_bins=10`, `split_polarity=true`) => `C=20` so use `model.in_chans=20`.
- `data_aug.preserve_input_size=true` なら crop/resize を無効化し、入力の解像度とアスペクト比を維持します。
  - default は `data_aug.pad_to_hw=[480,640]`（HxW）で letterbox padding を適用し、
    `data_aug.allowed_input_hw=[[480,640]]` を検証します。
- `data.crop_size=[480,640]`（HxW）を model/mask 側の基準グリッドとして使います。
  これにより、padding 後の入力サイズと mask/model のトークングリッドが一致します。
- `img_data.enabled=true` のとき、`vjepa2.1` と同様に rank split を有効化:
  - image rank は `img_data` 設定 (`dataset_fpcs=[1]` など) を使用
  - image rank で `img_mask` が指定されていれば `img_mask` を適用
  - video rank は通常の `data` と `mask` を継続使用
