# Stage-1 JEPA Training (Hydra)

## Entry point

- `scripts/train/run_train.py`
- Config root: `scripts/train/conf`

## Setup

```bash
python3 -m pip install -r requirements.txt
```

## Unified launcher (1GPU / multi-GPU 共通)

`scripts/launch_pretrain.py` を使うと、同じ起動形式で GPU 数を自動検出して `torchrun` します。

```bash
python3 scripts/launch_pretrain.py jepa \
  folder=outputs/stage1_dsec_auto \
  data.datasets=[/data/DSEC_voxels/train] \
  data.dataset_fpcs=[8] \
  data.batch_size=8
```

- 1GPU 環境: `nproc_per_node=1` で実行
- 複数GPU環境: `nproc_per_node=<GPU数>` で実行
- 明示指定したい場合: `--nproc-per-node 4` など

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
- `meta.use_tqdm=true` で、stepごとの logger 出力を抑えて tqdm 進捗表示に切り替えられます。

## ViT Tiny preset

`model=vit_tiny_2_1` を指定すると、ViT-Tiny + 軽量predictor設定に切り替えられます。

```bash
python3 scripts/train/run_train.py \
  model=vit_tiny_2_1 \
  folder=outputs/stage1_tiny
```

## Mask visualization

学習時と同じ `data` / `data_aug` / `mask` 設定から、実際に読み出された clip と
その場で生成された `encoder context mask / predictor mask` を PNG に保存できます。

```bash
python3 scripts/train/visualize_masks.py \
  --output-dir outputs/mask_debug \
  --num-samples 4 \
  data.datasets=[/data/DSEC_voxels/train] \
  data.dataset_fpcs=[16]
```

- `summary.txt`: clip index, keep率, mask設定, 時間方向の keep 率
- `sample_*.png`: `Activity / Context overlay / Predictor overlay`
- `contact_sheet.png`: 複数サンプルを書いたときの一覧
- `--sample-indices 0 10 42` で特定サンプルを固定できます。
- `--num-draws 3` にすると、同じ dataset index を複数回引いて mask のばらつきも見られます。
- `--branch image` を使うと、`img_data.enabled=true` の image branch 設定でも同じ確認ができます。
