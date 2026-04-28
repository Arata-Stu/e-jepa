# Preprocess Commands

基本は `dataset_root + output_root` で一括処理します。  
Hydra ランナー: `scripts/preprocess/run_preprocess.py`

デフォルト:

- `t_bins=10`
- `downsample_factor`: `dsec=1`, `eventscape=1`, `1mpx=2`, `m3ed=2`

## DSEC

想定例: `dsec/train/<sequence>/events/left/events.h5`

```bash
python3 scripts/preprocess/run_preprocess.py \
  dataset=dsec \
  dataset.dataset_root=/data/DSEC \
  dataset.output_root=/data/DSEC_voxels \
  dataset.splits=[train,test]
```

image middle:

```bash
python3 scripts/preprocess/run_preprocess.py \
  dataset=dsec \
  dataset.dataset_root=/data/DSEC \
  dataset.output_root=/data/DSEC_voxels \
  dataset.window_mode=image_middle
```

image middle + segmentation sync:

```bash
python3 scripts/preprocess/run_preprocess.py \
  dataset=dsec \
  dataset.dataset_root=/data/DSEC \
  dataset.output_root=/data/DSEC_voxels_seg \
  dataset.window_mode=image_middle \
  dataset.sync_segmentation=true \
  dataset.segmentation_root=/data/DSEC \
  dataset.segmentation_subdir=11classes \
  dataset.segmentation_tolerance_us=0
```

`image_root` は通常不要です（`<sequence>/images/...` を自動参照）。  
画像ディレクトリが別ルートに分離されている場合のみ `dataset.image_root=...` を指定します。

## 1MPX

```bash
python3 scripts/preprocess/run_preprocess.py \
  dataset=1mpx \
  dataset.dataset_root=/data/1mpx \
  dataset.output_root=/data/1mpx_voxels
```

split 指定:

```bash
python3 scripts/preprocess/run_preprocess.py \
  dataset=1mpx \
  dataset.dataset_root=/data/1mpx \
  dataset.output_root=/data/1mpx_voxels \
  dataset.splits=[train,test,val]
```

## M3ED

event + semantic:

```bash
python3 scripts/preprocess/run_preprocess.py \
  dataset=m3ed \
  dataset.dataset_root=/data/m3ed_left_event \
  dataset.output_root=/data/m3ed_voxels_semantic \
  dataset.window_mode=semantics_middle \
  dataset.output_suffix=_voxels_semantic.h5
```

event + depth:

```bash
python3 scripts/preprocess/run_preprocess.py \
  dataset=m3ed \
  dataset.dataset_root=/data/m3ed_left_event \
  dataset.output_root=/data/m3ed_voxels_depth \
  dataset.window_mode=depth_middle \
  dataset.output_suffix=_voxels_depth.h5
```

## EventScape

想定例: `event_scape/train/Town01/<sequence>/events/data/*.npz`

全split一括:

```bash
python3 scripts/preprocess/run_preprocess.py \
  dataset=eventscape \
  dataset.dataset_root=/data/EventScape \
  dataset.output_root=/data/EventScape_voxels
```

train のみ:

```bash
python3 scripts/preprocess/run_preprocess.py \
  dataset=eventscape \
  dataset.dataset_root=/data/EventScape \
  dataset.output_root=/data/EventScape_voxels_train \
  dataset.splits=[train]
```

## 共通上書き例

```bash
python3 scripts/preprocess/run_preprocess.py \
  dataset=m3ed \
  dataset.dataset_root=/data/m3ed_left_event \
  dataset.output_root=/data/m3ed_voxels_semantic \
  dataset.num_processes=8 \
  dataset.downsample_factor=2 \
  dataset.t_bins=10 \
  dataset.accum_time=50000 \
  dataset.output_dtype=float16
```

## 解析コマンド（voxel H5）

dataset_root 一括解析 + 可視化:

```bash
python3 scripts/preprocess/analyze_voxel_h5.py \
  --dataset_root /data/preprocessed_voxels \
  --output_dir /data/preprocessed_voxels_analysis
```

単一ファイル:

```bash
python3 scripts/preprocess/analyze_voxel_h5.py \
  --input_path /data/preprocessed_voxels/sample_voxels_2x.h5 \
  --output_dir /data/preprocessed_voxels_analysis_single
```
