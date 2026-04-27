# Preprocess README

このディレクトリの前処理は、基本的に `dataset_root + output_root` を指定して一括処理する運用を前提にしています。

対象スクリプト:

- `scripts/preprocess/preprocess_dsec.py`
- `scripts/preprocess/preprocess_1mpx.py`
- `scripts/preprocess/preprocess_m3ed.py`
- `scripts/preprocess/preprocess_eventscape.py`

Hydraランナー:

- `scripts/preprocess/run_preprocess.py`
- 設定: `scripts/preprocess/conf/`

## Quick Start (Hydra)

まずは Hydra ランナーを使うのが推奨です。  
デフォルト `t_bins=10` です。

```bash
python3 scripts/preprocess/run_preprocess.py \
  dataset=<dsec|1mpx|m3ed|eventscape> \
  dataset.dataset_root=/path/to/input_root \
  dataset.output_root=/path/to/output_root
```

Hydra設定ファイル:

```text
scripts/preprocess/conf/
  config.yaml
  dataset/
    dsec.yaml
    1mpx.yaml
    m3ed.yaml
    eventscape.yaml
```

よく使う上書き例:

```bash
python3 scripts/preprocess/run_preprocess.py \
  dataset=m3ed \
  dataset.dataset_root=/data/m3ed_left_event \
  dataset.output_root=/data/m3ed_voxels_semantic \
  dataset.window_mode=semantics_middle \
  dataset.output_suffix=_voxels_semantic.h5 \
  dataset.downsample_factor=2 \
  dataset.num_processes=8
```

```bash
python3 scripts/preprocess/run_preprocess.py \
  dataset=m3ed \
  dataset.dataset_root=/data/m3ed_left_event \
  dataset.output_root=/data/m3ed_voxels_depth \
  dataset.window_mode=depth_middle \
  dataset.output_suffix=_voxels_depth.h5 \
  dataset.downsample_factor=2 \
  dataset.num_processes=8
```

## Common Design

- 一括処理の主引数は `dataset_root` と `output_root`。
- 空間 downsample は `downsample_factor` で指定し、`1` または `2`（nearest-neighbor）。
- root mode の自動命名では `*_1x.h5` / `*_2x.h5` が付きます。
- 一時ファイルは `tmp_suffix`（既定 `.tmp`）で書き、成功時に原子的にリネームします。
- `split_polarity=true` のとき、出力チャネル数は `2 * t_bins`。
- 既定解像度:
  - DSEC: `640x480`
  - 1MPX: `1280x720`
  - M3ED: `1280x720`
  - EventScape: `512x256`

## DSEC (`preprocess_dsec.py`)

### Input Tree

```text
<dataset_root>/
  train/
    <sequence>/
      events/left/events.h5
  test/
    <sequence>/
      events/left/events.h5
```

または:

```text
<dataset_root>/
  train_events/
    <sequence>/
      events/left/events.h5
  test_events/
    <sequence>/
      events/left/events.h5
```

`image_middle` で使う時刻ファイル候補:

```text
<sequence>/images/timestamps.txt
<sequence>/images/left/timestamps.txt
<image_root>/<split>/<sequence>/images/timestamps.txt
<image_root>/<split>_images/<sequence>/images/timestamps.txt
```

### Input H5

```text
/
  events/
    x
    y
    p
    t
  t_offset             # optional
  ms_to_idx            # optional
```

### Output Tree

```text
<output_root>/
  <split or split_events>/
    <sequence>/
      events/
        left/
          voxels_1x.h5
```

### Output H5 (主要)

```text
/
  voxels
  window_t_start_us
  window_t_end_us
  window_event_count
  anchor_timestamp_us
  segmentation_*                 # sync_segmentation=true のとき
```

主要 attrs:

```text
@representation
@input_height
@input_width
@height
@width
@downsample_factor
@t_bins
@voxel_channels
@split_polarity
@window_mode                    # fixed | image_middle
@accum_time_us
@stride_time_us
@normalize
```

### Root Mode Command (Hydra)

```bash
python3 scripts/preprocess/run_preprocess.py \
  dataset=dsec \
  dataset.dataset_root=/data/DSEC \
  dataset.output_root=/data/DSEC_voxels \
  dataset.splits=[train,test] \
  dataset.output_suffix=_voxels.h5 \
  dataset.window_mode=image_middle \
  dataset.image_root=/data/DSEC \
  dataset.downsample_factor=2 \
  dataset.t_bins=10 \
  dataset.num_processes=8
```

セグメンテーション同期あり:

```bash
python3 scripts/preprocess/run_preprocess.py \
  dataset=dsec \
  dataset.dataset_root=/data/DSEC \
  dataset.output_root=/data/DSEC_voxels_seg \
  dataset.window_mode=image_middle \
  dataset.image_root=/data/DSEC \
  dataset.sync_segmentation=true \
  dataset.segmentation_root=/data/DSEC \
  dataset.segmentation_subdir=11classes_renamed \
  dataset.segmentation_tolerance_us=0
```

## 1MPX (`preprocess_1mpx.py`)

### Input Tree

```text
<dataset_root>/
  train/
    *.h5
  test/
    *.h5
  val/
    *.h5
```

### Input H5

```text
/
  events/
    x
    y
    p         # {0,1} or {-1,+1}
    t
    height    # optional
    width     # optional
  t_offset    # optional
  ms_to_idx   # optional
```

### Output Tree

```text
<output_root>/
  train/
    sample_0001_voxels_1x.h5
```

### Output H5 (主要)

```text
/
  voxels
  window_index
  window_t_start_us
  window_t_end_us
  window_rel_start_us
  window_rel_end_us
  anchor_timestamp_us
  anchor_rel_timestamp_us
  window_event_count
```

主要 attrs:

```text
@representation              # event_voxel_grid_1mpx
@input_height
@input_width
@height
@width
@downsample_factor
@t_bins
@voxel_channels
@split_polarity
@accum_time_us
@stride_time_us
@normalize
```

### Root Mode Command (Hydra)

```bash
python3 scripts/preprocess/run_preprocess.py \
  dataset=1mpx \
  dataset.dataset_root=/data/1mpx \
  dataset.output_root=/data/1mpx_voxels \
  dataset.splits=[train,test,val] \
  dataset.output_suffix=_voxels.h5 \
  dataset.downsample_factor=2 \
  dataset.t_bins=10 \
  dataset.accum_time=50000 \
  dataset.num_processes=8
```

## M3ED (`preprocess_m3ed.py`)

### Input Tree

```text
<dataset_root>/
  <sequence_name>/
    <sequence>_left_event.h5
```

`/prophesee/left/{x,y,t,p}` を持つ `.h5` を対象に処理します。

### Input H5

```text
/
  prophesee/
    left/
      x
      y
      t
      p
      ms_map_idx                     # optional
      calib/                         # optional
  ovc/
    ts_map_prophesee_left_t          # optional
  semantics/                         # optional
    ts                               # optional
    ts_map_prophesee_left_t          # optional
  depth_gt/                          # optional
    ts                               # optional
    ts_map_prophesee_left            # optional
    ts_map_prophesee_left_t          # optional
```

### Output Tree

```text
<output_root>/
  <sequence_name>/
    <sequence>_voxels_semantic_1x.h5
```

### Output H5 (主要)

```text
/
  voxels
  window_index
  window_t_start_us
  window_t_end_us
  window_rel_start_us
  window_rel_end_us
  anchor_timestamp_us
  anchor_rel_timestamp_us
  window_event_count
```

主要 attrs:

```text
@representation              # event_voxel_grid_m3ed
@source_event_group          # prophesee/left
@height
@width
@downsample_factor
@t_bins
@voxel_channels
@split_polarity
@sync_target                 # event_only | semantic | depth
@window_mode                 # fixed | semantics_middle | depth_middle
@semantics_ts_source
@depth_ts_source
@semantics_ts_divisor
@depth_ts_divisor
@num_semantic_timestamps
@num_depth_timestamps
@accum_time_us
@stride_time_us
```

### Root Mode Command (Hydra)

event + semantic:

```bash
python3 scripts/preprocess/run_preprocess.py \
  dataset=m3ed \
  dataset.dataset_root=/data/m3ed_left_event \
  dataset.output_root=/data/m3ed_voxels_semantic \
  dataset.window_mode=semantics_middle \
  dataset.output_suffix=_voxels_semantic.h5 \
  dataset.downsample_factor=2 \
  dataset.t_bins=10 \
  dataset.num_processes=8
```

event + depth:

```bash
python3 scripts/preprocess/run_preprocess.py \
  dataset=m3ed \
  dataset.dataset_root=/data/m3ed_left_event \
  dataset.output_root=/data/m3ed_voxels_depth \
  dataset.window_mode=depth_middle \
  dataset.output_suffix=_voxels_depth.h5 \
  dataset.downsample_factor=2 \
  dataset.t_bins=10 \
  dataset.num_processes=8
```

注意:

- `semantics_middle` と `depth_middle` は同時利用しません。
- `semantics_middle` では depth 系オプションをカスタム指定するとエラーになります。
- `depth_middle` では semantic 系オプションをカスタム指定するとエラーになります。

## EventScape (`preprocess_eventscape.py`)

### Input Tree

```text
<dataset_root>/
  Town05/
    sequence_0/
      events/
        data/
          05_000_0000_events.npz
          05_000_0001_events.npz
          ...
      semantic/
        data/
          05_000_0000_gt_labelIds.png
          ...
          timestamps.txt
      depth/
        data/
          05_000_0000_depth.npy
          ...
```

`events/data/*.npz` の生イベントを使って voxel 化します。

### Input NPZ

主要キー:

```text
x, y, t, p
```

互換キーも対応:

```text
xs/ys/ts/polarity, events (structured array), arr_0 (N,4)
```

### Output Tree

```text
<output_root>/
  Town05/
    sequence_0/
      sequence_0_voxels_1x.h5
```

### Output H5 (主要)

```text
/
  voxels
  window_index
  window_t_start_us
  window_t_end_us
  anchor_timestamp_us
  event_frame_index
  event_file_relpath
  semantic_available
  semantic_frame_index
  semantic_timestamp_us
  semantic_relpath
  depth_available
  depth_frame_index
  depth_timestamp_us
  depth_relpath
```

主要 attrs:

```text
@representation              # event_voxel_grid_eventscape
@input_height
@input_width
@height
@width
@downsample_factor
@t_bins
@voxel_channels
@split_polarity
@window_mode                 # event_file
@normalize
@num_event_files
@num_semantic_files
@num_depth_files
```

### Root Mode Command (Hydra)

```bash
python3 scripts/preprocess/run_preprocess.py \
  dataset=eventscape \
  dataset.dataset_root=/data/EventScape \
  dataset.output_root=/data/EventScape_voxels \
  dataset.output_suffix=_voxels.h5 \
  dataset.downsample_factor=2 \
  dataset.t_bins=10 \
  dataset.num_processes=8
```

## Legacy CLI (Argparse)

従来の `preprocess_*.py` 直接実行も引き続き利用できます。  
ただし、日常運用は Hydra (`run_preprocess.py`) を推奨します。
