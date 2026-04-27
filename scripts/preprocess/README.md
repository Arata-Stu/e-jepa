# Preprocess README

This directory contains dataset-specific preprocessing scripts:

- `scripts/preprocess/preprocess_dsec.py`
- `scripts/preprocess/preprocess_1mpx.py`
- `scripts/preprocess/preprocess_m3ed.py`
- `scripts/preprocess/preprocess_eventscape.py`

## Common Rules

- Single-file mode uses `--input_path` and `--output_path`.
- Root mode uses `--dataset_root` (DSEC also supports `--dsec_root`).
- `--downsample_factor` supports `1` or `2` (nearest-neighbor spatial mapping).
- In root mode, generated filenames include a scale tag (`_1x` or `_2x`) when filename is generated from suffix/default naming.
- Temporary files are written with `--tmp_suffix` and atomically renamed on success.

Default resolution presets:

- DSEC: `input_width=640`, `input_height=480` (VGA)
- 1MPX: `input_width=1280`, `input_height=720` (HD)
- M3ED: `input_width=1280`, `input_height=720` (HD)
- EventScape: `input_width=512`, `input_height=256`

## DSEC (`preprocess_dsec.py`)

### Input File Tree

Single-file mode (example):

```text
<workspace>/
  input/
    events.h5                 # --input_path
    timestamps.txt            # --image_timestamps_path (required for --window_mode image_middle)
    labels/                   # --segmentation_dir (optional, for --sync_segmentation)
      <timestamp>.png
  output/
    voxels.h5                 # --output_path
```

Root mode scans both layouts:

```text
<dataset_root>/
  train/
    <sequence>/
      events/left/events.h5
  test/
    <sequence>/
      events/left/events.h5
```

```text
<dataset_root>/
  train_events/
    <sequence>/
      events/left/events.h5
  test_events/
    <sequence>/
      events/left/events.h5
```

Auxiliary files for DSEC root mode:

```text
# image timestamps (used in --window_mode image_middle)
<sequence_dir>/images/timestamps.txt
<sequence_dir>/images/left/timestamps.txt
<image_root_or_dataset_root>/<split>/<sequence>/images/timestamps.txt
<image_root_or_dataset_root>/<split>_images/<sequence>/images/timestamps.txt

# segmentation labels (used with --sync_segmentation)
<sequence_dir>/semantic_segmentation/<segmentation_subdir>/<timestamp>.png
<segmentation_root_or_dataset_root>/semantic_segmentation/<split>/<sequence>/<segmentation_subdir>/<timestamp>.png
<segmentation_root_or_dataset_root>/<split>_semantic_segmentation/<sequence>/<segmentation_subdir>/<timestamp>.png
```

### Input H5 Structure

```text
/
  events/
    x                  (N,)
    y                  (N,)
    p                  (N,)
    t                  (N,)
  t_offset             ()     # optional
  ms_to_idx            (M,)   # optional
```

### Output File Tree

Without `--output_root`:

```text
<sequence_dir>/
  events/
    left/
      events.h5
      voxels_1x.h5            # default when downsample_factor=1
      voxels_2x.h5            # default when downsample_factor=2
      events_voxels_2x.h5     # example: --output_suffix _voxels.h5 + factor=2
```

With `--output_root`:

```text
<output_root>/
  <split or split_events>/
    <sequence>/
      events/
        left/
          voxels_1x.h5
```

Naming notes:

- Preferred: `--output_suffix` and optional `--output_subdir`.
- Backward-compatible: `--output_name` (exact filename, deprecated).
- `_1x`/`_2x` is auto-appended or updated when filename is generated from suffix/default naming.

### Output H5 Structure

Base datasets:

```text
/
  voxels                    float16|float32  shape=(N, t_bins, H, W)
  window_t_start_us         uint64           shape=(N,)
  window_t_end_us           uint64           shape=(N,)
  window_event_count        uint64           shape=(N,)
  anchor_timestamp_us       int64            shape=(N,)
```

Additional datasets when `--sync_segmentation`:

```text
/
  segmentation_available      uint8          shape=(N,)
  segmentation_timestamp_us   int64          shape=(N,)
  segmentation_time_delta_us  int64          shape=(N,)
  segmentation_relpath        string         shape=(N,)
```

Important attrs:

```text
/
  @representation
  @source_file
  @input_height
  @input_width
  @height
  @width
  @downsample_factor
  @spatial_resize_mode
  @t_bins
  @window_mode
  @accum_time_us
  @stride_time_us
  @time_origin_us
  @normalize
  @sync_segmentation
  @segmentation_tolerance_us
  @image_timestamps_path
  @segmentation_dir
  @num_image_timestamps      # image_middle mode only
```

### Commands

Single-file, fixed windows:

```bash
python3 scripts/preprocess/preprocess_dsec.py \
  --input_path /path/to/events.h5 \
  --output_path /path/to/voxels.h5 \
  --input_height 480 \
  --input_width 640 \
  --window_mode fixed \
  --downsample_factor 1 \
  --t_bins 5 \
  --accum_time 50000
```

Single-file, midpoint windows from image timestamps:

```bash
python3 scripts/preprocess/preprocess_dsec.py \
  --input_path /path/to/events.h5 \
  --output_path /path/to/voxels.h5 \
  --window_mode image_middle \
  --image_timestamps_path /path/to/timestamps.txt \
  --downsample_factor 2 \
  --t_bins 5
```

Root mode with segmentation sync metadata:

```bash
python3 scripts/preprocess/preprocess_dsec.py \
  --dataset_root /path/to/DSEC \
  --splits train test \
  --window_mode image_middle \
  --image_root /path/to/DSEC \
  --sync_segmentation \
  --segmentation_root /path/to/DSEC \
  --segmentation_subdir 11classes_renamed \
  --segmentation_tolerance_us 0 \
  --output_root /path/to/preprocessed \
  --output_suffix _voxels.h5 \
  --downsample_factor 2 \
  --num_processes 4
```

## 1MPX (`preprocess_1mpx.py`)

### Input File Tree

Typical root mode:

```text
<dataset_root>/
  train/
    sample_0001.h5
    sample_0002.h5
  test/
    sample_1001.h5
  val/
    sample_2001.h5
```

If `.h5` files are directly under root, use `--splits .`.

### Input H5 Structure

```text
/
  events/
    x                  (N,)
    y                  (N,)
    p                  (N,)   # {0,1} or {-1,+1}
    t                  (N,)
    height             ()     # optional
    width              ()     # optional
  t_offset             ()     # optional
  ms_to_idx            (M,)   # optional
```

### Output File Tree

Without `--output_root`:

```text
<dataset_root>/
  train/
    sample_0001.h5
    sample_0001_voxels_1x.h5
```

With `--output_root`:

```text
<output_root>/
  train/
    sample_0001_voxels_1x.h5
```

`_1x`/`_2x` tag is auto-managed from `downsample_factor` for root-mode generated filenames.

### Output H5 Structure

```text
/
  voxels                    float16|float32  shape=(N, t_bins, H, W)
  window_index              uint64           shape=(N,)
  window_t_start_us         uint64           shape=(N,)
  window_t_end_us           uint64           shape=(N,)
  window_rel_start_us       int64            shape=(N,)
  window_rel_end_us         int64            shape=(N,)
  anchor_timestamp_us       int64            shape=(N,)
  anchor_rel_timestamp_us   int64            shape=(N,)
  window_event_count        uint64           shape=(N,)
```

Important attrs:

```text
/
  @representation            # event_voxel_grid_1mpx
  @source_file
  @input_height
  @input_width
  @requested_output_height
  @requested_output_width
  @height
  @width
  @downsample_factor
  @spatial_resize_mode
  @t_bins
  @accum_time_us
  @stride_time_us
  @time_origin_us
  @normalize
  @num_windows_planned
```

### Commands

Single-file:

```bash
python3 scripts/preprocess/preprocess_1mpx.py \
  --input_path /path/to/sample.h5 \
  --output_path /path/to/sample_voxels.h5 \
  --downsample_factor 1 \
  --t_bins 5 \
  --accum_time 50000
```

Root mode:

```bash
python3 scripts/preprocess/preprocess_1mpx.py \
  --dataset_root /path/to/1mpx \
  --splits train test val \
  --output_suffix _voxels.h5 \
  --downsample_factor 2 \
  --t_bins 5 \
  --accum_time 50000 \
  --num_processes 4
```

Root mode (files directly under root):

```bash
python3 scripts/preprocess/preprocess_1mpx.py \
  --dataset_root /path/to/1mpx \
  --splits . \
  --recursive
```

## M3ED (`preprocess_m3ed.py`)

### Input File Tree

```text
<dataset_root>/
  <sequence_name>/
    <sequence_data>.h5
```

The script scans `.h5` directly under each `<sequence_name>/` and keeps files containing `/prophesee/left/{x,y,t,p}`.

### Input H5 Structure

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

Polarity note:

- `p` supports both `{0,1}` and `{-1,+1}`.
- Internally normalized as `p > 0 -> 1`, else `0`.

### Output File Tree

Without `--output_root`:

```text
<dataset_root>/
  <sequence_name>/
    <sequence_data>.h5
    <sequence_data>_voxels_1x.h5
```

With `--output_root`:

```text
<output_root>/
  <sequence_name>/
    <sequence_data>_voxels_1x.h5
```

`_1x`/`_2x` tag is auto-managed from `downsample_factor` for root-mode generated filenames.

### Output H5 Structure

```text
/
  voxels                    float16|float32  shape=(N, t_bins, H, W)
  window_index              uint64           shape=(N,)
  window_t_start_us         uint64           shape=(N,)
  window_t_end_us           uint64           shape=(N,)
  window_rel_start_us       int64            shape=(N,)
  window_rel_end_us         int64            shape=(N,)
  anchor_timestamp_us       int64            shape=(N,)
  anchor_rel_timestamp_us   int64            shape=(N,)
  window_event_count        uint64           shape=(N,)
```

Important attrs:

```text
/
  @representation            # event_voxel_grid_m3ed
  @source_file
  @source_event_group        # prophesee/left
  @input_height
  @input_width
  @requested_output_height
  @requested_output_width
  @height
  @width
  @downsample_factor
  @spatial_resize_mode
  @t_bins
  @accum_time_us
  @stride_time_us
  @sync_target               # event_only | semantic | depth
  @window_mode               # fixed | semantics_middle | depth_middle
  @semantics_ts_source
  @resolved_semantics_ts_source
  @semantics_ts_divisor
  @num_semantic_timestamps
  @depth_ts_source
  @resolved_depth_ts_source
  @depth_ts_divisor
  @num_depth_timestamps
  @time_origin_us
  @normalize
  @num_windows_planned
```

### Commands

Single-file, fixed windows:

```bash
python3 scripts/preprocess/preprocess_m3ed.py \
  --input_path /path/to/sequence/seq_data.h5 \
  --output_path /path/to/sequence/seq_data_voxels.h5 \
  --downsample_factor 1 \
  --t_bins 5 \
  --accum_time 50000
```

Root mode, fixed windows:

```bash
python3 scripts/preprocess/preprocess_m3ed.py \
  --dataset_root /path/to/m3ed \
  --output_suffix _voxels.h5 \
  --downsample_factor 2 \
  --t_bins 5 \
  --accum_time 50000 \
  --num_processes 4
```

Root mode, semantic midpoint windows:

```bash
python3 scripts/preprocess/preprocess_m3ed.py \
  --dataset_root /path/to/m3ed \
  --output_suffix _voxels_semantic.h5 \
  --window_mode semantics_middle \
  --semantics_ts_source auto \
  --semantics_ts_divisor 1 \
  --downsample_factor 2 \
  --t_bins 5 \
  --num_processes 4
```

Root mode, depth midpoint windows:

```bash
python3 scripts/preprocess/preprocess_m3ed.py \
  --dataset_root /path/to/m3ed \
  --output_suffix _voxels_depth.h5 \
  --window_mode depth_middle \
  --depth_ts_source auto \
  --depth_ts_divisor 1 \
  --downsample_factor 2 \
  --t_bins 5 \
  --num_processes 4
```

Notes:

- `event+semantic` (`semantics_middle`) and `event+depth` (`depth_middle`) are strictly separated modes.
- `semantics_middle` with depth timestamp options is rejected.
- `depth_middle` with semantic timestamp options is rejected.
- `semantics_middle` creates one window per semantic timestamp.
- Window boundaries are midpoints between adjacent semantic timestamps.
- `depth_middle` creates one window per depth timestamp (same midpoint-boundary logic).
- `window_index` corresponds to the selected anchor timestamp index (semantic or depth).
- If semantic timestamps are in ns, use `--semantics_ts_divisor 1000`.
- If depth timestamps are in ns, use `--depth_ts_divisor 1000`.

## EventScape (`preprocess_eventscape.py`)

### Input File Tree

Example root layout (`Town01`, `Town02`, `Town03`, `Town05`, ...):

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
          05_000_0001_gt_labelIds.png
          ...
          timestamps.txt
      depth/
        data/
          05_000_0000_depth.npy
          05_000_0001_depth.npy
          ...
```

The script uses raw event files in `events/data/*.npz` and writes one voxel window per event npz file.

### Input NPZ/H5 Structure

Event npz expected keys:

```text
x, y, t, p
```

Also supports common aliases such as:

```text
xs/ys/ts/polarity, events (structured array), arr_0 (N,4)
```

### Output File Tree

Without `--output_root`:

```text
<dataset_root>/
  Town05/
    sequence_0/
      sequence_0_voxels_1x.h5
```

With `--output_root`:

```text
<output_root>/
  Town05/
    sequence_0/
      sequence_0_voxels_1x.h5
```

`_1x`/`_2x` tag is auto-managed from `downsample_factor` for root-mode generated filenames.

### Output H5 Structure

```text
/
  voxels                    float16|float32  shape=(N, t_bins, H, W)
  window_index              uint64           shape=(N,)
  window_t_start_us         int64            shape=(N,)
  window_t_end_us           int64            shape=(N,)
  window_rel_start_us       int64            shape=(N,)
  window_rel_end_us         int64            shape=(N,)
  anchor_timestamp_us       int64            shape=(N,)
  anchor_rel_timestamp_us   int64            shape=(N,)
  window_event_count        uint64           shape=(N,)
  event_frame_index         int64            shape=(N,)
  event_file_relpath        string           shape=(N,)
  semantic_available        uint8            shape=(N,)
  semantic_frame_index      int64            shape=(N,)
  semantic_timestamp_us     int64            shape=(N,)
  semantic_relpath          string           shape=(N,)
  depth_available           uint8            shape=(N,)
  depth_frame_index         int64            shape=(N,)
  depth_timestamp_us        int64            shape=(N,)
  depth_relpath             string           shape=(N,)
```

Important attrs:

```text
/
  @representation            # event_voxel_grid_eventscape
  @source_sequence_dir
  @source_events_dir
  @source_semantic_dir
  @source_depth_dir
  @input_height
  @input_width
  @requested_output_height
  @requested_output_width
  @height
  @width
  @downsample_factor
  @spatial_resize_mode
  @t_bins
  @window_mode               # event_file
  @normalize
  @num_event_files
  @num_semantic_files
  @num_depth_files
  @has_semantic_timestamps
  @has_depth_timestamps
  @time_origin_us
  @num_windows_planned
```

### Commands

Single-sequence mode:

```bash
python3 scripts/preprocess/preprocess_eventscape.py \
  --input_path /path/to/Town05/sequence_0 \
  --output_path /path/to/Town05/sequence_0/voxels.h5 \
  --input_height 256 \
  --input_width 512 \
  --downsample_factor 1 \
  --t_bins 5
```

Root mode:

```bash
python3 scripts/preprocess/preprocess_eventscape.py \
  --dataset_root /path/to/EventScape \
  --output_suffix _voxels.h5 \
  --downsample_factor 2 \
  --t_bins 5 \
  --num_processes 4
```
