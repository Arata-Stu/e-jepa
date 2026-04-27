from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import weakref
from pathlib import Path

import h5py
import numpy as np
import torch
import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.representations import EventVoxelGrid
from scripts.preprocess.utils import (
    cleanup_tmp_file,
    ensure_scale_tag_in_filename,
    get_h5_compression_flags,
    normalized_output_subdir,
    normalized_output_suffix,
    tmp_output_path,
)
H5_COMPRESSION_FLAGS = get_h5_compression_flags()


def _read_t_offset(filehandle: h5py.File) -> int:
    if "t_offset" not in filehandle:
        return 0
    return int(filehandle["t_offset"][()])


def get_num_events(h5file: Path) -> int:
    with h5py.File(str(h5file), "r") as h5f:
        return len(h5f["events/t"])


def get_resolution(h5file: Path) -> tuple[int | None, int | None]:
    with h5py.File(str(h5file), "r") as h5f:
        events = h5f["events"]
        if "height" in events and "width" in events:
            height = int(events["height"][()])
            width = int(events["width"][()])
            return height, width
    return None, None


def _empty_events() -> dict[str, np.ndarray]:
    return {
        "x": np.empty((0,), dtype=np.uint16),
        "y": np.empty((0,), dtype=np.uint16),
        "p": np.empty((0,), dtype=np.int16),
        "t": np.empty((0,), dtype=np.int64),
    }


def _extract_from_h5_by_index(filehandle: h5py.File, ev_start_idx: int, ev_end_idx: int) -> dict[str, np.ndarray]:
    events = filehandle["events"]
    t_offset = _read_t_offset(filehandle)
    return {
        "x": np.asarray(events["x"][ev_start_idx:ev_end_idx]),
        "y": np.asarray(events["y"][ev_start_idx:ev_end_idx]),
        "p": np.asarray(events["p"][ev_start_idx:ev_end_idx]),
        "t": np.asarray(events["t"][ev_start_idx:ev_end_idx], dtype=np.int64) + t_offset,
    }


def _get_time_bounds_us(filehandle: h5py.File) -> tuple[int | None, int | None]:
    t = filehandle["events/t"]
    num_events = len(t)
    if num_events == 0:
        return None, None

    t_offset = _read_t_offset(filehandle)
    t_first = int(t[0]) + t_offset
    t_last_exclusive = int(t[num_events - 1]) + t_offset + 1
    return t_first, t_last_exclusive


def _load_ms_to_idx(filehandle: h5py.File) -> np.ndarray | None:
    if "ms_to_idx" not in filehandle:
        return None
    return filehandle["ms_to_idx"][()]


def _coarse_bounds_from_ms_to_idx(
    ms_to_idx: np.ndarray | None,
    num_events: int,
    start_us: int,
    end_us: int,
) -> tuple[int, int]:
    if ms_to_idx is None or ms_to_idx.size == 0:
        return 0, num_events

    start_ms = max(int(start_us // 1000), 0)
    end_ms_exclusive = max(int((end_us + 999) // 1000), start_ms + 1)

    start_ms = min(start_ms, ms_to_idx.size - 1)
    start_idx = int(ms_to_idx[start_ms])

    if end_ms_exclusive >= ms_to_idx.size:
        end_idx = num_events
    else:
        end_idx = int(ms_to_idx[end_ms_exclusive])

    start_idx = max(0, min(start_idx, num_events))
    end_idx = max(0, min(end_idx, num_events))
    return start_idx, end_idx


def _extract_events_by_time(
    filehandle: h5py.File,
    start_us: int,
    end_us: int,
    ms_to_idx: np.ndarray | None,
) -> dict[str, np.ndarray]:
    if end_us <= start_us:
        return _empty_events()

    t_ds = filehandle["events/t"]
    num_events = len(t_ds)
    if num_events == 0:
        return _empty_events()

    coarse_start, coarse_end = _coarse_bounds_from_ms_to_idx(
        ms_to_idx=ms_to_idx,
        num_events=num_events,
        start_us=start_us,
        end_us=end_us,
    )
    if coarse_end <= coarse_start:
        return _empty_events()

    t_offset = _read_t_offset(filehandle)
    t_coarse_abs = t_ds[coarse_start:coarse_end].astype(np.int64) + t_offset
    if t_coarse_abs.size == 0:
        return _empty_events()

    rel_start = int(np.searchsorted(t_coarse_abs, start_us, side="left"))
    rel_end = int(np.searchsorted(t_coarse_abs, end_us, side="left"))
    ev_start_idx = coarse_start + rel_start
    ev_end_idx = coarse_start + rel_end

    if ev_end_idx <= ev_start_idx:
        return _empty_events()
    return _extract_from_h5_by_index(filehandle=filehandle, ev_start_idx=ev_start_idx, ev_end_idx=ev_end_idx)


def _resolve_input_resolution(input_path: Path, input_height: int | None, input_width: int | None) -> tuple[int, int]:
    auto_height, auto_width = get_resolution(input_path)
    resolved_height = input_height if input_height is not None else auto_height
    resolved_width = input_width if input_width is not None else auto_width

    # Fallback to common 1MP settings if metadata is unavailable.
    if resolved_height is None:
        resolved_height = 720
    if resolved_width is None:
        resolved_width = 1280
    return int(resolved_height), int(resolved_width)


def _resolve_output_resolution(
    input_height: int,
    input_width: int,
    output_height: int,
    output_width: int,
    downsample_factor: int,
) -> tuple[int, int]:
    if int(downsample_factor) < 1:
        raise ValueError("downsample_factor must be >= 1")
    if int(downsample_factor) == 1:
        return int(output_height), int(output_width)

    if input_height % int(downsample_factor) != 0 or input_width % int(downsample_factor) != 0:
        raise ValueError(
            "input resolution must be divisible by downsample_factor for nearest downsample: "
            f"{input_width}x{input_height}, factor={downsample_factor}"
        )
    return int(input_height // int(downsample_factor)), int(input_width // int(downsample_factor))


def _spatially_normalize_events(
    events: dict[str, np.ndarray],
    input_height: int,
    input_width: int,
    output_height: int,
    output_width: int,
) -> dict[str, np.ndarray]:
    if len(events["t"]) == 0:
        return _empty_events()

    x_src = events["x"].astype(np.float32, copy=False)
    y_src = events["y"].astype(np.float32, copy=False)
    p_src = events["p"]
    t_src = events["t"]

    valid_src = (
        (x_src >= 0)
        & (x_src < float(input_width))
        & (y_src >= 0)
        & (y_src < float(input_height))
    )
    if not np.any(valid_src):
        return _empty_events()

    x_src = x_src[valid_src]
    y_src = y_src[valid_src]
    p_src = p_src[valid_src]
    t_src = t_src[valid_src]

    if output_width == input_width and output_height == input_height:
        x_out = x_src.astype(np.float32, copy=False)
        y_out = y_src.astype(np.float32, copy=False)
    elif input_width % output_width == 0 and input_height % output_height == 0:
        fx = input_width // output_width
        fy = input_height // output_height
        # Nearest-neighbor resize for integer downsample ratio.
        x_out = np.floor(x_src / float(fx)).astype(np.float32, copy=False)
        y_out = np.floor(y_src / float(fy)).astype(np.float32, copy=False)
    else:
        # Nearest-neighbor resize for generic ratio.
        x_out = np.floor(x_src * (float(output_width) / float(input_width))).astype(np.float32, copy=False)
        y_out = np.floor(y_src * (float(output_height) / float(input_height))).astype(np.float32, copy=False)

    valid_out = (
        (x_out >= 0)
        & (x_out < float(output_width))
        & (y_out >= 0)
        & (y_out < float(output_height))
    )
    if not np.any(valid_out):
        return _empty_events()

    # 1MPX source polarity may be either {0,1} or {-1,+1}.
    # EventVoxelGrid expects binary {0,1} because it applies (2*p-1) internally.
    p_bin = (p_src > 0).astype(np.float32, copy=False)

    return {
        "x": x_out[valid_out].astype(np.float32, copy=False),
        "y": y_out[valid_out].astype(np.float32, copy=False),
        "p": p_bin[valid_out].astype(np.float32, copy=False),
        "t": t_src[valid_out].astype(np.int64, copy=False),
    }


def _events_to_voxel_numpy(
    events: dict[str, np.ndarray],
    voxelizer: EventVoxelGrid,
    input_height: int,
    input_width: int,
    output_height: int,
    output_width: int,
) -> np.ndarray:
    normalized = _spatially_normalize_events(
        events=events,
        input_height=input_height,
        input_width=input_width,
        output_height=output_height,
        output_width=output_width,
    )
    if len(normalized["t"]) == 0:
        return np.zeros(voxelizer.voxel_grid.shape, dtype=np.float32)

    t_shifted = (normalized["t"] - normalized["t"][0]).astype(np.float32, copy=False)
    events_torch = {
        "x": torch.from_numpy(normalized["x"]),
        "y": torch.from_numpy(normalized["y"]),
        "p": torch.from_numpy(normalized["p"]),
        "t": torch.from_numpy(t_shifted),
    }
    voxel = voxelizer.convert(events_torch)
    return voxel.cpu().numpy().astype(np.float32, copy=False)


def _build_windows_from_start(
    start_us: int,
    end_exclusive_us: int,
    accum_time_us: int,
    stride_time_us: int,
) -> list[tuple[int, int, int, int]]:
    if end_exclusive_us <= start_us:
        return []

    starts = np.arange(start_us, end_exclusive_us, stride_time_us, dtype=np.int64)
    windows: list[tuple[int, int, int, int]] = []
    for i, win_start in enumerate(starts):
        win_start_int = int(win_start)
        win_end_int = min(win_start_int + int(accum_time_us), int(end_exclusive_us))
        anchor_int = win_start_int + (win_end_int - win_start_int) // 2
        windows.append((i, win_start_int, win_end_int, anchor_int))
    return windows


class VoxelH5Writer:
    def __init__(
        self,
        outfile: Path,
        t_bins: int,
        height: int,
        width: int,
        voxel_dtype: np.dtype,
        initial_capacity: int = 256,
    ):
        if outfile.exists():
            raise FileExistsError(f"output already exists: {outfile}")

        self.h5f = h5py.File(str(outfile), "a")
        self._finalizer = weakref.finalize(self, self.close_callback, self.h5f)
        self._capacity = int(initial_capacity)
        self._num_windows = 0
        self._datasets = (
            "voxels",
            "window_index",
            "window_t_start_us",
            "window_t_end_us",
            "window_rel_start_us",
            "window_rel_end_us",
            "anchor_timestamp_us",
            "anchor_rel_timestamp_us",
            "window_event_count",
        )

        voxel_chunks = (1, t_bins, min(height, 64), min(width, 64))
        scalar_chunks = (min(self._capacity, 4096),)
        self.h5f.create_dataset(
            "voxels",
            shape=(self._capacity, t_bins, height, width),
            maxshape=(None, t_bins, height, width),
            dtype=voxel_dtype,
            chunks=voxel_chunks,
            **H5_COMPRESSION_FLAGS,
        )
        self.h5f.create_dataset(
            "window_index",
            shape=(self._capacity,),
            maxshape=(None,),
            dtype="u8",
            chunks=scalar_chunks,
            **H5_COMPRESSION_FLAGS,
        )
        self.h5f.create_dataset(
            "window_t_start_us",
            shape=(self._capacity,),
            maxshape=(None,),
            dtype="u8",
            chunks=scalar_chunks,
            **H5_COMPRESSION_FLAGS,
        )
        self.h5f.create_dataset(
            "window_t_end_us",
            shape=(self._capacity,),
            maxshape=(None,),
            dtype="u8",
            chunks=scalar_chunks,
            **H5_COMPRESSION_FLAGS,
        )
        self.h5f.create_dataset(
            "window_rel_start_us",
            shape=(self._capacity,),
            maxshape=(None,),
            dtype="i8",
            chunks=scalar_chunks,
            **H5_COMPRESSION_FLAGS,
        )
        self.h5f.create_dataset(
            "window_rel_end_us",
            shape=(self._capacity,),
            maxshape=(None,),
            dtype="i8",
            chunks=scalar_chunks,
            **H5_COMPRESSION_FLAGS,
        )
        self.h5f.create_dataset(
            "anchor_timestamp_us",
            shape=(self._capacity,),
            maxshape=(None,),
            dtype="i8",
            chunks=scalar_chunks,
            **H5_COMPRESSION_FLAGS,
        )
        self.h5f.create_dataset(
            "anchor_rel_timestamp_us",
            shape=(self._capacity,),
            maxshape=(None,),
            dtype="i8",
            chunks=scalar_chunks,
            **H5_COMPRESSION_FLAGS,
        )
        self.h5f.create_dataset(
            "window_event_count",
            shape=(self._capacity,),
            maxshape=(None,),
            dtype="u8",
            chunks=scalar_chunks,
            **H5_COMPRESSION_FLAGS,
        )

    @staticmethod
    def close_callback(h5f: h5py.File):
        h5f.close()

    def _ensure_capacity(self, needed: int):
        if needed <= self._capacity:
            return
        new_capacity = self._capacity
        while new_capacity < needed:
            new_capacity *= 2
        for dset_name in self._datasets:
            self.h5f[dset_name].resize(new_capacity, axis=0)
        self._capacity = new_capacity

    def add_window(
        self,
        voxel: np.ndarray,
        window_index: int,
        t_start_us: int,
        t_end_us: int,
        rel_start_us: int,
        rel_end_us: int,
        anchor_timestamp_us: int,
        anchor_rel_timestamp_us: int,
        event_count: int,
    ):
        idx = self._num_windows
        self._ensure_capacity(idx + 1)

        self.h5f["voxels"][idx] = voxel
        self.h5f["window_index"][idx] = int(window_index)
        self.h5f["window_t_start_us"][idx] = int(t_start_us)
        self.h5f["window_t_end_us"][idx] = int(t_end_us)
        self.h5f["window_rel_start_us"][idx] = int(rel_start_us)
        self.h5f["window_rel_end_us"][idx] = int(rel_end_us)
        self.h5f["anchor_timestamp_us"][idx] = int(anchor_timestamp_us)
        self.h5f["anchor_rel_timestamp_us"][idx] = int(anchor_rel_timestamp_us)
        self.h5f["window_event_count"][idx] = int(event_count)
        self._num_windows += 1

    def _trim(self):
        for dset_name in self._datasets:
            self.h5f[dset_name].resize(self._num_windows, axis=0)

    def close(self):
        if self._finalizer.alive:
            self._trim()
            self._finalizer()


def process_single_file(
    input_path: Path,
    output_path: Path,
    input_height: int | None,
    input_width: int | None,
    output_height: int,
    output_width: int,
    downsample_factor: int,
    t_bins: int,
    accum_time: int,
    stride_time: int,
    start_time_us: int | None,
    normalize: bool,
    output_dtype: str,
    show_progress: bool,
    tmp_suffix: str,
) -> None:
    if output_height <= 0 or output_width <= 0:
        raise ValueError("output_height/output_width must be > 0")
    if int(downsample_factor) not in (1, 2):
        raise ValueError("downsample_factor must be 1 or 2")
    if t_bins <= 0:
        raise ValueError("t_bins must be > 0")
    if accum_time <= 0:
        raise ValueError("accum_time must be > 0")
    if stride_time <= 0:
        raise ValueError("stride_time must be > 0")

    resolved_input_height, resolved_input_width = _resolve_input_resolution(
        input_path=input_path,
        input_height=input_height,
        input_width=input_width,
    )
    effective_output_height, effective_output_width = _resolve_output_resolution(
        input_height=resolved_input_height,
        input_width=resolved_input_width,
        output_height=output_height,
        output_width=output_width,
        downsample_factor=downsample_factor,
    )
    voxel_dtype = np.float16 if output_dtype == "float16" else np.float32

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_output_path(output_path=output_path, tmp_suffix=tmp_suffix)
    cleanup_tmp_file(tmp_path=tmp_path, context=f"start processing {input_path}", strict=True)

    writer = None
    pbar = None
    try:
        with h5py.File(str(input_path), "r") as h5f:
            t_first, t_last_exclusive = _get_time_bounds_us(h5f)
            ms_to_idx = _load_ms_to_idx(h5f)

            writer = VoxelH5Writer(
                outfile=tmp_path,
                t_bins=t_bins,
                height=effective_output_height,
                width=effective_output_width,
                voxel_dtype=voxel_dtype,
            )
            writer.h5f.attrs["representation"] = "event_voxel_grid_1mpx"
            writer.h5f.attrs["source_file"] = str(input_path)
            writer.h5f.attrs["input_height"] = int(resolved_input_height)
            writer.h5f.attrs["input_width"] = int(resolved_input_width)
            writer.h5f.attrs["requested_output_height"] = int(output_height)
            writer.h5f.attrs["requested_output_width"] = int(output_width)
            writer.h5f.attrs["height"] = int(effective_output_height)
            writer.h5f.attrs["width"] = int(effective_output_width)
            writer.h5f.attrs["downsample_factor"] = int(downsample_factor)
            writer.h5f.attrs["spatial_resize_mode"] = "nearest"
            writer.h5f.attrs["t_bins"] = int(t_bins)
            writer.h5f.attrs["accum_time_us"] = int(accum_time)
            writer.h5f.attrs["stride_time_us"] = int(stride_time)
            writer.h5f.attrs["normalize"] = int(normalize)

            if t_first is None or t_last_exclusive is None:
                writer.h5f.attrs["time_origin_us"] = -1
                writer.h5f.attrs["num_windows_planned"] = 0
                writer.close()
                writer = None
                os.replace(tmp_path, output_path)
                return

            time_origin_us = int(t_first) if start_time_us is None else int(start_time_us)
            window_start_us = max(int(t_first), time_origin_us)
            windows = _build_windows_from_start(
                start_us=window_start_us,
                end_exclusive_us=int(t_last_exclusive),
                accum_time_us=accum_time,
                stride_time_us=stride_time,
            )
            writer.h5f.attrs["time_origin_us"] = int(time_origin_us)
            writer.h5f.attrs["num_windows_planned"] = int(len(windows))

            voxelizer = EventVoxelGrid(
                input_size=(t_bins, effective_output_height, effective_output_width),
                normalize=normalize,
            )
            if show_progress:
                pbar = tqdm.tqdm(total=len(windows), desc=input_path.name, leave=False)

            for window_index, start_us, end_us, anchor_us in windows:
                events = _extract_events_by_time(
                    filehandle=h5f,
                    start_us=start_us,
                    end_us=end_us,
                    ms_to_idx=ms_to_idx,
                )
                voxel = _events_to_voxel_numpy(
                    events=events,
                    voxelizer=voxelizer,
                    input_height=resolved_input_height,
                    input_width=resolved_input_width,
                    output_height=effective_output_height,
                    output_width=effective_output_width,
                )
                voxel = voxel.astype(voxel_dtype, copy=False)
                writer.add_window(
                    voxel=voxel,
                    window_index=window_index,
                    t_start_us=start_us,
                    t_end_us=end_us,
                    rel_start_us=start_us - time_origin_us,
                    rel_end_us=end_us - time_origin_us,
                    anchor_timestamp_us=anchor_us,
                    anchor_rel_timestamp_us=anchor_us - time_origin_us,
                    event_count=len(events["t"]),
                )
                if pbar is not None:
                    pbar.update(1)

        writer.close()
        writer = None
        os.replace(tmp_path, output_path)
    except Exception:
        if writer is not None:
            writer.close()
        cleanup_tmp_file(tmp_path=tmp_path, context=f"exception cleanup for {input_path}", strict=False)
        raise
    finally:
        if pbar is not None:
            pbar.close()


def _process_file_with_retry(
    input_path: Path,
    output_path: Path,
    input_height: int | None,
    input_width: int | None,
    output_height: int,
    output_width: int,
    downsample_factor: int,
    t_bins: int,
    accum_time: int,
    stride_time: int,
    start_time_us: int | None,
    normalize: bool,
    output_dtype: str,
    tmp_suffix: str,
) -> tuple[bool, str | None]:
    stale_tmp_path = tmp_output_path(output_path=output_path, tmp_suffix=tmp_suffix)
    if not cleanup_tmp_file(tmp_path=stale_tmp_path, context=f"resume prep for {input_path}", strict=False):
        return False, f"could not remove stale tmp file: {stale_tmp_path}"

    for attempt in (1, 2):
        try:
            process_single_file(
                input_path=input_path,
                output_path=output_path,
                input_height=input_height,
                input_width=input_width,
                output_height=output_height,
                output_width=output_width,
                downsample_factor=downsample_factor,
                t_bins=t_bins,
                accum_time=accum_time,
                stride_time=stride_time,
                start_time_us=start_time_us,
                normalize=normalize,
                output_dtype=output_dtype,
                show_progress=False,
                tmp_suffix=tmp_suffix,
            )
            return True, None
        except Exception as exc:
            if attempt == 1:
                cleanup_ok = cleanup_tmp_file(
                    tmp_path=stale_tmp_path,
                    context=f"retry prep for {input_path}",
                    strict=False,
                )
                if not cleanup_ok:
                    return False, f"retry cleanup failed for {stale_tmp_path}: {exc}"
                continue
            return False, str(exc)

    return False, "unknown failure"


def _worker_process_file(job: dict) -> tuple[str, bool, str | None]:
    input_path = Path(job["input_path"])
    output_path = Path(job["output_path"])
    ok, err = _process_file_with_retry(
        input_path=input_path,
        output_path=output_path,
        input_height=job["input_height"],
        input_width=job["input_width"],
        output_height=job["output_height"],
        output_width=job["output_width"],
        downsample_factor=job["downsample_factor"],
        t_bins=job["t_bins"],
        accum_time=job["accum_time"],
        stride_time=job["stride_time"],
        start_time_us=job["start_time_us"],
        normalize=job["normalize"],
        output_dtype=job["output_dtype"],
        tmp_suffix=job["tmp_suffix"],
    )
    return str(input_path), ok, err


def _find_h5_files(dataset_root: Path, splits: list[str], recursive: bool) -> list[Path]:
    input_files: list[Path] = []
    for split in splits:
        if split in {"", "."}:
            split_dir = dataset_root
        else:
            split_dir = dataset_root / split
        if not split_dir.exists():
            print(f"[WARN] missing split directory: {split_dir}")
            continue

        files = split_dir.rglob("*.h5") if recursive else split_dir.glob("*.h5")
        input_files.extend(sorted([p for p in files if p.is_file()]))
    return input_files


def _build_output_path(
    input_path: Path,
    dataset_root: Path,
    output_root: Path | None,
    normalized_suffix: str,
    normalized_subdir: str | None,
    downsample_factor: int,
) -> Path:
    output_name = f"{input_path.stem}{normalized_suffix}"
    output_name = ensure_scale_tag_in_filename(output_name, downsample_factor=downsample_factor)
    if output_root is None:
        output_dir = input_path.parent
    else:
        relative_input = input_path.relative_to(dataset_root)
        output_dir = output_root / relative_input.parent

    if normalized_subdir is not None:
        output_dir = output_dir / normalized_subdir
    return output_dir / output_name


def process_dataset_root(
    dataset_root: Path,
    splits: list[str],
    output_suffix: str,
    output_subdir: str | None,
    overwrite: bool,
    output_root: Path | None,
    input_height: int | None,
    input_width: int | None,
    output_height: int,
    output_width: int,
    downsample_factor: int,
    t_bins: int,
    accum_time: int,
    stride_time: int,
    start_time_us: int | None,
    normalize: bool,
    output_dtype: str,
    recursive: bool,
    tmp_suffix: str,
    num_processes: int,
) -> None:
    if int(num_processes) < 1:
        raise ValueError("num_processes must be >= 1")

    normalized_suffix = normalized_output_suffix(output_suffix)
    normalized_subdir = normalized_output_subdir(output_subdir)
    tagged_suffixes = {
        normalized_suffix,
        ensure_scale_tag_in_filename(f"dummy{normalized_suffix}", downsample_factor=1).removeprefix("dummy"),
        ensure_scale_tag_in_filename(f"dummy{normalized_suffix}", downsample_factor=2).removeprefix("dummy"),
    }
    input_files = _find_h5_files(dataset_root=dataset_root, splits=splits, recursive=recursive)
    if len(input_files) == 0:
        raise FileNotFoundError(f"No .h5 files found under {dataset_root} for splits={splits}")
    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)

    jobs: list[dict] = []
    num_done = 0
    num_skipped = 0
    num_failed = 0

    for input_path in tqdm.tqdm(input_files, desc="1mpx files"):
        if any(input_path.name.endswith(sfx) for sfx in tagged_suffixes):
            num_skipped += 1
            continue

        output_path = _build_output_path(
            input_path=input_path,
            dataset_root=dataset_root,
            output_root=output_root,
            normalized_suffix=normalized_suffix,
            normalized_subdir=normalized_subdir,
            downsample_factor=downsample_factor,
        )
        if output_path.exists():
            if overwrite:
                output_path.unlink()
            else:
                num_skipped += 1
                continue

        jobs.append(
            {
                "input_path": str(input_path),
                "output_path": str(output_path),
                "input_height": input_height,
                "input_width": input_width,
                "output_height": output_height,
                "output_width": output_width,
                "downsample_factor": downsample_factor,
                "t_bins": t_bins,
                "accum_time": accum_time,
                "stride_time": stride_time,
                "start_time_us": start_time_us,
                "normalize": normalize,
                "output_dtype": output_dtype,
                "tmp_suffix": tmp_suffix,
            }
        )

    if len(jobs) > 0:
        if int(num_processes) == 1:
            iterator = (_worker_process_file(job) for job in jobs)
            for input_name, success, err in tqdm.tqdm(iterator, total=len(jobs), desc="1mpx workers"):
                if success:
                    num_done += 1
                else:
                    num_failed += 1
                    print(f"[FAILED] {input_name}: {err}")
        else:
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=int(num_processes)) as pool:
                for input_name, success, err in tqdm.tqdm(
                    pool.imap_unordered(_worker_process_file, jobs),
                    total=len(jobs),
                    desc="1mpx workers",
                ):
                    if success:
                        num_done += 1
                    else:
                        num_failed += 1
                        print(f"[FAILED] {input_name}: {err}")

    print(f"[SUMMARY] done={num_done}, skipped={num_skipped}, failed={num_failed}")
    if num_failed > 0:
        raise RuntimeError(f"{num_failed} files failed while processing {dataset_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Build event voxel representations from 1MPX-style events H5")
    parser.add_argument("--input_path", type=Path, help="Input events .h5")
    parser.add_argument("--output_path", type=Path, help="Output voxel .h5")

    parser.add_argument("--dataset_root", type=Path, help="Dataset root with split directories")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test", "val"],
        help="Splits for --dataset_root mode. Use '.' to scan dataset_root directly.",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="_voxels.h5",
        help="Output suffix for --dataset_root mode (e.g. _voxels.h5). Scale tag (_1x/_2x) is auto-added.",
    )
    parser.add_argument(
        "--output_subdir",
        type=str,
        default=None,
        help="Optional output subdirectory under each target dir (e.g. voxel).",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=None,
        help="Optional output root dir for --dataset_root mode (preserves relative split paths)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    parser.add_argument("--recursive", action="store_true", help="Recursively search .h5 under split dirs")
    parser.add_argument(
        "--tmp_suffix",
        type=str,
        default=".tmp",
        help="Temporary suffix used while writing (renamed on success)",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="Parallel workers for --dataset_root mode (spawn).",
    )

    parser.add_argument("--input_height", type=int, default=720, help="Input event height (default: 720).")
    parser.add_argument("--input_width", type=int, default=1280, help="Input event width (default: 1280).")
    parser.add_argument("--output_height", type=int, default=720, help="Output voxel height (default: 720).")
    parser.add_argument("--output_width", type=int, default=1280, help="Output voxel width (default: 1280).")
    parser.add_argument(
        "--downsample_factor",
        type=int,
        choices=[1, 2],
        default=1,
        help="Nearest-neighbor spatial downsample factor (2 means 1/2 resolution).",
    )
    parser.add_argument("--t_bins", type=int, default=5, help="Number of temporal bins for voxel representation.")
    parser.add_argument(
        "--accum_time",
        type=int,
        default=50000,
        help="Accumulation window in microseconds.",
    )
    parser.add_argument(
        "--stride_time",
        type=int,
        default=None,
        help="Sliding stride in microseconds (default: same as --accum_time).",
    )
    parser.add_argument(
        "--start_time_us",
        type=int,
        default=None,
        help="Optional fixed time origin in us. Default uses the first event timestamp in each file.",
    )
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize non-zero voxel values per sample.",
    )
    parser.add_argument(
        "--output_dtype",
        choices=["float16", "float32"],
        default="float16",
        help="Stored dtype for voxel tensor in output HDF5.",
    )
    args = parser.parse_args()

    stride_time = args.accum_time if args.stride_time is None else args.stride_time

    is_single_mode = args.input_path is not None or args.output_path is not None
    is_root_mode = args.dataset_root is not None
    if is_single_mode and is_root_mode:
        parser.error("Use either single-file mode (--input_path/--output_path) or root mode (--dataset_root), not both.")

    if is_root_mode:
        process_dataset_root(
            dataset_root=args.dataset_root,
            splits=args.splits,
            output_suffix=args.output_suffix,
            output_subdir=args.output_subdir,
            overwrite=args.overwrite,
            output_root=args.output_root,
            input_height=args.input_height,
            input_width=args.input_width,
            output_height=args.output_height,
            output_width=args.output_width,
            downsample_factor=args.downsample_factor,
            t_bins=args.t_bins,
            accum_time=args.accum_time,
            stride_time=stride_time,
            start_time_us=args.start_time_us,
            normalize=args.normalize,
            output_dtype=args.output_dtype,
            recursive=args.recursive,
            tmp_suffix=args.tmp_suffix,
            num_processes=args.num_processes,
        )
    else:
        if args.input_path is None or args.output_path is None:
            parser.error("Single-file mode requires both --input_path and --output_path.")

        process_single_file(
            input_path=args.input_path,
            output_path=args.output_path,
            input_height=args.input_height,
            input_width=args.input_width,
            output_height=args.output_height,
            output_width=args.output_width,
            downsample_factor=args.downsample_factor,
            t_bins=args.t_bins,
            accum_time=args.accum_time,
            stride_time=stride_time,
            start_time_us=args.start_time_us,
            normalize=args.normalize,
            output_dtype=args.output_dtype,
            show_progress=True,
            tmp_suffix=args.tmp_suffix,
        )
