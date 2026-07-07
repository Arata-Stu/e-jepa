from __future__ import annotations

from collections import OrderedDict
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence

try:
    import hdf5plugin  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "hdf5plugin is required to load voxel HDF5 datasets. "
        "Please install dependencies (e.g. `pip install -r requirements.txt`)."
    ) from exc

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.datasets.m3ed_raw import (
    _align_anchor_timebase_to_events,
    _build_middle_windows,
    _discover_m3ed_h5_files,
    _h5_searchsorted_left,
    _read_t_offset,
    _resolve_companion_h5,
    _sequence_name_from_event_path,
)
from src.representations import EventVoxelGrid, accumulate_events_to_rgb


TaskTarget = Literal["semantic", "depth"]
DatasetKind = Literal["dsec", "m3ed", "m3ed_raw"]


def _parse_manifest_paths(manifest_path: Path) -> list[Path]:
    rows: list[Path] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for raw in reader:
            if len(raw) == 0:
                continue
            first = raw[0].strip()
            if not first or first.startswith("#"):
                continue
            token = first.split()[0]
            p = Path(token)
            if not p.is_absolute():
                p = (manifest_path.parent / p).resolve()
            rows.append(p)
    return rows


def discover_h5_files(
    roots: Sequence[str | Path],
    *,
    file_pattern: str = "*.h5",
    recursive: bool = True,
) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        p = Path(root).expanduser()
        suffix = p.suffix.lower()
        if suffix in {".csv", ".txt"}:
            files.extend(_parse_manifest_paths(p))
            continue
        if suffix in {".h5", ".hdf5"} and p.is_file():
            files.append(p.resolve())
            continue
        if p.is_dir():
            iterator = p.rglob(file_pattern) if recursive else p.glob(file_pattern)
            files.extend(sorted(x.resolve() for x in iterator if x.is_file()))
            continue
        raise FileNotFoundError(f"Unsupported downstream data path: {p}")

    uniq = sorted({f.resolve() for f in files})
    return [p for p in uniq if p.suffix.lower() in {".h5", ".hdf5"}]


def _decode_h5_string(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _decode_h5_string(value.item())
        if value.size == 1:
            return _decode_h5_string(value.reshape(-1)[0])
    return str(value)


def _load_h5_attr_str(h5f: h5py.File, key: str, default: str = "") -> str:
    if key not in h5f.attrs:
        return default
    return _decode_h5_string(h5f.attrs[key]).strip()


def _load_h5_attr_int(h5f: h5py.File, key: str) -> int | None:
    if key not in h5f.attrs:
        return None
    value = h5f.attrs[key]
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            value = value.item()
        elif value.size == 1:
            value = value.reshape(-1)[0]
    try:
        return int(value)
    except Exception:
        return None


def _maybe_rebase_split_embedded_window_index(
    *,
    h5f: h5py.File,
    window_index: np.ndarray,
    label_len: int,
    embedded_label_ds_path: str | None,
) -> np.ndarray:
    if embedded_label_ds_path is None or int(label_len) <= 0 or window_index.size == 0:
        return window_index

    split_start = _load_h5_attr_int(h5f, "split_source_start_index")
    split_end = _load_h5_attr_int(h5f, "split_source_end_index_exclusive")
    if split_start is None or split_end is None or int(split_end) <= int(split_start):
        return window_index

    if not np.all((window_index >= int(split_start)) & (window_index < int(split_end))):
        return window_index

    rebased = window_index - int(split_start)
    if not np.all((rebased >= 0) & (rebased < int(label_len))):
        return window_index
    return rebased.astype(np.int64, copy=False)


def _squeeze_to_hw(arr: np.ndarray, *, target: TaskTarget) -> np.ndarray:
    if arr.ndim == 2:
        return arr
    if arr.ndim != 3:
        raise ValueError(f"Expected 2D/3D label map, got shape={arr.shape}")
    # Common singleton-channel layouts.
    if arr.shape[0] == 1:
        return arr[0]
    if arr.shape[-1] == 1:
        return arr[..., 0]
    if target == "depth":
        # For depth, some datasets store (H, W, 1) or (1, H, W) only.
        raise ValueError(f"Unsupported depth map shape={arr.shape}; expected singleton-channel.")
    # For semantic we intentionally reject 3-channel color maps to avoid silent class corruption.
    raise ValueError(
        f"Unsupported semantic map shape={arr.shape}. "
        "Expected class-index map (H,W) or singleton-channel variant."
    )


def _resize_label_to_hw(
    label: torch.Tensor,
    *,
    target_h: int,
    target_w: int,
    target: TaskTarget,
) -> torch.Tensor:
    if int(label.shape[-2]) == target_h and int(label.shape[-1]) == target_w:
        return label
    mode = "nearest" if target == "semantic" else "bilinear"
    input_ = label.unsqueeze(0).unsqueeze(0).to(torch.float32)
    resized = F.interpolate(
        input_,
        size=(target_h, target_w),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )
    resized = resized.squeeze(0).squeeze(0)
    if target == "semantic":
        return torch.round(resized).to(torch.int64)
    return resized.to(torch.float32)


def _to_optional_hw_tuple(value, *, field_name: str) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, int):
        v = int(value)
        return (v, v)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise ValueError(f"{field_name} must be null, int, or [H, W], got: {value}")


def _resize_clip_cthw(
    clip: torch.Tensor,
    *,
    target_hw: tuple[int, int] | None,
    mode: str,
) -> torch.Tensor:
    if target_hw is None:
        return clip
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    if int(clip.shape[-2]) == target_h and int(clip.shape[-1]) == target_w:
        return clip

    clip_tchw = clip.permute(1, 0, 2, 3).contiguous()
    mode = str(mode).lower()
    kwargs = {"size": (target_h, target_w), "mode": mode}
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
    resized = F.interpolate(clip_tchw, **kwargs)
    return resized.permute(1, 0, 2, 3).contiguous()


def _resize_label_like_dense_target(
    label: torch.Tensor,
    *,
    target_h: int,
    target_w: int,
    target: TaskTarget,
) -> torch.Tensor:
    return _resize_label_to_hw(
        label,
        target_h=target_h,
        target_w=target_w,
        target=target,
    )


def _pad_or_crop_semantic_label_to_hw(
    label: torch.Tensor,
    *,
    target_h: int,
    target_w: int,
    pad_value: int,
) -> torch.Tensor:
    if label.ndim != 2:
        raise ValueError(f"expected semantic label [H,W], got shape={tuple(label.shape)}")
    if int(label.shape[-2]) == int(target_h) and int(label.shape[-1]) == int(target_w):
        return label.to(torch.int64)

    # DSEC semantic labels can be shorter than the event frame height (for example 440 vs 480).
    # Preserve the native top-left alignment and only pad/crop on the bottom/right.
    out = torch.full(
        (int(target_h), int(target_w)),
        int(pad_value),
        dtype=torch.int64,
        device=label.device,
    )
    src = label.to(torch.int64)
    copy_h = min(int(src.shape[-2]), int(target_h))
    copy_w = min(int(src.shape[-1]), int(target_w))
    out[:copy_h, :copy_w] = src[:copy_h, :copy_w]
    return out


def _build_centered_clip_indices(
    *,
    center_idx: int,
    num_total: int,
    num_frames: int,
    frame_stride: int,
) -> np.ndarray:
    if num_total <= 0:
        raise ValueError("num_total must be > 0")
    if num_frames <= 0:
        raise ValueError("num_frames must be > 0")
    if frame_stride <= 0:
        raise ValueError("frame_stride must be > 0")
    half = (num_frames - 1) // 2
    indices = []
    for i in range(num_frames):
        offset = (i - half) * frame_stride
        idx = int(center_idx + offset)
        idx = min(max(idx, 0), num_total - 1)
        indices.append(idx)
    return np.asarray(indices, dtype=np.int64)


def _read_voxels_tchw(voxels_ds: h5py.Dataset, indices: np.ndarray) -> np.ndarray:
    if indices.size == 0:
        raise ValueError("indices cannot be empty")
    if indices.size > 1 and np.all(indices[1:] - indices[:-1] == 1):
        start = int(indices[0])
        end = int(indices[-1]) + 1
        arr = np.asarray(voxels_ds[start:end])
    else:
        arr = np.stack([np.asarray(voxels_ds[int(i)]) for i in indices], axis=0)
    if arr.ndim != 4:
        raise ValueError(f"Expected voxel windows [T,C,H,W], got shape={arr.shape}")
    return arr.astype(np.float32, copy=False)


def _find_first_matching_dataset(
    h5f: h5py.File,
    *,
    candidates: Sequence[str],
    length0: int,
    min_ndim: int = 3,
) -> str | None:
    for path in candidates:
        if path not in h5f:
            continue
        ds = h5f[path]
        if _is_plausible_dense_label_dataset(ds, length0=length0, min_ndim=min_ndim):
            return path
    return None


def _is_plausible_dense_label_dataset(
    ds: h5py.Dataset,
    *,
    length0: int,
    min_ndim: int = 3,
    min_spatial_extent: int = 8,
) -> bool:
    if not isinstance(ds, h5py.Dataset):
        return False
    if ds.ndim < min_ndim or int(ds.shape[0]) != int(length0):
        return False
    tail = [int(v) for v in ds.shape[1:] if int(v) > 1]
    if len(tail) < 2:
        return False
    spatial_dims = sorted(tail)[-2:]
    return spatial_dims[0] >= int(min_spatial_extent) and spatial_dims[1] >= int(min_spatial_extent)


def _find_recursive_dataset_with_length(
    h5f: h5py.File,
    *,
    group_prefix: str,
    length0: int,
    min_ndim: int = 3,
) -> str | None:
    if group_prefix not in h5f:
        return None
    group = h5f[group_prefix]
    if not isinstance(group, h5py.Group):
        return None

    stack: list[tuple[str, h5py.Group]] = [(group_prefix, group)]
    while stack:
        prefix, g = stack.pop()
        for key in g.keys():
            full = f"{prefix}/{key}"
            obj = g[key]
            if isinstance(obj, h5py.Group):
                stack.append((full, obj))
                continue
            if _is_plausible_dense_label_dataset(obj, length0=length0, min_ndim=min_ndim):
                return full
    return None


@dataclass(frozen=True)
class _FileMeta:
    preprocessed_h5: Path
    num_windows: int
    # DSEC
    embedded_segmentation_dataset_path: str | None = None
    # M3ED
    window_index: np.ndarray | None = None
    label_lookup_index: np.ndarray | None = None
    label_dataset_path: str | None = None
    label_length: int = 0
    embedded_label_dataset_path: str | None = None


def _cityscapes_19_to_11_mapping(*, ignore_index: int) -> np.ndarray:
    mapping = np.full((256,), int(ignore_index), dtype=np.int64)
    mapping[:19] = np.asarray(
        [
            5,
            6,
            1,
            9,
            2,
            4,
            10,
            10,
            7,
            7,
            0,
            3,
            3,
            8,
            8,
            8,
            8,
            8,
            8,
        ],
        dtype=np.int64,
    )
    if 0 <= int(ignore_index) < mapping.size:
        mapping[int(ignore_index)] = int(ignore_index)
    return mapping


def _apply_semantic_label_remap(
    label: np.ndarray,
    *,
    remap: str,
    ignore_index: int,
) -> np.ndarray:
    label_i64 = np.asarray(label, dtype=np.int64)
    remap_name = str(remap).lower()
    if remap_name in {"", "none", "identity"}:
        return label_i64
    if remap_name != "cityscapes_19_to_11":
        raise ValueError(
            f"Unsupported semantic label_remap={remap!r}. "
            "Use 'none' or 'cityscapes_19_to_11'."
        )

    mapping = _cityscapes_19_to_11_mapping(ignore_index=int(ignore_index))
    out = np.full_like(label_i64, fill_value=int(ignore_index), dtype=np.int64)
    valid = (label_i64 >= 0) & (label_i64 < int(mapping.size))
    if np.any(valid):
        out[valid] = mapping[label_i64[valid]]
    return out


def _normalise_optional_sequence_names(value) -> set[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        names = [value]
    else:
        names = list(value)
    out = {str(name).strip() for name in names if str(name).strip()}
    return out if len(out) > 0 else None


@dataclass(frozen=True)
class _SequenceRange:
    start_fraction: float = 0.0
    stop_fraction: float = 1.0
    stride: int = 1


def _parse_sequence_range_spec(value) -> _SequenceRange:
    if isinstance(value, Mapping):
        if "f3_range" in value:
            return _parse_sequence_range_spec(value["f3_range"])
        start = float(value.get("start_fraction", value.get("start", 0.0)))
        stop = float(value.get("stop_fraction", value.get("stop", 1.0)))
        stride = int(value.get("stride", value.get("step", 1)))
        return _SequenceRange(
            start_fraction=start,
            stop_fraction=stop,
            stride=max(1, stride),
        )
    if isinstance(value, (list, tuple)) and len(value) == 3:
        # Match F3 segmentation configs: range = [start_fraction, step, stop_fraction].
        return _SequenceRange(
            start_fraction=float(value[0]),
            stop_fraction=float(value[2]),
            stride=max(1, int(value[1])),
        )
    raise ValueError(
        "sequence range must be a dict or F3-style "
        "[start_fraction, step, stop_fraction] list, got: "
        f"{value!r}"
    )


def _normalise_sequence_ranges(value) -> dict[str, _SequenceRange]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {
            str(name).strip(): _parse_sequence_range_spec(spec)
            for name, spec in value.items()
            if str(name).strip()
        }
    if isinstance(value, (list, tuple)):
        ranges: dict[str, _SequenceRange] = {}
        for entry in value:
            if not isinstance(entry, Mapping):
                raise ValueError(
                    "sequence range list entries must be dicts with a 'name' field"
                )
            name = str(entry.get("name", "")).strip()
            if not name:
                raise ValueError(f"sequence range entry is missing name: {entry!r}")
            if "range" in entry:
                ranges[name] = _parse_sequence_range_spec(entry["range"])
            else:
                ranges[name] = _parse_sequence_range_spec(entry)
        return ranges
    raise ValueError(f"Unsupported sequence_ranges value: {value!r}")


def _sequence_name_candidates(event_path: Path) -> set[str]:
    return {
        name
        for name in (
            _sequence_name_from_event_path(event_path),
            event_path.parent.name.strip(),
            event_path.stem.strip(),
        )
        if name
    }


def _sequence_is_selected(
    event_path: Path,
    *,
    include: set[str] | None,
    exclude: set[str] | None,
) -> bool:
    candidates = _sequence_name_candidates(event_path)
    if include is not None and candidates.isdisjoint(include):
        return False
    if exclude is not None and not candidates.isdisjoint(exclude):
        return False
    return True


def _sequence_range_for_path(
    event_path: Path,
    sequence_ranges: dict[str, _SequenceRange],
) -> _SequenceRange | None:
    for name in _sequence_name_candidates(event_path):
        if name in sequence_ranges:
            return sequence_ranges[name]
    return None


def _fractional_subset(
    indices: np.ndarray,
    *,
    start_fraction: float,
    stop_fraction: float,
    stride: int,
) -> np.ndarray:
    if indices.size == 0:
        return indices.astype(np.int64, copy=False)
    start = max(0.0, min(1.0, float(start_fraction)))
    stop = max(0.0, min(1.0, float(stop_fraction)))
    if stop < start:
        start, stop = stop, start
    lo = int(indices.size * start)
    hi = int(indices.size * stop)
    if stop >= 1.0:
        hi = int(indices.size)
    step = max(1, int(stride))
    return indices[lo:hi:step].astype(np.int64, copy=False)


class EventDenseTaskDataset(Dataset):
    """
    Downstream dataset for dense tasks on preprocessed event voxel H5:
      - DSEC semantic segmentation
      - M3ED semantic segmentation / depth estimation
    """

    def __init__(
        self,
        *,
        roots: Sequence[str | Path],
        dataset_kind: DatasetKind,
        target: TaskTarget,
        clip_num_frames: int,
        clip_frame_stride: int = 1,
        file_pattern: str = "*.h5",
        recursive: bool = True,
        ignore_index: int = 255,
        depth_scale: float = 1.0,
        require_labels: bool = True,
        input_size: Sequence[int] | int | None = None,
        input_resize_mode: str = "bilinear",
        return_eval_target: bool = False,
    ):
        self.dataset_kind = str(dataset_kind).lower()
        self.target = str(target).lower()
        if self.dataset_kind not in {"dsec", "m3ed"}:
            raise ValueError(f"Unsupported dataset_kind={dataset_kind}")
        if self.target not in {"semantic", "depth"}:
            raise ValueError(f"Unsupported target={target}")
        if self.dataset_kind == "dsec" and self.target != "semantic":
            raise ValueError("DSEC downstream currently supports semantic only.")

        self.clip_num_frames = int(clip_num_frames)
        self.clip_frame_stride = int(clip_frame_stride)
        self.ignore_index = int(ignore_index)
        self.depth_scale = float(depth_scale)
        self.require_labels = bool(require_labels)
        self.input_size = _to_optional_hw_tuple(input_size, field_name="input_size")
        self.input_resize_mode = str(input_resize_mode).lower()
        self.return_eval_target = bool(return_eval_target)

        self.files: list[_FileMeta] = []
        self.samples: list[tuple[int, int]] = []  # (file_idx, window_idx)

        self._pre_h5_cache: dict[str, h5py.File] = {}

        all_h5 = discover_h5_files(roots, file_pattern=file_pattern, recursive=recursive)
        if len(all_h5) == 0:
            raise FileNotFoundError("No downstream preprocessed H5 files found.")
        for h5_path in all_h5:
            meta, valid_window_indices = self._build_file_meta(h5_path)
            if meta is None or len(valid_window_indices) == 0:
                continue
            file_idx = len(self.files)
            self.files.append(meta)
            self.samples.extend((file_idx, int(wi)) for wi in valid_window_indices.tolist())

        if len(self.files) == 0 or len(self.samples) == 0:
            raise RuntimeError(
                "No valid downstream samples found. "
                "Check dataset roots, preprocessing mode, and label availability."
            )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_pre_h5_cache"] = {}
        return state

    def __del__(self):
        for h5f in getattr(self, "_pre_h5_cache", {}).values():
            try:
                h5f.close()
            except Exception:
                pass

    def __len__(self) -> int:
        return len(self.samples)

    def _get_pre_h5(self, path: Path) -> h5py.File:
        key = str(path)
        h5f = self._pre_h5_cache.get(key)
        if h5f is None:
            h5f = h5py.File(key, "r")
            self._pre_h5_cache[key] = h5f
        return h5f

    def _build_file_meta(self, h5_path: Path) -> tuple[_FileMeta | None, np.ndarray]:
        with h5py.File(str(h5_path), "r") as h5f:
            if "voxels" not in h5f:
                return None, np.empty((0,), dtype=np.int64)
            vox = h5f["voxels"]
            if vox.ndim != 4 or int(vox.shape[0]) <= 0:
                return None, np.empty((0,), dtype=np.int64)

            num_windows = int(vox.shape[0])
            all_indices = np.arange(num_windows, dtype=np.int64)

            if self.dataset_kind == "dsec":
                embedded_seg_ds_path = None
                embedded_label_ds_path = _load_h5_attr_str(h5f, "embedded_label_dataset", default="")
                if embedded_label_ds_path == "embedded_segmentation" and embedded_label_ds_path in h5f:
                    embedded_ds = h5f[embedded_label_ds_path]
                    if embedded_ds.ndim >= 3 and int(embedded_ds.shape[0]) == num_windows:
                        embedded_seg_ds_path = embedded_label_ds_path
                if embedded_seg_ds_path is None:
                    return None, np.empty((0,), dtype=np.int64)
                if "segmentation_available" in h5f:
                    avail = np.asarray(h5f["segmentation_available"][()], dtype=np.int64).reshape(-1)
                    if avail.shape[0] != num_windows:
                        return None, np.empty((0,), dtype=np.int64)
                else:
                    avail = np.ones((num_windows,), dtype=np.int64)

                valid = [idx for idx in range(num_windows) if int(avail[idx]) > 0]

                meta = _FileMeta(
                    preprocessed_h5=h5_path,
                    num_windows=num_windows,
                    embedded_segmentation_dataset_path=embedded_seg_ds_path,
                )
                return meta, np.asarray(valid, dtype=np.int64)

            if "window_index" in h5f:
                window_index = np.asarray(h5f["window_index"][()], dtype=np.int64).reshape(-1)
            else:
                window_index = all_indices.copy()
            if window_index.shape[0] != num_windows:
                return None, np.empty((0,), dtype=np.int64)

            embedded_label_ds_path = _load_h5_attr_str(h5f, "embedded_label_dataset", default="")
            label_ds_path = None
            label_len = 0
            if self.target == "semantic" and embedded_label_ds_path == "embedded_semantics" and embedded_label_ds_path in h5f:
                embedded_ds = h5f[embedded_label_ds_path]
                if _is_plausible_dense_label_dataset(embedded_ds, length0=num_windows, min_ndim=3):
                    label_ds_path = embedded_label_ds_path
                    label_len = int(embedded_ds.shape[0])
            elif self.target == "depth" and embedded_label_ds_path == "embedded_depth" and embedded_label_ds_path in h5f:
                embedded_ds = h5f[embedded_label_ds_path]
                if _is_plausible_dense_label_dataset(embedded_ds, length0=num_windows, min_ndim=3):
                    label_ds_path = embedded_label_ds_path
                    label_len = int(embedded_ds.shape[0])
            if label_ds_path is None or label_len <= 0:
                return None, np.empty((0,), dtype=np.int64)
            label_lookup_index = _maybe_rebase_split_embedded_window_index(
                h5f=h5f,
                window_index=window_index,
                label_len=int(label_len),
                embedded_label_ds_path=label_ds_path,
            )
            valid_mask = (label_lookup_index >= 0) & (label_lookup_index < int(label_len))
            if not np.any(valid_mask):
                return None, np.empty((0,), dtype=np.int64)
            valid_indices = all_indices[valid_mask]

            meta = _FileMeta(
                preprocessed_h5=h5_path,
                num_windows=num_windows,
                window_index=window_index,
                label_lookup_index=label_lookup_index,
                label_dataset_path=label_ds_path,
                label_length=int(label_len),
                embedded_label_dataset_path=label_ds_path,
            )
            return meta, valid_indices.astype(np.int64, copy=False)

    def __getitem__(self, index: int):
        file_idx, center_window = self.samples[index]
        meta = self.files[file_idx]

        pre_h5 = self._get_pre_h5(meta.preprocessed_h5)
        voxels_ds = pre_h5["voxels"]

        clip_indices = _build_centered_clip_indices(
            center_idx=int(center_window),
            num_total=meta.num_windows,
            num_frames=self.clip_num_frames,
            frame_stride=self.clip_frame_stride,
        )
        clip_tchw = _read_voxels_tchw(voxels_ds, clip_indices)  # [T,C,H,W]
        _, _, raw_h, raw_w = clip_tchw.shape
        clip_cthw = torch.from_numpy(clip_tchw).permute(1, 0, 2, 3).contiguous()  # [C,T,H,W]
        clip_cthw = _resize_clip_cthw(
            clip_cthw,
            target_hw=self.input_size,
            mode=self.input_resize_mode,
        )
        target_h, target_w = int(clip_cthw.shape[-2]), int(clip_cthw.shape[-1])

        if self.dataset_kind == "dsec":
            assert meta.embedded_segmentation_dataset_path is not None
            label_np = np.asarray(pre_h5[meta.embedded_segmentation_dataset_path][int(center_window)])
            label_np = _squeeze_to_hw(label_np, target="semantic")
            label = torch.from_numpy(label_np.astype(np.int64, copy=False))
            eval_label = _pad_or_crop_semantic_label_to_hw(
                label,
                target_h=raw_h,
                target_w=raw_w,
                pad_value=self.ignore_index,
            )
            label = _resize_label_to_hw(
                eval_label,
                target_h=target_h,
                target_w=target_w,
                target="semantic",
            )
            sample = {
                "input": clip_cthw.to(torch.float32),
                "target": label.to(torch.int64),
                "is_semantic": True,
            }
            if self.return_eval_target:
                sample["eval_target"] = eval_label.to(torch.int64)
            return sample

        assert meta.window_index is not None
        assert meta.label_lookup_index is not None
        assert meta.label_dataset_path is not None

        label_idx = int(meta.label_lookup_index[int(center_window)])
        assert meta.embedded_label_dataset_path is not None
        label_arr = np.asarray(pre_h5[meta.embedded_label_dataset_path][label_idx])
        label_arr = _squeeze_to_hw(label_arr, target=self.target)

        if self.target == "semantic":
            label = torch.from_numpy(label_arr.astype(np.int64, copy=False))
            eval_label = _resize_label_to_hw(
                label,
                target_h=raw_h,
                target_w=raw_w,
                target="semantic",
            )
            label = _resize_label_to_hw(
                eval_label,
                target_h=target_h,
                target_w=target_w,
                target="semantic",
            )
            sample = {
                "input": clip_cthw.to(torch.float32),
                "target": label.to(torch.int64),
                "is_semantic": True,
            }
            if self.return_eval_target:
                sample["eval_target"] = eval_label.to(torch.int64)
            return sample

        depth = torch.from_numpy(label_arr.astype(np.float32, copy=False) * self.depth_scale)
        eval_depth = _resize_label_to_hw(
            depth,
            target_h=raw_h,
            target_w=raw_w,
            target="depth",
        )
        depth = _resize_label_to_hw(
            eval_depth,
            target_h=target_h,
            target_w=target_w,
            target="depth",
        )
        sample = {
            "input": clip_cthw.to(torch.float32),
            "target": depth.to(torch.float32),
            "is_semantic": False,
        }
        if self.return_eval_target:
            sample["eval_target"] = eval_depth.to(torch.float32)
        return sample


@dataclass(frozen=True)
class _RawM3EDSemanticMeta:
    # Keep this compatibility name so visualization utilities can display
    # dataset.files[file_idx].preprocessed_h5 for both preprocessed and raw data.
    preprocessed_h5: Path
    semantics_h5: Path
    num_windows: int
    event_group_path: str
    t_offset_us: int
    anchors_us: np.ndarray
    window_starts_us: np.ndarray
    window_ends_us: np.ndarray
    label_dataset_path: str


class M3EDRawSemanticDataset(Dataset):
    """
    Dense semantic downstream dataset for downloaded M3ED layout.

    Event representations are generated on demand from ``<sequence>_data.h5``.
    Labels are read from the sibling ``<sequence>_semantics.h5`` file, usually
    from ``/predictions`` aligned by ``/ts``.
    """

    def __init__(
        self,
        *,
        roots: Sequence[str | Path],
        clip_num_frames: int,
        clip_frame_stride: int = 1,
        file_pattern: str = "*_data.h5",
        recursive: bool = True,
        ignore_index: int = 255,
        require_labels: bool = True,
        input_size: Sequence[int] | int | None = None,
        input_resize_mode: str = "bilinear",
        return_eval_target: bool = False,
        max_open_h5_files: int = 8,
        event_camera: str = "left",
        semantics_suffix: str | None = None,
        semantics_ts_path: str = "ts",
        semantics_ts_divisor: int = 1,
        label_dataset_path: str = "predictions",
        label_remap: str = "cityscapes_19_to_11",
        window_mode: str = "semantics_middle",
        accum_time_us: int = 50_000,
        representation: str = "voxel_grid",
        input_height: int = 720,
        input_width: int = 1280,
        output_height: int = 720,
        output_width: int = 1280,
        downsample_factor: int = 2,
        t_bins: int = 10,
        split_polarity: bool = True,
        normalize: bool = True,
        use_trilinear: bool = False,
        output_dtype: str = "float16",
        event_image_percentile: float = 99.0,
        sequence_include: Sequence[str] | str | None = None,
        sequence_exclude: Sequence[str] | str | None = None,
        sequence_ranges: Mapping[str, object] | Sequence[Mapping[str, object]] | None = None,
        sample_start_fraction: float = 0.0,
        sample_stop_fraction: float = 1.0,
        sample_stride: int = 1,
    ):
        if isinstance(roots, (str, Path)):
            roots = [roots]
        if len(roots) == 0:
            raise ValueError("roots must be non-empty for M3ED raw semantic dataset")
        if int(clip_num_frames) <= 0 or int(clip_frame_stride) <= 0:
            raise ValueError("clip_num_frames and clip_frame_stride must be > 0")
        if representation not in {"voxel_grid", "event_image"}:
            raise ValueError("representation must be 'voxel_grid' or 'event_image'")
        if window_mode not in {"semantics_middle", "fixed_before", "fixed_after"}:
            raise ValueError(
                "window_mode must be one of "
                "{'semantics_middle', 'fixed_before', 'fixed_after'}"
            )
        if int(accum_time_us) <= 0:
            raise ValueError("accum_time_us must be > 0")
        if int(downsample_factor) not in {1, 2}:
            raise ValueError("downsample_factor must be 1 or 2")
        if int(t_bins) <= 0:
            raise ValueError("t_bins must be > 0")
        if output_dtype not in {"float16", "float32"}:
            raise ValueError("output_dtype must be 'float16' or 'float32'")

        self.roots = [Path(root).expanduser() for root in roots]
        self.clip_num_frames = int(clip_num_frames)
        self.clip_frame_stride = int(clip_frame_stride)
        self.ignore_index = int(ignore_index)
        self.require_labels = bool(require_labels)
        self.input_size = _to_optional_hw_tuple(input_size, field_name="input_size")
        self.input_resize_mode = str(input_resize_mode).lower()
        self.return_eval_target = bool(return_eval_target)
        self.max_open_h5_files = max(1, int(max_open_h5_files))
        self.event_camera = str(event_camera).strip().lower()
        self.event_group_path = f"prophesee/{self.event_camera}"
        self.semantics_suffix = (
            str(semantics_suffix)
            if semantics_suffix is not None
            else (
                "semantics"
                if self.event_camera == "left"
                else f"semantics_{self.event_camera}"
            )
        )
        self.semantics_ts_path = str(semantics_ts_path).strip()
        self.semantics_ts_divisor = int(semantics_ts_divisor)
        if self.semantics_ts_divisor <= 0:
            raise ValueError("semantics_ts_divisor must be > 0")
        self.label_dataset_path = str(label_dataset_path).strip()
        self.label_remap = str(label_remap)
        self.window_mode = str(window_mode)
        self.accum_time_us = int(accum_time_us)
        self.representation = str(representation)
        self.input_height = int(input_height)
        self.input_width = int(input_width)
        self.downsample_factor = int(downsample_factor)
        self.t_bins = int(t_bins)
        self.split_polarity = bool(split_polarity)
        self.normalize = bool(normalize)
        self.use_trilinear = bool(use_trilinear)
        self.output_dtype = str(output_dtype)
        self.event_image_percentile = float(event_image_percentile)
        self.sequence_include = _normalise_optional_sequence_names(sequence_include)
        self.sequence_exclude = _normalise_optional_sequence_names(sequence_exclude)
        self.sequence_ranges = _normalise_sequence_ranges(sequence_ranges)
        self.sample_start_fraction = float(sample_start_fraction)
        self.sample_stop_fraction = float(sample_stop_fraction)
        self.sample_stride = max(1, int(sample_stride))

        if self.input_height <= 0 or self.input_width <= 0:
            raise ValueError("input_height and input_width must be > 0")
        if self.downsample_factor == 1:
            self.output_height = int(output_height)
            self.output_width = int(output_width)
        else:
            if (
                self.input_height % self.downsample_factor != 0
                or self.input_width % self.downsample_factor != 0
            ):
                raise ValueError(
                    "M3ED input resolution must be divisible by downsample_factor"
                )
            self.output_height = self.input_height // self.downsample_factor
            self.output_width = self.input_width // self.downsample_factor
        if self.output_height <= 0 or self.output_width <= 0:
            raise ValueError("resolved output resolution must be > 0")

        self.output_channels = (
            self.t_bins * (2 if self.split_polarity else 1)
            if self.representation == "voxel_grid"
            else 3
        )
        self._voxelizer = (
            EventVoxelGrid(
                input_size=(self.t_bins, self.output_height, self.output_width),
                normalize=self.normalize,
                separate_polarity=self.split_polarity,
                trilinear_interpolation=self.use_trilinear,
            )
            if self.representation == "voxel_grid"
            else None
        )

        self.files: list[_RawM3EDSemanticMeta] = []
        self.samples: list[tuple[int, int]] = []
        self._event_h5_cache: OrderedDict[str, h5py.File] = OrderedDict()
        self._semantics_h5_cache: OrderedDict[str, h5py.File] = OrderedDict()
        self._ms_idx_cache: OrderedDict[str, np.ndarray | None] = OrderedDict()

        for root in self.roots:
            for event_path in _discover_m3ed_h5_files(
                root,
                file_pattern=file_pattern,
                recursive=bool(recursive),
            ):
                if not _sequence_is_selected(
                    event_path,
                    include=self.sequence_include,
                    exclude=self.sequence_exclude,
                ):
                    continue
                sequence_range = _sequence_range_for_path(
                    event_path,
                    self.sequence_ranges,
                )
                if self.sequence_ranges and sequence_range is None:
                    continue
                semantics_path = _resolve_companion_h5(
                    event_path,
                    suffix=self.semantics_suffix,
                )
                if semantics_path is None:
                    if self.require_labels:
                        continue
                    continue
                meta, valid_indices = self._build_file_meta(
                    event_path=event_path,
                    semantics_path=semantics_path,
                    sequence_range=sequence_range,
                )
                if meta is None or valid_indices.size == 0:
                    continue
                file_idx = len(self.files)
                self.files.append(meta)
                self.samples.extend(
                    (file_idx, int(label_idx)) for label_idx in valid_indices.tolist()
                )

        if len(self.files) == 0 or len(self.samples) == 0:
            raise RuntimeError(
                "No valid raw M3ED semantic samples found. "
                "Check *_data.h5 paths, sibling *_semantics.h5 files, "
                "sequence filters, and label dataset names."
            )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_event_h5_cache"] = OrderedDict()
        state["_semantics_h5_cache"] = OrderedDict()
        state["_ms_idx_cache"] = OrderedDict()
        return state

    def __del__(self):
        for cache_name in ("_event_h5_cache", "_semantics_h5_cache"):
            cache = getattr(self, cache_name, None)
            if isinstance(cache, dict):
                for h5f in cache.values():
                    try:
                        h5f.close()
                    except Exception:
                        pass
                cache.clear()

    def __len__(self) -> int:
        return len(self.samples)

    def _build_file_meta(
        self,
        *,
        event_path: Path,
        semantics_path: Path,
        sequence_range: _SequenceRange | None,
    ) -> tuple[_RawM3EDSemanticMeta | None, np.ndarray]:
        with h5py.File(str(event_path), "r") as event_h5, h5py.File(str(semantics_path), "r") as sem_h5:
            if self.event_group_path not in event_h5:
                return None, np.empty((0,), dtype=np.int64)
            events = event_h5[self.event_group_path]
            if not all(key in events for key in ("x", "y", "t", "p")):
                return None, np.empty((0,), dtype=np.int64)
            if self.semantics_ts_path not in sem_h5 or self.label_dataset_path not in sem_h5:
                return None, np.empty((0,), dtype=np.int64)

            event_t = events["t"]
            label_ds = sem_h5[self.label_dataset_path]
            if int(len(event_t)) == 0 or label_ds.ndim < 3 or int(label_ds.shape[0]) <= 0:
                return None, np.empty((0,), dtype=np.int64)

            semantics_ts = np.asarray(
                sem_h5[self.semantics_ts_path][()],
                dtype=np.int64,
            ).reshape(-1)
            if semantics_ts.size == 0:
                return None, np.empty((0,), dtype=np.int64)
            if self.semantics_ts_divisor != 1:
                semantics_ts = np.floor_divide(
                    semantics_ts,
                    self.semantics_ts_divisor,
                ).astype(np.int64, copy=False)

            num_windows = min(int(label_ds.shape[0]), int(semantics_ts.size))
            if num_windows <= 0:
                return None, np.empty((0,), dtype=np.int64)
            semantics_ts = semantics_ts[:num_windows]

            t_offset_us = _read_t_offset(event_h5)
            t_first_us = int(event_t[0]) + int(t_offset_us)
            t_last_exclusive_us = int(event_t[-1]) + int(t_offset_us) + 1
            anchors_us = _align_anchor_timebase_to_events(
                semantics_ts,
                t_first_us=t_first_us,
                t_last_exclusive_us=t_last_exclusive_us,
                t_offset_us=int(t_offset_us),
            )

            if self.window_mode == "semantics_middle":
                starts_us, ends_us = _build_middle_windows(
                    anchors_us,
                    t_first_us=t_first_us,
                    t_last_exclusive_us=t_last_exclusive_us,
                )
            elif self.window_mode == "fixed_before":
                starts_us = anchors_us - int(self.accum_time_us)
                ends_us = anchors_us.copy()
                np.maximum(starts_us, int(t_first_us), out=starts_us)
                np.minimum(ends_us, int(t_last_exclusive_us), out=ends_us)
            else:
                starts_us = anchors_us.copy()
                ends_us = anchors_us + int(self.accum_time_us)
                np.maximum(starts_us, int(t_first_us), out=starts_us)
                np.minimum(ends_us, int(t_last_exclusive_us), out=ends_us)

        if anchors_us.size != num_windows or starts_us.size != num_windows or ends_us.size != num_windows:
            return None, np.empty((0,), dtype=np.int64)

        valid = np.nonzero(ends_us > starts_us)[0].astype(np.int64, copy=False)
        sample_start_fraction = self.sample_start_fraction
        sample_stop_fraction = self.sample_stop_fraction
        sample_stride = self.sample_stride
        if sequence_range is not None:
            sample_start_fraction = sequence_range.start_fraction
            sample_stop_fraction = sequence_range.stop_fraction
            sample_stride = sequence_range.stride
        valid = _fractional_subset(
            valid,
            start_fraction=sample_start_fraction,
            stop_fraction=sample_stop_fraction,
            stride=sample_stride,
        )
        if valid.size == 0:
            return None, np.empty((0,), dtype=np.int64)

        meta = _RawM3EDSemanticMeta(
            preprocessed_h5=event_path,
            semantics_h5=semantics_path,
            num_windows=int(num_windows),
            event_group_path=self.event_group_path,
            t_offset_us=int(t_offset_us),
            anchors_us=anchors_us.astype(np.int64, copy=False),
            window_starts_us=starts_us.astype(np.int64, copy=False),
            window_ends_us=ends_us.astype(np.int64, copy=False),
            label_dataset_path=self.label_dataset_path,
        )
        return meta, valid

    def _get_h5_from_cache(
        self,
        cache: OrderedDict[str, h5py.File],
        path: Path,
    ) -> h5py.File:
        key = str(path)
        h5f = cache.get(key)
        if h5f is not None:
            cache.move_to_end(key, last=True)
            return h5f

        h5f = h5py.File(key, "r")
        cache[key] = h5f
        while len(cache) > self.max_open_h5_files:
            old_key, old_h5 = cache.popitem(last=False)
            try:
                old_h5.close()
            except Exception:
                pass
            self._ms_idx_cache.pop(old_key, None)
        return h5f

    def _get_event_h5(self, path: Path) -> h5py.File:
        return self._get_h5_from_cache(self._event_h5_cache, path)

    def _get_semantics_h5(self, path: Path) -> h5py.File:
        return self._get_h5_from_cache(self._semantics_h5_cache, path)

    def _get_ms_idx(self, meta: _RawM3EDSemanticMeta) -> np.ndarray | None:
        key = str(meta.preprocessed_h5)
        if key in self._ms_idx_cache:
            value = self._ms_idx_cache[key]
            self._ms_idx_cache.move_to_end(key, last=True)
            return value

        h5f = self._get_event_h5(meta.preprocessed_h5)
        events = h5f[meta.event_group_path]
        if "ms_map_idx" in events:
            value = np.asarray(events["ms_map_idx"], dtype=np.int64)
        elif "ms_to_idx" in h5f:
            value = np.asarray(h5f["ms_to_idx"], dtype=np.int64)
        else:
            value = None
        self._ms_idx_cache[key] = value
        while len(self._ms_idx_cache) > self.max_open_h5_files:
            self._ms_idx_cache.popitem(last=False)
        return value

    def _extract_events_by_time(
        self,
        meta: _RawM3EDSemanticMeta,
        *,
        start_us: int,
        end_us: int,
    ) -> dict[str, np.ndarray]:
        empty = {
            "x": np.empty((0,), dtype=np.float32),
            "y": np.empty((0,), dtype=np.float32),
            "p": np.empty((0,), dtype=np.float32),
            "t": np.empty((0,), dtype=np.int64),
        }
        if end_us <= start_us:
            return empty

        h5f = self._get_event_h5(meta.preprocessed_h5)
        events = h5f[meta.event_group_path]
        timestamps = events["t"]
        num_events = int(len(timestamps))
        if num_events == 0:
            return empty

        ms_idx = self._get_ms_idx(meta)
        if ms_idx is not None and ms_idx.size > 0:
            start_rel_us = int(start_us) - int(meta.t_offset_us)
            end_rel_us = int(end_us) - int(meta.t_offset_us)
            if end_rel_us <= 0:
                return empty
            start_ms = max(start_rel_us // 1000, 0)
            end_ms_exclusive = max((end_rel_us + 999) // 1000, start_ms + 1)
            start_ms = min(start_ms, int(ms_idx.size) - 1)
            coarse_start = int(ms_idx[start_ms])
            coarse_end = (
                num_events
                if end_ms_exclusive >= int(ms_idx.size)
                else int(ms_idx[end_ms_exclusive])
            )
            coarse_start = max(0, min(coarse_start, num_events))
            coarse_end = max(0, min(coarse_end, num_events))
            if coarse_end <= coarse_start:
                return empty
            t_coarse = (
                np.asarray(timestamps[coarse_start:coarse_end], dtype=np.int64)
                + int(meta.t_offset_us)
            )
            rel_start = int(np.searchsorted(t_coarse, start_us, side="left"))
            rel_end = int(np.searchsorted(t_coarse, end_us, side="left"))
            event_start = coarse_start + rel_start
            event_end = coarse_start + rel_end
            selected_t = t_coarse[rel_start:rel_end]
        else:
            start_rel_us = int(start_us) - int(meta.t_offset_us)
            end_rel_us = int(end_us) - int(meta.t_offset_us)
            event_start = _h5_searchsorted_left(timestamps, start_rel_us)
            event_end = _h5_searchsorted_left(timestamps, end_rel_us)
            selected_t = (
                np.asarray(timestamps[event_start:event_end], dtype=np.int64)
                + int(meta.t_offset_us)
            )

        if event_end <= event_start:
            return empty
        return {
            "x": np.asarray(events["x"][event_start:event_end]),
            "y": np.asarray(events["y"][event_start:event_end]),
            "p": np.asarray(events["p"][event_start:event_end]),
            "t": selected_t,
        }

    def _normalize_event_coordinates(
        self,
        events: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        if events["t"].size == 0:
            return {
                "x": np.empty((0,), dtype=np.float32),
                "y": np.empty((0,), dtype=np.float32),
                "p": np.empty((0,), dtype=np.float32),
                "t": np.empty((0,), dtype=np.int64),
            }

        x = np.asarray(events["x"], dtype=np.float32)
        y = np.asarray(events["y"], dtype=np.float32)
        p = np.asarray(events["p"])
        t = np.asarray(events["t"], dtype=np.int64)
        valid = (
            (x >= 0)
            & (x < float(self.input_width))
            & (y >= 0)
            & (y < float(self.input_height))
        )
        if not np.any(valid):
            return {
                "x": np.empty((0,), dtype=np.float32),
                "y": np.empty((0,), dtype=np.float32),
                "p": np.empty((0,), dtype=np.float32),
                "t": np.empty((0,), dtype=np.int64),
            }

        x = x[valid]
        y = y[valid]
        p = p[valid]
        t = t[valid]
        if self.output_width == self.input_width and self.output_height == self.input_height:
            x_out = x
            y_out = y
        elif self.input_width % self.output_width == 0 and self.input_height % self.output_height == 0:
            x_out = np.floor(x / float(self.input_width // self.output_width))
            y_out = np.floor(y / float(self.input_height // self.output_height))
        else:
            x_out = np.floor(x * (float(self.output_width) / float(self.input_width)))
            y_out = np.floor(y * (float(self.output_height) / float(self.input_height)))

        valid_out = (
            (x_out >= 0)
            & (x_out < float(self.output_width))
            & (y_out >= 0)
            & (y_out < float(self.output_height))
        )
        return {
            "x": x_out[valid_out].astype(np.float32, copy=False),
            "y": y_out[valid_out].astype(np.float32, copy=False),
            "p": (p[valid_out] > 0).astype(np.float32, copy=False),
            "t": t[valid_out].astype(np.int64, copy=False),
        }

    def _make_representation(
        self,
        meta: _RawM3EDSemanticMeta,
        window_idx: int,
    ) -> torch.Tensor:
        events = self._extract_events_by_time(
            meta,
            start_us=int(meta.window_starts_us[window_idx]),
            end_us=int(meta.window_ends_us[window_idx]),
        )
        events = self._normalize_event_coordinates(events)

        if self.representation == "voxel_grid":
            assert self._voxelizer is not None
            if events["t"].size == 0:
                representation = torch.zeros(
                    (self.output_channels, self.output_height, self.output_width),
                    dtype=torch.float32,
                )
            else:
                shifted_t = (events["t"] - events["t"][0]).astype(np.float32, copy=False)
                representation = self._voxelizer.convert(
                    {
                        "x": torch.from_numpy(events["x"]),
                        "y": torch.from_numpy(events["y"]),
                        "p": torch.from_numpy(events["p"]),
                        "t": torch.from_numpy(shifted_t),
                    }
                ).cpu()
        else:
            image, _ = accumulate_events_to_rgb(
                events["x"],
                events["y"],
                events["p"],
                (self.output_height, self.output_width),
                percentile=self.event_image_percentile,
                dtype=np.float32,
            )
            representation = torch.from_numpy(image)

        representation = representation.to(torch.float32)
        if self.output_dtype == "float16":
            representation = representation.to(torch.float16).to(torch.float32)
        return representation.contiguous()

    def _read_semantic_label(
        self,
        meta: _RawM3EDSemanticMeta,
        label_idx: int,
    ) -> torch.Tensor:
        sem_h5 = self._get_semantics_h5(meta.semantics_h5)
        label_arr = np.asarray(sem_h5[meta.label_dataset_path][int(label_idx)])
        label_arr = _squeeze_to_hw(label_arr, target="semantic")
        label_arr = _apply_semantic_label_remap(
            label_arr,
            remap=self.label_remap,
            ignore_index=self.ignore_index,
        )
        return torch.from_numpy(label_arr.astype(np.int64, copy=False))

    def __getitem__(self, index: int):
        file_idx, center_window = self.samples[index]
        meta = self.files[file_idx]

        clip_indices = _build_centered_clip_indices(
            center_idx=int(center_window),
            num_total=meta.num_windows,
            num_frames=self.clip_num_frames,
            frame_stride=self.clip_frame_stride,
        )
        frame_cache: dict[int, torch.Tensor] = {}
        frames: list[torch.Tensor] = []
        for window_idx in clip_indices.tolist():
            cached = frame_cache.get(int(window_idx))
            if cached is None:
                cached = self._make_representation(meta, int(window_idx))
                frame_cache[int(window_idx)] = cached
            frames.append(cached)

        clip_tchw = torch.stack(frames, dim=0)
        clip_cthw = clip_tchw.permute(1, 0, 2, 3).contiguous()
        clip_cthw = _resize_clip_cthw(
            clip_cthw,
            target_hw=self.input_size,
            mode=self.input_resize_mode,
        )
        target_h, target_w = int(clip_cthw.shape[-2]), int(clip_cthw.shape[-1])

        eval_label = self._read_semantic_label(meta, int(center_window))
        label = _resize_label_like_dense_target(
            eval_label,
            target_h=target_h,
            target_w=target_w,
            target="semantic",
        )
        sample = {
            "input": clip_cthw.to(torch.float32),
            "target": label.to(torch.int64),
            "is_semantic": True,
        }
        if self.return_eval_target:
            sample["eval_target"] = eval_label.to(torch.int64)
        return sample


def _get_split_cfg_value(
    cfg_task: dict,
    split: str,
    key: str,
    default=None,
):
    split_key = f"{split}_{key}"
    if split_key in cfg_task and cfg_task.get(split_key) is not None:
        return cfg_task.get(split_key)
    return cfg_task.get(key, default)


def build_dense_task_dataset_from_config(
    *,
    cfg_task: dict,
    roots: Sequence[str | Path],
    split: str,
    return_eval_target: bool,
) -> Dataset:
    target = str(cfg_task.get("target", "semantic")).lower()
    dataset_kind = str(cfg_task.get("dataset_kind", "dsec")).lower()
    common_kwargs = {
        "roots": roots,
        "clip_num_frames": int(cfg_task.get("clip_num_frames", 2)),
        "clip_frame_stride": int(cfg_task.get("clip_frame_stride", 1)),
        "file_pattern": str(
            cfg_task.get(
                "file_pattern",
                "*_data.h5" if dataset_kind == "m3ed_raw" else "*.h5",
            )
        ),
        "recursive": bool(cfg_task.get("recursive", True)),
        "ignore_index": int(cfg_task.get("ignore_index", 255)),
        "require_labels": bool(cfg_task.get("require_labels", True)),
        "input_size": cfg_task.get("input_size", None),
        "input_resize_mode": str(cfg_task.get("input_resize_mode", "bilinear")),
        "return_eval_target": bool(return_eval_target),
    }

    if dataset_kind == "m3ed_raw":
        if target != "semantic":
            raise ValueError("dataset_kind=m3ed_raw currently supports target=semantic only")
        return M3EDRawSemanticDataset(
            **common_kwargs,
            max_open_h5_files=int(cfg_task.get("max_open_h5_files", 8)),
            event_camera=str(cfg_task.get("event_camera", "left")),
            semantics_suffix=cfg_task.get("semantics_suffix", None),
            semantics_ts_path=str(cfg_task.get("semantics_ts_path", "ts")),
            semantics_ts_divisor=int(cfg_task.get("semantics_ts_divisor", 1)),
            label_dataset_path=str(cfg_task.get("label_dataset_path", "predictions")),
            label_remap=str(cfg_task.get("label_remap", "cityscapes_19_to_11")),
            window_mode=str(cfg_task.get("window_mode", "semantics_middle")),
            accum_time_us=int(cfg_task.get("accum_time_us", 50_000)),
            representation=str(cfg_task.get("representation", "voxel_grid")),
            input_height=int(cfg_task.get("input_height", 720)),
            input_width=int(cfg_task.get("input_width", 1280)),
            output_height=int(cfg_task.get("output_height", 720)),
            output_width=int(cfg_task.get("output_width", 1280)),
            downsample_factor=int(cfg_task.get("downsample_factor", 2)),
            t_bins=int(cfg_task.get("t_bins", 10)),
            split_polarity=bool(cfg_task.get("split_polarity", True)),
            normalize=bool(cfg_task.get("normalize", True)),
            use_trilinear=bool(cfg_task.get("use_trilinear", False)),
            output_dtype=str(cfg_task.get("output_dtype", "float16")),
            event_image_percentile=float(cfg_task.get("event_image_percentile", 99.0)),
            sequence_include=_get_split_cfg_value(cfg_task, split, "sequence_include", None),
            sequence_exclude=_get_split_cfg_value(cfg_task, split, "sequence_exclude", None),
            sequence_ranges=_get_split_cfg_value(cfg_task, split, "sequence_ranges", None),
            sample_start_fraction=float(
                _get_split_cfg_value(cfg_task, split, "sample_start_fraction", 0.0)
            ),
            sample_stop_fraction=float(
                _get_split_cfg_value(cfg_task, split, "sample_stop_fraction", 1.0)
            ),
            sample_stride=int(
                _get_split_cfg_value(cfg_task, split, "sample_stride", 1)
            ),
        )

    return EventDenseTaskDataset(
        **common_kwargs,
        dataset_kind=dataset_kind,
        target=target,
        depth_scale=float(cfg_task.get("depth_scale", 1.0)),
    )
