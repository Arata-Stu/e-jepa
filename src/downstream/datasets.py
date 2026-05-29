from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

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


TaskTarget = Literal["semantic", "depth"]
DatasetKind = Literal["dsec", "m3ed"]


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
