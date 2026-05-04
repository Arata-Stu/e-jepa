from __future__ import annotations

import bisect
import csv
from pathlib import Path
from typing import Callable, Sequence

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

from .weighted_sampler import DistributedWeightedSampler


class ConcatIndices:
    """Map a global index to (dataset_idx, local_idx) for concatenated datasets."""

    def __init__(self, sizes: Sequence[int]):
        if len(sizes) == 0:
            raise ValueError("sizes must be non-empty")
        self.cumulative_sizes = np.cumsum(sizes)

    def __len__(self) -> int:
        return int(self.cumulative_sizes[-1])

    def __getitem__(self, idx: int) -> tuple[int, int]:
        if idx < 0 or idx >= len(self):
            raise ValueError(f"index out of bounds: idx={idx}, size={len(self)}")
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            return 0, idx
        return dataset_idx, idx - int(self.cumulative_sizes[dataset_idx - 1])


def _parse_manifest_paths(manifest_path: Path) -> list[Path]:
    """Accepts CSV/TXT manifests and returns the first column/token as a path."""
    rows: list[Path] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for raw in reader:
            if len(raw) == 0:
                continue
            first = raw[0].strip()
            if not first:
                continue
            if first.startswith("#"):
                continue
            # Handle plain whitespace-delimited manifests too.
            token = first.split()[0]
            p = Path(token)
            if not p.is_absolute():
                p = (manifest_path.parent / p).resolve()
            rows.append(p)
    return rows


def _is_voxel_h5(path: Path) -> bool:
    try:
        with h5py.File(str(path), "r") as h5f:
            if "voxels" not in h5f:
                return False
            ds = h5f["voxels"]
            return ds.ndim == 4 and ds.shape[0] > 0
    except Exception:
        return False


def _discover_h5_files(
    data_path: Path,
    *,
    file_pattern: str,
    recursive: bool,
    require_voxels_key: bool,
) -> list[Path]:
    suffix = data_path.suffix.lower()
    files: list[Path] = []

    if suffix in {".csv", ".txt"}:
        files = _parse_manifest_paths(data_path)
    elif suffix in {".h5", ".hdf5"} and data_path.is_file():
        files = [data_path.resolve()]
    elif data_path.is_dir():
        iterator = data_path.rglob(file_pattern) if recursive else data_path.glob(file_pattern)
        files = sorted(p.resolve() for p in iterator if p.is_file())
    else:
        raise FileNotFoundError(f"Unsupported data path: {data_path}")

    # If file_pattern is broad (*.h5), keep only preprocessed voxel H5.
    if require_voxels_key:
        files = [p for p in files if _is_voxel_h5(p)]
    return files


def _to_cthw_tensor(buffer: np.ndarray) -> torch.Tensor:
    """
    Convert [T, H, W, C] -> [C, T, H, W].
    """
    if buffer.ndim != 4:
        raise ValueError(f"buffer must be 4D [T,H,W,C], got shape={buffer.shape}")
    tensor = torch.from_numpy(buffer).to(torch.float32)
    return tensor.permute(3, 0, 1, 2).contiguous()


class EventVideoDataset(torch.utils.data.Dataset):
    """
    Stage-1 event representation dataset for JEPA-style pretraining.

    Output format is intentionally aligned with vjepa2 VideoDataset:
      (buffer, label, clip_indices)
    where `buffer` is a list of clips, each transformed to [C, T, H, W].
    """

    def __init__(
        self,
        data_paths: str | Path | Sequence[str | Path],
        *,
        datasets_weights: Sequence[float] | None = None,
        frames_per_clip: int = 8,
        dataset_fpcs: Sequence[int] | None = None,
        frame_step: int = 1,
        num_clips: int = 1,
        transform: Callable | None = None,
        shared_transform: Callable | None = None,
        random_clip_sampling: bool = True,
        allow_clip_overlap: bool = False,
        file_pattern: str = "*.h5",
        recursive: bool = True,
        require_voxels_key: bool = True,
    ):
        if isinstance(data_paths, (str, Path)):
            data_paths = [data_paths]
        if len(data_paths) == 0:
            raise ValueError("data_paths must be non-empty")
        if frame_step <= 0:
            raise ValueError("frame_step must be > 0")
        if num_clips <= 0:
            raise ValueError("num_clips must be > 0")
        if frames_per_clip <= 0:
            raise ValueError("frames_per_clip must be > 0")

        self.data_paths = [Path(p).expanduser() for p in data_paths]
        self.datasets_weights = datasets_weights
        self.frame_step = int(frame_step)
        self.num_clips = int(num_clips)
        self.transform = transform
        self.shared_transform = shared_transform
        self.random_clip_sampling = bool(random_clip_sampling)
        self.allow_clip_overlap = bool(allow_clip_overlap)

        if dataset_fpcs is None:
            self.dataset_fpcs = [int(frames_per_clip) for _ in self.data_paths]
        else:
            if len(dataset_fpcs) != len(self.data_paths):
                raise ValueError("dataset_fpcs length must match data_paths length")
            self.dataset_fpcs = [int(v) for v in dataset_fpcs]
        if any(fpc <= 0 for fpc in self.dataset_fpcs):
            raise ValueError("All dataset_fpcs must be > 0")

        samples: list[str] = []
        labels: list[int] = []
        self.num_samples_per_dataset: list[int] = []
        for path in self.data_paths:
            dataset_files = _discover_h5_files(
                path,
                file_pattern=file_pattern,
                recursive=recursive,
                require_voxels_key=require_voxels_key,
            )
            if len(dataset_files) == 0:
                raise FileNotFoundError(f"No voxel H5 files found in: {path}")
            samples += [str(p) for p in dataset_files]
            labels += [0] * len(dataset_files)
            self.num_samples_per_dataset.append(len(dataset_files))

        self.samples = samples
        self.labels = labels
        self.per_dataset_indices = ConcatIndices(self.num_samples_per_dataset)

        self.sample_weights: list[float] | None = None
        if self.datasets_weights is not None:
            if len(self.datasets_weights) != len(self.num_samples_per_dataset):
                raise ValueError("datasets_weights length must match number of datasets")
            self.sample_weights = []
            for dw, ns in zip(self.datasets_weights, self.num_samples_per_dataset):
                self.sample_weights += [float(dw) / float(ns)] * int(ns)

        # Worker-local lazy cache. We intentionally avoid sharing handles.
        self._h5_cache: dict[str, h5py.File] = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5_cache"] = {}
        return state

    def __del__(self):
        cache = getattr(self, "_h5_cache", None)
        if isinstance(cache, dict):
            for h5f in cache.values():
                try:
                    h5f.close()
                except Exception:
                    pass
            cache.clear()

    def __len__(self) -> int:
        return len(self.samples)

    def _get_h5(self, sample_path: str) -> h5py.File:
        h5f = self._h5_cache.get(sample_path)
        if h5f is None:
            h5f = h5py.File(sample_path, "r")
            self._h5_cache[sample_path] = h5f
        return h5f

    @staticmethod
    def _fit_indices_length(indices: np.ndarray, target_len: int, last_valid: int) -> np.ndarray:
        if indices.size >= target_len:
            pos = np.linspace(0, indices.size - 1, num=target_len, dtype=np.float64)
            pos = np.round(pos).astype(np.int64)
            return indices[pos]
        pad = np.full((target_len - indices.size,), int(last_valid), dtype=np.int64)
        return np.concatenate([indices, pad], axis=0)

    def _sample_clip_in_segment(
        self,
        *,
        segment_start: int,
        segment_length: int,
        total_windows: int,
        fpc: int,
    ) -> np.ndarray:
        if total_windows <= 0:
            return np.zeros((fpc,), dtype=np.int64)

        if segment_length <= 0:
            anchor = min(max(segment_start, 0), total_windows - 1)
            return np.full((fpc,), anchor, dtype=np.int64)

        clip_span = max(1, fpc * self.frame_step)
        if segment_length > clip_span:
            max_local_start = segment_length - clip_span
            if self.random_clip_sampling:
                local_start = int(np.random.randint(0, max_local_start + 1))
            else:
                local_start = max_local_start // 2
            local_indices = np.arange(
                local_start,
                local_start + clip_span,
                self.frame_step,
                dtype=np.int64,
            )
        else:
            local_indices = np.arange(0, segment_length, self.frame_step, dtype=np.int64)

        if local_indices.size == 0:
            local_indices = np.array([0], dtype=np.int64)
        local_indices = self._fit_indices_length(
            local_indices,
            target_len=fpc,
            last_valid=max(0, segment_length - 1),
        )

        global_indices = segment_start + local_indices
        np.clip(global_indices, 0, total_windows - 1, out=global_indices)
        return global_indices.astype(np.int64, copy=False)

    def _sample_clip_indices(self, total_windows: int, fpc: int) -> list[np.ndarray]:
        if self.num_clips == 1:
            return [
                self._sample_clip_in_segment(
                    segment_start=0,
                    segment_length=total_windows,
                    total_windows=total_windows,
                    fpc=fpc,
                )
            ]

        all_clip_indices: list[np.ndarray] = []
        if not self.allow_clip_overlap:
            partition_len = max(1, total_windows // self.num_clips)
            for clip_id in range(self.num_clips):
                start = clip_id * partition_len
                end = total_windows if clip_id == self.num_clips - 1 else min(total_windows, (clip_id + 1) * partition_len)
                length = max(1, end - start)
                all_clip_indices.append(
                    self._sample_clip_in_segment(
                        segment_start=start,
                        segment_length=length,
                        total_windows=total_windows,
                        fpc=fpc,
                    )
                )
        else:
            clip_span = max(1, fpc * self.frame_step)
            if self.num_clips == 1:
                starts = np.array([0], dtype=np.int64)
            elif total_windows <= clip_span:
                starts = np.linspace(0, max(0, total_windows - 1), num=self.num_clips, dtype=np.float64)
                starts = np.round(starts).astype(np.int64)
            else:
                starts = np.linspace(0, total_windows - clip_span, num=self.num_clips, dtype=np.float64)
                starts = np.round(starts).astype(np.int64)
            for start in starts.tolist():
                seg_len = min(total_windows - start, clip_span)
                all_clip_indices.append(
                    self._sample_clip_in_segment(
                        segment_start=int(start),
                        segment_length=int(seg_len),
                        total_windows=total_windows,
                        fpc=fpc,
                    )
                )
        return all_clip_indices

    @staticmethod
    def _read_windows(voxels_ds: h5py.Dataset, indices: np.ndarray) -> np.ndarray:
        if indices.size == 0:
            raise ValueError("indices cannot be empty")

        if indices.size > 1 and np.all(indices[1:] - indices[:-1] == 1):
            start = int(indices[0])
            end = int(indices[-1]) + 1
            arr = np.asarray(voxels_ds[start:end])
        else:
            arr = np.stack([np.asarray(voxels_ds[int(i)]) for i in indices], axis=0)

        if arr.ndim == 3:
            arr = arr[:, np.newaxis, :, :]
        if arr.ndim != 4:
            raise ValueError(f"Unexpected voxel window shape: {arr.shape}")
        return arr.astype(np.float32, copy=False)

    def get_item_event(self, index: int):
        sample_path = self.samples[index]
        dataset_idx, _ = self.per_dataset_indices[index]
        fpc = self.dataset_fpcs[dataset_idx]

        h5f = self._get_h5(sample_path)
        if "voxels" not in h5f:
            return None
        voxels_ds = h5f["voxels"]
        if voxels_ds.ndim != 4 or voxels_ds.shape[0] <= 0:
            return None

        total_windows = int(voxels_ds.shape[0])
        clip_indices = self._sample_clip_indices(total_windows=total_windows, fpc=fpc)
        all_indices = np.concatenate(clip_indices, axis=0).astype(np.int64, copy=False)
        windows = self._read_windows(voxels_ds=voxels_ds, indices=all_indices)  # [T,C,H,W]
        windows = np.transpose(windows, (0, 2, 3, 1))  # [T,H,W,C]

        if self.shared_transform is not None:
            windows = self.shared_transform(windows)

        split_clips = [windows[i * fpc : (i + 1) * fpc] for i in range(self.num_clips)]
        if self.transform is not None:
            split_clips = [self.transform(clip) for clip in split_clips]
        else:
            split_clips = [_to_cthw_tensor(clip) for clip in split_clips]

        label = self.labels[index]
        clip_indices = [ci.astype(np.int32, copy=False) for ci in clip_indices]
        return split_clips, label, clip_indices

    def __getitem__(self, index: int):
        num_trials = 8
        for _ in range(num_trials):
            loaded = self.get_item_event(index)
            if loaded is not None:
                return loaded
            index = int(np.random.randint(0, len(self.samples)))
        raise RuntimeError(f"Failed to load event sample after {num_trials} retries.")


def make_eventdataset(
    data_paths: str | Path | Sequence[str | Path],
    batch_size: int,
    *,
    frames_per_clip: int = 8,
    dataset_fpcs: Sequence[int] | None = None,
    frame_step: int = 1,
    num_clips: int = 1,
    random_clip_sampling: bool = True,
    allow_clip_overlap: bool = False,
    transform: Callable | None = None,
    shared_transform: Callable | None = None,
    rank: int = 0,
    world_size: int = 1,
    datasets_weights: Sequence[float] | None = None,
    collator: Callable | None = None,
    drop_last: bool = True,
    num_workers: int = 10,
    pin_mem: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int | None = None,
    file_pattern: str = "*.h5",
    recursive: bool = True,
    require_voxels_key: bool = True,
):
    dataset = EventVideoDataset(
        data_paths=data_paths,
        datasets_weights=datasets_weights,
        frames_per_clip=frames_per_clip,
        dataset_fpcs=dataset_fpcs,
        frame_step=frame_step,
        num_clips=num_clips,
        transform=transform,
        shared_transform=shared_transform,
        random_clip_sampling=random_clip_sampling,
        allow_clip_overlap=allow_clip_overlap,
        file_pattern=file_pattern,
        recursive=recursive,
        require_voxels_key=require_voxels_key,
    )

    if datasets_weights is not None:
        sampler = DistributedWeightedSampler(
            dataset=dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
    else:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset=dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )

    dataloader_kwargs = dict(
        dataset=dataset,
        collate_fn=collator,
        sampler=sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
    )
    if num_workers > 0 and prefetch_factor is not None:
        dataloader_kwargs["prefetch_factor"] = int(prefetch_factor)

    data_loader = torch.utils.data.DataLoader(**dataloader_kwargs)
    return dataset, data_loader, sampler
