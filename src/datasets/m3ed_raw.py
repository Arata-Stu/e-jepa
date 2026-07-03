from __future__ import annotations

from collections import OrderedDict
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import h5py
import numpy as np
import torch

from src.representations import EventVoxelGrid, accumulate_events_to_rgb

from .event_dataset import (
    _expand_optional_per_dataset,
    _parse_manifest_paths,
)
from .weighted_sampler import DistributedWeightedSampler


EVENT_GROUP_PATH = "prophesee/left"
# Retained as an opt-in compatibility filter for the previously extracted
# semantic subset. Downloaded M3ED data can use OVC timestamps for every
# sequence, so the raw-data presets do not enable this filter by default.
KNOWN_SEMANTIC_SUPPORTED_M3ED_SEQUENCES = frozenset(
    {
        "car_urban_day_city_hall",
        "car_urban_day_horse",
        "car_urban_day_penno_big_loop",
        "car_urban_day_penno_small_loop",
        "car_urban_day_rittenhouse",
        "car_urban_day_ucity_small_loop",
    }
)


@dataclass(frozen=True)
class _M3EDSequence:
    path: str
    dataset_idx: int
    t_first_us: int
    t_last_exclusive_us: int
    t_offset_us: int
    anchors_us: np.ndarray
    window_starts_us: np.ndarray
    window_ends_us: np.ndarray


@dataclass(frozen=True)
class _VirtualChunk:
    sequence_idx: int
    dataset_idx: int
    window_start: int
    window_end: int

    @property
    def num_windows(self) -> int:
        return int(self.window_end - self.window_start)


def _sequence_name_from_event_path(input_path: Path) -> str:
    stem = input_path.stem.strip()
    for suffix in ("_data", "_left_event", "_event", "_events"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _candidate_sequence_names(input_path: Path) -> tuple[str, ...]:
    names: list[str] = []
    if input_path.parent.name.strip():
        names.append(input_path.parent.name.strip())

    stem = _sequence_name_from_event_path(input_path)
    if stem:
        names.append(stem)
    return tuple(dict.fromkeys(names))


def _is_known_semantic_sequence(input_path: Path) -> bool:
    return any(
        name in KNOWN_SEMANTIC_SUPPORTED_M3ED_SEQUENCES
        for name in _candidate_sequence_names(input_path)
    )


def _is_valid_m3ed_h5(path: Path) -> bool:
    try:
        with h5py.File(str(path), "r") as h5f:
            if EVENT_GROUP_PATH not in h5f:
                return False
            events = h5f[EVENT_GROUP_PATH]
            return all(key in events for key in ("x", "y", "t", "p"))
    except Exception:
        return False


def _resolve_companion_h5(
    event_path: Path,
    *,
    suffix: str,
) -> Path | None:
    """Resolve an official M3ED sibling such as ``<seq>_semantics.h5``."""
    sequence_name = _sequence_name_from_event_path(event_path)
    candidates = [
        event_path.with_name(f"{sequence_name}_{suffix}{event_path.suffix}"),
        event_path.with_name(f"{sequence_name}_{suffix}.h5"),
        event_path.with_name(f"{sequence_name}_{suffix}.hdf5"),
    ]
    for candidate in dict.fromkeys(candidates):
        if candidate != event_path and candidate.is_file():
            return candidate.resolve()
    return None


def _discover_m3ed_h5_files(
    data_path: Path,
    *,
    file_pattern: str,
    recursive: bool,
) -> list[Path]:
    suffix = data_path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        candidates = _parse_manifest_paths(data_path)
    elif suffix in {".h5", ".hdf5"} and data_path.is_file():
        candidates = [data_path.resolve()]
    elif data_path.is_dir():
        iterator = (
            data_path.rglob(file_pattern)
            if recursive
            else data_path.glob(file_pattern)
        )
        candidates = sorted(path.resolve() for path in iterator if path.is_file())
    else:
        raise FileNotFoundError(f"Unsupported M3ED data path: {data_path}")

    return [path for path in candidates if _is_valid_m3ed_h5(path)]


def _read_t_offset(h5f: h5py.File) -> int:
    if "t_offset" not in h5f:
        return 0
    return int(h5f["t_offset"][()])


def _read_anchor_timestamps(
    h5f: h5py.File,
    *,
    event_path: Path,
    window_mode: str,
    semantics_ts_source: str,
    semantics_ts_divisor: int,
    depth_ts_source: str,
    depth_ts_divisor: int,
) -> np.ndarray:
    with ExitStack() as stack:
        if window_mode == "semantics_middle":
            companion_path = _resolve_companion_h5(
                event_path,
                suffix="semantics",
            )
            companion_h5 = (
                stack.enter_context(h5py.File(str(companion_path), "r"))
                if companion_path is not None
                else None
            )
            candidates = {
                # Downloaded layout:
                #   <sequence>_semantics.h5:/ts
                # Legacy extracted layout:
                #   <event-file>:/semantics/ts
                "semantics_ts": (
                    (companion_h5, "ts", companion_path),
                    (h5f, "semantics/ts", event_path),
                ),
                # The downloaded _data.h5 contains the same OVC timestamps,
                # so semantic labels are not required for frame-aligned
                # self-supervised pretraining.
                "ovc_ts": ((h5f, "ovc/ts", event_path),),
                "semantics_ts_map": (
                    (
                        companion_h5,
                        "ts_map_prophesee_left_t",
                        companion_path,
                    ),
                    (
                        h5f,
                        "semantics/ts_map_prophesee_left_t",
                        event_path,
                    ),
                ),
                "ovc_ts_map": (
                    (h5f, "ovc/ts_map_prophesee_left_t", event_path),
                ),
            }
            ordered = (
                (
                    "semantics_ts",
                    "ovc_ts",
                    "semantics_ts_map",
                    "ovc_ts_map",
                )
                if semantics_ts_source == "auto"
                else (semantics_ts_source,)
            )
            divisor = int(semantics_ts_divisor)
        elif window_mode == "depth_middle":
            companion_path = _resolve_companion_h5(
                event_path,
                suffix="depth_gt",
            )
            companion_h5 = (
                stack.enter_context(h5py.File(str(companion_path), "r"))
                if companion_path is not None
                else None
            )
            candidates = {
                "depth_ts": (
                    (companion_h5, "ts", companion_path),
                    (h5f, "depth_gt/ts", event_path),
                ),
                "depth_ts_map_left_t": (
                    (
                        companion_h5,
                        "ts_map_prophesee_left_t",
                        companion_path,
                    ),
                    (
                        h5f,
                        "depth_gt/ts_map_prophesee_left_t",
                        event_path,
                    ),
                ),
                "depth_ts_map_left": (
                    (
                        companion_h5,
                        "ts_map_prophesee_left",
                        companion_path,
                    ),
                    (
                        h5f,
                        "depth_gt/ts_map_prophesee_left",
                        event_path,
                    ),
                ),
            }
            ordered = (
                ("depth_ts", "depth_ts_map_left_t", "depth_ts_map_left")
                if depth_ts_source == "auto"
                else (depth_ts_source,)
            )
            divisor = int(depth_ts_divisor)
        else:
            raise ValueError(
                f"Anchor timestamps are not used for window_mode={window_mode}"
            )

        if divisor <= 0:
            raise ValueError("timestamp divisor must be > 0")

        checked: list[str] = []
        for source in ordered:
            if source not in candidates:
                raise ValueError(
                    f"Unsupported timestamp source '{source}' for "
                    f"window_mode={window_mode}"
                )
            for source_h5, dataset_path, source_path in candidates[source]:
                if source_h5 is None:
                    continue
                checked.append(f"{source_path}:{dataset_path}")
                if dataset_path not in source_h5:
                    continue
                timestamps = np.atleast_1d(
                    source_h5[dataset_path][()]
                ).astype(np.int64, copy=False).reshape(-1)
                if timestamps.size == 0:
                    continue
                if divisor != 1:
                    timestamps = np.floor_divide(
                        timestamps,
                        divisor,
                    ).astype(np.int64, copy=False)
                return timestamps

    raise FileNotFoundError(
        f"No non-empty M3ED timestamps found for window_mode={window_mode}. "
        f"checked: {', '.join(checked)}"
    )


def _in_range_ratio(
    values: np.ndarray,
    start_us: int,
    end_exclusive_us: int,
) -> float:
    if values.size == 0:
        return 0.0
    in_range = (values >= int(start_us)) & (values < int(end_exclusive_us))
    return float(np.count_nonzero(in_range)) / float(values.size)


def _align_anchor_timebase_to_events(
    anchors_us: np.ndarray,
    *,
    t_first_us: int,
    t_last_exclusive_us: int,
    t_offset_us: int,
) -> np.ndarray:
    anchors = np.asarray(anchors_us, dtype=np.int64).reshape(-1)
    if anchors.size == 0 or t_offset_us == 0:
        return anchors

    candidates = (
        anchors,
        anchors + int(t_offset_us),
        anchors - int(t_offset_us),
    )

    def _score(values: np.ndarray) -> tuple[float, int]:
        ratio = _in_range_ratio(values, t_first_us, t_last_exclusive_us)
        edge = abs(int(values[0]) - int(t_first_us)) + abs(
            int(values[-1]) - int(t_last_exclusive_us - 1)
        )
        return ratio, -edge

    as_is_score = _score(candidates[0])
    scores = [_score(values) for values in candidates]
    best_idx = max(range(len(candidates)), key=lambda idx: scores[idx])
    best_score = scores[best_idx]
    if best_idx != 0 and (
        (as_is_score[0] == 0.0 and best_score[0] > 0.0)
        or best_score[0] >= as_is_score[0] + 0.25
    ):
        return candidates[best_idx].astype(np.int64, copy=False)
    return anchors


def _build_middle_windows(
    anchors_us: np.ndarray,
    *,
    t_first_us: int,
    t_last_exclusive_us: int,
) -> tuple[np.ndarray, np.ndarray]:
    anchors = np.asarray(anchors_us, dtype=np.int64).reshape(-1)
    if anchors.size == 0:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )
    if anchors.size > 1 and np.any(np.diff(anchors) < 0):
        raise ValueError("M3ED anchor timestamps must be non-decreasing")

    midpoints = anchors[:-1] + np.floor_divide(
        anchors[1:] - anchors[:-1], 2
    )
    starts = np.empty_like(anchors)
    ends = np.empty_like(anchors)
    starts[0] = int(t_first_us)
    if anchors.size > 1:
        starts[1:] = midpoints
        ends[:-1] = midpoints
    ends[-1] = int(t_last_exclusive_us)
    np.maximum(starts, int(t_first_us), out=starts)
    np.minimum(ends, int(t_last_exclusive_us), out=ends)
    return starts, ends


def _build_fixed_windows(
    *,
    t_first_us: int,
    t_last_exclusive_us: int,
    start_time_us: int | None,
    accum_time_us: int,
    stride_time_us: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time_origin_us = (
        int(t_first_us) if start_time_us is None else int(start_time_us)
    )
    window_start_us = max(int(t_first_us), time_origin_us)
    starts = np.arange(
        window_start_us,
        int(t_last_exclusive_us),
        int(stride_time_us),
        dtype=np.int64,
    )
    ends = np.minimum(
        starts + int(accum_time_us),
        int(t_last_exclusive_us),
    ).astype(np.int64, copy=False)
    anchors = starts + np.floor_divide(ends - starts, 2)
    return anchors, starts, ends


def _virtual_chunk_ranges(
    anchors_us: np.ndarray,
    *,
    chunk_duration_s: float | None,
    min_windows_per_chunk: int,
) -> list[tuple[int, int]]:
    if anchors_us.size == 0:
        return []
    if chunk_duration_s is None:
        ranges = [(0, int(anchors_us.size))]
    else:
        chunk_duration_us = int(round(float(chunk_duration_s) * 1_000_000.0))
        if chunk_duration_us <= 0:
            raise ValueError("virtual_chunk_duration_s must be > 0 or null")
        chunk_ids = np.floor_divide(
            anchors_us - int(anchors_us[0]),
            chunk_duration_us,
        )
        changes = np.where(chunk_ids[1:] != chunk_ids[:-1])[0] + 1
        starts = np.concatenate(([0], changes))
        ends = np.concatenate((changes, [anchors_us.size]))
        ranges = [
            (int(start), int(end))
            for start, end in zip(starts, ends)
            if end > start
        ]
    return [
        (start, end)
        for start, end in ranges
        if end - start >= int(min_windows_per_chunk)
    ]


def _h5_searchsorted_left(timestamps: h5py.Dataset, value: int) -> int:
    lo = 0
    hi = int(len(timestamps))
    while lo < hi:
        mid = (lo + hi) // 2
        if int(timestamps[mid]) < int(value):
            lo = mid + 1
        else:
            hi = mid
    return lo


class M3EDRawEventDataset(torch.utils.data.Dataset):
    """Create M3ED event representations on demand from raw event HDF5 files."""

    def __init__(
        self,
        data_paths: str | Path | Sequence[str | Path],
        *,
        datasets_weights: Sequence[float] | None = None,
        frames_per_clip: int = 16,
        dataset_fpcs: Sequence[int] | None = None,
        frame_step: int = 1,
        fps=None,
        num_clips: int = 1,
        transform: Callable | None = None,
        shared_transform: Callable | None = None,
        random_clip_sampling: bool = True,
        allow_clip_overlap: bool = False,
        file_pattern: str = "*.h5",
        recursive: bool = True,
        max_open_h5_files: int = 8,
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
        window_mode: str = "semantics_middle",
        accum_time_us: int = 50_000,
        stride_time_us: int | None = None,
        start_time_us: int | None = None,
        semantics_ts_source: str = "auto",
        semantics_ts_divisor: int = 1,
        depth_ts_source: str = "auto",
        depth_ts_divisor: int = 1,
        filter_known_semantic_sequences: bool = False,
        virtual_chunk_duration_s: float | None = 20.0,
        min_windows_per_chunk: int = 1,
        activity_filter_enabled: bool = False,
        activity_filter_min_clip_mean_active_pixel_ratio=None,
        activity_filter_min_clip_mean_activity_score=None,
        activity_filter_min_clip_active_window_ratio=None,
        activity_filter_active_window_threshold=None,
        activity_filter_max_trials: int = 8,
    ):
        if isinstance(data_paths, (str, Path)):
            data_paths = [data_paths]
        if len(data_paths) == 0:
            raise ValueError("data_paths must be non-empty")
        if frames_per_clip <= 0 or frame_step <= 0 or num_clips <= 0:
            raise ValueError(
                "frames_per_clip, frame_step, and num_clips must all be > 0"
            )
        if representation not in {"voxel_grid", "event_image"}:
            raise ValueError(
                "representation must be one of {'voxel_grid', 'event_image'}"
            )
        if window_mode not in {"fixed", "semantics_middle", "depth_middle"}:
            raise ValueError(
                "window_mode must be one of "
                "{'fixed', 'semantics_middle', 'depth_middle'}"
            )
        if int(accum_time_us) <= 0:
            raise ValueError("accum_time_us must be > 0")
        if stride_time_us is None:
            stride_time_us = int(accum_time_us)
        if int(stride_time_us) <= 0:
            raise ValueError("stride_time_us must be > 0")
        if int(downsample_factor) not in {1, 2}:
            raise ValueError("downsample_factor must be 1 or 2")
        if int(t_bins) <= 0:
            raise ValueError("t_bins must be > 0")
        if output_dtype not in {"float16", "float32"}:
            raise ValueError("output_dtype must be 'float16' or 'float32'")
        if int(min_windows_per_chunk) <= 0:
            raise ValueError("min_windows_per_chunk must be > 0")

        self.data_paths = [Path(path).expanduser() for path in data_paths]
        self.datasets_weights = datasets_weights
        self.frame_step = int(frame_step)
        self.num_clips = int(num_clips)
        self.transform = transform
        self.shared_transform = shared_transform
        self.random_clip_sampling = bool(random_clip_sampling)
        self.allow_clip_overlap = bool(allow_clip_overlap)
        self.max_open_h5_files = max(1, int(max_open_h5_files))
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
        self.window_mode = str(window_mode)
        self.accum_time_us = int(accum_time_us)
        self.stride_time_us = int(stride_time_us)
        self.start_time_us = (
            None if start_time_us is None else int(start_time_us)
        )
        self.semantics_ts_source = str(semantics_ts_source)
        self.semantics_ts_divisor = int(semantics_ts_divisor)
        self.depth_ts_source = str(depth_ts_source)
        self.depth_ts_divisor = int(depth_ts_divisor)
        self.filter_known_semantic_sequences = bool(
            filter_known_semantic_sequences
        )
        self.virtual_chunk_duration_s = (
            None
            if virtual_chunk_duration_s is None
            else float(virtual_chunk_duration_s)
        )
        self.min_windows_per_chunk = int(min_windows_per_chunk)
        self.activity_filter_enabled = bool(activity_filter_enabled)
        self.activity_filter_max_trials = max(1, int(activity_filter_max_trials))

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

        if dataset_fpcs is None:
            self.dataset_fpcs = [
                int(frames_per_clip) for _ in self.data_paths
            ]
        else:
            if len(dataset_fpcs) != len(self.data_paths):
                raise ValueError("dataset_fpcs length must match data_paths length")
            self.dataset_fpcs = [int(value) for value in dataset_fpcs]
        if any(value <= 0 for value in self.dataset_fpcs):
            raise ValueError("All dataset_fpcs values must be > 0")

        num_datasets = len(self.data_paths)
        self.target_fps_per_dataset = _expand_optional_per_dataset(
            fps,
            num_datasets=num_datasets,
            field_name="fps",
        )
        if any(
            value is not None and float(value) <= 0.0
            for value in self.target_fps_per_dataset
        ):
            raise ValueError("fps values must be > 0 when provided")

        self.activity_filter_min_clip_mean_active_pixel_ratio = (
            _expand_optional_per_dataset(
                activity_filter_min_clip_mean_active_pixel_ratio,
                num_datasets=num_datasets,
                field_name="activity_filter_min_clip_mean_active_pixel_ratio",
            )
        )
        self.activity_filter_min_clip_mean_activity_score = (
            _expand_optional_per_dataset(
                activity_filter_min_clip_mean_activity_score,
                num_datasets=num_datasets,
                field_name="activity_filter_min_clip_mean_activity_score",
            )
        )
        self.activity_filter_min_clip_active_window_ratio = (
            _expand_optional_per_dataset(
                activity_filter_min_clip_active_window_ratio,
                num_datasets=num_datasets,
                field_name="activity_filter_min_clip_active_window_ratio",
            )
        )
        self.activity_filter_active_window_threshold = (
            _expand_optional_per_dataset(
                activity_filter_active_window_threshold,
                num_datasets=num_datasets,
                field_name="activity_filter_active_window_threshold",
            )
        )
        for dataset_idx in range(num_datasets):
            if (
                self.activity_filter_min_clip_active_window_ratio[dataset_idx]
                is not None
                and self.activity_filter_active_window_threshold[dataset_idx]
                is None
            ):
                self.activity_filter_active_window_threshold[dataset_idx] = 1e-6

        self.sequences: list[_M3EDSequence] = []
        self.virtual_chunks: list[_VirtualChunk] = []
        self.num_samples_per_dataset: list[int] = []
        for dataset_idx, data_path in enumerate(self.data_paths):
            files = _discover_m3ed_h5_files(
                data_path,
                file_pattern=file_pattern,
                recursive=recursive,
            )
            if self.window_mode == "semantics_middle" and (
                self.filter_known_semantic_sequences
            ):
                files = [
                    path for path in files if _is_known_semantic_sequence(path)
                ]
            if len(files) == 0:
                raise FileNotFoundError(
                    f"No compatible M3ED raw H5 files found in: {data_path}"
                )

            sample_count_before = len(self.virtual_chunks)
            for path in files:
                sequence = self._build_sequence_metadata(
                    path=path,
                    dataset_idx=dataset_idx,
                )
                if sequence is None:
                    continue
                sequence_idx = len(self.sequences)
                self.sequences.append(sequence)
                for start, end in _virtual_chunk_ranges(
                    sequence.anchors_us,
                    chunk_duration_s=self.virtual_chunk_duration_s,
                    min_windows_per_chunk=self.min_windows_per_chunk,
                ):
                    self.virtual_chunks.append(
                        _VirtualChunk(
                            sequence_idx=sequence_idx,
                            dataset_idx=dataset_idx,
                            window_start=start,
                            window_end=end,
                        )
                    )
            num_samples = len(self.virtual_chunks) - sample_count_before
            if num_samples == 0:
                raise FileNotFoundError(
                    f"No non-empty M3ED virtual chunks found in: {data_path}"
                )
            self.num_samples_per_dataset.append(num_samples)

        # Preserve the path-oriented `samples` attribute expected by the
        # repository's visualization utilities.
        self.samples = [
            self.sequences[chunk.sequence_idx].path
            for chunk in self.virtual_chunks
        ]
        self.labels = [0] * len(self.virtual_chunks)
        self.sample_weights: list[float] | None = None
        if self.datasets_weights is not None:
            if len(self.datasets_weights) != len(self.num_samples_per_dataset):
                raise ValueError(
                    "datasets_weights length must match number of data_paths"
                )
            self.sample_weights = []
            for dataset_weight, num_samples in zip(
                self.datasets_weights,
                self.num_samples_per_dataset,
            ):
                self.sample_weights.extend(
                    [float(dataset_weight) / float(num_samples)] * num_samples
                )

        out_channels = (
            self.t_bins * (2 if self.split_polarity else 1)
            if self.representation == "voxel_grid"
            else 3
        )
        self.output_channels = int(out_channels)
        self._voxelizer = (
            EventVoxelGrid(
                input_size=(
                    self.t_bins,
                    self.output_height,
                    self.output_width,
                ),
                normalize=self.normalize,
                separate_polarity=self.split_polarity,
                trilinear_interpolation=self.use_trilinear,
            )
            if self.representation == "voxel_grid"
            else None
        )
        self._h5_cache: OrderedDict[str, h5py.File] = OrderedDict()
        self._ms_idx_cache: OrderedDict[str, np.ndarray | None] = OrderedDict()
        self._sampling_step_cache: dict[int, int] = {}

    def _build_sequence_metadata(
        self,
        *,
        path: Path,
        dataset_idx: int,
    ) -> _M3EDSequence | None:
        with h5py.File(str(path), "r") as h5f:
            events = h5f[EVENT_GROUP_PATH]
            timestamps = events["t"]
            if len(timestamps) == 0:
                return None
            t_offset_us = _read_t_offset(h5f)
            t_first_us = int(timestamps[0]) + t_offset_us
            t_last_exclusive_us = int(timestamps[-1]) + t_offset_us + 1

            if self.window_mode == "fixed":
                anchors_us, starts_us, ends_us = _build_fixed_windows(
                    t_first_us=t_first_us,
                    t_last_exclusive_us=t_last_exclusive_us,
                    start_time_us=self.start_time_us,
                    accum_time_us=self.accum_time_us,
                    stride_time_us=self.stride_time_us,
                )
            else:
                anchors_us = _read_anchor_timestamps(
                    h5f,
                    event_path=path,
                    window_mode=self.window_mode,
                    semantics_ts_source=self.semantics_ts_source,
                    semantics_ts_divisor=self.semantics_ts_divisor,
                    depth_ts_source=self.depth_ts_source,
                    depth_ts_divisor=self.depth_ts_divisor,
                )
                anchors_us = _align_anchor_timebase_to_events(
                    anchors_us,
                    t_first_us=t_first_us,
                    t_last_exclusive_us=t_last_exclusive_us,
                    t_offset_us=t_offset_us,
                )
                starts_us, ends_us = _build_middle_windows(
                    anchors_us,
                    t_first_us=t_first_us,
                    t_last_exclusive_us=t_last_exclusive_us,
                )

        if anchors_us.size == 0:
            return None
        return _M3EDSequence(
            path=str(path),
            dataset_idx=int(dataset_idx),
            t_first_us=int(t_first_us),
            t_last_exclusive_us=int(t_last_exclusive_us),
            t_offset_us=int(t_offset_us),
            anchors_us=anchors_us.astype(np.int64, copy=False),
            window_starts_us=starts_us.astype(np.int64, copy=False),
            window_ends_us=ends_us.astype(np.int64, copy=False),
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5_cache"] = OrderedDict()
        state["_ms_idx_cache"] = OrderedDict()
        state["_sampling_step_cache"] = {}
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
        return len(self.virtual_chunks)

    def _get_h5(self, path: str) -> h5py.File:
        cached = self._h5_cache.get(path)
        if cached is not None:
            self._h5_cache.move_to_end(path, last=True)
            return cached

        h5f = h5py.File(path, "r")
        self._h5_cache[path] = h5f
        while len(self._h5_cache) > self.max_open_h5_files:
            old_path, old_h5 = self._h5_cache.popitem(last=False)
            try:
                old_h5.close()
            except Exception:
                pass
            self._ms_idx_cache.pop(old_path, None)
        return h5f

    def _get_ms_idx(self, path: str) -> np.ndarray | None:
        if path in self._ms_idx_cache:
            value = self._ms_idx_cache[path]
            self._ms_idx_cache.move_to_end(path, last=True)
            return value

        h5f = self._get_h5(path)
        events = h5f[EVENT_GROUP_PATH]
        if "ms_map_idx" in events:
            value = np.asarray(events["ms_map_idx"], dtype=np.int64)
        elif "ms_to_idx" in h5f:
            value = np.asarray(h5f["ms_to_idx"], dtype=np.int64)
        else:
            value = None
        self._ms_idx_cache[path] = value
        while len(self._ms_idx_cache) > self.max_open_h5_files:
            self._ms_idx_cache.popitem(last=False)
        return value

    def _extract_events_by_time(
        self,
        sequence: _M3EDSequence,
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

        h5f = self._get_h5(sequence.path)
        events = h5f[EVENT_GROUP_PATH]
        timestamps = events["t"]
        num_events = int(len(timestamps))
        if num_events == 0:
            return empty

        ms_idx = self._get_ms_idx(sequence.path)
        if ms_idx is not None and ms_idx.size > 0:
            start_rel_us = int(start_us) - int(sequence.t_offset_us)
            end_rel_us = int(end_us) - int(sequence.t_offset_us)
            if end_rel_us <= 0:
                return empty
            start_ms = max(start_rel_us // 1000, 0)
            end_ms_exclusive = max(
                (end_rel_us + 999) // 1000,
                start_ms + 1,
            )
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
                np.asarray(
                    timestamps[coarse_start:coarse_end],
                    dtype=np.int64,
                )
                + int(sequence.t_offset_us)
            )
            rel_start = int(np.searchsorted(t_coarse, start_us, side="left"))
            rel_end = int(np.searchsorted(t_coarse, end_us, side="left"))
            event_start = coarse_start + rel_start
            event_end = coarse_start + rel_end
            selected_t = t_coarse[rel_start:rel_end]
        else:
            start_rel_us = int(start_us) - int(sequence.t_offset_us)
            end_rel_us = int(end_us) - int(sequence.t_offset_us)
            event_start = _h5_searchsorted_left(timestamps, start_rel_us)
            event_end = _h5_searchsorted_left(timestamps, end_rel_us)
            selected_t = (
                np.asarray(
                    timestamps[event_start:event_end],
                    dtype=np.int64,
                )
                + int(sequence.t_offset_us)
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
        if (
            self.output_width == self.input_width
            and self.output_height == self.input_height
        ):
            x_out = x
            y_out = y
        elif (
            self.input_width % self.output_width == 0
            and self.input_height % self.output_height == 0
        ):
            x_out = np.floor(
                x / float(self.input_width // self.output_width)
            )
            y_out = np.floor(
                y / float(self.input_height // self.output_height)
            )
        else:
            x_out = np.floor(
                x * (float(self.output_width) / float(self.input_width))
            )
            y_out = np.floor(
                y * (float(self.output_height) / float(self.input_height))
            )

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

    @staticmethod
    def _activity_metrics_from_volume(
        activity_volume: torch.Tensor,
    ) -> tuple[float, float]:
        activity = activity_volume.detach()
        score = float(
            torch.count_nonzero(activity).item() / max(1, activity.numel())
        )
        active_hw = activity.sum(dim=0) > 0
        active_pixel_ratio = float(
            torch.count_nonzero(active_hw).item() / max(1, active_hw.numel())
        )
        return score, active_pixel_ratio

    def _make_representation(
        self,
        sequence: _M3EDSequence,
        window_idx: int,
    ) -> tuple[torch.Tensor, float, float]:
        events = self._extract_events_by_time(
            sequence,
            start_us=int(sequence.window_starts_us[window_idx]),
            end_us=int(sequence.window_ends_us[window_idx]),
        )
        events = self._normalize_event_coordinates(events)

        if self.representation == "voxel_grid":
            assert self._voxelizer is not None
            if events["t"].size == 0:
                voxel = torch.zeros(
                    (
                        self.output_channels,
                        self.output_height,
                        self.output_width,
                    ),
                    dtype=torch.float32,
                )
            else:
                shifted_t = (events["t"] - events["t"][0]).astype(
                    np.float32, copy=False
                )
                voxel = self._voxelizer.convert(
                    {
                        "x": torch.from_numpy(events["x"]),
                        "y": torch.from_numpy(events["y"]),
                        "p": torch.from_numpy(events["p"]),
                        "t": torch.from_numpy(shifted_t),
                    }
                ).cpu()

            if self._activity_filter_is_enabled(sequence.dataset_idx):
                if self.split_polarity:
                    activity_volume = (
                        voxel.abs()
                        .reshape(
                            2,
                            self.t_bins,
                            self.output_height,
                            self.output_width,
                        )
                        .sum(dim=0)
                    )
                else:
                    activity_volume = voxel.abs()
                activity_score, active_pixel_ratio = (
                    self._activity_metrics_from_volume(activity_volume)
                )
            else:
                activity_score, active_pixel_ratio = 0.0, 0.0
            representation = voxel
        else:
            image, activity_np = accumulate_events_to_rgb(
                events["x"],
                events["y"],
                events["p"],
                (self.output_height, self.output_width),
                percentile=self.event_image_percentile,
                dtype=np.float32,
            )
            representation = torch.from_numpy(image)
            if self._activity_filter_is_enabled(sequence.dataset_idx):
                activity_score, active_pixel_ratio = (
                    self._activity_metrics_from_volume(
                        torch.from_numpy(activity_np)
                    )
                )
            else:
                activity_score, active_pixel_ratio = 0.0, 0.0

        representation = representation.to(torch.float32)
        if self.output_dtype == "float16":
            # Match the preprocessed path: float32 generation -> float16 storage
            # -> float32 loading.
            representation = representation.to(torch.float16).to(torch.float32)
        return (
            representation.permute(1, 2, 0).contiguous(),
            activity_score,
            active_pixel_ratio,
        )

    @staticmethod
    def _fit_indices_length(
        indices: np.ndarray,
        target_len: int,
        last_valid: int,
    ) -> np.ndarray:
        if indices.size >= target_len:
            positions = np.linspace(
                0,
                indices.size - 1,
                num=target_len,
                dtype=np.float64,
            )
            return indices[np.round(positions).astype(np.int64)]
        padding = np.full(
            (target_len - indices.size,),
            int(last_valid),
            dtype=np.int64,
        )
        return np.concatenate([indices, padding])

    def _resolve_sampling_step(self, sample_idx: int) -> int:
        cached = self._sampling_step_cache.get(int(sample_idx))
        if cached is not None:
            return cached

        sample = self.virtual_chunks[sample_idx]
        target_fps = self.target_fps_per_dataset[sample.dataset_idx]
        if target_fps is None:
            step = self.frame_step
        else:
            sequence = self.sequences[sample.sequence_idx]
            anchors = sequence.anchors_us[
                sample.window_start : sample.window_end
            ]
            deltas = np.diff(anchors)
            deltas = deltas[deltas > 0]
            median_delta_us = (
                float(np.median(deltas))
                if deltas.size > 0
                else float(self.stride_time_us)
            )
            source_fps = int(np.ceil(1_000_000.0 / median_delta_us))
            step = max(int(source_fps // float(target_fps)), 1)
        self._sampling_step_cache[int(sample_idx)] = int(step)
        return int(step)

    def _sample_clip_in_segment(
        self,
        *,
        sample_idx: int,
        segment_start: int,
        segment_length: int,
        fpc: int,
    ) -> np.ndarray:
        total_windows = self.virtual_chunks[sample_idx].num_windows
        if total_windows <= 0:
            return np.zeros((fpc,), dtype=np.int64)
        if segment_length <= 0:
            anchor = min(max(segment_start, 0), total_windows - 1)
            return np.full((fpc,), anchor, dtype=np.int64)

        sampling_step = self._resolve_sampling_step(sample_idx)
        clip_span = max(1, fpc * sampling_step)
        if segment_length > clip_span:
            max_local_start = segment_length - clip_span
            local_start = (
                int(np.random.randint(0, max_local_start + 1))
                if self.random_clip_sampling
                else max_local_start // 2
            )
            local_indices = np.arange(
                local_start,
                local_start + clip_span,
                sampling_step,
                dtype=np.int64,
            )
        else:
            local_indices = np.arange(
                0,
                segment_length,
                sampling_step,
                dtype=np.int64,
            )
        if local_indices.size == 0:
            local_indices = np.array([0], dtype=np.int64)
        local_indices = self._fit_indices_length(
            local_indices,
            target_len=fpc,
            last_valid=max(0, segment_length - 1),
        )
        indices = segment_start + local_indices
        np.clip(indices, 0, total_windows - 1, out=indices)
        return indices.astype(np.int64, copy=False)

    def _sample_clip_indices(
        self,
        *,
        sample_idx: int,
        fpc: int,
    ) -> list[np.ndarray]:
        total_windows = self.virtual_chunks[sample_idx].num_windows
        if self.num_clips == 1:
            return [
                self._sample_clip_in_segment(
                    sample_idx=sample_idx,
                    segment_start=0,
                    segment_length=total_windows,
                    fpc=fpc,
                )
            ]

        clips: list[np.ndarray] = []
        if not self.allow_clip_overlap:
            partition_len = max(1, total_windows // self.num_clips)
            for clip_idx in range(self.num_clips):
                start = clip_idx * partition_len
                end = (
                    total_windows
                    if clip_idx == self.num_clips - 1
                    else min(total_windows, (clip_idx + 1) * partition_len)
                )
                clips.append(
                    self._sample_clip_in_segment(
                        sample_idx=sample_idx,
                        segment_start=start,
                        segment_length=max(1, end - start),
                        fpc=fpc,
                    )
                )
        else:
            sampling_step = self._resolve_sampling_step(sample_idx)
            clip_span = max(1, fpc * sampling_step)
            if total_windows <= clip_span:
                starts = np.linspace(
                    0,
                    max(0, total_windows - 1),
                    num=self.num_clips,
                )
            else:
                starts = np.linspace(
                    0,
                    total_windows - clip_span,
                    num=self.num_clips,
                )
            for start_value in np.round(starts).astype(np.int64).tolist():
                clips.append(
                    self._sample_clip_in_segment(
                        sample_idx=sample_idx,
                        segment_start=int(start_value),
                        segment_length=min(
                            total_windows - int(start_value),
                            clip_span,
                        ),
                        fpc=fpc,
                    )
                )
        return clips

    def _activity_filter_is_enabled(self, dataset_idx: int) -> bool:
        if not self.activity_filter_enabled:
            return False
        return any(
            value is not None
            for value in (
                self.activity_filter_min_clip_mean_active_pixel_ratio[
                    dataset_idx
                ],
                self.activity_filter_min_clip_mean_activity_score[dataset_idx],
                self.activity_filter_min_clip_active_window_ratio[dataset_idx],
            )
        )

    def _passes_activity_filter(
        self,
        *,
        dataset_idx: int,
        clip_indices: list[np.ndarray],
        activity_scores: list[float],
        active_pixel_ratios: list[float],
    ) -> bool:
        if not self._activity_filter_is_enabled(dataset_idx):
            return True

        cursor = 0
        for indices in clip_indices:
            length = len(indices)
            scores = np.asarray(
                activity_scores[cursor : cursor + length],
                dtype=np.float32,
            )
            active_ratios = np.asarray(
                active_pixel_ratios[cursor : cursor + length],
                dtype=np.float32,
            )
            cursor += length

            min_active = (
                self.activity_filter_min_clip_mean_active_pixel_ratio[
                    dataset_idx
                ]
            )
            if min_active is not None and float(active_ratios.mean()) < float(
                min_active
            ):
                return False
            min_score = self.activity_filter_min_clip_mean_activity_score[
                dataset_idx
            ]
            if min_score is not None and float(scores.mean()) < float(min_score):
                return False
            min_active_window_ratio = (
                self.activity_filter_min_clip_active_window_ratio[dataset_idx]
            )
            if min_active_window_ratio is not None:
                threshold = self.activity_filter_active_window_threshold[
                    dataset_idx
                ]
                threshold = 0.0 if threshold is None else float(threshold)
                ratio = float(np.mean(active_ratios > threshold))
                if ratio < float(min_active_window_ratio):
                    return False
        return True

    def get_item_event(self, index: int):
        sample = self.virtual_chunks[index]
        sequence = self.sequences[sample.sequence_idx]
        fpc = self.dataset_fpcs[sample.dataset_idx]

        trials = (
            self.activity_filter_max_trials
            if self._activity_filter_is_enabled(sample.dataset_idx)
            else 1
        )
        for _ in range(trials):
            clip_indices = self._sample_clip_indices(
                sample_idx=index,
                fpc=fpc,
            )
            all_local_indices = np.concatenate(clip_indices).astype(
                np.int64, copy=False
            )
            representation_cache: dict[
                int, tuple[torch.Tensor, float, float]
            ] = {}
            frames: list[torch.Tensor] = []
            activity_scores: list[float] = []
            active_pixel_ratios: list[float] = []
            for local_idx in all_local_indices.tolist():
                global_idx = int(sample.window_start + int(local_idx))
                represented = representation_cache.get(global_idx)
                if represented is None:
                    represented = self._make_representation(
                        sequence,
                        global_idx,
                    )
                    representation_cache[global_idx] = represented
                frame, activity_score, active_pixel_ratio = represented
                frames.append(frame)
                activity_scores.append(activity_score)
                active_pixel_ratios.append(active_pixel_ratio)

            if not self._passes_activity_filter(
                dataset_idx=sample.dataset_idx,
                clip_indices=clip_indices,
                activity_scores=activity_scores,
                active_pixel_ratios=active_pixel_ratios,
            ):
                continue

            windows = torch.stack(frames, dim=0)
            if self.shared_transform is not None:
                windows = self.shared_transform(windows)
            split_clips = [
                windows[clip_idx * fpc : (clip_idx + 1) * fpc]
                for clip_idx in range(self.num_clips)
            ]
            if self.transform is not None:
                split_clips = [self.transform(clip) for clip in split_clips]
            else:
                split_clips = [
                    clip.to(torch.float32)
                    .permute(3, 0, 1, 2)
                    .contiguous()
                    for clip in split_clips
                ]

            return (
                split_clips,
                self.labels[index],
                [
                    indices.astype(np.int32, copy=False)
                    for indices in clip_indices
                ],
            )
        return None

    def __getitem__(self, index: int):
        num_trials = 8
        for _ in range(num_trials):
            loaded = self.get_item_event(index)
            if loaded is not None:
                return loaded
            index = int(np.random.randint(0, len(self.virtual_chunks)))
        raise RuntimeError(
            f"Failed to load M3ED raw event sample after {num_trials} retries."
        )


def make_m3ed_raw_eventdataset(
    data_paths: str | Path | Sequence[str | Path],
    batch_size: int,
    *,
    frames_per_clip: int = 16,
    dataset_fpcs: Sequence[int] | None = None,
    frame_step: int = 1,
    fps=None,
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
    num_workers: int = 8,
    pin_mem: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int | None = None,
    **dataset_kwargs,
):
    dataset = M3EDRawEventDataset(
        data_paths=data_paths,
        datasets_weights=datasets_weights,
        frames_per_clip=frames_per_clip,
        dataset_fpcs=dataset_fpcs,
        frame_step=frame_step,
        fps=fps,
        num_clips=num_clips,
        random_clip_sampling=random_clip_sampling,
        allow_clip_overlap=allow_clip_overlap,
        transform=transform,
        shared_transform=shared_transform,
        **dataset_kwargs,
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

    dataloader_kwargs = {
        "dataset": dataset,
        "collate_fn": collator,
        "sampler": sampler,
        "batch_size": batch_size,
        "drop_last": drop_last,
        "pin_memory": pin_mem,
        "num_workers": num_workers,
        "persistent_workers": (num_workers > 0) and persistent_workers,
    }
    if num_workers > 0 and prefetch_factor is not None:
        dataloader_kwargs["prefetch_factor"] = int(prefetch_factor)
    data_loader = torch.utils.data.DataLoader(**dataloader_kwargs)
    return dataset, data_loader, sampler
