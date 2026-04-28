from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np


@dataclass
class TimeRange:
    start_us: int
    end_exclusive_us: int

    @property
    def duration_s(self) -> float:
        return max(0.0, float(self.end_exclusive_us - self.start_us) / 1e6)


def _read_t_offset(h5f: h5py.File) -> int:
    if "t_offset" not in h5f:
        return 0
    return int(h5f["t_offset"][()])


def _event_time_range(h5f: h5py.File) -> TimeRange | None:
    if "events/t" not in h5f:
        raise KeyError("missing dataset events/t")
    t_ds = h5f["events/t"]
    n = len(t_ds)
    if n == 0:
        return None
    off = _read_t_offset(h5f)
    start_us = int(t_ds[0]) + off
    end_exclusive_us = int(t_ds[n - 1]) + off + 1
    return TimeRange(start_us=start_us, end_exclusive_us=end_exclusive_us)


def _load_ms_to_idx(h5f: h5py.File) -> np.ndarray | None:
    if "ms_to_idx" in h5f:
        return np.asarray(h5f["ms_to_idx"][()], dtype=np.int64)
    if "events/ms_map_idx" in h5f:
        return np.asarray(h5f["events/ms_map_idx"][()], dtype=np.int64)
    return None


def _coarse_bounds_from_ms_to_idx(
    ms_to_idx: np.ndarray | None,
    num_events: int,
    t_offset_us: int,
    start_us: int,
    end_us: int,
) -> tuple[int, int]:
    if ms_to_idx is None or ms_to_idx.size == 0:
        return 0, num_events

    start_us_rel = int(start_us) - int(t_offset_us)
    end_us_rel = int(end_us) - int(t_offset_us)
    start_ms = max(int(start_us_rel // 1000), 0)
    end_ms_exclusive = max(int((end_us_rel + 999) // 1000), start_ms + 1)
    start_ms = min(start_ms, ms_to_idx.size - 1)
    start_idx = int(ms_to_idx[start_ms])
    if end_ms_exclusive >= ms_to_idx.size:
        end_idx = num_events
    else:
        end_idx = int(ms_to_idx[end_ms_exclusive])
    start_idx = max(0, min(start_idx, num_events))
    end_idx = max(0, min(end_idx, num_events))
    return start_idx, end_idx


def _count_events_in_window(
    *,
    t_ds: h5py.Dataset,
    t_offset: int,
    ms_to_idx: np.ndarray | None,
    start_us: int,
    end_us: int,
) -> int:
    if end_us <= start_us:
        return 0
    num_events = len(t_ds)
    if num_events == 0:
        return 0

    coarse_start, coarse_end = _coarse_bounds_from_ms_to_idx(
        ms_to_idx=ms_to_idx,
        num_events=num_events,
        t_offset_us=t_offset,
        start_us=start_us,
        end_us=end_us,
    )
    if coarse_end <= coarse_start:
        return 0

    t_coarse_abs = t_ds[coarse_start:coarse_end].astype(np.int64) + int(t_offset)
    if t_coarse_abs.size == 0:
        return 0

    rel_start = int(np.searchsorted(t_coarse_abs, start_us, side="left"))
    rel_end = int(np.searchsorted(t_coarse_abs, end_us, side="left"))
    return max(0, rel_end - rel_start)


def _load_timestamps(path: Path, divisor: int) -> np.ndarray:
    arr = np.loadtxt(str(path), dtype=np.int64)
    arr = np.atleast_1d(arr).astype(np.int64, copy=False).reshape(-1)
    if arr.size == 0:
        return arr
    if int(divisor) <= 0:
        raise ValueError("divisor must be > 0")
    if int(divisor) != 1:
        arr = np.floor_divide(arr, int(divisor)).astype(np.int64, copy=False)
    return arr


def _build_image_middle_windows(
    *,
    event_range: TimeRange,
    image_timestamps_us: np.ndarray,
) -> list[tuple[int, int, int]]:
    if image_timestamps_us.size == 0:
        return []
    if image_timestamps_us.size > 1 and np.any(image_timestamps_us[1:] < image_timestamps_us[:-1]):
        raise ValueError("image timestamps must be non-decreasing")

    midpoints = np.empty((max(0, image_timestamps_us.size - 1),), dtype=np.int64)
    for i in range(image_timestamps_us.size - 1):
        midpoints[i] = image_timestamps_us[i] + (image_timestamps_us[i + 1] - image_timestamps_us[i]) // 2

    windows: list[tuple[int, int, int]] = []
    for i in range(image_timestamps_us.size):
        if i == 0:
            start_us = int(event_range.start_us)
        else:
            start_us = int(midpoints[i - 1])
        if i < image_timestamps_us.size - 1:
            end_us = int(midpoints[i])
        else:
            end_us = int(event_range.end_exclusive_us)
        start_us = max(start_us, int(event_range.start_us))
        end_us = min(end_us, int(event_range.end_exclusive_us))
        windows.append((start_us, end_us, int(image_timestamps_us[i])))
    return windows


def _stats_int(name: str, values: np.ndarray) -> str:
    if values.size == 0:
        return f"{name}: empty"
    return (
        f"{name}: n={values.size}, min={int(values.min())}, "
        f"max={int(values.max())}, mean={float(values.mean()):.2f}"
    )


def _suggest_timestamp_paths(sequence_dir: Path) -> list[Path]:
    return [
        sequence_dir / "images/timestamps.txt",
        sequence_dir / "images/left/timestamps.txt",
    ]


def _resolve_sequence_events_h5(dataset_root: Path, split: str, sequence: str) -> Path:
    rel_candidates = [
        Path("events/left/events.h5"),
        Path("events_left/events.h5"),
        Path("events/events.h5"),
        Path("events.h5"),
    ]
    base_candidates = [
        dataset_root / split / sequence,
        dataset_root / f"{split}_events" / sequence,
        dataset_root / sequence,
    ]

    candidates: list[Path] = []
    for base in base_candidates:
        for rel in rel_candidates:
            candidates.append(base / rel)

    for p in candidates:
        if p.exists():
            return p

    # Fallback: recursive search for events.h5 that contains the sequence directory name.
    search_roots = [
        dataset_root / split,
        dataset_root / f"{split}_events",
        dataset_root,
    ]
    found_recursive: list[Path] = []
    for root in search_roots:
        if not root.exists() or not root.is_dir():
            continue
        for p in root.rglob("events.h5"):
            parts = set(p.parts)
            if sequence in parts:
                found_recursive.append(p)

    if len(found_recursive) == 1:
        return found_recursive[0]
    if len(found_recursive) > 1:
        found_recursive = sorted(found_recursive)
        chosen = found_recursive[0]
        print("[WARN] multiple events.h5 candidates found; using the first one:")
        print(f"  -> {chosen}")
        for extra in found_recursive[1:10]:
            print(f"     {extra}")
        if len(found_recursive) > 10:
            print(f"     ... and {len(found_recursive) - 10} more")
        return chosen

    tried = "\n".join([f"  - {p}" for p in candidates])
    raise FileNotFoundError(
        "events.h5 not found. tried direct candidates and recursive scan.\n"
        f"split={split}, sequence={sequence}\n"
        f"direct candidates:\n{tried}\n"
        "tip: pass --events_h5 explicitly if your layout is custom."
    )


def _resolve_sequence_dir_from_events(events_h5: Path) -> Path:
    # .../<sequence>/events/left/events.h5 -> .../<sequence>
    return events_h5.parent.parent.parent


def _compare_with_preprocessed(preprocessed_h5: Path) -> tuple[np.ndarray, str]:
    with h5py.File(str(preprocessed_h5), "r") as out_h5:
        if "window_event_count" not in out_h5:
            raise KeyError("preprocessed file has no window_event_count dataset")
        counts = np.asarray(out_h5["window_event_count"][()], dtype=np.int64)
        txt = _stats_int("preprocessed.window_event_count", counts)
        return counts, txt


def _run_one_divisor(
    *,
    events_h5_path: Path,
    timestamps_path: Path,
    divisor: int,
    max_windows: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(str(events_h5_path), "r") as h5f:
        ev_range = _event_time_range(h5f)
        if ev_range is None:
            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
        t_offset = _read_t_offset(h5f)
        t_ds = h5f["events/t"]
        ms_to_idx = _load_ms_to_idx(h5f)

        image_ts = _load_timestamps(timestamps_path, divisor=divisor)
        windows = _build_image_middle_windows(event_range=ev_range, image_timestamps_us=image_ts)
        if max_windows is not None and max_windows > 0:
            windows = windows[: int(max_windows)]

        counts = np.zeros((len(windows),), dtype=np.int64)
        anchors = np.zeros((len(windows),), dtype=np.int64)
        for i, (s_us, e_us, a_us) in enumerate(windows):
            counts[i] = _count_events_in_window(
                t_ds=t_ds,
                t_offset=t_offset,
                ms_to_idx=ms_to_idx,
                start_us=s_us,
                end_us=e_us,
            )
            anchors[i] = int(a_us)
        return counts, anchors


def main() -> None:
    parser = argparse.ArgumentParser("Debug DSEC image-middle synchronization before/after preprocess.")
    parser.add_argument("--dataset_root", type=Path, default=None, help="DSEC root.")
    parser.add_argument("--split", type=str, default=None, help="Split name, e.g. train/test.")
    parser.add_argument("--sequence", type=str, default=None, help="Sequence name, e.g. interlaken_00_a.")
    parser.add_argument("--events_h5", type=Path, default=None, help="Direct path to raw events.h5.")
    parser.add_argument("--timestamps_path", type=Path, default=None, help="Direct path to image timestamps.txt.")
    parser.add_argument("--preprocessed_h5", type=Path, default=None, help="Optional preprocessed voxel h5 to compare.")
    parser.add_argument(
        "--divisors",
        nargs="+",
        type=int,
        default=[1, 1000],
        help="Candidate divisors for timestamps unit conversion (e.g. 1 for us, 1000 for ns->us).",
    )
    parser.add_argument("--max_windows", type=int, default=None, help="Optional cap for analyzed windows.")
    args = parser.parse_args()

    if args.events_h5 is not None:
        events_h5 = args.events_h5
    else:
        if args.dataset_root is None or args.split is None or args.sequence is None:
            raise ValueError("use --events_h5 OR provide --dataset_root --split --sequence")
        events_h5 = _resolve_sequence_events_h5(args.dataset_root, args.split, args.sequence)
    if not events_h5.exists():
        raise FileNotFoundError(events_h5)

    sequence_dir = _resolve_sequence_dir_from_events(events_h5)
    if args.timestamps_path is not None:
        timestamps_path = args.timestamps_path
    else:
        candidates = _suggest_timestamp_paths(sequence_dir)
        timestamps_path = None
        for p in candidates:
            if p.exists():
                timestamps_path = p
                break
        if timestamps_path is None:
            tried = "\n".join([f"  - {p}" for p in candidates])
            raise FileNotFoundError(f"timestamps.txt not found. tried:\n{tried}")

    print(f"[INFO] events_h5={events_h5}")
    print(f"[INFO] timestamps_path={timestamps_path}")
    if args.preprocessed_h5 is not None:
        print(f"[INFO] preprocessed_h5={args.preprocessed_h5}")

    with h5py.File(str(events_h5), "r") as h5f:
        ev_range = _event_time_range(h5f)
        if ev_range is None:
            print("[INFO] no events in events.h5")
            return
        print(
            "[EVENTS] "
            f"start_us={ev_range.start_us}, end_exclusive_us={ev_range.end_exclusive_us}, "
            f"duration_s={ev_range.duration_s:.3f}, num_events={len(h5f['events/t'])}, "
            f"t_offset={_read_t_offset(h5f)}"
        )
        ms_to_idx = _load_ms_to_idx(h5f)
        if ms_to_idx is None:
            print("[EVENTS] ms_to_idx: missing (coarse search fallback disabled)")
        else:
            print(f"[EVENTS] ms_to_idx: n={len(ms_to_idx)}")

    for div in args.divisors:
        counts, anchors = _run_one_divisor(
            events_h5_path=events_h5,
            timestamps_path=timestamps_path,
            divisor=int(div),
            max_windows=args.max_windows,
        )
        if anchors.size > 0:
            print(
                "[TIMESTAMPS] "
                f"divisor={div} | anchor_min={int(anchors.min())}, anchor_max={int(anchors.max())}, "
                f"n={anchors.size}"
            )
        else:
            print(f"[TIMESTAMPS] divisor={div} | no windows")
        if counts.size > 0:
            zero = int(np.count_nonzero(counts == 0))
            nz = int(np.count_nonzero(counts > 0))
            print(
                "[WINDOWS] "
                f"divisor={div} | zero={zero}/{counts.size}, nonzero={nz}/{counts.size}, "
                f"min={int(counts.min())}, max={int(counts.max())}, mean={float(counts.mean()):.2f}"
            )
            sample_idx = np.linspace(0, counts.size - 1, num=min(5, counts.size), dtype=int)
            sample_txt = ", ".join([f"{int(i)}:{int(counts[i])}" for i in sample_idx])
            print(f"[WINDOWS] divisor={div} | sample_counts={sample_txt}")
        else:
            print(f"[WINDOWS] divisor={div} | empty")

    if args.preprocessed_h5 is not None:
        if not args.preprocessed_h5.exists():
            raise FileNotFoundError(args.preprocessed_h5)
        out_counts, txt = _compare_with_preprocessed(args.preprocessed_h5)
        print(f"[PREPROCESSED] {txt}")
        for div in args.divisors:
            raw_counts, _ = _run_one_divisor(
                events_h5_path=events_h5,
                timestamps_path=timestamps_path,
                divisor=int(div),
                max_windows=None,
            )
            if raw_counts.size != out_counts.size:
                print(
                    "[COMPARE] "
                    f"divisor={div} | length mismatch raw={raw_counts.size} vs preprocessed={out_counts.size}"
                )
                continue
            match = int(np.count_nonzero(raw_counts == out_counts))
            print(
                "[COMPARE] "
                f"divisor={div} | exact_match={match}/{raw_counts.size}, "
                f"mean_abs_diff={float(np.mean(np.abs(raw_counts - out_counts))):.3f}"
            )


if __name__ == "__main__":
    main()
