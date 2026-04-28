from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from pathlib import Path
import sys

import h5py
import hdf5plugin  # noqa: F401 (registers hdf5 plugins)
import numpy as np
import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.preprocess.utils import cleanup_tmp_file, get_h5_compression_flags, tmp_output_path


H5_COMPRESSION_FLAGS = get_h5_compression_flags()


def _is_voxel_h5(path: Path) -> bool:
    try:
        with h5py.File(str(path), "r") as h5f:
            return "voxels" in h5f and h5f["voxels"].ndim == 4 and h5f["voxels"].shape[0] > 0
    except Exception:
        return False


def _select_input_files(
    input_path: Path | None,
    dataset_root: Path | None,
    recursive: bool,
    file_pattern: str,
    max_files: int | None,
) -> list[Path]:
    if input_path is not None and dataset_root is not None:
        raise ValueError("use either --input_path or --dataset_root, not both")
    if input_path is None and dataset_root is None:
        raise ValueError("either --input_path or --dataset_root is required")

    if input_path is not None:
        files = [input_path]
    else:
        assert dataset_root is not None
        iterator = dataset_root.rglob(file_pattern) if recursive else dataset_root.glob(file_pattern)
        files = sorted([p for p in iterator if p.is_file()])

    files = [p for p in files if _is_voxel_h5(p)]
    if max_files is not None and int(max_files) > 0:
        files = files[: int(max_files)]
    return files


def _resolve_output_path(
    input_path: Path,
    dataset_root: Path | None,
    output_root: Path | None,
) -> Path:
    if output_root is None:
        return input_path.parent / f"{input_path.stem}_split" / input_path.name

    if dataset_root is None:
        return output_root / input_path.name

    rel = input_path.relative_to(dataset_root)
    return output_root / rel


def _anchor_times_us(h5f: h5py.File, n_samples: int) -> np.ndarray:
    if n_samples <= 0:
        return np.empty((0,), dtype=np.int64)

    if "anchor_timestamp_us" in h5f and len(h5f["anchor_timestamp_us"]) >= n_samples:
        return np.asarray(h5f["anchor_timestamp_us"][:n_samples], dtype=np.int64)

    if "window_t_start_us" in h5f and "window_t_end_us" in h5f:
        starts = np.asarray(h5f["window_t_start_us"][:n_samples], dtype=np.int64)
        ends = np.asarray(h5f["window_t_end_us"][:n_samples], dtype=np.int64)
        return starts + ((ends - starts) // 2)

    if "window_rel_start_us" in h5f and "window_rel_end_us" in h5f and "time_origin_us" in h5f.attrs:
        origin = int(h5f.attrs["time_origin_us"])
        rel_starts = np.asarray(h5f["window_rel_start_us"][:n_samples], dtype=np.int64)
        rel_ends = np.asarray(h5f["window_rel_end_us"][:n_samples], dtype=np.int64)
        return origin + rel_starts + ((rel_ends - rel_starts) // 2)

    raise KeyError("cannot resolve anchor timestamps from H5 (missing anchor/window time datasets)")


def _contiguous_chunk_ranges(anchors_us: np.ndarray, chunk_duration_us: int) -> list[tuple[int, int]]:
    if anchors_us.size == 0:
        return []
    if int(chunk_duration_us) <= 0:
        raise ValueError("chunk_duration_us must be > 0")
    if anchors_us.size > 1 and np.any(anchors_us[1:] < anchors_us[:-1]):
        raise ValueError("anchor timestamps must be non-decreasing")

    first = int(anchors_us[0])
    chunk_ids = ((anchors_us - first) // int(chunk_duration_us)).astype(np.int64, copy=False)
    change = np.where(chunk_ids[1:] != chunk_ids[:-1])[0] + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [anchors_us.size]))
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e > s]


def _dataset_create_kwargs(src_ds: h5py.Dataset, out_shape: tuple[int, ...]) -> dict:
    if len(out_shape) == 0:
        return {}

    kwargs: dict = {}
    if src_ds.ndim > 0 and src_ds.shape[0] > 0:
        if src_ds.chunks is not None:
            chunks = list(src_ds.chunks)
            chunks[0] = min(max(1, chunks[0]), max(1, out_shape[0]))
            for i in range(1, min(len(chunks), len(out_shape))):
                chunks[i] = min(max(1, chunks[i]), max(1, out_shape[i]))
            kwargs["chunks"] = tuple(chunks[: len(out_shape)])
        else:
            kwargs["chunks"] = True

    kwargs.update(H5_COMPRESSION_FLAGS)
    return kwargs


def _copy_attrs(src, dst) -> None:
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def _copy_dataset_slice(
    src_ds: h5py.Dataset,
    dst_ds: h5py.Dataset,
    src_start: int,
    src_end: int,
    copy_batch_size: int,
) -> None:
    if src_ds.ndim == 0:
        dst_ds[()] = src_ds[()]
        return
    if src_end <= src_start:
        return

    copy_batch_size = max(1, int(copy_batch_size))
    out_pos = 0
    for s in range(int(src_start), int(src_end), copy_batch_size):
        e = min(int(src_end), s + copy_batch_size)
        data = src_ds[s:e]
        dst_ds[out_pos : out_pos + (e - s)] = data
        out_pos += (e - s)


def _create_dataset_with_fallback(
    out_h5: h5py.File,
    name: str,
    shape: tuple[int, ...],
    dtype,
    create_kwargs: dict,
) -> h5py.Dataset:
    try:
        return out_h5.create_dataset(name, shape=shape, dtype=dtype, **create_kwargs)
    except Exception:
        # Some HDF5 dtypes (notably variable-length strings) may not support
        # filter/chunk combinations consistently across environments.
        return out_h5.create_dataset(name, shape=shape, dtype=dtype)


def _write_chunk_file(
    *,
    src_h5_path: Path,
    out_h5_path: Path,
    src_start: int,
    src_end: int,
    copy_batch_size: int,
    chunk_index: int,
    total_chunks: int,
    chunk_duration_s: float,
) -> None:
    out_h5_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_output_path(out_h5_path, ".tmp")
    cleanup_tmp_file(tmp_path, context=f"start chunk write {out_h5_path}", strict=True)

    with h5py.File(str(src_h5_path), "r") as src_h5:
        with h5py.File(str(tmp_path), "w") as out_h5:
            if "voxels" not in src_h5:
                raise KeyError(f"missing voxels in source: {src_h5_path}")
            n_samples = int(src_h5["voxels"].shape[0])
            if src_end > n_samples:
                raise ValueError(
                    f"slice exceeds source size: [{src_start}:{src_end}) vs n_samples={n_samples}, file={src_h5_path}"
                )
            _copy_attrs(src_h5, out_h5)

            group_names: list[str] = []
            dataset_names: list[str] = []

            def _visitor(name: str, obj) -> None:
                if isinstance(obj, h5py.Group):
                    group_names.append(name)
                elif isinstance(obj, h5py.Dataset):
                    dataset_names.append(name)

            src_h5.visititems(_visitor)

            for gname in sorted(group_names):
                out_group = out_h5.require_group(gname)
                _copy_attrs(src_h5[gname], out_group)

            for dname in sorted(dataset_names):
                src_ds = src_h5[dname]
                row_aligned = src_ds.ndim > 0 and src_ds.shape[0] == n_samples

                if row_aligned:
                    out_shape = (int(src_end - src_start),) + tuple(src_ds.shape[1:])
                else:
                    out_shape = tuple(src_ds.shape)

                parent_name = dname.rsplit("/", 1)[0] if "/" in dname else ""
                if parent_name:
                    out_h5.require_group(parent_name)

                create_kwargs = _dataset_create_kwargs(src_ds, out_shape=out_shape)
                out_ds = _create_dataset_with_fallback(
                    out_h5=out_h5,
                    name=dname,
                    shape=out_shape,
                    dtype=src_ds.dtype,
                    create_kwargs=create_kwargs,
                )
                _copy_attrs(src_ds, out_ds)

                if src_ds.ndim == 0:
                    out_ds[()] = src_ds[()]
                elif row_aligned:
                    _copy_dataset_slice(
                        src_ds=src_ds,
                        dst_ds=out_ds,
                        src_start=src_start,
                        src_end=src_end,
                        copy_batch_size=copy_batch_size,
                    )
                else:
                    out_ds[...] = src_ds[...]

            out_h5.attrs["split_parent_file"] = str(src_h5_path)
            out_h5.attrs["split_chunk_index"] = int(chunk_index)
            out_h5.attrs["split_num_chunks"] = int(total_chunks)
            out_h5.attrs["split_chunk_duration_s"] = float(chunk_duration_s)
            out_h5.attrs["split_source_start_index"] = int(src_start)
            out_h5.attrs["split_source_end_index_exclusive"] = int(src_end)
            out_h5.attrs["split_source_num_samples"] = int(n_samples)

    os.replace(tmp_path, out_h5_path)


def _process_one_file(
    *,
    input_path: Path,
    output_base_path: Path,
    chunk_duration_s: float,
    copy_batch_size: int,
    min_windows_per_chunk: int,
    chunk_index_pad: int,
    overwrite: bool,
) -> tuple[str, int, int]:
    with h5py.File(str(input_path), "r") as h5f:
        n_samples = int(h5f["voxels"].shape[0])
        anchors = _anchor_times_us(h5f=h5f, n_samples=n_samples)

    chunk_duration_us = int(round(float(chunk_duration_s) * 1e6))
    ranges = _contiguous_chunk_ranges(anchors_us=anchors, chunk_duration_us=chunk_duration_us)
    if int(min_windows_per_chunk) > 1:
        ranges = [(s, e) for (s, e) in ranges if (e - s) >= int(min_windows_per_chunk)]

    done = 0
    for chunk_idx, (src_start, src_end) in enumerate(ranges):
        out_name = f"{output_base_path.stem}_part{chunk_idx:0{int(chunk_index_pad)}d}{output_base_path.suffix}"
        out_path = output_base_path.with_name(out_name)

        if out_path.exists():
            if overwrite:
                out_path.unlink()
            else:
                continue

        _write_chunk_file(
            src_h5_path=input_path,
            out_h5_path=out_path,
            src_start=src_start,
            src_end=src_end,
            copy_batch_size=copy_batch_size,
            chunk_index=chunk_idx,
            total_chunks=len(ranges),
            chunk_duration_s=float(chunk_duration_s),
        )
        done += 1

    return str(input_path), int(n_samples), int(done)


def _worker(job: dict) -> tuple[str, bool, str | None, int, int]:
    try:
        input_path = Path(job["input_path"])
        output_base_path = Path(job["output_base_path"])
        file_path, n_samples, n_written = _process_one_file(
            input_path=input_path,
            output_base_path=output_base_path,
            chunk_duration_s=float(job["chunk_duration_s"]),
            copy_batch_size=int(job["copy_batch_size"]),
            min_windows_per_chunk=int(job["min_windows_per_chunk"]),
            chunk_index_pad=int(job["chunk_index_pad"]),
            overwrite=bool(job["overwrite"]),
        )
        return file_path, True, None, n_samples, n_written
    except Exception as exc:
        return str(job.get("input_path", "")), False, str(exc), 0, 0


def main() -> None:
    parser = argparse.ArgumentParser("Split preprocessed voxel H5 files into shorter files by duration.")
    parser.add_argument("--input_path", type=Path, default=None, help="Single input voxel H5.")
    parser.add_argument("--dataset_root", type=Path, default=None, help="Root directory for batch split.")
    parser.add_argument("--output_root", type=Path, default=None, help="Optional output root (recommended).")
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True, help="Recursive scan.")
    parser.add_argument("--file_pattern", type=str, default="*.h5", help="Glob pattern under dataset_root.")
    parser.add_argument("--max_files", type=int, default=None, help="Optional cap on files to process.")
    parser.add_argument(
        "--chunk_duration_s",
        type=float,
        default=20.0,
        help="Target chunk duration in seconds.",
    )
    parser.add_argument(
        "--min_windows_per_chunk",
        type=int,
        default=1,
        help="Drop chunks with fewer than this number of windows.",
    )
    parser.add_argument("--chunk_index_pad", type=int, default=4, help="Zero padding for part index.")
    parser.add_argument(
        "--copy_batch_size",
        type=int,
        default=8,
        help="Rows copied per I/O batch while slicing datasets (smaller uses less RAM).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing split files.")
    parser.add_argument("--num_processes", type=int, default=1, help="Parallel workers.")
    args = parser.parse_args()

    if float(args.chunk_duration_s) <= 0:
        raise ValueError("--chunk_duration_s must be > 0")
    if int(args.num_processes) < 1:
        raise ValueError("--num_processes must be >= 1")
    if int(args.min_windows_per_chunk) < 1:
        raise ValueError("--min_windows_per_chunk must be >= 1")
    if int(args.copy_batch_size) < 1:
        raise ValueError("--copy_batch_size must be >= 1")

    input_files = _select_input_files(
        input_path=args.input_path,
        dataset_root=args.dataset_root,
        recursive=bool(args.recursive),
        file_pattern=str(args.file_pattern),
        max_files=args.max_files,
    )
    if len(input_files) == 0:
        raise FileNotFoundError("no voxel h5 files found to split")

    jobs: list[dict] = []
    for input_path in input_files:
        output_base_path = _resolve_output_path(
            input_path=input_path,
            dataset_root=args.dataset_root,
            output_root=args.output_root,
        )
        output_base_path.parent.mkdir(parents=True, exist_ok=True)
        jobs.append(
            {
                "input_path": str(input_path),
                "output_base_path": str(output_base_path),
                "chunk_duration_s": float(args.chunk_duration_s),
                "copy_batch_size": int(args.copy_batch_size),
                "min_windows_per_chunk": int(args.min_windows_per_chunk),
                "chunk_index_pad": int(args.chunk_index_pad),
                "overwrite": bool(args.overwrite),
            }
        )

    num_ok = 0
    num_fail = 0
    total_input_samples = 0
    total_output_files = 0

    if int(args.num_processes) == 1:
        iterator = (_worker(job) for job in jobs)
        pbar = tqdm.tqdm(iterator, total=len(jobs), desc="split voxel h5")
    else:
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(processes=int(args.num_processes))
        pbar = tqdm.tqdm(pool.imap_unordered(_worker, jobs), total=len(jobs), desc="split voxel h5")

    try:
        for file_path, ok, err, n_samples, n_written in pbar:
            if ok:
                num_ok += 1
                total_input_samples += int(n_samples)
                total_output_files += int(n_written)
            else:
                num_fail += 1
                print(f"[FAILED] {file_path}: {err}")
    finally:
        if int(args.num_processes) > 1:
            pool.close()
            pool.join()

    print(
        "[SUMMARY] "
        f"files_ok={num_ok}, files_failed={num_fail}, "
        f"input_samples={total_input_samples}, output_files={total_output_files}"
    )
    if num_fail > 0:
        raise RuntimeError(f"{num_fail} files failed during split")


if __name__ == "__main__":
    main()
