from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _safe_attr(attrs: h5py.AttributeManager, key: str, default: Any = None) -> Any:
    if key not in attrs:
        return default
    value = attrs[key]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _is_voxel_h5(path: Path) -> bool:
    try:
        with h5py.File(str(path), "r") as h5f:
            return "voxels" in h5f and h5f["voxels"].ndim == 4
    except Exception:
        return False


def _select_files(input_path: Path | None, dataset_root: Path | None, recursive: bool) -> list[Path]:
    if input_path is not None and dataset_root is not None:
        raise ValueError("use either --input_path or --dataset_root, not both")
    if input_path is None and dataset_root is None:
        raise ValueError("either --input_path or --dataset_root is required")

    if input_path is not None:
        return [input_path]

    assert dataset_root is not None
    pattern = "**/*.h5" if recursive else "*.h5"
    return sorted([p for p in dataset_root.glob(pattern) if p.is_file()])


def _start_end_us(h5f: h5py.File, n_samples: int) -> tuple[int | None, int | None]:
    if n_samples <= 0:
        return None, None

    if "window_t_start_us" in h5f and "window_t_end_us" in h5f:
        starts = h5f["window_t_start_us"]
        ends = h5f["window_t_end_us"]
        if len(starts) >= n_samples and len(ends) >= n_samples:
            return int(starts[0]), int(ends[n_samples - 1])

    if "anchor_timestamp_us" in h5f:
        anchors = h5f["anchor_timestamp_us"]
        if len(anchors) >= n_samples:
            return int(anchors[0]), int(anchors[n_samples - 1])

    origin = _safe_attr(h5f.attrs, "time_origin_us", None)
    if (
        origin is not None
        and int(origin) >= 0
        and "window_rel_start_us" in h5f
        and "window_rel_end_us" in h5f
    ):
        rel_starts = h5f["window_rel_start_us"]
        rel_ends = h5f["window_rel_end_us"]
        if len(rel_starts) >= n_samples and len(rel_ends) >= n_samples:
            return int(origin) + int(rel_starts[0]), int(origin) + int(rel_ends[n_samples - 1])

    return None, None


def _infer_split_polarity(h5f: h5py.File, channels: int) -> bool:
    attr = _safe_attr(h5f.attrs, "split_polarity", None)
    if attr is not None:
        return bool(int(attr))
    return channels % 2 == 0


def _voxel_to_rgb(voxel: np.ndarray, split_polarity: bool) -> np.ndarray:
    if voxel.ndim != 3:
        raise ValueError(f"voxel must be (C,H,W), got shape={voxel.shape}")

    channels = voxel.shape[0]
    if channels == 0:
        return np.zeros((voxel.shape[1], voxel.shape[2], 3), dtype=np.uint8)

    if split_polarity and channels >= 2:
        half = channels // 2
        pos_map = np.abs(voxel[:half]).sum(axis=0)
        neg_map = np.abs(voxel[half:]).sum(axis=0)
    else:
        signed = voxel.sum(axis=0)
        pos_map = np.clip(signed, 0.0, None)
        neg_map = np.clip(-signed, 0.0, None)

    intensity = pos_map + neg_map
    scale = float(np.percentile(intensity, 99.0)) if intensity.size > 0 else 0.0
    if scale <= 1e-8:
        scale = float(intensity.max()) if intensity.size > 0 else 0.0
    if scale <= 1e-8:
        scale = 1.0

    pos = np.clip(pos_map / scale, 0.0, 1.0)
    neg = np.clip(neg_map / scale, 0.0, 1.0)
    inten = np.clip(intensity / scale, 0.0, 1.0)

    # Gamma for visibility.
    rgb = np.stack([pos, 0.5 * inten, neg], axis=-1)
    rgb = np.sqrt(np.clip(rgb, 0.0, 1.0))
    return (rgb * 255.0).astype(np.uint8)


def _pick_indices(n_samples: int, num_visualizations: int, explicit_indices: list[int] | None) -> list[int]:
    if n_samples <= 0:
        return []

    if explicit_indices is not None and len(explicit_indices) > 0:
        out = sorted(set([int(i) for i in explicit_indices if 0 <= int(i) < n_samples]))
        return out

    k = min(max(0, int(num_visualizations)), n_samples)
    if k == 0:
        return []
    if k == 1:
        return [0]
    return sorted(set(np.linspace(0, n_samples - 1, num=k, dtype=int).tolist()))


def _relative_or_name(path: Path, root: Path | None) -> str:
    if root is None:
        return path.name
    try:
        return str(path.relative_to(root))
    except Exception:
        return path.name


def _write_preview_images(
    *,
    h5f: h5py.File,
    file_path: Path,
    output_dir: Path,
    dataset_root: Path | None,
    n_samples: int,
    num_visualizations: int,
    explicit_indices: list[int] | None,
) -> list[str]:
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("Pillow is required for visualization. Install via `pip install pillow`.") from exc

    voxels = h5f["voxels"]
    channels = int(voxels.shape[1])
    split_polarity = _infer_split_polarity(h5f, channels)
    indices = _pick_indices(
        n_samples=n_samples,
        num_visualizations=num_visualizations,
        explicit_indices=explicit_indices,
    )
    if len(indices) == 0:
        return []

    relative_name = _relative_or_name(file_path, dataset_root).replace("/", "__")
    relative_name = relative_name.replace("\\", "__")
    stem = Path(relative_name).with_suffix("").name
    preview_dir = output_dir / "previews" / stem
    preview_dir.mkdir(parents=True, exist_ok=True)

    written_paths: list[str] = []
    for idx in indices:
        voxel = np.asarray(voxels[idx], dtype=np.float32)
        rgb = _voxel_to_rgb(voxel=voxel, split_polarity=split_polarity)
        out_path = preview_dir / f"window_{idx:06d}.png"
        Image.fromarray(rgb, mode="RGB").save(out_path)
        written_paths.append(str(out_path))
    return written_paths


def _analyze_file(
    *,
    file_path: Path,
    output_dir: Path,
    dataset_root: Path | None,
    with_visualization: bool,
    num_visualizations: int,
    explicit_indices: list[int] | None,
) -> dict[str, Any]:
    with h5py.File(str(file_path), "r") as h5f:
        if "voxels" not in h5f:
            raise ValueError("missing dataset 'voxels'")
        voxels = h5f["voxels"]
        if voxels.ndim != 4:
            raise ValueError(f"voxels must be 4D, got shape={voxels.shape}")

        n_samples, channels, height, width = [int(v) for v in voxels.shape]
        start_us, end_us = _start_end_us(h5f=h5f, n_samples=n_samples)
        duration_us = None if start_us is None or end_us is None else max(0, int(end_us) - int(start_us))
        duration_s = None if duration_us is None else float(duration_us) / 1e6
        windows_per_second = None
        if duration_s is not None and duration_s > 0:
            windows_per_second = float(n_samples) / float(duration_s)

        preview_paths: list[str] = []
        if with_visualization:
            preview_paths = _write_preview_images(
                h5f=h5f,
                file_path=file_path,
                output_dir=output_dir,
                dataset_root=dataset_root,
                n_samples=n_samples,
                num_visualizations=num_visualizations,
                explicit_indices=explicit_indices,
            )

        result = {
            "file": str(file_path),
            "file_size_mb": round(float(file_path.stat().st_size) / (1024.0 * 1024.0), 3),
            "representation": _safe_attr(h5f.attrs, "representation", ""),
            "dtype": str(voxels.dtype),
            "shape": [n_samples, channels, height, width],
            "samples": n_samples,
            "channels": channels,
            "height": height,
            "width": width,
            "t_bins": _safe_attr(h5f.attrs, "t_bins", None),
            "split_polarity": _infer_split_polarity(h5f, channels),
            "window_mode": _safe_attr(h5f.attrs, "window_mode", ""),
            "recording_start_us": start_us,
            "recording_end_us": end_us,
            "estimated_recording_duration_s": duration_s,
            "estimated_windows_per_second": windows_per_second,
            "preview_images": preview_paths,
        }
        return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "file",
        "file_size_mb",
        "representation",
        "dtype",
        "samples",
        "channels",
        "height",
        "width",
        "t_bins",
        "split_polarity",
        "window_mode",
        "recording_start_us",
        "recording_end_us",
        "estimated_recording_duration_s",
        "estimated_windows_per_second",
        "preview_images",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            out = {k: row.get(k, "") for k in columns}
            out["preview_images"] = ";".join(row.get("preview_images", []))
            writer.writerow(out)


def main() -> None:
    parser = argparse.ArgumentParser("Analyze preprocessed voxel H5 files and save quick RGB previews.")
    parser.add_argument("--input_path", type=Path, default=None, help="Single voxel H5 file.")
    parser.add_argument("--dataset_root", type=Path, default=None, help="Root directory to scan for voxel H5 files.")
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recursively scan --dataset_root for *.h5 files.",
    )
    parser.add_argument("--max_files", type=int, default=None, help="Optional cap for number of files to analyze.")
    parser.add_argument("--output_dir", type=Path, default=Path("tmp/voxel_h5_analysis"), help="Output directory.")
    parser.add_argument(
        "--with_visualization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save quick RGB preview images from voxel windows.",
    )
    parser.add_argument(
        "--num_visualizations",
        type=int,
        default=3,
        help="Number of preview windows per file when --visualization_indices is not set.",
    )
    parser.add_argument(
        "--visualization_indices",
        nargs="+",
        type=int,
        default=None,
        help="Explicit window indices to visualize (e.g. 0 10 100).",
    )
    parser.add_argument(
        "--report_json",
        type=Path,
        default=None,
        help="Optional JSON report path (default: <output_dir>/summary.json).",
    )
    parser.add_argument(
        "--report_csv",
        type=Path,
        default=None,
        help="Optional CSV report path (default: <output_dir>/summary.csv).",
    )
    args = parser.parse_args()

    files = _select_files(
        input_path=args.input_path,
        dataset_root=args.dataset_root,
        recursive=bool(args.recursive),
    )
    if args.max_files is not None:
        files = files[: max(0, int(args.max_files))]
    if len(files) == 0:
        raise FileNotFoundError("no input files found")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filtered_files: list[Path] = []
    for file_path in files:
        if _is_voxel_h5(file_path):
            filtered_files.append(file_path)
    if len(filtered_files) == 0:
        raise FileNotFoundError("no voxel h5 files found (requires root dataset 'voxels' with 4D shape)")

    results: list[dict[str, Any]] = []
    for file_path in filtered_files:
        try:
            result = _analyze_file(
                file_path=file_path,
                output_dir=output_dir,
                dataset_root=args.dataset_root,
                with_visualization=bool(args.with_visualization),
                num_visualizations=int(args.num_visualizations),
                explicit_indices=args.visualization_indices,
            )
            results.append(result)
            duration_s = result["estimated_recording_duration_s"]
            duration_txt = "n/a" if duration_s is None else f"{duration_s:.3f}s"
            print(
                "[OK] "
                f"{file_path} | samples={result['samples']} | "
                f"shape={tuple(result['shape'])} | duration={duration_txt}"
            )
        except Exception as exc:
            print(f"[FAILED] {file_path}: {exc}")

    if len(results) == 0:
        raise RuntimeError("all files failed to analyze")

    report_json = args.report_json if args.report_json is not None else (output_dir / "summary.json")
    report_csv = args.report_csv if args.report_csv is not None else (output_dir / "summary.csv")
    report_json.parent.mkdir(parents=True, exist_ok=True)

    with report_json.open("w") as f:
        json.dump(results, f, indent=2)
    _write_csv(path=report_csv, rows=results)

    total_samples = sum(int(r["samples"]) for r in results)
    durations = [r["estimated_recording_duration_s"] for r in results if r["estimated_recording_duration_s"] is not None]
    total_duration = float(sum(durations)) if len(durations) > 0 else None
    total_duration_txt = "n/a" if total_duration is None else f"{total_duration:.3f}s"
    print(
        "[SUMMARY] "
        f"files={len(results)} | total_samples={total_samples} | "
        f"total_estimated_duration={total_duration_txt} | "
        f"json={report_json} | csv={report_csv}"
    )


if __name__ == "__main__":
    main()
