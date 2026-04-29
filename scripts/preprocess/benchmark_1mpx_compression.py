from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def _find_input_files(dataset_root: Path, splits: list[str], recursive: bool) -> list[Path]:
    files: list[Path] = []
    for split in splits:
        if split in {"", ".", "./"}:
            split_dir = dataset_root
        else:
            split_dir = dataset_root / split
        if not split_dir.exists():
            print(f"[WARN] missing split dir: {split_dir}")
            continue

        iterator = split_dir.rglob("*.h5") if recursive else split_dir.glob("*.h5")
        for path in iterator:
            if not path.is_file():
                continue
            # Keep raw inputs only when possible.
            if "_voxels" in path.name or path.name.endswith("_1x.h5") or path.name.endswith("_2x.h5"):
                continue
            files.append(path)
    return sorted(files)


def _select_subset(files: list[Path], max_files: int, shuffle: bool, seed: int) -> list[Path]:
    if max_files <= 0 or max_files >= len(files):
        return files

    work = list(files)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(work)
    return sorted(work[:max_files])


def _bytes_to_gib(value: int) -> float:
    return float(value) / (1024.0 ** 3)


def _bytes_to_mib(value: int) -> float:
    return float(value) / (1024.0 ** 2)


def main() -> None:
    parser = argparse.ArgumentParser("Benchmark 1MPX preprocess output size across compression levels")
    parser.add_argument("--dataset_root", type=Path, required=True, help="Root containing split directories")
    parser.add_argument("--output_root", type=Path, required=True, help="Benchmark output root")
    parser.add_argument("--splits", nargs="+", default=["train", "test", "val"], help="Split names")
    parser.add_argument("--recursive", action="store_true", help="Recursively search .h5 under split dirs")
    parser.add_argument("--max_files", type=int, default=3, help="Number of files used for benchmark")
    parser.add_argument("--shuffle_files", action="store_true", help="Shuffle before subset selection")
    parser.add_argument("--seed", type=int, default=42, help="Seed for shuffled subset selection")
    parser.add_argument(
        "--compression_levels",
        nargs="+",
        type=int,
        default=[1, 3, 5, 7, 9],
        help="Compression levels in [0,9] to compare",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="_voxels.h5",
        help="Output suffix before auto scale tag insertion",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")

    parser.add_argument("--input_height", type=int, default=720)
    parser.add_argument("--input_width", type=int, default=1280)
    parser.add_argument("--output_height", type=int, default=720)
    parser.add_argument("--output_width", type=int, default=1280)
    parser.add_argument("--downsample_factor", type=int, choices=[1, 2], default=2)
    parser.add_argument("--t_bins", type=int, default=10)
    parser.add_argument("--split_polarity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--accum_time", type=int, default=50000)
    parser.add_argument("--stride_time", type=int, default=None)
    parser.add_argument("--start_time_us", type=int, default=None)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--tmp_suffix", type=str, default=".tmp")
    parser.add_argument("--report_csv", type=Path, default=None, help="Optional CSV output path")
    args = parser.parse_args()

    from scripts.preprocess.preprocess_1mpx import process_single_file
    from scripts.preprocess.utils import ensure_scale_tag_in_filename

    levels = [int(x) for x in args.compression_levels]
    if len(levels) == 0:
        raise ValueError("compression_levels must not be empty")
    for level in levels:
        if level < 0 or level > 9:
            raise ValueError(f"compression level must be in [0,9], got {level}")

    all_files = _find_input_files(
        dataset_root=args.dataset_root,
        splits=[str(x) for x in args.splits],
        recursive=bool(args.recursive),
    )
    if len(all_files) == 0:
        raise FileNotFoundError(f"No input .h5 files found under {args.dataset_root} with splits={args.splits}")

    selected_files = _select_subset(
        files=all_files,
        max_files=int(args.max_files),
        shuffle=bool(args.shuffle_files),
        seed=int(args.seed),
    )
    if len(selected_files) == 0:
        raise RuntimeError("No files selected for benchmark")

    args.output_root.mkdir(parents=True, exist_ok=True)
    stride_time = args.accum_time if args.stride_time is None else int(args.stride_time)

    print("[INFO] selected files:")
    for path in selected_files:
        print(f"  - {path}")

    results: list[dict[str, float | int]] = []
    baseline_total: int | None = None

    for level in levels:
        level_total_bytes = 0
        level_dir = args.output_root / f"level_{level}"
        level_dir.mkdir(parents=True, exist_ok=True)

        print(f"[RUN] compression_level={level}")
        for input_path in selected_files:
            rel = input_path.relative_to(args.dataset_root)
            out_dir = level_dir / rel.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            output_name = ensure_scale_tag_in_filename(
                f"{input_path.stem}{args.output_suffix}",
                downsample_factor=int(args.downsample_factor),
            )
            output_path = out_dir / output_name

            if output_path.exists() and not bool(args.overwrite):
                size_bytes = output_path.stat().st_size
                level_total_bytes += size_bytes
                print(f"  [SKIP] {output_path} ({_bytes_to_mib(size_bytes):.2f} MiB)")
                continue

            process_single_file(
                input_path=input_path,
                output_path=output_path,
                input_height=int(args.input_height),
                input_width=int(args.input_width),
                output_height=int(args.output_height),
                output_width=int(args.output_width),
                downsample_factor=int(args.downsample_factor),
                t_bins=int(args.t_bins),
                split_polarity=bool(args.split_polarity),
                accum_time=int(args.accum_time),
                stride_time=int(stride_time),
                start_time_us=None if args.start_time_us is None else int(args.start_time_us),
                normalize=bool(args.normalize),
                output_dtype=str(args.output_dtype),
                compression_level=int(level),
                show_progress=False,
                tmp_suffix=str(args.tmp_suffix),
            )

            size_bytes = output_path.stat().st_size
            level_total_bytes += size_bytes
            print(f"  [OK] {output_path} ({_bytes_to_mib(size_bytes):.2f} MiB)")

        if baseline_total is None:
            baseline_total = level_total_bytes
        ratio = (float(level_total_bytes) / float(baseline_total)) if baseline_total > 0 else 1.0
        results.append(
            {
                "compression_level": int(level),
                "num_files": int(len(selected_files)),
                "total_bytes": int(level_total_bytes),
                "total_gib": _bytes_to_gib(level_total_bytes),
                "avg_mib_per_file": _bytes_to_mib(level_total_bytes) / max(len(selected_files), 1),
                "ratio_vs_first_level": ratio,
            }
        )

    print("\n[SUMMARY]")
    print("level | files | total_GiB | avg_MiB/file | ratio_vs_first_level")
    for row in results:
        print(
            f"{row['compression_level']:>5} | {row['num_files']:>5} | "
            f"{row['total_gib']:>9.3f} | {row['avg_mib_per_file']:>12.2f} | "
            f"{row['ratio_vs_first_level']:.3f}"
        )

    if args.report_csv is not None:
        args.report_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "compression_level",
            "num_files",
            "total_bytes",
            "total_gib",
            "avg_mib_per_file",
            "ratio_vs_first_level",
        ]
        with open(args.report_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        print(f"[REPORT] csv={args.report_csv}")


if __name__ == "__main__":
    main()
