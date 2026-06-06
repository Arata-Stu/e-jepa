from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def _read_numeric_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            if not row:
                continue
            epoch = str(row.get("epoch", "")).strip()
            if epoch == "" or epoch.lower() == "epoch":
                continue
            rows.append({str(k).strip(): str(v).strip() for k, v in row.items()})
    return rows


def _to_int(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def _to_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _resolve_csv(path: Path) -> Path:
    if path.is_file():
        return path

    pretrain_csv = path / "log_r0.csv"
    downstream_csv = path / "downstream_log.csv"
    if pretrain_csv.exists():
        return pretrain_csv
    if downstream_csv.exists():
        return downstream_csv

    raise FileNotFoundError(
        f"No supported CSV found in {path}. Expected log_r0.csv or downstream_log.csv."
    )


def _kind_from_header(rows: list[dict[str, str]], csv_path: Path) -> str:
    if len(rows) == 0:
        raise ValueError(f"No numeric rows found in {csv_path}")

    keys = set(rows[0].keys())
    if {"epoch", "itr", "loss", "iter-time(ms)", "gpu-time(ms)", "dataload-time(ms)"}.issubset(keys):
        return "pretrain"
    if {"epoch", "train_loss", "val_loss", "lr"}.issubset(keys):
        return "downstream"

    raise ValueError(f"Unsupported CSV schema in {csv_path}: {sorted(keys)}")


def _write_pretrain(writer, rows: list[dict[str, str]], *, ipe: int | None) -> dict[str, int]:
    if ipe is None:
        ipe = max(_to_int(row, "itr") for row in rows) + 1

    losses_by_epoch: dict[int, list[float]] = defaultdict(list)
    step_count = 0
    for row_index, row in enumerate(rows):
        epoch = _to_int(row, "epoch")
        itr = _to_int(row, "itr")
        step = (epoch - 1) * ipe + itr
        loss = _to_float(row, "loss")

        writer.add_scalar("train/loss", loss, step)
        writer.add_scalar("train/loss_total", loss, step)
        writer.add_scalar("time/iter_ms", _to_float(row, "iter-time(ms)"), step)
        writer.add_scalar("time/gpu_ms", _to_float(row, "gpu-time(ms)"), step)
        writer.add_scalar("time/data_ms", _to_float(row, "dataload-time(ms)"), step)
        writer.add_scalar("rebuilt/row_index", row_index, step)

        losses_by_epoch[epoch].append(loss)
        step_count += 1

    for epoch, values in sorted(losses_by_epoch.items()):
        writer.add_scalar("epoch/loss_avg", sum(values) / max(len(values), 1), epoch)
        writer.add_scalar("rebuilt/epoch_row_count", len(values), epoch)

    return {"rows": step_count, "epochs": len(losses_by_epoch)}


def _write_downstream(writer, rows: list[dict[str, str]]) -> dict[str, int]:
    has_semantic = "miou" in rows[0]
    has_depth = "mae" in rows[0] and "rmse" in rows[0]

    for row in rows:
        epoch = _to_int(row, "epoch")
        writer.add_scalar("train/loss", _to_float(row, "train_loss"), epoch)
        writer.add_scalar("val/loss", _to_float(row, "val_loss"), epoch)
        writer.add_scalar("train/lr", _to_float(row, "lr"), epoch)

        if has_semantic:
            writer.add_scalar("val/pixel_acc", _to_float(row, "pixel_acc"), epoch)
            writer.add_scalar("val/miou", _to_float(row, "miou"), epoch)
        if has_depth:
            writer.add_scalar("val/mae", _to_float(row, "mae"), epoch)
            writer.add_scalar("val/rmse", _to_float(row, "rmse"), epoch)

    return {"rows": len(rows), "epochs": len({int(float(row["epoch"])) for row in rows})}


def rebuild_one(path: Path, *, output_name: str, ipe: int | None) -> None:
    from torch.utils.tensorboard import SummaryWriter

    csv_path = _resolve_csv(path)
    run_dir = csv_path.parent
    output_dir = run_dir / output_name
    rows = _read_numeric_rows(csv_path)
    kind = _kind_from_header(rows, csv_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(output_dir))
    try:
        if kind == "pretrain":
            stats = _write_pretrain(writer, rows, ipe=ipe)
        else:
            stats = _write_downstream(writer, rows)
    finally:
        writer.flush()
        writer.close()

    print(
        f"rebuilt {kind}: csv={csv_path} rows={stats['rows']} "
        f"epochs={stats['epochs']} tensorboard={output_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild TensorBoard scalar events from pretrain/downstream CSV logs."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Run directories or CSV files. Directories may contain log_r0.csv or downstream_log.csv.",
    )
    parser.add_argument(
        "--output-name",
        default="tensorboard_rebuilt",
        help="Output directory name created under each run directory.",
    )
    parser.add_argument(
        "--ipe",
        type=int,
        default=None,
        help="Iterations per epoch for pretrain logs. By default inferred from max itr + 1.",
    )
    args = parser.parse_args()

    for raw_path in args.paths:
        rebuild_one(Path(raw_path), output_name=str(args.output_name), ipe=args.ipe)


if __name__ == "__main__":
    main()
