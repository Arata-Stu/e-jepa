from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
JEPA_ENTRY = PROJECT_ROOT / "scripts" / "train" / "run_train.py"
MAE_ENTRY = PROJECT_ROOT / "scripts" / "mae" / "run_mae.py"


def _detect_visible_gpus() -> int:
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cuda_visible_devices:
        # CUDA_VISIBLE_DEVICES can include index remapping tokens.
        tokens = [t.strip() for t in cuda_visible_devices.split(",")]
        tokens = [t for t in tokens if t and t != "-1"]
        if len(tokens) > 0:
            return len(tokens)

    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return int(torch.cuda.device_count())
    except Exception:
        pass

    return 0


def _resolve_nproc_per_node(mode: str) -> int:
    normalized = str(mode).strip().lower()
    if normalized in {"auto", "gpu"}:
        return max(1, _detect_visible_gpus())
    if normalized == "cpu":
        return 1
    try:
        value = int(normalized)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported --nproc-per-node='{mode}'. Use auto|gpu|cpu|<int>."
        ) from exc
    if value < 1:
        raise ValueError("--nproc-per-node must be >= 1")
    return value


def _entry_path(task: str) -> Path:
    if task == "jepa":
        return JEPA_ENTRY
    if task == "mae":
        return MAE_ENTRY
    raise ValueError(f"Unsupported task='{task}'. Use jepa or mae.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Unified launcher for JEPA/MAE pretraining. "
            "Uses torchrun with auto GPU detection so the same command works on 1 or many GPUs."
        )
    )
    parser.add_argument(
        "task",
        choices=["jepa", "mae"],
        help="Pretraining target: jepa or mae.",
    )
    parser.add_argument(
        "--nproc-per-node",
        default="auto",
        help="auto|gpu|cpu|<int> (default: auto)",
    )
    parser.add_argument(
        "--nnodes",
        type=int,
        default=1,
        help="Number of nodes (default: 1).",
    )
    parser.add_argument(
        "--node-rank",
        type=int,
        default=0,
        help="Current node rank for multi-node runs (default: 0).",
    )
    parser.add_argument(
        "--master-addr",
        default="127.0.0.1",
        help="Master address for multi-node runs (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=29500,
        help="Master port for rendezvous (default: 29500).",
    )
    parser.add_argument(
        "--no-standalone",
        action="store_true",
        help="Disable torchrun --standalone (useful for explicit multi-node rendezvous).",
    )

    args, passthrough = parser.parse_known_args()
    nproc_per_node = _resolve_nproc_per_node(args.nproc_per_node)
    entry = _entry_path(args.task)

    if not entry.exists():
        raise FileNotFoundError(f"Entry script not found: {entry}")

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nnodes={args.nnodes}",
        f"--nproc_per_node={nproc_per_node}",
        f"--node_rank={args.node_rank}",
        f"--master_addr={args.master_addr}",
        f"--master_port={args.master_port}",
    ]
    if not args.no_standalone and int(args.nnodes) == 1:
        cmd.append("--standalone")

    cmd.append(str(entry))
    cmd.extend(passthrough)

    print(
        "[launch_pretrain] "
        f"task={args.task} nproc_per_node={nproc_per_node} nnodes={args.nnodes} "
        f"entry={entry}"
    )
    raise SystemExit(subprocess.call(cmd, cwd=str(PROJECT_ROOT)))


if __name__ == "__main__":
    main()
