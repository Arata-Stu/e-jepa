from __future__ import annotations

import os
import sys
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _save_resolved_config(cfg: DictConfig) -> None:
    rank = int(os.environ.get("RANK", "0"))
    if rank != 0:
        return

    folder = cfg.get("folder", None)
    if folder is None:
        return

    output_dir = Path(str(folder))
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = output_dir / "params-mae-resolved.yaml"
    cfg_path.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg.folder = HydraConfig.get().runtime.output_dir
    _save_resolved_config(cfg)

    from scripts.mae.train import main as train_main

    args = OmegaConf.to_container(cfg, resolve=True)
    train_main(args=args, resume_preempt=False)


if __name__ == "__main__":
    main()
