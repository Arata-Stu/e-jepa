from __future__ import annotations

import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

import src.models.vision_transformer as video_vit
from src.downstream.datasets import EventDenseTaskDataset
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.distributed import init_distributed
from src.utils.logging import CSVLogger, AverageMeter, get_logger

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


logger = get_logger(__name__, force=True)


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def _to_dtype(dtype_name: str) -> tuple[torch.dtype, bool]:
    name = str(dtype_name).lower()
    if name == "bfloat16":
        return torch.bfloat16, True
    if name == "float16":
        return torch.float16, True
    return torch.float32, False


def _setup_device() -> torch.device:
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        return device
    return torch.device("cpu")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _all_reduce_tensor(x: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
    if _is_distributed():
        dist.all_reduce(x, op=op)
    return x


def _save_checkpoint(
    *,
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.cuda.amp.GradScaler | None,
    epoch: int,
    best_metric: float,
    rank: int,
) -> None:
    if rank != 0:
        return
    model_to_save = model.module if isinstance(model, DistributedDataParallel) else model
    save_dict = {
        "model": model_to_save.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "epoch": int(epoch),
        "best_metric": float(best_metric),
    }
    torch.save(save_dict, str(path))


def _load_train_checkpoint(
    *,
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.cuda.amp.GradScaler | None,
) -> tuple[int, float]:
    ckpt = robust_checkpoint_loader(str(path), map_location=torch.device("cpu"))
    model_to_load = model.module if isinstance(model, DistributedDataParallel) else model
    msg = model_to_load.load_state_dict(ckpt["model"], strict=False)
    logger.info(f"Loaded downstream train checkpoint with msg: {msg}")
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler", None) is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler", None) is not None:
        scaler.load_state_dict(ckpt["scaler"])
    epoch = int(ckpt.get("epoch", 0))
    best_metric = float(ckpt.get("best_metric", -1e18))
    return epoch, best_metric


def _strip_encoder_key(key: str) -> str:
    out = key
    changed = True
    while changed:
        changed = False
        for prefix in ("module.", "backbone."):
            if out.startswith(prefix):
                out = out[len(prefix) :]
                changed = True
    return out


def _load_pretrained_encoder(
    *,
    encoder: nn.Module,
    checkpoint_path: str,
    checkpoint_key: str = "encoder",
) -> None:
    ckpt = robust_checkpoint_loader(checkpoint_path, map_location=torch.device("cpu"))
    loaded = ckpt.get(checkpoint_key, ckpt)
    if not isinstance(loaded, dict):
        raise ValueError(
            f"Expected dict at checkpoint key '{checkpoint_key}', got {type(loaded)}"
        )
    loaded = {_strip_encoder_key(k): v for k, v in loaded.items()}

    model_state = encoder.state_dict()
    aligned = {}
    missing = 0
    mismatch = 0
    for k, v in model_state.items():
        if k not in loaded:
            aligned[k] = v
            missing += 1
            continue
        if loaded[k].shape != v.shape:
            aligned[k] = v
            mismatch += 1
            continue
        aligned[k] = loaded[k]

    msg = encoder.load_state_dict(aligned, strict=False)
    logger.info(
        "Loaded pretrained encoder '%s' (missing=%d, shape_mismatch=%d) msg=%s",
        checkpoint_path,
        missing,
        mismatch,
        msg,
    )


class DenseLinearProbe(nn.Module):
    def __init__(
        self,
        *,
        encoder: nn.Module,
        num_output_channels: int,
        freeze_encoder: bool,
        patch_size: int,
        head_dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.freeze_encoder = bool(freeze_encoder)
        self.patch_size = int(patch_size)
        self.head_dropout = nn.Dropout2d(head_dropout) if head_dropout > 0 else nn.Identity()
        self.head = nn.Conv2d(
            in_channels=int(getattr(encoder, "embed_dim")),
            out_channels=int(num_output_channels),
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        if self.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def _token_time_dim(self, x: torch.Tensor) -> int:
        if hasattr(self.encoder, "check_temporal_dim") and callable(self.encoder.check_temporal_dim):
            if self.encoder.check_temporal_dim(x.shape):
                return int(x.shape[2])
        tubelet_size = int(getattr(self.encoder, "tubelet_size", 1))
        return max(1, int(x.shape[2]) // max(1, tubelet_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,T,H,W]
        b, _, _, h, w = x.shape
        hp = int(h) // self.patch_size
        wp = int(w) // self.patch_size
        if hp <= 0 or wp <= 0:
            raise ValueError(f"Invalid input HxW=({h},{w}) for patch_size={self.patch_size}")

        if self.freeze_encoder:
            self.encoder.eval()
            with torch.no_grad():
                tokens = self.encoder(x, masks=None, training=False)
        else:
            tokens = self.encoder(x, masks=None, training=False)

        if isinstance(tokens, list):
            tokens = tokens[-1]
        if tokens.ndim != 3:
            raise ValueError(f"Expected encoder output [B,N,D], got shape={tokens.shape}")

        _, n, d = tokens.shape
        tp = self._token_time_dim(x)
        expected = tp * hp * wp
        if n != expected:
            if hp * wp > 0 and (n % (hp * wp) == 0):
                tp = n // (hp * wp)
            else:
                raise ValueError(
                    f"Token shape mismatch: N={n}, expected {expected} (=Tp*Hp*Wp) "
                    f"with Tp={tp}, Hp={hp}, Wp={wp}"
                )

        feat = tokens.view(b, tp, hp, wp, d).mean(dim=1).permute(0, 3, 1, 2).contiguous()
        feat = self.head_dropout(feat)
        out = self.head(feat)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
        return out


def _build_encoder_from_cfg(cfg_model: dict) -> nn.Module:
    model_name = str(cfg_model.get("model_name", "vit_base"))
    img_size = cfg_model.get("img_size", [480, 640])
    if isinstance(img_size, int):
        img_size = (int(img_size), int(img_size))
    else:
        img_size = (int(img_size[0]), int(img_size[1]))
    patch_size = int(cfg_model.get("patch_size", 16))
    num_frames = int(cfg_model.get("num_frames", 16))
    tubelet_size = int(cfg_model.get("tubelet_size", 2))
    in_chans = int(cfg_model.get("in_chans", 20))
    uniform_power = bool(cfg_model.get("uniform_power", True))
    use_sdpa = bool(cfg_model.get("use_sdpa", True))
    use_rope = bool(cfg_model.get("use_rope", True))
    use_silu = bool(cfg_model.get("use_silu", False))
    wide_silu = bool(cfg_model.get("wide_silu", True))
    is_causal = bool(cfg_model.get("is_causal", False))
    init_type = str(cfg_model.get("init_type", "default"))
    img_temporal_dim_size = cfg_model.get("img_temporal_dim_size", None)
    if img_temporal_dim_size is not None:
        img_temporal_dim_size = int(img_temporal_dim_size)
    n_registers = int(cfg_model.get("n_registers", 0))
    has_cls_first = bool(cfg_model.get("has_cls_first", False))
    interpolate_rope = bool(cfg_model.get("interpolate_rope", False))
    modality_embedding = bool(cfg_model.get("modality_embedding", False))
    use_activation_checkpointing = bool(cfg_model.get("use_activation_checkpointing", False))

    encoder = video_vit.__dict__[model_name](
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        use_rope=use_rope,
        use_silu=use_silu,
        wide_silu=wide_silu,
        is_causal=is_causal,
        init_type=init_type,
        img_temporal_dim_size=img_temporal_dim_size,
        n_registers=n_registers,
        has_cls_first=has_cls_first,
        interpolate_rope=interpolate_rope,
        modality_embedding=modality_embedding,
        use_activation_checkpointing=use_activation_checkpointing,
    )
    return encoder


def _semantic_confusion_update(
    conf: torch.Tensor,
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    *,
    num_classes: int,
    ignore_index: int,
) -> None:
    pred = pred_logits.argmax(dim=1)
    mask = target != int(ignore_index)
    if not torch.any(mask):
        return
    t = target[mask].reshape(-1)
    p = pred[mask].reshape(-1)
    valid = (t >= 0) & (t < int(num_classes))
    if not torch.any(valid):
        return
    t = t[valid]
    p = p[valid]
    idx = t * int(num_classes) + p
    bincount = torch.bincount(idx, minlength=int(num_classes) * int(num_classes))
    conf += bincount.reshape(int(num_classes), int(num_classes))


def _semantic_metrics_from_confusion(conf: torch.Tensor) -> tuple[float, float]:
    conf = conf.to(torch.float64)
    diag = torch.diag(conf)
    gt = conf.sum(dim=1)
    pred = conf.sum(dim=0)
    union = gt + pred - diag
    iou = torch.zeros_like(diag)
    valid = union > 0
    iou[valid] = diag[valid] / union[valid]
    miou = float(iou[valid].mean().item()) if torch.any(valid) else 0.0
    pix_acc = float((diag.sum() / gt.sum()).item()) if gt.sum() > 0 else 0.0
    return pix_acc, miou


def _resize_logits_to_target(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    mode: str,
) -> torch.Tensor:
    target_hw = tuple(int(v) for v in target.shape[-2:])
    if tuple(int(v) for v in logits.shape[-2:]) == target_hw:
        return logits
    mode = str(mode).lower()
    kwargs = {"size": target_hw, "mode": mode}
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
    return F.interpolate(logits.float(), **kwargs)


def _depth_metrics_reduce(
    abs_err_sum: torch.Tensor,
    sq_err_sum: torch.Tensor,
    valid_count: torch.Tensor,
) -> tuple[float, float]:
    abs_err_sum = _all_reduce_tensor(abs_err_sum)
    sq_err_sum = _all_reduce_tensor(sq_err_sum)
    valid_count = _all_reduce_tensor(valid_count)
    count = float(valid_count.item())
    if count <= 0:
        return 0.0, 0.0
    mae = float((abs_err_sum / count).item())
    rmse = float(torch.sqrt(sq_err_sum / count).item())
    return mae, rmse


def _infer_num_classes(
    dataset: EventDenseTaskDataset,
    *,
    ignore_index: int,
    max_samples: int = 256,
) -> int:
    if len(dataset) == 0:
        raise RuntimeError("Cannot infer num_classes from empty dataset")
    step = max(1, len(dataset) // max(1, int(max_samples)))
    observed = -1
    scanned = 0
    for i in range(0, len(dataset), step):
        y = dataset[i]["target"]
        if y.ndim != 2:
            continue
        valid = y != int(ignore_index)
        if torch.any(valid):
            observed = max(observed, int(y[valid].max().item()))
        scanned += 1
        if scanned >= int(max_samples):
            break
    if observed < 0:
        raise RuntimeError(
            "Failed to infer num_classes. No valid semantic labels found in sampled subset."
        )
    return int(observed + 1)


def main(args: dict):
    cfg_meta = args.get("meta", {})
    cfg_model = args.get("model", {})
    cfg_task = args.get("task", {})
    cfg_opt = args.get("optimization", {})
    folder = str(args.get("folder", "outputs/downstream"))

    dtype, mixed_precision = _to_dtype(str(cfg_meta.get("dtype", "float32")))
    enable_tensorboard = bool(cfg_meta.get("tensorboard", True))
    log_freq = int(cfg_meta.get("log_freq", 20))
    checkpoint_freq = int(cfg_meta.get("checkpoint_freq", 1))
    save_every_freq = int(cfg_meta.get("save_every_freq", -1))
    load_checkpoint = bool(cfg_meta.get("load_checkpoint", False))
    auto_resume_latest = bool(cfg_meta.get("auto_resume_latest", False))
    read_checkpoint = cfg_meta.get("read_checkpoint", None)
    dist_port = int(cfg_meta.get("dist_port", 37139))
    seed = int(cfg_meta.get("seed", 0))

    _seed_everything(seed)
    if cfg_meta.get("use_tf32", True):
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass
    torch.backends.cudnn.benchmark = True

    world_size, rank = init_distributed(port=dist_port)
    device = _setup_device()

    if rank != 0 and not bool(cfg_meta.get("log_all_ranks", False)):
        import logging

        logger.setLevel(logging.WARNING)

    output_dir = Path(folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / "latest_downstream.pth.tar"
    best_path = output_dir / "best_downstream.pth.tar"

    target = str(cfg_task.get("target", "semantic")).lower()
    dataset_kind = str(cfg_task.get("dataset_kind", "dsec")).lower()
    clip_num_frames = int(cfg_task.get("clip_num_frames", 2))
    clip_frame_stride = int(cfg_task.get("clip_frame_stride", 1))

    tubelet_size = int(cfg_model.get("tubelet_size", 2))
    img_temporal_dim_size = cfg_model.get("img_temporal_dim_size", None)
    if img_temporal_dim_size is not None:
        img_temporal_dim_size = int(img_temporal_dim_size)
    if clip_num_frames < tubelet_size and (
        img_temporal_dim_size is None or clip_num_frames != img_temporal_dim_size
    ):
        raise ValueError(
            f"task.clip_num_frames={clip_num_frames} is smaller than model.tubelet_size={tubelet_size}. "
            "Increase clip_num_frames or set model.img_temporal_dim_size to use the image branch."
        )

    train_roots = cfg_task.get("train_roots", [])
    val_roots = cfg_task.get("val_roots", [])
    if len(train_roots) == 0 or len(val_roots) == 0:
        raise ValueError("task.train_roots and task.val_roots must be non-empty lists.")

    train_dataset = EventDenseTaskDataset(
        roots=train_roots,
        dataset_kind=dataset_kind,
        target=target,
        clip_num_frames=clip_num_frames,
        clip_frame_stride=clip_frame_stride,
        file_pattern=str(cfg_task.get("file_pattern", "*.h5")),
        recursive=bool(cfg_task.get("recursive", True)),
        ignore_index=int(cfg_task.get("ignore_index", 255)),
        depth_scale=float(cfg_task.get("depth_scale", 1.0)),
        require_labels=bool(cfg_task.get("require_labels", True)),
        input_size=cfg_task.get("input_size", None),
        input_resize_mode=str(cfg_task.get("input_resize_mode", "bilinear")),
        return_eval_target=False,
    )
    val_dataset = EventDenseTaskDataset(
        roots=val_roots,
        dataset_kind=dataset_kind,
        target=target,
        clip_num_frames=clip_num_frames,
        clip_frame_stride=clip_frame_stride,
        file_pattern=str(cfg_task.get("file_pattern", "*.h5")),
        recursive=bool(cfg_task.get("recursive", True)),
        ignore_index=int(cfg_task.get("ignore_index", 255)),
        depth_scale=float(cfg_task.get("depth_scale", 1.0)),
        require_labels=bool(cfg_task.get("require_labels", True)),
        input_size=cfg_task.get("input_size", None),
        input_resize_mode=str(cfg_task.get("input_resize_mode", "bilinear")),
        return_eval_target=bool(cfg_task.get("eval_original_resolution", True)),
    )

    if _is_distributed():
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
    else:
        train_sampler = None
        val_sampler = None

    batch_size = int(cfg_task.get("batch_size", 4))
    num_workers = int(cfg_task.get("num_workers", 4))
    pin_mem = bool(cfg_task.get("pin_mem", True))
    persistent_workers = bool(cfg_task.get("persistent_workers", True))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=(num_workers > 0) and persistent_workers,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=(num_workers > 0) and persistent_workers,
        drop_last=False,
    )
    logger.info(
        f"Downstream dataset setup: task={dataset_kind}/{target}, "
        f"train_samples={len(train_dataset)}, val_samples={len(val_dataset)}, "
        f"batch_size={batch_size}, num_workers={num_workers}, "
        f"persistent_workers={persistent_workers}, "
        f"clip_num_frames={clip_num_frames}, input_size={cfg_task.get('input_size', None)}"
    )

    num_classes = int(cfg_task.get("num_classes", 0))
    ignore_index = int(cfg_task.get("ignore_index", 255))
    if target == "semantic" and num_classes <= 0:
        num_classes = _infer_num_classes(
            train_dataset,
            ignore_index=ignore_index,
            max_samples=int(cfg_task.get("infer_num_classes_max_samples", 256)),
        )
        logger.info(f"Inferred num_classes={num_classes}")

    encoder = _build_encoder_from_cfg(cfg_model).to(device)
    checkpoint_path = cfg_model.get("checkpoint_path", None)
    if checkpoint_path is not None and str(checkpoint_path).strip() != "":
        _load_pretrained_encoder(
            encoder=encoder,
            checkpoint_path=str(checkpoint_path),
            checkpoint_key=str(cfg_model.get("checkpoint_key", "encoder")),
        )
    else:
        logger.warning("model.checkpoint_path is empty. Downstream training starts from random encoder weights.")

    freeze_encoder = bool(cfg_model.get("freeze_encoder", True))
    model = DenseLinearProbe(
        encoder=encoder,
        num_output_channels=(num_classes if target == "semantic" else 1),
        freeze_encoder=freeze_encoder,
        patch_size=int(cfg_model.get("patch_size", 16)),
        head_dropout=float(cfg_model.get("head_dropout", 0.0)),
    ).to(device)

    if _is_distributed():
        model = DistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()] if torch.cuda.is_available() else None,
            output_device=torch.cuda.current_device() if torch.cuda.is_available() else None,
            static_graph=False,
            find_unused_parameters=False,
        )

    module_ref = model.module if isinstance(model, DistributedDataParallel) else model
    encoder_params = [p for p in module_ref.encoder.parameters() if p.requires_grad]
    enc_param_ids = {id(p) for p in encoder_params}
    head_params = [p for p in module_ref.parameters() if p.requires_grad and id(p) not in enc_param_ids]
    if len(head_params) == 0:
        raise RuntimeError("No trainable parameters found in downstream head.")

    lr = float(cfg_opt.get("lr", 1e-3))
    encoder_lr = float(cfg_opt.get("encoder_lr", lr * 0.1))
    weight_decay = float(cfg_opt.get("weight_decay", 1e-4))
    if len(encoder_params) > 0:
        optimizer = torch.optim.AdamW(
            [
                {"params": head_params, "lr": lr},
                {"params": encoder_params, "lr": encoder_lr},
            ],
            weight_decay=weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            [{"params": head_params, "lr": lr}],
            weight_decay=weight_decay,
        )

    epochs = int(cfg_opt.get("epochs", 20))
    min_lr = float(cfg_opt.get("min_lr", 1e-6))
    warmup_epochs = int(cfg_opt.get("warmup_epochs", 0))

    def _lr_lambda(epoch_idx: int) -> float:
        if warmup_epochs > 0 and epoch_idx < warmup_epochs:
            return float(epoch_idx + 1) / float(max(1, warmup_epochs))
        cosine_total = max(1, epochs - warmup_epochs)
        cosine_step = max(0, epoch_idx - warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * cosine_step / cosine_total))
        floor_ratio = min_lr / max(lr, 1e-12)
        return floor_ratio + (1.0 - floor_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

    if target == "semantic":
        criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    else:
        criterion = None

    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16 and torch.cuda.is_available()))
    use_autocast = bool(mixed_precision and torch.cuda.is_available())
    autocast_device_type = "cuda" if torch.cuda.is_available() else "cpu"

    csv_logger = None
    if rank == 0:
        if target == "semantic":
            csv_logger = CSVLogger(
                str(output_dir / "downstream_log.csv"),
                ("%d", "epoch"),
                ("%.6f", "train_loss"),
                ("%.6f", "val_loss"),
                ("%.6f", "pixel_acc"),
                ("%.6f", "miou"),
                ("%.8f", "lr"),
            )
        else:
            csv_logger = CSVLogger(
                str(output_dir / "downstream_log.csv"),
                ("%d", "epoch"),
                ("%.6f", "train_loss"),
                ("%.6f", "val_loss"),
                ("%.6f", "mae"),
                ("%.6f", "rmse"),
                ("%.8f", "lr"),
            )

    tb_writer = None
    if rank == 0 and enable_tensorboard:
        if SummaryWriter is None:
            logger.warning("tensorboard is unavailable; SummaryWriter import failed.")
        else:
            tb_writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

    start_epoch = 0
    best_metric = -1e18 if target == "semantic" else 1e18
    should_resume_latest = auto_resume_latest and latest_path.exists()
    if load_checkpoint or should_resume_latest:
        load_path = Path(str(read_checkpoint)) if read_checkpoint else latest_path
        if should_resume_latest:
            load_path = latest_path
        if load_path.exists():
            start_epoch, best_metric = _load_train_checkpoint(
                path=load_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
            )
            logger.info(f"Resumed downstream training from {load_path} at epoch={start_epoch}")
            if target == "depth" and best_metric < 0:
                best_metric = 1e18
        else:
            logger.warning(f"Downstream checkpoint not found at {load_path}; starting from scratch.")

    clip_grad = float(cfg_opt.get("clip_grad", 0.0))
    depth_valid_min = float(cfg_task.get("depth_valid_min", 0.0))
    depth_valid_max = float(cfg_task.get("depth_valid_max", 1e9))
    eval_logits_resize_mode = str(cfg_task.get("eval_logits_resize_mode", "bilinear"))

    for epoch in range(start_epoch, epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        if freeze_encoder:
            module_ref.encoder.eval()

        train_loss_meter = AverageMeter()
        for itr, batch in enumerate(train_loader):
            x = batch["input"].to(device, non_blocking=True)
            y = batch["target"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=autocast_device_type,
                dtype=dtype,
                enabled=use_autocast,
            ):
                pred = model(x)
                if target == "semantic":
                    loss = criterion(pred, y)
                else:
                    pred_depth = pred[:, 0]
                    valid = (
                        torch.isfinite(y)
                        & (y > depth_valid_min)
                        & (y < depth_valid_max)
                    )
                    if torch.any(valid):
                        loss = torch.mean(torch.abs(pred_depth[valid] - y[valid]))
                    else:
                        loss = pred_depth.mean() * 0.0

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if clip_grad > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()

            train_loss_meter.update(float(loss.item()), n=int(x.size(0)))
            if rank == 0 and (itr % log_freq == 0 or itr == len(train_loader) - 1):
                logger.info(
                    f"[epoch {epoch + 1}/{epochs}][itr {itr}/{len(train_loader)}] "
                    f"train_loss={train_loss_meter.avg:.6f} lr={optimizer.param_groups[0]['lr']:.3e}"
                )

        model.eval()
        val_loss_meter = AverageMeter()
        if target == "semantic":
            conf = torch.zeros((num_classes, num_classes), dtype=torch.float64, device=device)
        else:
            depth_abs_sum = torch.zeros((1,), dtype=torch.float64, device=device)
            depth_sq_sum = torch.zeros((1,), dtype=torch.float64, device=device)
            depth_valid_count = torch.zeros((1,), dtype=torch.float64, device=device)

        with torch.no_grad():
            for batch in val_loader:
                x = batch["input"].to(device, non_blocking=True)
                y = batch["target"].to(device, non_blocking=True)
                with torch.autocast(
                    device_type=autocast_device_type,
                    dtype=dtype,
                    enabled=use_autocast,
                ):
                    pred = model(x)
                    if target == "semantic":
                        loss = criterion(pred, y)
                    else:
                        pred_depth = pred[:, 0]
                        valid = (
                            torch.isfinite(y)
                            & (y > depth_valid_min)
                            & (y < depth_valid_max)
                        )
                        if torch.any(valid):
                            loss = torch.mean(torch.abs(pred_depth[valid] - y[valid]))
                        else:
                            loss = pred_depth.mean() * 0.0
                val_loss_meter.update(float(loss.item()), n=int(x.size(0)))

                if target == "semantic":
                    y_eval = batch.get("eval_target", batch["target"]).to(
                        device,
                        non_blocking=True,
                    )
                    pred_eval = _resize_logits_to_target(
                        pred,
                        y_eval,
                        mode=eval_logits_resize_mode,
                    )
                    _semantic_confusion_update(
                        conf=conf,
                        pred_logits=pred_eval,
                        target=y_eval,
                        num_classes=num_classes,
                        ignore_index=ignore_index,
                    )
                else:
                    pred_depth = pred[:, 0]
                    valid = (
                        torch.isfinite(y)
                        & (y > depth_valid_min)
                        & (y < depth_valid_max)
                    )
                    if torch.any(valid):
                        err = pred_depth[valid] - y[valid]
                        depth_abs_sum += torch.sum(torch.abs(err), dtype=torch.float64)
                        depth_sq_sum += torch.sum(err * err, dtype=torch.float64)
                        depth_valid_count += torch.sum(valid, dtype=torch.float64)

        val_loss_t = torch.tensor([val_loss_meter.sum, val_loss_meter.count], dtype=torch.float64, device=device)
        val_loss_t = _all_reduce_tensor(val_loss_t)
        val_loss_avg = float((val_loss_t[0] / max(val_loss_t[1], 1.0)).item())

        if target == "semantic":
            conf = _all_reduce_tensor(conf)
            pixel_acc, miou = _semantic_metrics_from_confusion(conf)
            current_metric = miou
        else:
            mae, rmse = _depth_metrics_reduce(depth_abs_sum, depth_sq_sum, depth_valid_count)
            current_metric = mae

        scheduler.step()

        if rank == 0:
            lr_now = float(optimizer.param_groups[0]["lr"])
            if target == "semantic":
                logger.info(
                    f"[epoch {epoch + 1}/{epochs}] train_loss={train_loss_meter.avg:.6f} "
                    f"val_loss={val_loss_avg:.6f} pixel_acc={pixel_acc:.4f} miou={miou:.4f}"
                )
                if csv_logger is not None:
                    csv_logger.log(epoch + 1, train_loss_meter.avg, val_loss_avg, pixel_acc, miou, lr_now)
                if tb_writer is not None:
                    tb_writer.add_scalar("train/loss", train_loss_meter.avg, epoch + 1)
                    tb_writer.add_scalar("val/loss", val_loss_avg, epoch + 1)
                    tb_writer.add_scalar("val/pixel_acc", pixel_acc, epoch + 1)
                    tb_writer.add_scalar("val/miou", miou, epoch + 1)
                    tb_writer.add_scalar("train/lr", lr_now, epoch + 1)
            else:
                logger.info(
                    f"[epoch {epoch + 1}/{epochs}] train_loss={train_loss_meter.avg:.6f} "
                    f"val_loss={val_loss_avg:.6f} mae={mae:.6f} rmse={rmse:.6f}"
                )
                if csv_logger is not None:
                    csv_logger.log(epoch + 1, train_loss_meter.avg, val_loss_avg, mae, rmse, lr_now)
                if tb_writer is not None:
                    tb_writer.add_scalar("train/loss", train_loss_meter.avg, epoch + 1)
                    tb_writer.add_scalar("val/loss", val_loss_avg, epoch + 1)
                    tb_writer.add_scalar("val/mae", mae, epoch + 1)
                    tb_writer.add_scalar("val/rmse", rmse, epoch + 1)
                    tb_writer.add_scalar("train/lr", lr_now, epoch + 1)

        if (epoch + 1) % checkpoint_freq == 0 or epoch == (epochs - 1):
            _save_checkpoint(
                path=latest_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch + 1,
                best_metric=best_metric,
                rank=rank,
            )
            if save_every_freq > 0 and (epoch + 1) % save_every_freq == 0:
                _save_checkpoint(
                    path=output_dir / f"e{epoch + 1}_downstream.pth.tar",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch + 1,
                    best_metric=best_metric,
                    rank=rank,
                )

        is_better = (target == "semantic" and current_metric > best_metric) or (
            target == "depth" and current_metric < best_metric
        )
        if is_better:
            best_metric = current_metric
            _save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch + 1,
                best_metric=best_metric,
                rank=rank,
            )

    if tb_writer is not None:
        tb_writer.close()

    if _is_distributed():
        dist.barrier()
        dist.destroy_process_group()

    return {
        "rank": rank,
        "world_size": world_size,
        "output_dir": str(output_dir),
    }
