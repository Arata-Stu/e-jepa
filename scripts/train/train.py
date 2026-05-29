from __future__ import annotations

import copy
import gc
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from src.datasets.data_manager import init_data
from src.datasets.transforms import make_event_transforms
from src.masks.multiseq_multiblock3d import MaskCollator
from src.masks.utils import apply_masks
from src.models.vision_transformer import VIT_EMBED_DIMS
from src.models.utils.masks_dist import compute_mask_distance
from src.models.utils.modules import Lambda_LinearWarmupHold
from src.training.jepa21_utils import (
    init_opt,
    init_video_model,
    load_checkpoint,
    normalize_nested,
)
from src.utils.distributed import init_distributed
from src.utils.logging import AverageMeter, CSVLogger, get_logger, gpu_timer

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


logger = get_logger(__name__, force=True)


def _to_dtype(which_dtype: str) -> tuple[torch.dtype, bool]:
    which = str(which_dtype).lower()
    if which == "bfloat16":
        return torch.bfloat16, True
    if which == "float16":
        return torch.float16, True
    return torch.float32, False


def _resolve_interpolation(mode: str):
    from torchvision.transforms import InterpolationMode

    lookup = {
        "nearest": InterpolationMode.NEAREST,
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC,
    }
    key = str(mode).lower()
    if key not in lookup:
        raise ValueError(f"unsupported interpolation: {mode}")
    return lookup[key]


def _to_hw_tuple(value, field_name: str) -> tuple[int, int]:
    if isinstance(value, int):
        v = int(value)
        return (v, v)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise ValueError(f"{field_name} must be int or [H, W], got: {value}")


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def _setup_device() -> torch.device:
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        return device
    return torch.device("cpu")


def _encoder_embed_dim(model_name: str) -> int:
    aliases = {
        "vit_large_rope": "vit_large",
        "vit_giant_xformers": "vit_giant",
        "vit_gigantic_xformers": "vit_gigantic",
    }
    canonical_name = aliases.get(model_name, model_name)
    if canonical_name in VIT_EMBED_DIMS:
        return int(VIT_EMBED_DIMS[canonical_name])
    supported = sorted(set(VIT_EMBED_DIMS.keys()) | set(aliases.keys()))
    raise ValueError(
        f"unsupported model_name for embed dim inference: {model_name}. "
        f"Supported: {supported}"
    )


def _maybe_ddp(model: torch.nn.Module, find_unused_parameters: bool = False):
    if not _is_distributed():
        return model
    return DistributedDataParallel(
        model,
        device_ids=[torch.cuda.current_device()] if torch.cuda.is_available() else None,
        output_device=torch.cuda.current_device() if torch.cuda.is_available() else None,
        static_graph=not find_unused_parameters,
        find_unused_parameters=find_unused_parameters,
    )


def _world_info() -> tuple[int, int]:
    if _is_distributed():
        return dist.get_world_size(), dist.get_rank()
    return 1, 0


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ensure_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _extract_loader_len(loader) -> int:
    try:
        return len(loader)
    except Exception:
        return -1


def main(args, resume_preempt: bool = False):
    folder = str(args.get("folder"))

    cfgs_meta = args.get("meta", {})
    cfgs_mask = args.get("mask", [])
    cfgs_model = args.get("model", {})
    cfgs_data = args.get("data", {})
    cfgs_data_aug = args.get("data_aug", {})
    cfgs_loss = args.get("loss", {})
    cfgs_opt = args.get("optimization", {})

    which_dtype = cfgs_meta.get("dtype", "float32")
    dtype, mixed_precision = _to_dtype(which_dtype)
    log_freq = int(cfgs_meta.get("log_freq", 10))
    use_tqdm = bool(cfgs_meta.get("use_tqdm", False))
    checkpoint_freq = int(cfgs_meta.get("checkpoint_freq", 1))
    save_every_freq = int(cfgs_meta.get("save_every_freq", -1))
    skip_batches = int(cfgs_meta.get("skip_batches", -1))
    sync_gc = bool(cfgs_meta.get("sync_gc", False))
    gc_collect_itr_freq = int(cfgs_meta.get("gc_collect_itr_freq", 50))
    max_loader_retries = int(cfgs_meta.get("max_loader_retries", 5))
    enable_tensorboard = bool(cfgs_meta.get("tensorboard", True))
    load_model = bool(cfgs_meta.get("load_checkpoint", False)) or bool(resume_preempt)
    auto_resume_latest = bool(cfgs_meta.get("auto_resume_latest", False))
    read_checkpoint = cfgs_meta.get("read_checkpoint", None)
    seed = int(cfgs_meta.get("seed", 0))
    dist_port = int(cfgs_meta.get("dist_port", 37129))

    _seed_everything(seed)
    torch.backends.cudnn.benchmark = True
    if cfgs_meta.get("use_tf32", True):
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass
    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    init_distributed(port=dist_port)
    world_size, rank = _world_info()
    device = _setup_device()

    if rank != 0 and not bool(cfgs_meta.get("log_all_ranks", False)):
        import logging

        logger.setLevel(logging.WARNING)

    logger.info(f"Initialized rank/world_size={rank}/{world_size}")
    logger.info(f"Using device={device}, dtype={dtype}, mixed_precision={mixed_precision}")

    output_dir = Path(folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / "latest.pth.tar"

    if load_model:
        load_path = Path(str(read_checkpoint)) if read_checkpoint else latest_path
        if not load_path.exists():
            logger.warning(f"Checkpoint not found at {load_path}; starting from scratch.")
            load_model = False
            load_path = None
    else:
        load_path = None

    csv_logger = CSVLogger(
        str(output_dir / f"log_r{rank}.csv"),
        ("%d", "epoch"),
        ("%d", "itr"),
        ("%.6f", "loss"),
        ("%d", "iter-time(ms)"),
        ("%d", "gpu-time(ms)"),
        ("%d", "dataload-time(ms)"),
    )

    tb_writer = None
    if rank == 0 and enable_tensorboard:
        if SummaryWriter is None:
            logger.warning("tensorboard is unavailable; SummaryWriter import failed.")
        else:
            tb_writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

    # -- Model params
    compile_model = bool(cfgs_model.get("compile_model", False))
    use_activation_checkpointing = bool(
        cfgs_model.get("use_activation_checkpointing", False)
    )
    model_name = str(cfgs_model.get("model_name", "vit_base"))
    in_chans = int(cfgs_model.get("in_chans", 3))
    pred_depth = int(cfgs_model.get("pred_depth", 12))
    pred_num_heads = cfgs_model.get("pred_num_heads", None)
    if pred_num_heads is not None:
        pred_num_heads = int(pred_num_heads)
    pred_embed_dim = int(cfgs_model.get("pred_embed_dim", 384))
    uniform_power = bool(cfgs_model.get("uniform_power", True))
    use_mask_tokens = bool(cfgs_model.get("use_mask_tokens", True))
    zero_init_mask_tokens = bool(cfgs_model.get("zero_init_mask_tokens", True))
    use_rope = bool(cfgs_model.get("use_rope", True))
    use_silu = bool(cfgs_model.get("use_silu", False))
    use_pred_silu = bool(cfgs_model.get("use_pred_silu", False))
    wide_silu = bool(cfgs_model.get("wide_silu", True))
    is_causal = bool(cfgs_model.get("is_causal", False))
    pred_is_causal = bool(cfgs_model.get("pred_is_causal", False))
    init_type = str(cfgs_model.get("init_type", "default"))
    img_temporal_dim_size = cfgs_model.get("img_temporal_dim_size", None)
    if img_temporal_dim_size is not None:
        img_temporal_dim_size = int(img_temporal_dim_size)
    n_registers = int(cfgs_model.get("n_registers", 0))
    has_cls_first = bool(cfgs_model.get("has_cls_first", False))
    interpolate_rope = bool(cfgs_model.get("interpolate_rope", False))
    n_registers_predictor = int(cfgs_model.get("n_registers_predictor", 0))
    lambda_value_img = float(cfgs_model.get("lambda_value_img", 0.5))
    lambda_value_vid = float(cfgs_model.get("lambda_value_vid", 0.5))
    lambda_progressive = bool(cfgs_model.get("lambda_progressive", True))
    normalize_predictor = bool(cfgs_model.get("normalize_predictor", False))
    modality_embedding = bool(cfgs_model.get("modality_embedding", False))
    levels_predictor = int(cfgs_model.get("levels_predictor", 4))

    # -- Data params
    dataset_type = str(cfgs_data.get("dataset_type", "eventdataset"))
    dataset_paths = _ensure_list(cfgs_data.get("datasets", []))
    if len(dataset_paths) == 0:
        raise ValueError("data.datasets must contain at least one path")
    datasets_weights = cfgs_data.get("datasets_weights", None)
    dataset_fpcs = [int(v) for v in _ensure_list(cfgs_data.get("dataset_fpcs", [8]))]
    if len(dataset_fpcs) == 0:
        raise ValueError("data.dataset_fpcs must contain at least one value")
    max_num_frames = int(max(dataset_fpcs))

    batch_size = int(cfgs_data.get("batch_size", 8))
    tubelet_size = int(cfgs_data.get("tubelet_size", 2))
    fps = cfgs_data.get("fps", None)
    crop_size = _to_hw_tuple(cfgs_data.get("crop_size", 224), "data.crop_size")
    patch_size = int(cfgs_data.get("patch_size", 16))
    pin_mem = bool(cfgs_data.get("pin_mem", True))
    num_workers = int(cfgs_data.get("num_workers", 4))
    persistent_workers = bool(cfgs_data.get("persistent_workers", True))
    prefetch_factor = cfgs_data.get("prefetch_factor", None)
    if prefetch_factor is not None:
        prefetch_factor = int(prefetch_factor)
        if prefetch_factor < 1:
            raise ValueError("data.prefetch_factor must be >= 1 or null")
    max_open_h5_files = int(cfgs_data.get("max_open_h5_files", 32))
    if max_open_h5_files < 1:
        raise ValueError("data.max_open_h5_files must be >= 1")
    num_clips = int(cfgs_data.get("num_clips", 1))
    random_clip_sampling = bool(cfgs_data.get("random_clip_sampling", True))
    allow_clip_overlap = bool(cfgs_data.get("allow_clip_overlap", False))
    frame_sample_rate = int(cfgs_data.get("frame_sample_rate", 1))
    file_pattern = str(cfgs_data.get("file_pattern", "*.h5"))
    recursive = bool(cfgs_data.get("recursive", True))
    activity_filter_enabled = bool(cfgs_data.get("activity_filter_enabled", False))
    activity_filter_min_clip_mean_active_pixel_ratio = cfgs_data.get(
        "activity_filter_min_clip_mean_active_pixel_ratio",
        None,
    )
    activity_filter_min_clip_mean_activity_score = cfgs_data.get(
        "activity_filter_min_clip_mean_activity_score",
        None,
    )
    activity_filter_min_clip_active_window_ratio = cfgs_data.get(
        "activity_filter_min_clip_active_window_ratio",
        None,
    )
    activity_filter_active_window_threshold = cfgs_data.get(
        "activity_filter_active_window_threshold",
        None,
    )

    # Optional image branch config (vjepa2.1 style rank split)
    cfgs_img_data = args.get("img_data", None)
    img_mask = args.get("img_mask", None)
    img_enabled = isinstance(cfgs_img_data, dict) and bool(
        cfgs_img_data.get("enabled", False)
    )

    # -- Data augmentation params
    ar_range = tuple(float(v) for v in cfgs_data_aug.get("random_resize_aspect_ratio", [0.75, 1.333333]))
    rr_scale = tuple(float(v) for v in cfgs_data_aug.get("random_resize_scale", [0.3, 1.0]))
    random_horizontal_flip = bool(cfgs_data_aug.get("random_horizontal_flip", True))
    interpolation = _resolve_interpolation(str(cfgs_data_aug.get("interpolation", "bilinear")))
    antialias = bool(cfgs_data_aug.get("antialias", True))
    preserve_input_size = bool(cfgs_data_aug.get("preserve_input_size", False))
    pad_to_hw_raw = cfgs_data_aug.get("pad_to_hw", None)
    pad_to_hw = None
    if pad_to_hw_raw is not None:
        if not isinstance(pad_to_hw_raw, (list, tuple)) or len(pad_to_hw_raw) != 2:
            raise ValueError("data_aug.pad_to_hw must be [H, W] or null")
        pad_to_hw = (int(pad_to_hw_raw[0]), int(pad_to_hw_raw[1]))
    pad_value = float(cfgs_data_aug.get("pad_value", 0.0))
    allowed_input_hw_raw = cfgs_data_aug.get("allowed_input_hw", None)
    allowed_input_hw = None
    if allowed_input_hw_raw is not None:
        allowed_input_hw = set()
        for hw in allowed_input_hw_raw:
            if not isinstance(hw, (list, tuple)) or len(hw) != 2:
                raise ValueError(
                    "data_aug.allowed_input_hw must be a list of [H, W] pairs"
                )
            allowed_input_hw.add((int(hw[0]), int(hw[1])))

    # -- Loss params
    loss_exp = float(cfgs_loss.get("loss_exp", 1.0))
    shift_by_n = int(cfgs_loss.get("shift_by_n", 0))
    predict_all = bool(cfgs_loss.get("predict_all", True))
    weight_distance_loss = bool(cfgs_loss.get("weight_distance_loss", True))
    offset_context_loss = bool(cfgs_loss.get("offset_context_loss", False))

    # -- Optimization params
    is_anneal = bool(cfgs_opt.get("is_anneal", False))
    anneal_ckpt = cfgs_opt.get("anneal_ckpt", None)
    if is_anneal and anneal_ckpt is None:
        raise ValueError("optimization.anneal_ckpt must be set when optimization.is_anneal=true")
    resume_anneal = bool(cfgs_opt.get("resume_anneal", False)) or (
        is_anneal and bool(resume_preempt)
    )

    ipe = cfgs_opt.get("ipe", None)
    if ipe is not None:
        ipe = int(ipe)
    ipe_scale = float(cfgs_opt.get("ipe_scale", 1.0))

    wd = float(cfgs_opt.get("weight_decay", 0.04))
    final_wd = float(cfgs_opt.get("final_weight_decay", 0.04))
    num_epochs = int(cfgs_opt.get("epochs", 1000))
    warmup = float(cfgs_opt.get("warmup", 40))
    start_lr = float(cfgs_opt.get("start_lr", 1e-4))
    lr = float(cfgs_opt.get("lr", 6e-4))
    final_lr = float(cfgs_opt.get("final_lr", 6e-4))
    ema = [float(v) for v in cfgs_opt.get("ema", [0.99925, 0.99925])]
    if len(ema) != 2:
        raise ValueError("optimization.ema must contain [start, end]")
    use_radamw = bool(cfgs_opt.get("use_radamw", False))
    betas = tuple(float(v) for v in cfgs_opt.get("betas", [0.9, 0.999]))
    eps = float(cfgs_opt.get("eps", 1e-8))
    clip_grad = cfgs_opt.get("clip_grad", None)
    if clip_grad is not None:
        clip_grad = float(clip_grad)
        if clip_grad <= 0.0:
            clip_grad = None

    loss_reg_std_mult = cfgs_opt.get("loss_reg_std_mult", None)
    if loss_reg_std_mult is not None:
        loss_reg_std_mult = float(loss_reg_std_mult)
    loss_reg_num_tracking_steps = int(cfgs_opt.get("loss_reg_num_tracking_steps", 300))
    loss_reg_min_epoch = int(cfgs_opt.get("loss_reg_min_epoch", 50))

    data_rank, data_world_size = rank, world_size
    lambda_value = lambda_value_vid
    model_fpcs = list(dataset_fpcs)
    model_cfgs_mask = list(cfgs_mask)
    model_tubelet_size = int(tubelet_size)

    if img_enabled:
        if world_size <= 1:
            logger.warning(
                "img_data.enabled=true but world_size=1. Rank split is skipped and video branch is used."
            )
        else:
            img_rank_ratio = float(cfgs_img_data.get("rank_ratio", 0.25))
            if not (0.0 < img_rank_ratio < 1.0):
                raise ValueError("img_data.rank_ratio must be in (0, 1)")

            img_dataset_type = str(cfgs_img_data.get("dataset_type", "eventdataset"))
            img_dataset_paths = _ensure_list(cfgs_img_data.get("datasets", []))
            img_dataset_weights = cfgs_img_data.get("datasets_weights", None)
            img_dataset_fpcs = [
                int(v) for v in _ensure_list(cfgs_img_data.get("dataset_fpcs", [1]))
            ]
            img_dataset_batch_size = int(cfgs_img_data.get("batch_size", batch_size))
            img_num_workers = int(cfgs_img_data.get("num_workers", num_workers))

            if len(img_dataset_paths) == 0:
                raise ValueError("img_data.datasets must contain at least one path")
            if len(img_dataset_fpcs) == 0:
                raise ValueError("img_data.dataset_fpcs must contain at least one value")

            img_world_size = int(world_size * img_rank_ratio)
            if img_world_size <= 0 or img_world_size >= world_size:
                raise ValueError(
                    "img_data.rank_ratio leads to an empty image or video rank group. "
                    f"world_size={world_size}, rank_ratio={img_rank_ratio}, img_world_size={img_world_size}"
                )
            num_video_ranks = world_size - img_world_size

            img_total_batch_size = img_dataset_batch_size * world_size
            video_total_batch_size = batch_size * world_size
            if img_total_batch_size % img_world_size != 0:
                raise ValueError(
                    f"img_total_batch_size ({img_total_batch_size}) must be divisible by num_img_ranks ({img_world_size})"
                )
            if video_total_batch_size % num_video_ranks != 0:
                raise ValueError(
                    f"video_total_batch_size ({video_total_batch_size}) must be divisible by num_video_ranks ({num_video_ranks})"
                )

            # Keep total video batch size unchanged when some ranks are reassigned to image data.
            batch_size = video_total_batch_size // num_video_ranks

            if rank < img_world_size:
                crop_size = _to_hw_tuple(
                    cfgs_img_data.get("crop_size", crop_size),
                    "img_data.crop_size",
                )
                if img_temporal_dim_size is not None:
                    if img_dataset_fpcs[0] != 1:
                        raise NotImplementedError(
                            "Image branch requires img_data.dataset_fpcs[0]=1 when model.img_temporal_dim_size is set."
                        )
                    tubelet_size = 1

                dataset_type = img_dataset_type
                dataset_paths = img_dataset_paths
                datasets_weights = img_dataset_weights
                dataset_fpcs = img_dataset_fpcs
                batch_size = img_dataset_batch_size
                num_workers = img_num_workers
                if img_mask is not None:
                    cfgs_mask = img_mask

                data_rank = rank
                data_world_size = img_world_size
                lambda_value = lambda_value_img
            else:
                data_rank = rank - img_world_size
                data_world_size = world_size - img_world_size
                lambda_value = lambda_value_vid

            logger.info(
                f"Modality split active: rank={rank}/{world_size}, "
                f"data_rank={data_rank}/{data_world_size}, "
                f"dataset_type={dataset_type}, dataset_fpcs={dataset_fpcs}, "
                f"batch_size={batch_size}, num_workers={num_workers}, lambda={lambda_value}"
            )

    logger.info(
        f"Dataset setup: type={dataset_type}, paths={dataset_paths}, dataset_fpcs={dataset_fpcs}, "
        f"batch_size={batch_size}, activity_filter_enabled={activity_filter_enabled}, "
        f"min_clip_mean_active={activity_filter_min_clip_mean_active_pixel_ratio}, "
        f"min_clip_mean_score={activity_filter_min_clip_mean_activity_score}, "
        f"min_clip_active_window_ratio={activity_filter_min_clip_active_window_ratio}, "
        f"active_window_threshold={activity_filter_active_window_threshold}"
    )

    transform = make_event_transforms(
        random_horizontal_flip=random_horizontal_flip,
        random_resize_aspect_ratio=ar_range,
        random_resize_scale=rr_scale,
        crop_size=crop_size,
        interpolation=interpolation,
        antialias=antialias,
        apply_random_resized_crop=not preserve_input_size,
        pad_to_hw=pad_to_hw,
        pad_value=pad_value,
    )

    mask_collator = MaskCollator(
        cfgs_mask=cfgs_mask,
        dataset_fpcs=dataset_fpcs,
        crop_size=crop_size,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
    )

    encoder, predictor = init_video_model(
        device=device,
        in_chans=in_chans,
        patch_size=patch_size,
        max_num_frames=int(max(model_fpcs)),
        tubelet_size=model_tubelet_size,
        model_name=model_name,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_num_heads=pred_num_heads,
        pred_embed_dim=pred_embed_dim,
        uniform_power=uniform_power,
        use_mask_tokens=use_mask_tokens,
        num_mask_tokens=int(len(model_cfgs_mask) * len(model_fpcs)),
        zero_init_mask_tokens=zero_init_mask_tokens,
        use_sdpa=bool(cfgs_meta.get("use_sdpa", True)),
        use_rope=use_rope,
        use_silu=use_silu,
        use_pred_silu=use_pred_silu,
        wide_silu=wide_silu,
        is_causal=is_causal,
        pred_is_causal=pred_is_causal,
        use_activation_checkpointing=use_activation_checkpointing,
        return_all_tokens=predict_all,
        chop_last_n_tokens=shift_by_n,
        init_type=init_type,
        img_temporal_dim_size=img_temporal_dim_size,
        n_registers=n_registers,
        n_registers_predictor=n_registers_predictor,
        has_cls_first=has_cls_first,
        interpolate_rope=interpolate_rope,
        modality_embedding=modality_embedding,
    )
    target_encoder = copy.deepcopy(encoder)

    if compile_model:
        logger.info("Compiling encoder/predictor/target_encoder")
        torch._dynamo.config.optimize_ddp = False
        encoder.compile()
        predictor.compile()
        target_encoder.compile()

    data_loader, data_sampler = init_data(
        data=dataset_type,
        root_path=dataset_paths,
        batch_size=batch_size,
        training=True,
        dataset_fpcs=dataset_fpcs,
        frame_sample_rate=frame_sample_rate,
        fps=fps,
        clip_len=max_num_frames,
        num_clips=num_clips,
        random_clip_sampling=random_clip_sampling,
        allow_clip_overlap=allow_clip_overlap,
        datasets_weights=datasets_weights,
        transform=transform,
        collator=mask_collator,
        rank=data_rank,
        world_size=data_world_size,
        num_workers=num_workers,
        pin_mem=pin_mem,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        max_open_h5_files=max_open_h5_files,
        file_pattern=file_pattern,
        recursive=recursive,
        activity_filter_enabled=activity_filter_enabled,
        activity_filter_min_clip_mean_active_pixel_ratio=activity_filter_min_clip_mean_active_pixel_ratio,
        activity_filter_min_clip_mean_activity_score=activity_filter_min_clip_mean_activity_score,
        activity_filter_min_clip_active_window_ratio=activity_filter_min_clip_active_window_ratio,
        activity_filter_active_window_threshold=activity_filter_active_window_threshold,
    )

    dlen = _extract_loader_len(data_loader)
    if ipe is None:
        ipe = dlen
    if ipe is None or ipe <= 0:
        raise ValueError(f"optimization.ipe must be > 0, got {ipe}, loader_len={dlen}")

    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        is_anneal=is_anneal,
        encoder=encoder,
        predictor=predictor,
        use_radamw=use_radamw,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        mixed_precision=mixed_precision,
        betas=betas,
        eps=eps,
    )

    encoder = _maybe_ddp(encoder, find_unused_parameters=False)
    predictor = _maybe_ddp(predictor, find_unused_parameters=True)
    target_encoder = _maybe_ddp(target_encoder, find_unused_parameters=False)

    for p in target_encoder.parameters():
        p.requires_grad = False

    momentum_scheduler = (
        ema[0] + i * (ema[1] - ema[0]) / (ipe * num_epochs * ipe_scale)
        for i in range(int(ipe * num_epochs) + 1)
    )
    lambda_sched = Lambda_LinearWarmupHold(lambda_value=lambda_value)

    start_epoch = 0
    should_resume_latest = auto_resume_latest and latest_path.exists()
    if load_model or should_resume_latest:
        resolved_load_path = load_path
        if resolved_load_path is None:
            resolved_load_path = latest_path
        if is_anneal:
            if latest_path.exists() and resume_anneal:
                resolved_load_path = latest_path
            else:
                resolved_load_path = Path(str(anneal_ckpt))
                resume_anneal = False

        (
            encoder,
            predictor,
            target_encoder,
            optimizer,
            scaler,
            start_epoch,
        ) = load_checkpoint(
            r_path=str(resolved_load_path),
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler,
            is_anneal=is_anneal and not resume_anneal,
        )

        if not is_anneal or resume_anneal:
            for _ in range(start_epoch * ipe):
                scheduler.step()
                wd_scheduler.step()
                next(momentum_scheduler)
                mask_collator.step()

    def save_checkpoint(epoch: int, path: Path, loss_val: float):
        if rank != 0:
            return
        save_dict = {
            "encoder": encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "target_encoder": target_encoder.state_dict(),
            "opt": optimizer.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "epoch": int(epoch),
            "loss": float(loss_val),
            "batch_size": int(batch_size),
            "world_size": int(world_size),
            "lr": float(lr),
        }
        try:
            torch.save(save_dict, str(path))
        except Exception as exc:
            logger.warning(f"Failed to save checkpoint to {path}: {exc}")

    if data_sampler is not None and hasattr(data_sampler, "set_epoch"):
        data_sampler.set_epoch(start_epoch)

    loader = iter(data_loader)

    if skip_batches > 0:
        logger.info(f"Skipping first {skip_batches} batches")
        for itr in range(skip_batches):
            if itr % 10 == 0:
                logger.info(f"Skipped {itr}/{skip_batches} batches")
            try:
                _ = next(loader)
            except Exception:
                loader = iter(data_loader)
                _ = next(loader)

    if sync_gc:
        gc.disable()
        gc.collect()

    trailing_losses = []
    embed_dim_encoder = _encoder_embed_dim(model_name)
    if crop_size[0] % patch_size != 0 or crop_size[1] % patch_size != 0:
        raise ValueError(
            f"crop_size={crop_size} must be divisible by patch_size={patch_size}"
        )
    grid_h = crop_size[0] // patch_size
    grid_w = crop_size[1] // patch_size

    for epoch in range(start_epoch, num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        pbar = None
        if rank == 0 and use_tqdm:
            pbar = tqdm(
                total=ipe,
                desc=f"Epoch {epoch + 1}/{num_epochs}",
                dynamic_ncols=True,
                leave=True,
            )

        if data_sampler is not None and hasattr(data_sampler, "set_epoch"):
            data_sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        loss_pred_meter = AverageMeter()
        loss_context_meter = AverageMeter()
        loss_context_weighted_meter = AverageMeter()
        context_lambda_meter = AverageMeter()
        mask_meters = {fpc: AverageMeter() for fpc in dataset_fpcs}
        iter_time_meter = AverageMeter()
        gpu_time_meter = AverageMeter()
        data_elapsed_time_meter = AverageMeter()

        for itr in range(ipe):
            itr_start_time = time.time()

            iter_retries = 0
            iter_successful = False
            while not iter_successful:
                try:
                    sample = next(loader)
                    iter_successful = True
                except StopIteration:
                    if data_sampler is not None and hasattr(data_sampler, "set_epoch"):
                        data_sampler.set_epoch(epoch)
                    loader = iter(data_loader)
                except Exception as exc:
                    if iter_retries < max_loader_retries:
                        logger.warning(
                            f"Data loading exception (retry {iter_retries + 1}/{max_loader_retries}): {exc}"
                        )
                        iter_retries += 1
                        time.sleep(1.0)
                    else:
                        raise RuntimeError(
                            f"Exceeded max retries ({max_loader_retries}) when loading data"
                        ) from exc

            for fpc_sample in sample:
                bs, fpc = fpc_sample[0][-1][0].size()
                mask_meters[fpc].update(bs / batch_size)

            all_clips, all_masks_enc, all_masks_pred = [], [], []
            for fpc_sample in sample:
                udata, masks_enc, masks_pred = fpc_sample
                all_clips.append(udata[0][0].to(device, non_blocking=True))
                all_masks_enc.append([m.to(device, non_blocking=True) for m in masks_enc])
                all_masks_pred.append([m.to(device, non_blocking=True) for m in masks_pred])

            clips, masks_enc, masks_pred = all_clips, all_masks_enc, all_masks_pred
            data_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0

            # Fail fast with a clear message when model/input channel configs mismatch.
            for clip in clips:
                clip_in_chans = int(clip.shape[1])
                if clip_in_chans != in_chans:
                    raise ValueError(
                        f"Input channel mismatch: model.in_chans={in_chans}, "
                        f"but batch clip has C={clip_in_chans}. "
                        "Please set model.in_chans to match the preprocessed voxel channel count."
                    )
                if preserve_input_size and allowed_input_hw:
                    clip_h = int(clip.shape[-2])
                    clip_w = int(clip.shape[-1])
                    if (clip_h, clip_w) not in allowed_input_hw:
                        raise ValueError(
                            f"Unexpected input resolution HxW=({clip_h},{clip_w}). "
                            f"Allowed values: {sorted(allowed_input_hw)}"
                        )

            if sync_gc and (itr + 1) % gc_collect_itr_freq == 0:
                gc.collect()

            def train_step():
                new_lr = scheduler.step()
                new_wd = wd_scheduler.step()

                def forward_target(c, embed_dim=embed_dim_encoder):
                    with torch.no_grad():
                        h = target_encoder(c, gram_mode=False, training_mode=True)
                        new_h = []
                        for hi in h:
                            if levels_predictor > 1:
                                hi_0 = F.layer_norm(hi[:, :, :embed_dim], (embed_dim,))
                                hi_1 = F.layer_norm(
                                    hi[:, :, embed_dim : embed_dim * 2],
                                    (embed_dim,),
                                )
                                hi_2 = F.layer_norm(
                                    hi[:, :, embed_dim * 2 : embed_dim * 3],
                                    (embed_dim,),
                                )
                                hi_3 = F.layer_norm(hi[:, :, -embed_dim:], (embed_dim,))
                                hi_norm = torch.cat([hi_0, hi_1, hi_2, hi_3], dim=2)
                                new_h.append(hi_norm)
                            else:
                                new_h.append(F.layer_norm(hi, (hi.size(-1),)))
                        return new_h

                def forward_context(_clips, embed_dim=embed_dim_encoder):
                    modality = "video"
                    if img_temporal_dim_size is not None and _clips[0].shape[2] == img_temporal_dim_size:
                        modality = "image"
                    z = encoder(_clips, masks_enc, gram_mode=False, training_mode=True)
                    z_pred, z_context = predictor(z, masks_enc, masks_pred, mod=modality)
                    if normalize_predictor:
                        z_pred = normalize_nested(z_pred, embed_dim)
                        if predict_all:
                            z_context = normalize_nested(z_context, embed_dim)
                    return z_pred, z_context

                def loss_fn(z, h, masks_to_apply, cls_loss, d_weights):
                    if cls_loss:
                        h_cls = [hi[:, 0].unsqueeze(1) for hi in h]
                        h_masked = [
                            apply_masks(hi[:, 1:], mi, concat=False)
                            for hi, mi in zip(h, masks_to_apply)
                        ]
                        loss_v, n = 0.0, 0
                        for zi, hi, hi_cls in zip(z, h_masked, h_cls):
                            for zij, hij in zip(zi, hi):
                                h_term = torch.cat([hi_cls, hij], dim=1)
                                loss_v += torch.mean(torch.abs(zij - h_term) ** loss_exp) / loss_exp
                                n += 1
                        return loss_v / max(n, 1)

                    h_masked = [
                        apply_masks(hi, mi, concat=False)
                        for hi, mi in zip(h, masks_to_apply)
                    ]

                    if d_weights is not None:
                        loss_v, n = 0.0, 0
                        for zi, hi, d_i in zip(z, h_masked, d_weights):
                            for zij, hij, d_ij in zip(zi, hi, d_i):
                                loss_n = torch.abs(zij - hij) ** loss_exp * (1 / d_ij.unsqueeze(2))
                                loss_v += torch.mean(loss_n) / loss_exp
                                n += 1
                        return loss_v / max(n, 1)

                    loss_v, n = 0.0, 0
                    for zi, hi in zip(z, h_masked):
                        for zij, hij in zip(zi, hi):
                            loss_v += torch.mean(torch.abs(zij - hij) ** loss_exp) / loss_exp
                            n += 1
                    return loss_v / max(n, 1)

                with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                    h = forward_target(clips)
                    z_pred, z_context = forward_context(clips)

                    loss_context = None
                    loss_context_weighted = None
                    lambda_value_step = 0.0

                    loss_pred = loss_fn(
                        z_pred,
                        h,
                        masks_pred,
                        cls_loss=has_cls_first,
                        d_weights=None,
                    )
                    loss = loss_pred

                    if predict_all:
                        distance_weights = compute_mask_distance(
                            masks_pred,
                            masks_enc,
                            offset_context_loss=offset_context_loss,
                            h_patches=grid_h,
                            w_patches=grid_w,
                        )
                        d_weights = distance_weights if weight_distance_loss else None
                        loss_context = loss_fn(
                            z_context,
                            h,
                            masks_enc,
                            cls_loss=False,
                            d_weights=d_weights,
                        )
                        lambda_value_step = (
                            lambda_sched.value(epoch * ipe + itr)
                            if lambda_progressive
                            else lambda_value
                        )
                        loss_context_weighted = loss_context * lambda_value_step
                        loss = loss + loss_context_weighted

                run_step = True
                if loss_reg_std_mult is not None and len(trailing_losses) > 0:
                    meanval = np.mean(trailing_losses)
                    stdval = np.std(trailing_losses)
                    max_bound = meanval + loss_reg_std_mult * stdval
                    if (
                        loss > max_bound
                        and epoch > loss_reg_min_epoch
                        and len(trailing_losses) > int(0.5 * loss_reg_num_tracking_steps)
                    ):
                        run_step = False

                if run_step:
                    if mixed_precision:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        if clip_grad is not None:
                            torch.nn.utils.clip_grad_norm_(
                                (
                                    p
                                    for group in optimizer.param_groups
                                    for p in group["params"]
                                    if p.grad is not None
                                ),
                                clip_grad,
                            )
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        if clip_grad is not None:
                            torch.nn.utils.clip_grad_norm_(
                                (
                                    p
                                    for group in optimizer.param_groups
                                    for p in group["params"]
                                    if p.grad is not None
                                ),
                                clip_grad,
                            )
                        optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                m = min(next(momentum_scheduler), ema[1])
                with torch.no_grad():
                    params_k = []
                    params_q = []
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        params_k.append(param_k)
                        params_q.append(param_q)
                    torch._foreach_mul_(params_k, m)
                    torch._foreach_add_(params_k, params_q, alpha=1 - m)

                loss_context_value = (
                    0.0 if loss_context is None else float(loss_context.detach())
                )
                loss_context_weighted_value = (
                    0.0
                    if loss_context_weighted is None
                    else float(loss_context_weighted.detach())
                )
                loss_details = {
                    "loss_pred": float(loss_pred.detach()),
                    "loss_context": loss_context_value,
                    "loss_context_weighted": loss_context_weighted_value,
                    "context_lambda": float(lambda_value_step),
                }

                return float(loss.detach()), float(new_lr), float(new_wd), run_step, loss_details

            (loss, new_lr, new_wd, run_step, loss_details), gpu_etime_ms = gpu_timer(train_step)
            iter_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0

            loss_meter.update(loss)
            loss_pred_meter.update(loss_details["loss_pred"])
            loss_context_meter.update(loss_details["loss_context"])
            loss_context_weighted_meter.update(loss_details["loss_context_weighted"])
            context_lambda_meter.update(loss_details["context_lambda"])
            iter_time_meter.update(iter_elapsed_time_ms)
            gpu_time_meter.update(gpu_etime_ms)
            data_elapsed_time_meter.update(data_elapsed_time_ms)

            if loss_reg_std_mult is not None:
                if run_step:
                    trailing_losses.append(loss)
                    if len(trailing_losses) > loss_reg_num_tracking_steps:
                        trailing_losses = trailing_losses[1:]
                else:
                    pass

            csv_logger.log(
                epoch + 1,
                itr,
                loss,
                int(iter_elapsed_time_ms),
                int(gpu_etime_ms),
                int(data_elapsed_time_ms),
            )

            if tb_writer is not None:
                global_step = epoch * ipe + itr
                tb_writer.add_scalar("train/loss", loss, global_step)
                tb_writer.add_scalar("train/loss_total", loss, global_step)
                tb_writer.add_scalar(
                    "train/loss_pred", loss_details["loss_pred"], global_step
                )
                tb_writer.add_scalar(
                    "train/loss_context", loss_details["loss_context"], global_step
                )
                tb_writer.add_scalar(
                    "train/loss_context_weighted",
                    loss_details["loss_context_weighted"],
                    global_step,
                )
                tb_writer.add_scalar(
                    "train/context_lambda",
                    loss_details["context_lambda"],
                    global_step,
                )
                tb_writer.add_scalar("train/lr", new_lr, global_step)
                tb_writer.add_scalar("train/wd", new_wd, global_step)
                tb_writer.add_scalar("time/iter_ms", iter_elapsed_time_ms, global_step)
                tb_writer.add_scalar("time/gpu_ms", gpu_etime_ms, global_step)
                tb_writer.add_scalar("time/data_ms", data_elapsed_time_ms, global_step)

            max_mem_mb = (
                torch.cuda.max_memory_allocated() / (1024.0**2)
                if torch.cuda.is_available()
                else 0.0
            )
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    loss=f"{loss_meter.avg:.4f}",
                    lr=f"{new_lr:.2e}",
                    wd=f"{new_wd:.2e}",
                    mem=f"{max_mem_mb:.0f}MB",
                    iter=f"{iter_time_meter.avg:.0f}ms",
                    gpu=f"{gpu_time_meter.avg:.0f}ms",
                    data=f"{data_elapsed_time_meter.avg:.0f}ms",
                )

            if (
                (itr % log_freq == 0 or itr == ipe - 1 or np.isnan(loss) or np.isinf(loss))
                and not use_tqdm
            ):
                logger.info(
                    "[%d, %5d] loss: %.4f masks: %s [wd: %.2e] [lr: %.2e] [mem: %.2f MB] [iter: %.1f ms] [gpu: %.1f ms] [data: %.1f ms]"
                    % (
                        epoch + 1,
                        itr,
                        loss_meter.avg,
                        "["
                        + ", ".join([f"{k}: {mask_meters[k].avg:.2f}" for k in mask_meters])
                        + "]",
                        new_wd,
                        new_lr,
                        max_mem_mb,
                        iter_time_meter.avg,
                        gpu_time_meter.avg,
                        data_elapsed_time_meter.avg,
                    )
                )

            if np.isnan(loss) or np.isinf(loss):
                raise RuntimeError("Loss became NaN/Inf")

        if pbar is not None:
            pbar.close()

        if tb_writer is not None:
            tb_writer.add_scalar("epoch/loss_avg", loss_meter.avg, epoch + 1)
            tb_writer.add_scalar("epoch/loss_pred_avg", loss_pred_meter.avg, epoch + 1)
            tb_writer.add_scalar(
                "epoch/loss_context_avg", loss_context_meter.avg, epoch + 1
            )
            tb_writer.add_scalar(
                "epoch/loss_context_weighted_avg",
                loss_context_weighted_meter.avg,
                epoch + 1,
            )
            tb_writer.add_scalar(
                "epoch/context_lambda_avg", context_lambda_meter.avg, epoch + 1
            )

        logger.info(f"Epoch {epoch + 1} avg loss: {loss_meter.avg:.6f}")
        if (epoch + 1) % checkpoint_freq == 0 or epoch == (num_epochs - 1):
            save_checkpoint(epoch + 1, latest_path, loss_meter.avg)
            if save_every_freq > 0 and (epoch + 1) % save_every_freq == 0:
                save_checkpoint(epoch + 1, output_dir / f"e{epoch + 1}.pth.tar", loss_meter.avg)

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
