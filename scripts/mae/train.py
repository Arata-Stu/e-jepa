from __future__ import annotations

import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

import src.models.vision_transformer as video_vit
from src.datasets.data_manager import init_data
from src.datasets.transforms import make_event_transforms
from src.models.utils.modules import Block
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.distributed import init_distributed
from src.utils.logging import AverageMeter, CSVLogger, get_logger, gpu_timer
from src.utils.pretrain_debug_vis import make_mae_debug_images
from src.utils.schedulers import CosineWDSchedule, WarmupCosineSchedule

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


def _extract_loader_len(loader) -> int:
    try:
        return len(loader)
    except Exception:
        return -1


def mae_collate(batch):
    clips: list[torch.Tensor] = []
    for item in batch:
        if item is None or len(item) < 1:
            continue
        split_clips = item[0]
        if split_clips is None:
            continue
        for clip in split_clips:
            clips.append(clip)
    if len(clips) == 0:
        raise RuntimeError("Empty batch in mae_collate")
    return torch.stack(clips, dim=0)


def _sin_cos_1d(embed_dim: int, positions: torch.Tensor) -> torch.Tensor:
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even for sin-cos encoding, got {embed_dim}")
    omega = torch.arange(embed_dim // 2, dtype=positions.dtype, device=positions.device)
    omega = omega / float(embed_dim // 2)
    omega = 1.0 / (10000**omega)
    out = torch.einsum("n,d->nd", positions, omega)
    emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)
    return emb


def _get_3d_pos_embed_non_square(
    *,
    embed_dim: int,
    t_patches: int,
    h_patches: int,
    w_patches: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    # Token order matches Conv3D flatten order: T-major, H-major, W-minor.
    ids = torch.arange(t_patches * h_patches * w_patches, device=device, dtype=dtype)
    hw = h_patches * w_patches
    t = torch.div(ids, hw, rounding_mode="floor")
    rem = ids - t * hw
    h = torch.div(rem, w_patches, rounding_mode="floor")
    w = rem - h * w_patches
    return _sin_cos_1d(embed_dim, t) + _sin_cos_1d(embed_dim, h) + _sin_cos_1d(embed_dim, w)


class MAEModel(nn.Module):
    def __init__(
        self,
        *,
        encoder: nn.Module,
        patch_size: int,
        tubelet_size: int,
        in_chans: int,
        mask_ratio: float = 0.75,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 4,
        decoder_num_heads: int = 8,
        decoder_mlp_ratio: float = 4.0,
        norm_pix_loss: bool = False,
        loss_type: str = "l2",
        use_sdpa: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.patch_size = int(patch_size)
        self.tubelet_size = int(tubelet_size)
        self.in_chans = int(in_chans)
        self.mask_ratio = float(mask_ratio)
        self.norm_pix_loss = bool(norm_pix_loss)
        self.loss_type = str(loss_type).lower()
        if self.loss_type not in {"l1", "l2"}:
            raise ValueError(f"Unsupported loss_type={loss_type}. Use 'l1' or 'l2'.")

        enc_embed_dim = int(getattr(self.encoder, "embed_dim"))
        self.decoder_embed = nn.Linear(enc_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    dim=decoder_embed_dim,
                    num_heads=decoder_num_heads,
                    mlp_ratio=decoder_mlp_ratio,
                    qkv_bias=True,
                    drop=0.0,
                    attn_drop=0.0,
                    drop_path=0.0,
                    norm_layer=nn.LayerNorm,
                    use_sdpa=use_sdpa,
                    use_rope=False,
                )
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        patch_dim = self.in_chans * self.tubelet_size * self.patch_size * self.patch_size
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_dim, bias=True)
        nn.init.normal_(self.mask_token, std=0.02)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, H, W] -> [B, N, tubelet*patch*patch*C]
        b, c, t, h, w = x.shape
        p = self.patch_size
        u = self.tubelet_size
        if t % u != 0:
            raise ValueError(f"Input T={t} not divisible by tubelet_size={u}")
        if h % p != 0 or w % p != 0:
            raise ValueError(f"Input HxW=({h},{w}) not divisible by patch_size={p}")
        tp = t // u
        hp = h // p
        wp = w // p
        x = x.reshape(b, c, tp, u, hp, p, wp, p)
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
        x = x.reshape(b, tp * hp * wp, u * p * p * c)
        return x

    @staticmethod
    def _random_masking(x_tokens: torch.Tensor, mask_ratio: float):
        b, n, _ = x_tokens.shape
        len_keep = max(1, int(n * (1.0 - mask_ratio)))
        noise = torch.rand(b, n, device=x_tokens.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        mask = torch.ones((b, n), device=x_tokens.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return ids_keep, ids_restore, mask

    def _forward_encoder(self, x: torch.Tensor, ids_keep: torch.Tensor) -> torch.Tensor:
        # Encoder removes masked tokens via `masks=ids_keep`.
        latent = self.encoder(x, masks=ids_keep, training=False)
        return latent

    def _forward_decoder(
        self,
        latent: torch.Tensor,
        ids_restore: torch.Tensor,
        t_patches: int,
        h_patches: int,
        w_patches: int,
    ) -> torch.Tensor:
        x = self.decoder_embed(latent)
        b, len_keep, d = x.shape
        n = ids_restore.shape[1]
        if n < len_keep:
            raise ValueError(f"ids_restore has N={n} < len_keep={len_keep}")

        mask_tokens = self.mask_token.repeat(b, n - len_keep, 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        index = ids_restore.unsqueeze(-1).repeat(1, 1, d)
        x_ = torch.gather(x_, dim=1, index=index)

        pos = _get_3d_pos_embed_non_square(
            embed_dim=d,
            t_patches=t_patches,
            h_patches=h_patches,
            w_patches=w_patches,
            device=x_.device,
            dtype=x_.dtype,
        )
        x_ = x_ + pos.unsqueeze(0)

        for blk in self.decoder_blocks:
            x_, _ = blk(x_)
        x_ = self.decoder_norm(x_)
        pred = self.decoder_pred(x_)
        return pred

    def _forward_loss(
        self,
        x: torch.Tensor,
        pred: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        target = self._patchify(x)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True, unbiased=False)
            target = (target - mean) / torch.sqrt(var + 1e-6)

        if self.loss_type == "l1":
            loss_patch = torch.abs(pred - target).mean(dim=-1)
        else:
            loss_patch = (pred - target).pow(2).mean(dim=-1)

        denom = torch.sum(mask).clamp_min(1.0)
        loss = torch.sum(loss_patch * mask) / denom
        return loss

    def forward(self, x: torch.Tensor):
        # x: [B,C,T,H,W]
        _, _, t, h, w = x.shape
        p = self.patch_size
        u = self.tubelet_size
        if t % u != 0:
            raise ValueError(f"Input T={t} not divisible by tubelet_size={u}")
        if h % p != 0 or w % p != 0:
            raise ValueError(f"Input HxW=({h},{w}) not divisible by patch_size={p}")
        t_patches = t // u
        h_patches = h // p
        w_patches = w // p
        n = t_patches * h_patches * w_patches

        x_tokens = torch.empty((x.shape[0], n, 1), device=x.device, dtype=x.dtype)
        ids_keep, ids_restore, mask = self._random_masking(x_tokens, self.mask_ratio)
        latent = self._forward_encoder(x, ids_keep)
        pred = self._forward_decoder(latent, ids_restore, t_patches, h_patches, w_patches)
        loss = self._forward_loss(x, pred, mask)
        return loss, pred, mask


def main(args, resume_preempt: bool = False):
    folder = str(args.get("folder"))

    cfgs_meta = args.get("meta", {})
    cfgs_model = args.get("model", {})
    cfgs_data = args.get("data", {})
    cfgs_data_aug = args.get("data_aug", {})
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
    dist_port = int(cfgs_meta.get("dist_port", 37149))

    _seed_everything(seed)
    torch.backends.cudnn.benchmark = True
    if cfgs_meta.get("use_tf32", True):
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
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
    latest_path = output_dir / "latest_mae.pth.tar"

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
    vis_interval = int(cfgs_meta.get("vis_interval", 0) or 0)
    vis_max_temporal_slices = int(cfgs_meta.get("vis_max_temporal_slices", 8) or 8)

    # -- Data params
    dataset_type = str(cfgs_data.get("dataset_type", "eventdataset"))
    dataset_paths = cfgs_data.get("datasets", [])
    datasets_weights = cfgs_data.get("datasets_weights", None)
    dataset_fpcs = list(cfgs_data.get("dataset_fpcs", [16]))
    if len(dataset_fpcs) == 0:
        raise ValueError("data.dataset_fpcs must be non-empty")
    dataset_fpcs = [int(v) for v in dataset_fpcs]
    if len(set(dataset_fpcs)) != 1:
        raise ValueError(
            f"MAE training currently requires a single temporal length across datasets. got dataset_fpcs={dataset_fpcs}"
        )
    batch_size = int(cfgs_data.get("batch_size", 8))
    tubelet_size = int(cfgs_data.get("tubelet_size", 2))
    fps = cfgs_data.get("fps", None)
    crop_size = _to_hw_tuple(cfgs_data.get("crop_size", [480, 640]), "data.crop_size")
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
    frame_sample_rate = int(cfgs_data.get("frame_sample_rate", 1))
    num_clips = int(cfgs_data.get("num_clips", 1))
    random_clip_sampling = bool(cfgs_data.get("random_clip_sampling", True))
    allow_clip_overlap = bool(cfgs_data.get("allow_clip_overlap", False))
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

    if len(dataset_paths) == 0:
        raise ValueError("data.datasets is empty. Please provide one or more dataset roots/manifests/H5 files.")

    # -- Data aug params
    random_horizontal_flip = bool(cfgs_data_aug.get("random_horizontal_flip", True))
    ar_range = tuple(float(v) for v in cfgs_data_aug.get("random_resize_aspect_ratio", [0.75, 1.333333]))
    rr_scale = tuple(float(v) for v in cfgs_data_aug.get("random_resize_scale", [0.3, 1.0]))
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
                raise ValueError("data_aug.allowed_input_hw must be a list of [H, W] pairs")
            allowed_input_hw.add((int(hw[0]), int(hw[1])))

    # -- Model params
    compile_model = bool(cfgs_model.get("compile_model", False))
    use_activation_checkpointing = bool(cfgs_model.get("use_activation_checkpointing", False))
    model_name = str(cfgs_model.get("model_name", "vit_base"))
    in_chans = int(cfgs_model.get("in_chans", 20))
    uniform_power = bool(cfgs_model.get("uniform_power", True))
    use_rope = bool(cfgs_model.get("use_rope", True))
    use_silu = bool(cfgs_model.get("use_silu", False))
    wide_silu = bool(cfgs_model.get("wide_silu", True))
    is_causal = bool(cfgs_model.get("is_causal", False))
    init_type = str(cfgs_model.get("init_type", "default"))
    img_temporal_dim_size = cfgs_model.get("img_temporal_dim_size", None)
    if img_temporal_dim_size is not None:
        img_temporal_dim_size = int(img_temporal_dim_size)
        raise NotImplementedError(
            "MAE pretraining currently supports video branch only. "
            "Set model.img_temporal_dim_size=null."
        )
    n_registers = int(cfgs_model.get("n_registers", 0))
    has_cls_first = bool(cfgs_model.get("has_cls_first", False))
    interpolate_rope = bool(cfgs_model.get("interpolate_rope", False))
    modality_embedding = bool(cfgs_model.get("modality_embedding", False))

    # -- MAE params
    mask_ratio = float(cfgs_model.get("mask_ratio", 0.75))
    decoder_embed_dim = int(cfgs_model.get("decoder_embed_dim", 512))
    decoder_depth = int(cfgs_model.get("decoder_depth", 4))
    decoder_num_heads = int(cfgs_model.get("decoder_num_heads", 8))
    decoder_mlp_ratio = float(cfgs_model.get("decoder_mlp_ratio", 4.0))
    norm_pix_loss = bool(cfgs_model.get("norm_pix_loss", False))
    loss_type = str(cfgs_model.get("loss_type", "l2"))

    # -- Optimization params
    ipe = cfgs_opt.get("ipe", None)
    if ipe is not None:
        ipe = int(ipe)
    num_epochs = int(cfgs_opt.get("epochs", 100))
    wd = float(cfgs_opt.get("weight_decay", 0.04))
    final_wd = float(cfgs_opt.get("final_weight_decay", wd))
    lr = float(cfgs_opt.get("lr", 1.5e-4))
    start_lr = float(cfgs_opt.get("start_lr", 1e-6))
    final_lr = float(cfgs_opt.get("final_lr", 1e-6))
    warmup = float(cfgs_opt.get("warmup", 0.1))
    ipe_scale = float(cfgs_opt.get("ipe_scale", 1.0))
    betas = tuple(float(v) for v in cfgs_opt.get("betas", [0.9, 0.95]))
    eps = float(cfgs_opt.get("eps", 1.0e-8))
    clip_grad = cfgs_opt.get("clip_grad", None)
    if clip_grad is not None:
        clip_grad = float(clip_grad)
        if clip_grad <= 0.0:
            clip_grad = None

    logger.info(
        f"Dataset setup: type={dataset_type}, paths={dataset_paths}, "
        f"dataset_fpcs={dataset_fpcs}, batch_size={batch_size}, num_clips={num_clips}, "
        f"activity_filter_enabled={activity_filter_enabled}, "
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

    max_num_frames = int(dataset_fpcs[0])
    if max_num_frames < tubelet_size and (
        img_temporal_dim_size is None or max_num_frames != img_temporal_dim_size
    ):
        raise ValueError(
            f"dataset_fpcs={dataset_fpcs} incompatible with tubelet_size={tubelet_size}. "
            "Increase fpc or set model.img_temporal_dim_size for image branch."
        )

    encoder = video_vit.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        in_chans=in_chans,
        num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=bool(cfgs_meta.get("use_sdpa", True)),
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
    ).to(device)

    mae_model = MAEModel(
        encoder=encoder,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        in_chans=in_chans,
        mask_ratio=mask_ratio,
        decoder_embed_dim=decoder_embed_dim,
        decoder_depth=decoder_depth,
        decoder_num_heads=decoder_num_heads,
        decoder_mlp_ratio=decoder_mlp_ratio,
        norm_pix_loss=norm_pix_loss,
        loss_type=loss_type,
        use_sdpa=bool(cfgs_meta.get("use_sdpa", True)),
    ).to(device)

    if compile_model:
        logger.info("Compiling MAE model")
        torch._dynamo.config.optimize_ddp = False
        mae_model.compile()

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
        collator=mae_collate,
        rank=rank,
        world_size=world_size,
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

    optimizer = torch.optim.AdamW(
        [p for p in mae_model.parameters() if p.requires_grad],
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=wd,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(mixed_precision and torch.cuda.is_available()))
    autocast_device_type = "cuda" if torch.cuda.is_available() else "cpu"
    scheduler = WarmupCosineSchedule(
        optimizer=optimizer,
        warmup_steps=int(warmup * ipe),
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        T_max=int(ipe_scale * num_epochs * ipe),
    )
    wd_scheduler = CosineWDSchedule(
        optimizer=optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale * num_epochs * ipe),
    )

    mae_model = _maybe_ddp(mae_model, find_unused_parameters=False)

    start_epoch = 0
    should_resume_latest = auto_resume_latest and latest_path.exists()
    if load_model or should_resume_latest:
        resolved_load_path = load_path if load_path is not None else latest_path
        checkpoint = robust_checkpoint_loader(str(resolved_load_path), map_location=torch.device("cpu"))
        model_to_load = mae_model.module if isinstance(mae_model, DistributedDataParallel) else mae_model
        msg = model_to_load.load_state_dict(checkpoint["mae_model"], strict=False)
        logger.info(f"loaded MAE model from {resolved_load_path} with msg: {msg}")
        optimizer.load_state_dict(checkpoint["opt"])
        if scaler is not None and checkpoint.get("scaler", None) is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint.get("epoch", 0))
        for _ in range(start_epoch * ipe):
            scheduler.step()
            wd_scheduler.step()

    def save_checkpoint(epoch: int, path: Path, loss_val: float):
        if rank != 0:
            return
        model_to_save = mae_model.module if isinstance(mae_model, DistributedDataParallel) else mae_model
        save_dict = {
            "mae_model": model_to_save.state_dict(),
            # Keep an `encoder` entry for downstream compatibility.
            "encoder": model_to_save.encoder.state_dict(),
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
        import gc

        gc.disable()
        gc.collect()

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
        iter_time_meter = AverageMeter()
        gpu_time_meter = AverageMeter()
        data_elapsed_time_meter = AverageMeter()

        for itr in range(ipe):
            itr_start_time = time.time()

            iter_retries = 0
            iter_successful = False
            while not iter_successful:
                try:
                    batch_clips = next(loader)
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

            clips = batch_clips.to(device, non_blocking=True)
            data_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0

            clip_in_chans = int(clips.shape[1])
            if clip_in_chans != in_chans:
                raise ValueError(
                    f"Input channel mismatch: model.in_chans={in_chans}, but batch clip has C={clip_in_chans}. "
                    "Please set model.in_chans to match the preprocessed voxel channel count."
                )
            if preserve_input_size and allowed_input_hw:
                clip_h = int(clips.shape[-2])
                clip_w = int(clips.shape[-1])
                if (clip_h, clip_w) not in allowed_input_hw:
                    raise ValueError(
                        f"Unexpected input resolution HxW=({clip_h},{clip_w}). "
                        f"Allowed values: {sorted(allowed_input_hw)}"
                    )

            if sync_gc and (itr + 1) % gc_collect_itr_freq == 0:
                import gc

                gc.collect()

            global_step = epoch * ipe + itr
            should_visualize = (
                tb_writer is not None
                and vis_interval > 0
                and (global_step == 0 or global_step % vis_interval == 0)
            )

            def train_step():
                new_lr = scheduler.step()
                new_wd = wd_scheduler.step()
                with torch.autocast(
                    device_type=autocast_device_type,
                    dtype=dtype,
                    enabled=(mixed_precision and torch.cuda.is_available()),
                ):
                    loss, pred, mask = mae_model(clips)
                    debug_images = None
                    if should_visualize:
                        with torch.no_grad():
                            debug_images = make_mae_debug_images(
                                clips=clips,
                                pred=pred,
                                mask=mask,
                                patch_size=patch_size,
                                tubelet_size=tubelet_size,
                                loss_type=loss_type,
                                norm_pix_loss=norm_pix_loss,
                                max_slices=vis_max_temporal_slices,
                            )

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if clip_grad is not None:
                        torch.nn.utils.clip_grad_norm_(
                            mae_model.parameters(),
                            clip_grad,
                        )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if clip_grad is not None:
                        torch.nn.utils.clip_grad_norm_(
                            mae_model.parameters(),
                            clip_grad,
                        )
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                return float(loss), float(new_lr), float(new_wd), debug_images

            (loss, new_lr, new_wd, debug_images), gpu_etime_ms = gpu_timer(train_step)
            iter_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0

            loss_meter.update(loss)
            iter_time_meter.update(iter_elapsed_time_ms)
            gpu_time_meter.update(gpu_etime_ms)
            data_elapsed_time_meter.update(data_elapsed_time_ms)

            csv_logger.log(
                epoch + 1,
                itr,
                loss,
                int(iter_elapsed_time_ms),
                int(gpu_etime_ms),
                int(data_elapsed_time_ms),
            )

            if tb_writer is not None:
                tb_writer.add_scalar("train/loss", loss, global_step)
                tb_writer.add_scalar("train/lr", new_lr, global_step)
                tb_writer.add_scalar("train/wd", new_wd, global_step)
                tb_writer.add_scalar("time/iter_ms", iter_elapsed_time_ms, global_step)
                tb_writer.add_scalar("time/gpu_ms", gpu_etime_ms, global_step)
                tb_writer.add_scalar("time/data_ms", data_elapsed_time_ms, global_step)
                if debug_images is not None:
                    for tag, image in debug_images.items():
                        tb_writer.add_image(tag, image, global_step)

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

            if (itr % log_freq == 0 or itr == ipe - 1 or np.isnan(loss) or np.isinf(loss)) and not use_tqdm:
                logger.info(
                    "[%d, %5d] mae_loss: %.4f [wd: %.2e] [lr: %.2e] [mem: %.2f MB] [iter: %.1f ms] [gpu: %.1f ms] [data: %.1f ms]"
                    % (
                        epoch + 1,
                        itr,
                        loss_meter.avg,
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

        logger.info(f"Epoch {epoch + 1} avg MAE loss: {loss_meter.avg:.6f}")
        if (epoch + 1) % checkpoint_freq == 0 or epoch == (num_epochs - 1):
            save_checkpoint(epoch + 1, latest_path, loss_meter.avg)
            if save_every_freq > 0 and (epoch + 1) % save_every_freq == 0:
                save_checkpoint(epoch + 1, output_dir / f"e{epoch + 1}_mae.pth.tar", loss_meter.avg)

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
