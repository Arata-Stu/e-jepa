from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.models.vision_transformer as video_vit
from scripts.mae.train import MAEModel, _resolve_interpolation, _to_hw_tuple
from src.datasets.event_dataset import EventVideoDataset
from src.datasets.transforms import make_event_transforms
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.pretrain_debug_vis import patchify_video, unpatchify_video


_RESAMPLING = getattr(Image, "Resampling", Image)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize MAE input/reconstruction/error from a saved MAE checkpoint. "
            "Use a resolved MAE config, usually <run-dir>/params-mae-resolved.yaml."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Run directory containing params-mae-resolved.yaml and latest_mae.pth.tar.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Resolved MAE config. Defaults to <run-dir>/params-mae-resolved.yaml.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="MAE checkpoint. Defaults to <run-dir>/latest_mae.pth.tar.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <run-dir>/mae_reconstruction_visualizations.",
    )
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--sample-indices", nargs="*", type=int, default=None)
    parser.add_argument(
        "--sampling",
        choices=("spread", "first", "random"),
        default="spread",
        help="How to choose samples when --sample-indices is omitted.",
    )
    parser.add_argument("--clip-id", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--panel-width", type=int, default=190)
    parser.add_argument("--seed", type=int, default=239)
    parser.add_argument("--mask-seed", type=int, default=1234)
    parser.add_argument(
        "--mask-ratio",
        type=float,
        default=None,
        help="Optional visualization-time mask ratio override.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--use-config-random-clip-sampling",
        action="store_true",
        help="Use random clip sampling from the resolved config. Default is deterministic center sampling.",
    )
    parser.add_argument(
        "--use-config-random-horizontal-flip",
        action="store_true",
        help="Use random horizontal flip from the resolved config. Default disables flip for easier comparison.",
    )
    parser.add_argument(
        "--contact-sheet-columns",
        type=int,
        default=2,
        help="Number of columns in contact_sheet.png.",
    )
    return parser.parse_args()


def _resolve_path(path: Path, *, base_dir: Path | None = None) -> Path:
    out = path.expanduser()
    if not out.is_absolute() and base_dir is not None:
        out = base_dir / out
    return out.resolve()


def _load_config(args: argparse.Namespace) -> tuple[dict[str, Any], Path | None, Path]:
    run_dir = None if args.run_dir is None else _resolve_path(args.run_dir)
    config_path = args.config
    if config_path is None:
        if run_dir is None:
            raise ValueError("Either --run-dir or --config must be provided.")
        config_path = run_dir / "params-mae-resolved.yaml"
    config_path = _resolve_path(config_path, base_dir=run_dir)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected dict config at {config_path}, got {type(cfg)}")
    return cfg, run_dir, config_path


def _resolve_checkpoint_path(args: argparse.Namespace, *, run_dir: Path | None) -> Path:
    checkpoint = args.checkpoint
    if checkpoint is None:
        if run_dir is None:
            raise ValueError("Either --run-dir or --checkpoint must be provided.")
        checkpoint = run_dir / "latest_mae.pth.tar"
    checkpoint = _resolve_path(checkpoint, base_dir=run_dir)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    return checkpoint


def _resolve_output_dir(args: argparse.Namespace, *, run_dir: Path | None, cfg: dict[str, Any]) -> Path:
    if args.output_dir is not None:
        return _resolve_path(args.output_dir, base_dir=run_dir)
    if run_dir is not None:
        return (run_dir / "mae_reconstruction_visualizations").resolve()
    folder = Path(str(cfg.get("folder", "outputs/mae_reconstruction_visualizations")))
    return (folder / "mae_reconstruction_visualizations").resolve()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device=cuda was requested, but CUDA is unavailable.")
        return torch.device("cuda:0")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _build_dataset(
    *,
    cfg: dict[str, Any],
    use_config_random_clip_sampling: bool,
    use_config_random_horizontal_flip: bool,
) -> EventVideoDataset:
    cfg_data = dict(cfg.get("data", {}))
    cfg_aug = dict(cfg.get("data_aug", {}))

    dataset_type = str(cfg_data.get("dataset_type", "eventdataset")).lower()
    if dataset_type not in {"eventdataset", "eventvoxel", "videodataset"}:
        raise ValueError(f"Only event datasets are supported, got data.dataset_type={dataset_type}")

    dataset_paths = _ensure_list(cfg_data.get("datasets", []))
    if len(dataset_paths) == 0:
        raise ValueError("data.datasets is empty in the resolved config.")

    dataset_fpcs = [int(v) for v in _ensure_list(cfg_data.get("dataset_fpcs", [8]))]
    crop_size = _to_hw_tuple(cfg_data.get("crop_size", [240, 320]), "data.crop_size")

    pad_to_hw_raw = cfg_aug.get("pad_to_hw", None)
    pad_to_hw = None
    if pad_to_hw_raw is not None:
        if not isinstance(pad_to_hw_raw, (list, tuple)) or len(pad_to_hw_raw) != 2:
            raise ValueError("data_aug.pad_to_hw must be [H, W] or null")
        pad_to_hw = (int(pad_to_hw_raw[0]), int(pad_to_hw_raw[1]))

    transform = make_event_transforms(
        random_horizontal_flip=(
            bool(cfg_aug.get("random_horizontal_flip", True))
            and bool(use_config_random_horizontal_flip)
        ),
        random_resize_aspect_ratio=tuple(
            float(v) for v in cfg_aug.get("random_resize_aspect_ratio", [0.75, 1.3333333333])
        ),
        random_resize_scale=tuple(float(v) for v in cfg_aug.get("random_resize_scale", [0.3, 1.0])),
        crop_size=crop_size,
        interpolation=_resolve_interpolation(str(cfg_aug.get("interpolation", "bilinear"))),
        antialias=bool(cfg_aug.get("antialias", True)),
        apply_random_resized_crop=not bool(cfg_aug.get("preserve_input_size", False)),
        pad_to_hw=pad_to_hw,
        pad_value=float(cfg_aug.get("pad_value", 0.0)),
    )

    return EventVideoDataset(
        data_paths=dataset_paths,
        datasets_weights=cfg_data.get("datasets_weights", None),
        frames_per_clip=int(max(dataset_fpcs)),
        dataset_fpcs=dataset_fpcs,
        frame_step=int(cfg_data.get("frame_sample_rate", 1)),
        fps=cfg_data.get("fps", None),
        num_clips=int(cfg_data.get("num_clips", 1)),
        transform=transform,
        shared_transform=None,
        random_clip_sampling=(
            bool(cfg_data.get("random_clip_sampling", True))
            and bool(use_config_random_clip_sampling)
        ),
        allow_clip_overlap=bool(cfg_data.get("allow_clip_overlap", False)),
        file_pattern=str(cfg_data.get("file_pattern", "*.h5")),
        recursive=bool(cfg_data.get("recursive", True)),
        require_voxels_key=True,
        max_open_h5_files=int(cfg_data.get("max_open_h5_files", 32)),
        activity_filter_enabled=bool(cfg_data.get("activity_filter_enabled", False)),
        activity_filter_min_clip_mean_active_pixel_ratio=cfg_data.get(
            "activity_filter_min_clip_mean_active_pixel_ratio",
            None,
        ),
        activity_filter_min_clip_mean_activity_score=cfg_data.get(
            "activity_filter_min_clip_mean_activity_score",
            None,
        ),
        activity_filter_min_clip_active_window_ratio=cfg_data.get(
            "activity_filter_min_clip_active_window_ratio",
            None,
        ),
        activity_filter_active_window_threshold=cfg_data.get(
            "activity_filter_active_window_threshold",
            None,
        ),
    )


def _build_model(cfg: dict[str, Any], *, device: torch.device, mask_ratio: float | None) -> MAEModel:
    cfg_data = dict(cfg.get("data", {}))
    cfg_model = dict(cfg.get("model", {}))
    cfg_meta = dict(cfg.get("meta", {}))

    dataset_fpcs = [int(v) for v in _ensure_list(cfg_data.get("dataset_fpcs", [8]))]
    if len(dataset_fpcs) == 0:
        raise ValueError("data.dataset_fpcs is empty.")
    if len(set(dataset_fpcs)) != 1:
        raise ValueError(f"MAE visualization expects one fpc, got {dataset_fpcs}")

    crop_size = _to_hw_tuple(cfg_data.get("crop_size", [240, 320]), "data.crop_size")
    patch_size = int(cfg_data.get("patch_size", 16))
    tubelet_size = int(cfg_data.get("tubelet_size", 2))
    max_num_frames = int(dataset_fpcs[0])
    model_name = str(cfg_model.get("model_name", "vit_tiny"))

    if model_name not in video_vit.__dict__:
        raise KeyError(f"Unknown video ViT model_name={model_name}")

    encoder = video_vit.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        in_chans=int(cfg_model.get("in_chans", 2)),
        num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        uniform_power=bool(cfg_model.get("uniform_power", True)),
        use_sdpa=bool(cfg_meta.get("use_sdpa", True)),
        use_rope=bool(cfg_model.get("use_rope", True)),
        use_silu=bool(cfg_model.get("use_silu", False)),
        wide_silu=bool(cfg_model.get("wide_silu", True)),
        is_causal=bool(cfg_model.get("is_causal", False)),
        init_type=str(cfg_model.get("init_type", "default")),
        img_temporal_dim_size=cfg_model.get("img_temporal_dim_size", None),
        n_registers=int(cfg_model.get("n_registers", 0)),
        has_cls_first=bool(cfg_model.get("has_cls_first", False)),
        interpolate_rope=bool(cfg_model.get("interpolate_rope", False)),
        modality_embedding=bool(cfg_model.get("modality_embedding", False)),
        use_activation_checkpointing=False,
    )

    mae_model = MAEModel(
        encoder=encoder,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        in_chans=int(cfg_model.get("in_chans", 2)),
        mask_ratio=float(cfg_model.get("mask_ratio", 0.75) if mask_ratio is None else mask_ratio),
        decoder_embed_dim=int(cfg_model.get("decoder_embed_dim", 512)),
        decoder_depth=int(cfg_model.get("decoder_depth", 4)),
        decoder_num_heads=int(cfg_model.get("decoder_num_heads", 8)),
        decoder_mlp_ratio=float(cfg_model.get("decoder_mlp_ratio", 4.0)),
        norm_pix_loss=bool(cfg_model.get("norm_pix_loss", False)),
        loss_type=str(cfg_model.get("loss_type", "l2")),
        use_sdpa=bool(cfg_meta.get("use_sdpa", True)),
    ).to(device)
    mae_model.eval()
    return mae_model


def _load_checkpoint(model: MAEModel, checkpoint_path: Path) -> dict[str, Any]:
    ckpt = robust_checkpoint_loader(str(checkpoint_path), map_location=torch.device("cpu"))
    state = ckpt.get("mae_model", None)
    if state is None:
        raise KeyError(
            f"{checkpoint_path} does not contain 'mae_model'. "
            "The encoder-only checkpoint cannot reconstruct MAE outputs."
        )
    if not isinstance(state, dict):
        raise ValueError(f"Expected mae_model state dict, got {type(state)}")
    cleaned = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned[key] = value
    msg = model.load_state_dict(cleaned, strict=False)
    if len(msg.missing_keys) > 0 or len(msg.unexpected_keys) > 0:
        print(f"Checkpoint load summary: missing={msg.missing_keys}, unexpected={msg.unexpected_keys}")
    return ckpt


def _pick_indices(
    total: int,
    *,
    num_samples: int,
    sample_indices: list[int] | None,
    sampling: str,
    seed: int,
) -> list[int]:
    if total <= 0:
        return []
    if sample_indices:
        out = []
        for idx in sample_indices:
            if idx < 0 or idx >= total:
                raise IndexError(f"sample index out of range: {idx} (dataset size={total})")
            out.append(int(idx))
        return out

    count = max(1, min(int(num_samples), total))
    if sampling == "first":
        return list(range(count))
    if sampling == "random":
        rng = random.Random(seed)
        return sorted(rng.sample(range(total), k=count))
    if count == 1:
        return [0]
    return sorted({int(round(float(v))) for v in np.linspace(0, total - 1, num=count)})


def _normalize_uint8(values: np.ndarray, *, upper: float | None = None, percentile: float = 99.5) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    if upper is None:
        upper = float(np.percentile(arr, percentile))
    if upper <= 0.0:
        upper = float(arr.max()) if arr.size > 0 else 0.0
    if upper <= 0.0:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.clip(np.round(np.clip(arr / upper, 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)


def _gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    gray = np.asarray(gray, dtype=np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def _mask_to_pixel(mask_2d: np.ndarray, *, patch_size: int, height: int, width: int) -> np.ndarray:
    up = np.repeat(np.repeat(mask_2d.astype(bool), int(patch_size), axis=0), int(patch_size), axis=1)
    return up[:height, :width]


def _patch_grid_to_pixel(values_2d: np.ndarray, *, patch_size: int, height: int, width: int) -> np.ndarray:
    values = np.asarray(values_2d, dtype=np.float32)
    up = np.repeat(np.repeat(values, int(patch_size), axis=0), int(patch_size), axis=1)
    return up[:height, :width]


def _overlay_mask(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    base = np.asarray(rgb, dtype=np.float32)
    out = base.copy()
    tint = np.asarray((230, 63, 63), dtype=np.float32)
    out[mask] = base[mask] * 0.30 + tint * 0.70
    out[~mask] = base[~mask] * 0.80
    return np.clip(out, 0, 255).astype(np.uint8)


def _resize_panel(image: Image.Image, *, panel_width: int, nearest: bool = False) -> Image.Image:
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {(width, height)}")
    scale = float(panel_width) / float(width)
    size = (int(panel_width), max(1, int(round(height * scale))))
    return image.resize(size, resample=_RESAMPLING.NEAREST if nearest else _RESAMPLING.BILINEAR)


def _annotate(image: Image.Image, label: str) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default()
    try:
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = int(bbox[2] - bbox[0])
        text_h = int(bbox[3] - bbox[1])
    except Exception:
        text_w = max(1, len(label) * 6)
        text_h = 11
    pad = 4
    draw.rectangle([(0, 0), (text_w + pad * 2, text_h + pad * 2)], fill=(0, 0, 0))
    draw.text((pad, pad), label, fill=(255, 255, 255), font=font)
    return out


def _make_row(panels: list[tuple[str, np.ndarray, bool]], *, panel_width: int) -> Image.Image:
    rendered = []
    for label, rgb, nearest in panels:
        img = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
        img = _resize_panel(img, panel_width=panel_width, nearest=nearest)
        rendered.append(_annotate(img, label))

    gap = 8
    row_w = sum(img.width for img in rendered) + gap * max(0, len(rendered) - 1)
    row_h = max(img.height for img in rendered)
    row = Image.new("RGB", (row_w, row_h), color=(248, 248, 248))
    x = 0
    for img in rendered:
        row.paste(img, (x, 0))
        x += img.width + gap
    return row


def _make_text_block(lines: list[str], *, width: int) -> Image.Image:
    font = ImageFont.load_default()
    draw_probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    wrapped: list[str] = []
    for line in lines:
        if len(line) <= 120:
            wrapped.append(line)
            continue
        for start in range(0, len(line), 120):
            wrapped.append(line[start : start + 120])

    heights = []
    for line in wrapped:
        try:
            bbox = draw_probe.textbbox((0, 0), line, font=font)
            heights.append(int(bbox[3] - bbox[1]))
        except Exception:
            heights.append(11)
    pad = 8
    line_gap = 4
    height = pad * 2 + sum(heights) + line_gap * max(0, len(heights) - 1)
    image = Image.new("RGB", (width, max(28, height)), color=(248, 248, 248))
    draw = ImageDraw.Draw(image)
    y = pad
    for line, line_h in zip(wrapped, heights):
        draw.text((pad, y), line, fill=(30, 34, 42), font=font)
        y += line_h + line_gap
    return image


def _stack(rows: list[Image.Image]) -> Image.Image:
    gap = 10
    pad = 12
    width = max(row.width for row in rows)
    height = sum(row.height for row in rows) + gap * max(0, len(rows) - 1)
    canvas = Image.new("RGB", (width + pad * 2, height + pad * 2), color=(238, 238, 238))
    y = pad
    for row in rows:
        x = pad + (width - row.width) // 2
        canvas.paste(row, (x, y))
        y += row.height + gap
    return canvas


def _write_contact_sheet(image_paths: list[Path], output_path: Path, *, columns: int) -> None:
    if len(image_paths) == 0:
        return
    images = [Image.open(path).convert("RGB") for path in image_paths]
    max_w = max(img.width for img in images)
    max_h = max(img.height for img in images)
    cols = max(1, int(columns))
    rows = int(math.ceil(len(images) / cols))
    gap = 10
    sheet = Image.new(
        "RGB",
        (cols * max_w + (cols + 1) * gap, rows * max_h + (rows + 1) * gap),
        color=(238, 238, 238),
    )
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = gap + col * (max_w + gap)
        y = gap + row * (max_h + gap)
        sheet.paste(img, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    for img in images:
        img.close()


def _frame_ids(num_frames: int, max_frames: int) -> list[int]:
    ids = list(range(int(num_frames)))
    if max_frames > 0 and len(ids) > max_frames:
        ids = sorted({int(round(float(v))) for v in np.linspace(0, num_frames - 1, num=max_frames)})
    return ids


def _reconstruct_from_pred(
    *,
    clips: torch.Tensor,
    pred: torch.Tensor,
    patch_size: int,
    tubelet_size: int,
    norm_pix_loss: bool,
) -> torch.Tensor:
    target = patchify_video(clips.detach().float(), patch_size=patch_size, tubelet_size=tubelet_size)
    pred_for_recon = pred.detach().float()
    if norm_pix_loss:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True, unbiased=False)
        pred_for_recon = pred_for_recon * torch.sqrt(var + 1.0e-6) + mean
    return unpatchify_video(
        pred_for_recon,
        channels=int(clips.shape[1]),
        frames=int(clips.shape[2]),
        height=int(clips.shape[3]),
        width=int(clips.shape[4]),
        patch_size=patch_size,
        tubelet_size=tubelet_size,
    )


def _patch_losses(
    *,
    clips: torch.Tensor,
    pred: torch.Tensor,
    patch_size: int,
    tubelet_size: int,
    norm_pix_loss: bool,
    loss_type: str,
) -> torch.Tensor:
    target = patchify_video(clips.detach().float(), patch_size=patch_size, tubelet_size=tubelet_size)
    if norm_pix_loss:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True, unbiased=False)
        target = (target - mean) / torch.sqrt(var + 1.0e-6)
    pred = pred.detach().float()
    if str(loss_type).lower() == "l1":
        return torch.abs(pred - target).mean(dim=-1)
    return (pred - target).pow(2).mean(dim=-1)


def _render_sample(
    *,
    cfg: dict[str, Any],
    checkpoint_path: Path,
    sample_path: str,
    sample_index: int,
    clip_indices: list[int],
    clip: torch.Tensor,
    recon: torch.Tensor,
    mask: torch.Tensor,
    patch_loss: torch.Tensor,
    loss_value: float,
    output_path: Path,
    panel_width: int,
    max_frames: int,
) -> list[str]:
    cfg_data = dict(cfg.get("data", {}))
    cfg_model = dict(cfg.get("model", {}))
    patch_size = int(cfg_data.get("patch_size", 16))
    tubelet_size = int(cfg_data.get("tubelet_size", 2))

    clip_np = np.asarray(clip.detach().cpu(), dtype=np.float32)
    recon_np = np.asarray(recon.detach().cpu(), dtype=np.float32)
    mask_np = np.asarray(mask.detach().cpu(), dtype=np.float32)
    patch_loss_np = np.asarray(patch_loss.detach().cpu(), dtype=np.float32)

    channels, num_frames, height, width = [int(v) for v in clip_np.shape]
    tp = num_frames // tubelet_size
    hp = height // patch_size
    wp = width // patch_size
    mask_grid = mask_np.reshape(tp, hp, wp)
    patch_loss_grid = patch_loss_np.reshape(tp, hp, wp)

    input_activity = np.abs(clip_np).sum(axis=0)
    recon_activity = np.abs(recon_np).sum(axis=0)
    err_activity = np.abs(recon_np - clip_np).sum(axis=0)

    act_upper = float(np.percentile(np.concatenate([input_activity.reshape(-1), recon_activity.reshape(-1)]), 99.5))
    err_upper = float(np.percentile(err_activity.reshape(-1), 99.5))
    patch_loss_upper = float(np.percentile(patch_loss_grid.reshape(-1), 99.5))

    visible = mask_np < 0.5
    masked = mask_np >= 0.5
    masked_loss = float(patch_loss_np[masked].mean()) if np.any(masked) else 0.0
    visible_loss = float(patch_loss_np[visible].mean()) if np.any(visible) else 0.0
    input_nonzero = float(np.mean(np.abs(clip_np) > 0.0))
    recon_nonzero = float(np.mean(np.abs(recon_np) > 0.0))

    rows: list[Image.Image] = []
    title_lines = [
        f"MAE reconstruction | sample={sample_index} | loss={loss_value:.6f} | masked_patch_loss={masked_loss:.6f}",
        f"checkpoint={checkpoint_path}",
        f"path={sample_path}",
        f"shape=[C={channels}, T={num_frames}, H={height}, W={width}] mask_ratio={float(mask_np.mean()):.4f} visible_loss={visible_loss:.6f}",
        f"input_nonzero={input_nonzero:.6f} recon_nonzero={recon_nonzero:.6f} model_mask_ratio={float(cfg_model.get('mask_ratio', 0.75))}",
    ]
    rows.append(_make_text_block(title_lines, width=max(6 * panel_width + 5 * 8, 900)))

    for frame_idx in _frame_ids(num_frames, max_frames):
        patch_t = min(tp - 1, frame_idx // max(1, tubelet_size))
        pixel_mask = _mask_to_pixel(
            mask_grid[patch_t],
            patch_size=patch_size,
            height=height,
            width=width,
        )

        input_gray = _normalize_uint8(input_activity[frame_idx], upper=act_upper)
        recon_gray = _normalize_uint8(recon_activity[frame_idx], upper=act_upper)
        err_gray = _normalize_uint8(err_activity[frame_idx], upper=err_upper)
        loss_gray = _normalize_uint8(
            _patch_grid_to_pixel(
                patch_loss_grid[patch_t],
                patch_size=patch_size,
                height=height,
                width=width,
            ),
            upper=patch_loss_upper,
        )

        input_rgb = _gray_to_rgb(input_gray)
        recon_rgb = _gray_to_rgb(recon_gray)
        masked_recon_gray = input_gray.copy()
        masked_recon_gray[pixel_mask] = recon_gray[pixel_mask]
        mask_rgb = _overlay_mask(input_rgb, pixel_mask)

        window_idx = clip_indices[frame_idx] if frame_idx < len(clip_indices) else -1
        rows.append(
            _make_row(
                [
                    (f"input t={frame_idx} win={window_idx}", input_rgb, False),
                    ("reconstruction", recon_rgb, False),
                    ("masked recon only", _gray_to_rgb(masked_recon_gray), False),
                    ("abs error", _gray_to_rgb(err_gray), False),
                    (f"mask patch_t={patch_t}", mask_rgb, True),
                    ("patch loss", _gray_to_rgb(loss_gray), True),
                ],
                panel_width=panel_width,
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = _stack(rows)
    image.save(output_path)
    return title_lines


def main() -> None:
    args = _parse_args()
    cfg, run_dir, config_path = _load_config(args)
    checkpoint_path = _resolve_checkpoint_path(args, run_dir=run_dir)
    output_dir = _resolve_output_dir(args, run_dir=run_dir, cfg=cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    _seed_everything(int(args.seed))
    device = _resolve_device(str(args.device))
    dataset = _build_dataset(
        cfg=cfg,
        use_config_random_clip_sampling=bool(args.use_config_random_clip_sampling),
        use_config_random_horizontal_flip=bool(args.use_config_random_horizontal_flip),
    )
    model = _build_model(cfg, device=device, mask_ratio=args.mask_ratio)
    ckpt = _load_checkpoint(model, checkpoint_path)

    cfg_data = dict(cfg.get("data", {}))
    cfg_model = dict(cfg.get("model", {}))
    patch_size = int(cfg_data.get("patch_size", 16))
    tubelet_size = int(cfg_data.get("tubelet_size", 2))
    norm_pix_loss = bool(cfg_model.get("norm_pix_loss", False))
    loss_type = str(cfg_model.get("loss_type", "l2"))

    indices = _pick_indices(
        len(dataset),
        num_samples=int(args.num_samples),
        sample_indices=args.sample_indices,
        sampling=str(args.sampling),
        seed=int(args.seed),
    )
    if len(indices) == 0:
        raise RuntimeError("No samples selected.")

    image_paths: list[Path] = []
    summary_lines = [
        f"config={config_path}",
        f"checkpoint={checkpoint_path}",
        f"output_dir={output_dir}",
        f"device={device}",
        f"checkpoint_epoch={ckpt.get('epoch', 'unknown')}",
        f"indices={indices}",
        "",
    ]

    model.eval()
    for draw_id, sample_index in enumerate(indices):
        _seed_everything(int(args.seed) + int(sample_index))
        split_clips, _, clip_indices_list = dataset[int(sample_index)]
        clip_id = int(args.clip_id)
        if clip_id < 0 or clip_id >= len(split_clips):
            raise IndexError(f"--clip-id={clip_id} out of range for sample {sample_index}: {len(split_clips)} clips")
        clip = split_clips[clip_id].detach().float()
        clip_indices = [int(v) for v in np.asarray(clip_indices_list[clip_id]).reshape(-1).tolist()]

        _seed_everything(int(args.mask_seed) + int(sample_index))
        clips = clip.unsqueeze(0).to(device, non_blocking=True)
        with torch.no_grad():
            loss, pred, mask = model(clips)

        clips_cpu = clips.detach().cpu()
        pred_cpu = pred.detach().cpu()
        mask_cpu = mask.detach().cpu()
        recon = _reconstruct_from_pred(
            clips=clips_cpu,
            pred=pred_cpu,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            norm_pix_loss=norm_pix_loss,
        )[0]
        patch_loss = _patch_losses(
            clips=clips_cpu,
            pred=pred_cpu,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            norm_pix_loss=norm_pix_loss,
            loss_type=loss_type,
        )[0]

        sample_path = str(dataset.samples[int(sample_index)])
        image_path = output_dir / f"sample_{draw_id:03d}_idx_{int(sample_index):06d}.png"
        lines = _render_sample(
            cfg=cfg,
            checkpoint_path=checkpoint_path,
            sample_path=sample_path,
            sample_index=int(sample_index),
            clip_indices=clip_indices,
            clip=clip,
            recon=recon,
            mask=mask_cpu[0],
            patch_loss=patch_loss,
            loss_value=float(loss.detach().cpu().item()),
            output_path=image_path,
            panel_width=int(args.panel_width),
            max_frames=int(args.max_frames),
        )
        image_paths.append(image_path)
        summary_lines.extend(lines)
        summary_lines.append(f"wrote={image_path}")
        summary_lines.append("")

    contact_sheet = output_dir / "contact_sheet.png"
    _write_contact_sheet(
        image_paths,
        contact_sheet,
        columns=int(args.contact_sheet_columns),
    )
    summary_path = output_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Wrote {len(image_paths)} sample visualizations to {output_dir}")
    print(f"Contact sheet: {contact_sheet}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
