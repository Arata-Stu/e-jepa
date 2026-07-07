from __future__ import annotations

import argparse
import colorsys
import csv
import math
import sys
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.downstream.train import (
    DenseLinearProbe,
    _build_encoder_from_cfg,
    _resize_logits_to_target,
    _to_dtype,
)
from scripts.preprocess.utils import RgbMp4Writer
from src.downstream.datasets import build_dense_task_dataset_from_config
from src.utils.checkpoint_loader import robust_checkpoint_loader


_RESAMPLING = getattr(Image, "Resampling", Image)
_PANEL_WIDTH = 220
_ROW_GAP = 10
_CANVAS_PAD = 14


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run inference on downstream validation samples and save visualizations "
            "using a trained downstream checkpoint."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "Training run directory that contains params-downstream-resolved.yaml and "
            "best_downstream.pth.tar / latest_downstream.pth.tar."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional resolved downstream config path. Defaults to <run-dir>/params-downstream-resolved.yaml.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint path. Defaults to <run-dir>/<checkpoint-tag>_downstream.pth.tar.",
    )
    parser.add_argument(
        "--checkpoint-tag",
        choices=("best", "latest"),
        default="best",
        help="Checkpoint tag to use when --checkpoint is omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to <run-dir>/val_visualizations_<checkpoint-tag>.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=12,
        help="How many validation samples to visualize when --sample-indices is not set.",
    )
    parser.add_argument(
        "--sample-indices",
        type=int,
        nargs="*",
        default=None,
        help="Explicit dataset indices from the validation set to visualize.",
    )
    parser.add_argument(
        "--sample-mode",
        choices=("spread", "contiguous"),
        default="spread",
        help=(
            "How to choose samples when --sample-indices is not set. "
            "'spread' keeps the old evenly-spaced behavior; 'contiguous' is useful for MP4 export."
        ),
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First dataset index when --sample-mode=contiguous.",
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=1,
        help="Dataset-index stride when --sample-mode=contiguous.",
    )
    parser.add_argument(
        "--split",
        choices=("val", "train"),
        default="val",
        help="Dataset split to visualize. Validation is the default.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device.",
    )
    parser.add_argument(
        "--write-video",
        action="store_true",
        help="Also write an MP4 from the rendered sample visualizations.",
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        default=None,
        help="Optional MP4 output path. Setting this also enables video export.",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=8.0,
        help="Frames per second for --write-video.",
    )
    parser.add_argument(
        "--video-width",
        type=int,
        default=0,
        help="Resize rendered video frames to this width before encoding. Use 0 to keep original size.",
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
        config_path = run_dir / "params-downstream-resolved.yaml"
    config_path = _resolve_path(config_path, base_dir=run_dir)
    if not config_path.exists():
        raise FileNotFoundError(f"Resolved config not found: {config_path}")
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected dict config at {config_path}, got {type(cfg)}")
    return cfg, run_dir, config_path


def _resolve_checkpoint_path(args: argparse.Namespace, *, run_dir: Path | None) -> Path:
    checkpoint = args.checkpoint
    if checkpoint is None:
        if run_dir is None:
            raise ValueError("Either --run-dir or --checkpoint must be provided.")
        checkpoint = run_dir / f"{args.checkpoint_tag}_downstream.pth.tar"
    checkpoint = _resolve_path(checkpoint, base_dir=run_dir)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    return checkpoint


def _resolve_output_dir(args: argparse.Namespace, *, run_dir: Path | None, cfg: dict[str, Any]) -> Path:
    if args.output_dir is not None:
        base_dir = run_dir if run_dir is not None else None
        return _resolve_path(args.output_dir, base_dir=base_dir)
    if run_dir is not None:
        return (run_dir / f"{args.split}_visualizations_{args.checkpoint_tag}").resolve()
    folder = Path(str(cfg.get("folder", Path.cwd()))).resolve()
    return (folder / f"{args.split}_visualizations_{args.checkpoint_tag}").resolve()


def _resolve_video_path(args: argparse.Namespace, *, output_dir: Path) -> Path:
    if args.video_path is None:
        return (output_dir / f"{args.split}_visualizations_{args.checkpoint_tag}.mp4").resolve()
    return _resolve_path(args.video_path, base_dir=output_dir)


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device=cuda was requested, but CUDA is unavailable.")
        return torch.device("cuda:0")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _pick_sample_indices(
    total: int,
    *,
    num_samples: int,
    sample_indices: list[int] | None,
    sample_mode: str,
    start_index: int,
    sample_stride: int,
) -> list[int]:
    if total <= 0:
        return []
    if sample_indices:
        picked = []
        for idx in sample_indices:
            if idx < 0 or idx >= total:
                raise IndexError(f"sample index out of range: {idx} (dataset size={total})")
            picked.append(int(idx))
        return picked

    count = max(1, min(int(num_samples), total))
    if sample_mode == "contiguous":
        start = int(start_index)
        stride = int(sample_stride)
        if start < 0 or start >= total:
            raise IndexError(f"--start-index out of range: {start} (dataset size={total})")
        if stride <= 0:
            raise ValueError(f"--sample-stride must be positive, got {stride}")
        return list(range(start, total, stride))[:count]

    if sample_mode != "spread":
        raise ValueError(f"Unsupported sample mode: {sample_mode}")

    if count == total:
        return list(range(total))
    grid = np.linspace(0, total - 1, num=count)
    indices = []
    seen = set()
    for value in grid:
        idx = int(round(float(value)))
        if idx not in seen:
            seen.add(idx)
            indices.append(idx)
    if len(indices) < count:
        for idx in range(total):
            if idx in seen:
                continue
            seen.add(idx)
            indices.append(idx)
            if len(indices) >= count:
                break
    return indices[:count]


def _find_state_key_suffix(state_dict: dict[str, Any], suffix: str) -> str:
    matches = [key for key in state_dict.keys() if key.endswith(suffix)]
    if len(matches) != 1:
        raise KeyError(f"Expected exactly one state key ending with '{suffix}', found {matches}")
    return matches[0]


def _load_model(
    *,
    cfg_model: dict[str, Any],
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    ckpt = robust_checkpoint_loader(str(checkpoint_path), map_location=torch.device("cpu"))
    state_dict = ckpt.get("model", ckpt)
    if not isinstance(state_dict, dict):
        raise ValueError(f"Expected downstream checkpoint dict at {checkpoint_path}, got {type(state_dict)}")

    head_weight_key = _find_state_key_suffix(state_dict, "head.weight")
    head_weight = torch.as_tensor(state_dict[head_weight_key])
    if head_weight.ndim != 4:
        raise ValueError(f"Unexpected head weight shape={tuple(head_weight.shape)} in {checkpoint_path}")
    num_output_channels = int(head_weight.shape[0])

    encoder = _build_encoder_from_cfg(cfg_model).to(device)
    model = DenseLinearProbe(
        encoder=encoder,
        num_output_channels=num_output_channels,
        freeze_encoder=bool(cfg_model.get("freeze_encoder", True)),
        patch_size=int(cfg_model.get("patch_size", 16)),
        head_dropout=float(cfg_model.get("head_dropout", 0.0)),
    ).to(device)
    msg = model.load_state_dict(state_dict, strict=False)
    if len(msg.missing_keys) > 0 or len(msg.unexpected_keys) > 0:
        print(f"Checkpoint load summary: missing={msg.missing_keys}, unexpected={msg.unexpected_keys}")
    model.eval()
    return model, ckpt


def _normalize_to_uint8(arr: np.ndarray, *, percentile: float = 99.5) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    upper = float(np.percentile(arr, percentile))
    if upper <= 0:
        upper = float(arr.max()) if arr.size > 0 else 0.0
    if upper <= 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = np.clip(arr / upper, 0.0, 1.0)
    return np.round(scaled * 255.0).astype(np.uint8)


def _voxel_frame_to_activity_rgb(voxel_chw: np.ndarray) -> np.ndarray:
    activity = np.abs(np.asarray(voxel_chw, dtype=np.float32)).sum(axis=0)
    gray = _normalize_to_uint8(activity, percentile=99.5)
    return np.stack([gray, gray, gray], axis=-1)


def _semantic_palette(ignore_index: int) -> np.ndarray:
    palette = np.zeros((256, 3), dtype=np.uint8)
    colors = [
        (220, 20, 60),
        (255, 127, 80),
        (255, 215, 0),
        (50, 205, 50),
        (0, 170, 0),
        (70, 130, 180),
        (65, 105, 225),
        (138, 43, 226),
        (255, 105, 180),
        (255, 140, 0),
        (0, 206, 209),
        (205, 92, 92),
        (154, 205, 50),
        (123, 104, 238),
        (244, 164, 96),
        (46, 139, 87),
        (210, 180, 140),
        (30, 144, 255),
        (199, 21, 133),
        (188, 143, 143),
    ]
    for idx, color in enumerate(colors):
        palette[idx] = np.asarray(color, dtype=np.uint8)
    for idx in range(len(colors), 255):
        hue = float((idx * 0.61803398875) % 1.0)
        sat = 0.55 + 0.25 * float((idx % 5) / 4.0)
        val = 0.70 + 0.20 * float((idx % 7) / 6.0)
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        palette[idx] = np.asarray(
            [int(round(r * 255.0)), int(round(g * 255.0)), int(round(b * 255.0))],
            dtype=np.uint8,
        )
    if 0 <= int(ignore_index) < 256:
        palette[int(ignore_index)] = np.asarray((24, 24, 24), dtype=np.uint8)
    palette[255] = np.asarray((24, 24, 24), dtype=np.uint8)
    return palette


def _semantic_to_rgb(label: np.ndarray, *, ignore_index: int) -> np.ndarray:
    palette = _semantic_palette(ignore_index)
    label = np.asarray(label, dtype=np.int64)
    rgb = palette[np.clip(label, 0, 255)]
    if 0 <= int(ignore_index) < 256:
        rgb[label == int(ignore_index)] = palette[int(ignore_index)]
    return rgb


def _depth_palette_stops() -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray([0.0, 0.2, 0.4, 0.65, 0.82, 1.0], dtype=np.float32)
    colors = np.asarray(
        [
            (13, 8, 135),
            (75, 3, 161),
            (125, 3, 168),
            (187, 55, 84),
            (249, 142, 8),
            (240, 249, 33),
        ],
        dtype=np.float32,
    )
    return positions, colors


def _colorize_scalar_map(
    values: np.ndarray,
    valid_mask: np.ndarray,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    background: int = 24,
) -> tuple[np.ndarray, dict[str, float | None]]:
    values = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    rgb = np.full((values.shape[0], values.shape[1], 3), int(background), dtype=np.uint8)
    if not np.any(valid):
        return rgb, {"min": None, "max": None, "vmin": None, "vmax": None, "valid_ratio": 0.0}

    valid_values = values[valid]
    lo = float(np.percentile(valid_values, 5.0)) if vmin is None else float(vmin)
    hi = float(np.percentile(valid_values, 95.0)) if vmax is None else float(vmax)
    if hi <= lo:
        lo = float(valid_values.min()) if vmin is None else float(vmin)
        hi = float(valid_values.max()) if vmax is None else float(vmax)
    if hi <= lo:
        hi = lo + 1e-6

    normalized = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    pos, col = _depth_palette_stops()
    colorized = np.zeros((values.shape[0], values.shape[1], 3), dtype=np.float32)
    for channel in range(3):
        colorized[..., channel] = np.interp(normalized, pos, col[:, channel])
    rgb[valid] = np.clip(np.round(colorized[valid]), 0, 255).astype(np.uint8)
    return rgb, {
        "min": float(valid_values.min()),
        "max": float(valid_values.max()),
        "vmin": lo,
        "vmax": hi,
        "valid_ratio": float(np.mean(valid)),
    }


def _semantic_diagnostic_rgb(
    center_rgb: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    ignore_index: int,
) -> np.ndarray:
    base = np.asarray(center_rgb, dtype=np.float32) * 0.28
    target = np.asarray(target, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.int64)
    valid = target != int(ignore_index)
    correct = valid & (target == prediction)
    wrong = valid & (target != prediction)

    out = np.clip(np.round(base), 0, 255).astype(np.uint8)
    if np.any(correct):
        out[correct] = np.asarray((46, 196, 82), dtype=np.uint8)
    if np.any(wrong):
        out[wrong] = np.asarray((228, 58, 58), dtype=np.uint8)
    return out


def _sample_semantic_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    ignore_index: int,
) -> dict[str, float]:
    target = np.asarray(target, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.int64)
    valid = target != int(ignore_index)
    valid_count = int(np.count_nonzero(valid))
    if valid_count == 0:
        return {"pixel_acc": 0.0, "miou": 0.0, "valid_ratio": 0.0}

    pixel_acc = float(np.mean(prediction[valid] == target[valid]))
    classes = np.unique(np.concatenate([target[valid], prediction[valid]], axis=0))
    ious = []
    for cls in classes.tolist():
        cls = int(cls)
        inter = np.count_nonzero(valid & (target == cls) & (prediction == cls))
        union = np.count_nonzero(valid & ((target == cls) | (prediction == cls)))
        if union > 0:
            ious.append(float(inter) / float(union))
    miou = float(np.mean(ious)) if len(ious) > 0 else 0.0
    return {
        "pixel_acc": pixel_acc,
        "miou": miou,
        "valid_ratio": float(valid_count) / float(target.size),
    }


def _sample_depth_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    valid_mask: np.ndarray,
) -> dict[str, float]:
    target = np.asarray(target, dtype=np.float32)
    prediction = np.asarray(prediction, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    valid_count = int(np.count_nonzero(valid))
    if valid_count == 0:
        return {"mae": 0.0, "rmse": 0.0, "valid_ratio": 0.0}
    err = prediction[valid] - target[valid]
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "valid_ratio": float(valid_count) / float(target.size),
    }


def _to_pil(rgb: np.ndarray) -> Image.Image:
    return Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")


def _pil_to_video_rgb(image: Image.Image, *, target_width: int) -> np.ndarray:
    frame = image.convert("RGB")
    width, height = frame.size
    if int(target_width) > 0 and width != int(target_width):
        scale = float(target_width) / float(width)
        width = int(target_width)
        height = max(1, int(round(float(height) * scale)))
        frame = frame.resize((width, height), resample=_RESAMPLING.BILINEAR)

    width, height = frame.size
    even_width = width if width % 2 == 0 else width + 1
    even_height = height if height % 2 == 0 else height + 1
    if even_width != width or even_height != height:
        padded = Image.new("RGB", (even_width, even_height), color=(248, 248, 248))
        padded.paste(frame, (0, 0))
        frame = padded

    return np.asarray(frame, dtype=np.uint8)


def _resize_video_rgb_to_size(frame_rgb: np.ndarray, *, size: tuple[int, int]) -> np.ndarray:
    target_w, target_h = int(size[0]), int(size[1])
    if int(frame_rgb.shape[1]) == target_w and int(frame_rgb.shape[0]) == target_h:
        return np.asarray(frame_rgb, dtype=np.uint8)
    image = _to_pil(frame_rgb)
    resized = image.resize((target_w, target_h), resample=_RESAMPLING.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def _resize_rgb_to_hw(rgb: np.ndarray, *, target_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    if int(rgb.shape[0]) == target_h and int(rgb.shape[1]) == target_w:
        return np.asarray(rgb, dtype=np.uint8)
    image = _to_pil(rgb)
    resized = image.resize((target_w, target_h), resample=_RESAMPLING.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def _resize_panel(image: Image.Image, *, panel_width: int, is_label: bool) -> Image.Image:
    w, h = image.size
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid image size: {(w, h)}")
    scale = float(panel_width) / float(w)
    panel_height = max(1, int(round(float(h) * scale)))
    resample = _RESAMPLING.NEAREST if is_label else _RESAMPLING.BILINEAR
    return image.resize((int(panel_width), int(panel_height)), resample=resample)


def _annotate_panel(image: Image.Image, label: str) -> Image.Image:
    font = ImageFont.load_default()
    draw = ImageDraw.Draw(image)
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
    return image


def _make_panel_row(entries: list[tuple[str, Image.Image, bool]], *, panel_width: int) -> Image.Image:
    if len(entries) == 0:
        raise ValueError("entries cannot be empty")

    rendered: list[Image.Image] = []
    max_h = 0
    for label, image, is_label in entries:
        panel = _resize_panel(image, panel_width=panel_width, is_label=is_label)
        panel = _annotate_panel(panel, label)
        rendered.append(panel)
        max_h = max(max_h, panel.height)

    row_w = sum(panel.width for panel in rendered) + _ROW_GAP * max(0, len(rendered) - 1)
    row = Image.new("RGB", (row_w, max_h), color=(248, 248, 248))
    x = 0
    for panel in rendered:
        row.paste(panel, (x, 0))
        x += panel.width + _ROW_GAP
    return row


def _make_text_block(lines: list[str], *, width: int) -> Image.Image:
    font = ImageFont.load_default()
    draw_probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    wrapped_lines: list[str] = []
    for line in lines:
        wrapped = textwrap.wrap(line, width=100) or [line]
        wrapped_lines.extend(wrapped)

    line_heights = []
    for line in wrapped_lines:
        try:
            bbox = draw_probe.textbbox((0, 0), line, font=font)
            line_heights.append(int(bbox[3] - bbox[1]))
        except Exception:
            line_heights.append(11)

    content_h = sum(line_heights) + 4 * max(0, len(wrapped_lines) - 1)
    canvas_h = max(24, content_h + 6)
    canvas = Image.new("RGB", (width, canvas_h), color=(248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    y = 3
    for line, h in zip(wrapped_lines, line_heights):
        draw.text((0, y), line, fill=(28, 32, 40), font=font)
        y += h + 4
    return canvas


def _compose_sample_image(
    *,
    title_lines: list[str],
    clip_entries: list[tuple[str, Image.Image, bool]],
    panel_entries: list[tuple[str, Image.Image, bool]],
) -> Image.Image:
    clip_row = _make_panel_row(clip_entries, panel_width=_PANEL_WIDTH)
    panel_row = _make_panel_row(panel_entries, panel_width=_PANEL_WIDTH)
    inner_w = max(clip_row.width, panel_row.width)
    header = _make_text_block(title_lines, width=inner_w)

    total_h = _CANVAS_PAD * 2 + header.height + clip_row.height + panel_row.height + _ROW_GAP * 2
    total_w = inner_w + _CANVAS_PAD * 2
    canvas = Image.new("RGB", (total_w, total_h), color=(248, 248, 248))

    y = _CANVAS_PAD
    canvas.paste(header, (_CANVAS_PAD, y))
    y += header.height + _ROW_GAP
    canvas.paste(clip_row, (_CANVAS_PAD + (inner_w - clip_row.width) // 2, y))
    y += clip_row.height + _ROW_GAP
    canvas.paste(panel_row, (_CANVAS_PAD + (inner_w - panel_row.width) // 2, y))
    return canvas


def _make_contact_sheet(images: list[Image.Image], *, columns: int = 2) -> Image.Image:
    if len(images) == 0:
        raise ValueError("images cannot be empty")
    columns = max(1, int(columns))
    rows = int(math.ceil(float(len(images)) / float(columns)))
    col_widths = [0] * columns
    row_heights = [0] * rows
    for idx, image in enumerate(images):
        row = idx // columns
        col = idx % columns
        col_widths[col] = max(col_widths[col], image.width)
        row_heights[row] = max(row_heights[row], image.height)

    gap = 16
    outer_pad = 18
    width = outer_pad * 2 + sum(col_widths) + gap * max(0, columns - 1)
    height = outer_pad * 2 + sum(row_heights) + gap * max(0, rows - 1)
    canvas = Image.new("RGB", (width, height), color=(242, 242, 242))

    y = outer_pad
    for row in range(rows):
        x = outer_pad
        for col in range(columns):
            idx = row * columns + col
            if idx >= len(images):
                break
            image = images[idx]
            canvas.paste(image, (x, y))
            x += col_widths[col] + gap
        y += row_heights[row] + gap
    return canvas


def _make_clip_entries(clip_cthw: np.ndarray) -> tuple[list[tuple[str, Image.Image, bool]], np.ndarray]:
    clip_entries: list[tuple[str, Image.Image, bool]] = []
    num_frames = int(clip_cthw.shape[1])
    center_idx = max(0, num_frames // 2)
    center_rgb = None
    for frame_idx in range(num_frames):
        frame_rgb = _voxel_frame_to_activity_rgb(clip_cthw[:, frame_idx])
        if frame_idx == center_idx:
            center_rgb = frame_rgb
        clip_entries.append((f"Input t={frame_idx}", _to_pil(frame_rgb), False))
    if center_rgb is None:
        raise RuntimeError("Failed to choose center frame for visualization.")
    return clip_entries, center_rgb


def _render_semantic_sample(
    *,
    sample: dict[str, Any],
    sample_idx: int,
    file_path: Path,
    center_window: int,
    prediction: np.ndarray,
    ignore_index: int,
) -> tuple[Image.Image, dict[str, Any]]:
    clip = sample["input"].detach().cpu().numpy()
    target_tensor = sample.get("eval_target", sample["target"])
    target = target_tensor.detach().cpu().numpy().astype(np.int64, copy=False)
    clip_entries, center_rgb = _make_clip_entries(clip)
    diag_base_rgb = _resize_rgb_to_hw(center_rgb, target_hw=target.shape[-2:])

    metrics = _sample_semantic_metrics(target, prediction, ignore_index=ignore_index)
    target_rgb = _semantic_to_rgb(target, ignore_index=ignore_index)
    pred_rgb = _semantic_to_rgb(prediction, ignore_index=ignore_index)
    diag_rgb = _semantic_diagnostic_rgb(diag_base_rgb, target, prediction, ignore_index=ignore_index)

    title_lines = [
        f"sample={sample_idx} file={file_path.name} window={center_window}",
        (
            f"semantic valid={metrics['valid_ratio'] * 100.0:.1f}% "
            f"pixel_acc={metrics['pixel_acc'] * 100.0:.2f}% "
            f"miou={metrics['miou'] * 100.0:.2f}%"
        ),
    ]
    panel_entries = [
        ("Center activity", _to_pil(center_rgb), False),
        ("Target", _to_pil(target_rgb), True),
        ("Prediction", _to_pil(pred_rgb), True),
        ("Green=correct / Red=wrong", _to_pil(diag_rgb), False),
    ]
    image = _compose_sample_image(title_lines=title_lines, clip_entries=clip_entries, panel_entries=panel_entries)
    manifest_row = {
        "sample_index": int(sample_idx),
        "file_path": str(file_path),
        "center_window": int(center_window),
        "valid_ratio": float(metrics["valid_ratio"]),
        "pixel_acc": float(metrics["pixel_acc"]),
        "miou": float(metrics["miou"]),
    }
    return image, manifest_row


def _render_depth_sample(
    *,
    sample: dict[str, Any],
    sample_idx: int,
    file_path: Path,
    center_window: int,
    prediction: np.ndarray,
    depth_valid_min: float,
    depth_valid_max: float,
) -> tuple[Image.Image, dict[str, Any]]:
    clip = sample["input"].detach().cpu().numpy()
    target = sample["target"].detach().cpu().numpy().astype(np.float32, copy=False)
    clip_entries, center_rgb = _make_clip_entries(clip)

    valid_mask = np.isfinite(target) & (target > float(depth_valid_min)) & (target < float(depth_valid_max))
    metrics = _sample_depth_metrics(target, prediction, valid_mask=valid_mask)

    if np.any(valid_mask):
        joint_values = np.concatenate([target[valid_mask], prediction[valid_mask]], axis=0)
        lo = float(np.percentile(joint_values, 5.0))
        hi = float(np.percentile(joint_values, 95.0))
        if hi <= lo:
            lo = float(joint_values.min())
            hi = float(joint_values.max())
    else:
        lo, hi = 0.0, 1.0
    if hi <= lo:
        hi = lo + 1e-6

    target_rgb, target_stats = _colorize_scalar_map(target, valid_mask, vmin=lo, vmax=hi)
    pred_rgb, _ = _colorize_scalar_map(prediction, valid_mask, vmin=lo, vmax=hi)
    abs_error = np.abs(prediction - target)
    if np.any(valid_mask):
        error_hi = float(np.percentile(abs_error[valid_mask], 95.0))
    else:
        error_hi = 1.0
    if error_hi <= 0:
        error_hi = 1e-6
    error_rgb, _ = _colorize_scalar_map(abs_error, valid_mask, vmin=0.0, vmax=error_hi)

    title_lines = [
        f"sample={sample_idx} file={file_path.name} window={center_window}",
        (
            f"depth valid={metrics['valid_ratio'] * 100.0:.1f}% "
            f"mae={metrics['mae']:.6f} rmse={metrics['rmse']:.6f} "
            f"range=[{target_stats['vmin']:.4f}, {target_stats['vmax']:.4f}]"
        ),
    ]
    panel_entries = [
        ("Center activity", _to_pil(center_rgb), False),
        ("Target depth", _to_pil(target_rgb), False),
        ("Prediction depth", _to_pil(pred_rgb), False),
        ("Absolute error", _to_pil(error_rgb), False),
    ]
    image = _compose_sample_image(title_lines=title_lines, clip_entries=clip_entries, panel_entries=panel_entries)
    manifest_row = {
        "sample_index": int(sample_idx),
        "file_path": str(file_path),
        "center_window": int(center_window),
        "valid_ratio": float(metrics["valid_ratio"]),
        "mae": float(metrics["mae"]),
        "rmse": float(metrics["rmse"]),
        "depth_vmin": float(lo),
        "depth_vmax": float(hi),
        "error_vmax": float(error_hi),
    }
    return image, manifest_row


def main() -> None:
    args = _parse_args()
    cfg, run_dir, config_path = _load_config(args)
    checkpoint_path = _resolve_checkpoint_path(args, run_dir=run_dir)
    output_dir = _resolve_output_dir(args, run_dir=run_dir, cfg=cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg_meta = dict(cfg.get("meta", {}))
    cfg_model = dict(cfg.get("model", {}))
    cfg_task = dict(cfg.get("task", {}))

    if bool(cfg_meta.get("use_tf32", True)):
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass

    split_name = str(args.split)
    roots_key = f"{split_name}_roots"
    roots = cfg_task.get(roots_key, [])
    if len(roots) == 0:
        raise ValueError(f"Config field task.{roots_key} is empty.")

    target = str(cfg_task.get("target", "semantic")).lower()
    dataset = build_dense_task_dataset_from_config(
        cfg_task=cfg_task,
        roots=roots,
        split=split_name,
        return_eval_target=bool(cfg_task.get("eval_original_resolution", True)),
    )

    device = _resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dtype, mixed_precision = _to_dtype(str(cfg_meta.get("dtype", "float32")))
    use_autocast = bool(mixed_precision and device.type == "cuda")
    autocast_device_type = "cuda" if device.type == "cuda" else "cpu"

    model, ckpt = _load_model(
        cfg_model=cfg_model,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    picked_indices = _pick_sample_indices(
        len(dataset),
        num_samples=int(args.num_samples),
        sample_indices=args.sample_indices,
        sample_mode=str(args.sample_mode),
        start_index=int(args.start_index),
        sample_stride=int(args.sample_stride),
    )
    if len(picked_indices) == 0:
        raise RuntimeError("No samples were selected for visualization.")

    manifest_rows: list[dict[str, Any]] = []
    rendered_images: list[Image.Image] = []
    write_video = bool(args.write_video or args.video_path is not None)
    video_path = _resolve_video_path(args, output_dir=output_dir) if write_video else None
    video_writer: RgbMp4Writer | None = None
    video_frame_size: tuple[int, int] | None = None
    ignore_index = int(cfg_task.get("ignore_index", 255))
    depth_valid_min = float(cfg_task.get("depth_valid_min", 0.0))
    depth_valid_max = float(cfg_task.get("depth_valid_max", 1e9))
    eval_logits_resize_mode = str(cfg_task.get("eval_logits_resize_mode", "bilinear"))

    print(f"Loaded config from: {config_path}")
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Writing visualizations to: {output_dir}")
    if write_video:
        if float(args.video_fps) <= 0.0:
            raise ValueError(f"--video-fps must be positive, got {args.video_fps}")
        if int(args.video_width) < 0:
            raise ValueError(f"--video-width must be non-negative, got {args.video_width}")
        assert video_path is not None
        print(f"Writing MP4 to: {video_path}")
    if "epoch" in ckpt:
        print(f"Checkpoint epoch: {ckpt['epoch']}")

    try:
        with torch.inference_mode():
            for order, dataset_idx in enumerate(picked_indices):
                sample = dataset[dataset_idx]
                file_idx, center_window = dataset.samples[dataset_idx]
                file_path = dataset.files[file_idx].preprocessed_h5

                x = sample["input"].unsqueeze(0).to(device)
                with torch.autocast(
                    device_type=autocast_device_type,
                    dtype=dtype,
                    enabled=use_autocast,
                ):
                    pred = model(x)

                if target == "semantic":
                    eval_target = sample.get("eval_target", sample["target"]).unsqueeze(0).to(device)
                    pred_eval = _resize_logits_to_target(
                        pred,
                        eval_target,
                        mode=eval_logits_resize_mode,
                    )
                    pred_map = pred_eval.argmax(dim=1)[0].detach().cpu().numpy().astype(np.int64, copy=False)
                    image, row = _render_semantic_sample(
                        sample=sample,
                        sample_idx=dataset_idx,
                        file_path=file_path,
                        center_window=int(center_window),
                        prediction=pred_map,
                        ignore_index=ignore_index,
                    )
                else:
                    pred_map = pred[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)
                    image, row = _render_depth_sample(
                        sample=sample,
                        sample_idx=dataset_idx,
                        file_path=file_path,
                        center_window=int(center_window),
                        prediction=pred_map,
                        depth_valid_min=depth_valid_min,
                        depth_valid_max=depth_valid_max,
                    )

                output_name = f"sample_{order:03d}_idx{dataset_idx:06d}_{file_path.stem}_w{int(center_window):06d}.png"
                output_path = output_dir / output_name
                image.save(output_path)
                rendered_images.append(image)

                if write_video:
                    assert video_path is not None
                    frame_rgb = _pil_to_video_rgb(image, target_width=int(args.video_width))
                    if video_writer is None:
                        video_frame_size = (int(frame_rgb.shape[1]), int(frame_rgb.shape[0]))
                        video_writer = RgbMp4Writer(
                            video_path,
                            fps=float(args.video_fps),
                            width=video_frame_size[0],
                            height=video_frame_size[1],
                        )
                    elif video_frame_size is not None:
                        frame_rgb = _resize_video_rgb_to_size(frame_rgb, size=video_frame_size)
                    video_writer.write_rgb(frame_rgb)

                row["order"] = int(order)
                row["output_png"] = str(output_path)
                manifest_rows.append(row)
                print(f"[{order + 1}/{len(picked_indices)}] wrote {output_path.name}")
    finally:
        if video_writer is not None:
            video_writer.close()

    contact_sheet = _make_contact_sheet(rendered_images, columns=2 if len(rendered_images) > 1 else 1)
    contact_sheet_path = output_dir / "contact_sheet.png"
    contact_sheet.save(contact_sheet_path)

    manifest_path = output_dir / "manifest.csv"
    fieldnames = sorted({key for row in manifest_rows for key in row.keys()})
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Wrote {len(rendered_images)} sample image(s) to {output_dir}")
    print(f"Contact sheet: {contact_sheet_path}")
    print(f"Manifest: {manifest_path}")
    if write_video:
        assert video_path is not None
        print(f"MP4: {video_path}")


if __name__ == "__main__":
    main()
