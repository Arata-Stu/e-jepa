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

from src.masks.multiseq_multiblock3d import MaskCollator
from scripts.train.visualize_masks import (
    _RESAMPLING,
    _annotate_panel,
    _build_config,
    _build_dataset,
    _draw_patch_grid,
    _ensure_list,
    _frame_to_temporal_index,
    _indices_to_mask_volume,
    _overlay_patch_mask,
    _pick_dataset_indices,
    _resolve_branch_settings,
    _seed_everything,
    _stack_images,
    _to_hw_tuple,
    _voxel_to_activity_rgb,
    _write_contact_sheet,
)


DEFAULT_MASK_GROUPS = [
    "stage1_event_random",
    "stage1_event_activity_adaptive",
    "stage1_event_strategic",
]

DEFAULT_STRATEGY_NAMES = [
    "Random",
    "Adaptive Area",
    "Strategic",
]

PRED_COLOR = (217, 48, 37)
CONTEXT_COLOR = (52, 168, 83)


def _strip_mask_overrides(overrides: list[str]) -> list[str]:
    stripped: list[str] = []
    for override in overrides:
        text = str(override)
        if text.startswith("mask=") or text.startswith("+mask=") or text.startswith("~mask"):
            continue
        stripped.append(text)
    return stripped


def _make_text_panel(lines: list[str], *, width: int = 1000) -> Image.Image:
    font = ImageFont.load_default()
    line_h = 14
    pad = 10
    height = pad * 2 + max(1, len(lines)) * line_h
    panel = Image.new("RGB", (int(width), int(height)), color=(250, 250, 250))
    draw = ImageDraw.Draw(panel)
    y = pad
    for line in lines:
        draw.text((pad, y), str(line), fill=(25, 30, 38), font=font)
        y += line_h
    return panel


def _resize_panel(image: Image.Image, *, frame_display_height: int) -> Image.Image:
    scale = max(1.0, float(frame_display_height) / float(max(1, image.height)))
    size = (
        max(1, int(round(image.width * scale))),
        max(1, int(round(image.height * scale))),
    )
    return image.resize(size, resample=_RESAMPLING.NEAREST)


def _make_panel_grid(
    *,
    title: str,
    panels: list[list[Image.Image]],
    row_labels: list[str],
    col_labels: list[str],
    frame_display_height: int,
) -> Image.Image:
    if len(panels) == 0 or len(panels[0]) == 0:
        raise ValueError("panels must be a non-empty 2D list")

    font = ImageFont.load_default()
    resized = [
        [_resize_panel(img, frame_display_height=frame_display_height) for img in row]
        for row in panels
    ]
    rows = len(resized)
    cols = max(len(row) for row in resized)
    if len(row_labels) != rows:
        raise ValueError("row_labels length must match panel rows")
    if len(col_labels) != cols:
        raise ValueError("col_labels length must match panel columns")

    cell_w = max(img.width for row in resized for img in row)
    cell_h = max(img.height for row in resized for img in row)
    row_label_w = max(70, max(len(label) for label in row_labels) * 7 + 12)
    col_label_h = 22
    title_h = 24
    gap = 6
    pad = 8

    width = pad * 2 + row_label_w + cols * cell_w + max(0, cols - 1) * gap
    height = (
        pad * 2
        + title_h
        + col_label_h
        + rows * cell_h
        + max(0, rows - 1) * gap
    )
    canvas = Image.new("RGB", (width, height), color=(250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, pad), title, fill=(25, 30, 38), font=font)

    x0 = pad + row_label_w
    y0 = pad + title_h
    for col, label in enumerate(col_labels):
        draw.text((x0 + col * (cell_w + gap), y0), label, fill=(25, 30, 38), font=font)

    y = y0 + col_label_h
    for row_idx, row in enumerate(resized):
        draw.text((pad, y + 4), row_labels[row_idx], fill=(25, 30, 38), font=font)
        x = x0
        for img in row:
            canvas.paste(img, (x, y))
            x += cell_w + gap
        y += cell_h + gap
    return canvas


def _select_frame_ids(num_frames: int, max_frames: int) -> list[int]:
    frame_ids = list(range(int(num_frames)))
    if max_frames > 0 and len(frame_ids) > max_frames:
        frame_ids = sorted(
            {
                int(v)
                for v in np.linspace(
                    0,
                    len(frame_ids) - 1,
                    num=int(max_frames),
                    dtype=int,
                ).tolist()
            }
        )
    return frame_ids


def _load_strategy_configs(
    *,
    common_overrides: list[str],
    mask_groups: list[str],
) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
    args_by_strategy: list[dict[str, Any]] = []
    mask_cfgs_by_strategy: list[list[dict[str, Any]]] = []
    for mask_group in mask_groups:
        args = _build_config([*common_overrides, f"mask={mask_group}"])
        cfg_mask = list(args.get("mask", []))
        if len(cfg_mask) == 0:
            raise ValueError(f"mask group {mask_group!r} resolved to an empty mask list")
        args_by_strategy.append(args)
        mask_cfgs_by_strategy.append(cfg_mask)
    return args_by_strategy, mask_cfgs_by_strategy


def _make_collator(
    *,
    cfg_mask: list[dict[str, Any]],
    dataset_fpcs: list[int],
    crop_size: tuple[int, int],
    patch_size: int,
    tubelet_size: int,
) -> MaskCollator:
    return MaskCollator(
        cfgs_mask=cfg_mask,
        dataset_fpcs=dataset_fpcs,
        crop_size=crop_size,
        patch_size=patch_size,
        tubelet_size=int(tubelet_size),
    )


def _collate_masks_for_strategy(
    *,
    loaded_sample,
    cfg_mask: list[dict[str, Any]],
    dataset_fpcs: list[int],
    crop_size: tuple[int, int],
    patch_size: int,
    tubelet_size: int,
    mask_seed: int,
) -> tuple[Any, list[torch.Tensor], list[torch.Tensor]]:
    _seed_everything(int(mask_seed))
    collator = _make_collator(
        cfg_mask=cfg_mask,
        dataset_fpcs=dataset_fpcs,
        crop_size=crop_size,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
    )
    collations = collator([loaded_sample])
    if len(collations) != 1:
        raise RuntimeError(
            f"Expected exactly one collated fpc bucket for a single sample, got {len(collations)}"
        )
    return collations[0]


def _mask_volume_from_batch(
    mask_batch: torch.Tensor,
    *,
    temporal_dim: int,
    grid_h: int,
    grid_w: int,
) -> np.ndarray:
    return _indices_to_mask_volume(
        torch.as_tensor(mask_batch[0]).reshape(-1),
        temporal_dim=temporal_dim,
        grid_h=grid_h,
        grid_w=grid_w,
    )


def _format_cfg_short(cfg: dict[str, Any]) -> str:
    keys = [
        "num_blocks",
        "spatial_scale",
        "temporal_scale",
        "aspect_ratio",
        "activity_adaptive",
        "location_strategy",
    ]
    parts = []
    for key in keys:
        if key in cfg:
            parts.append(f"{key}={cfg[key]}")
    return ", ".join(parts)


def _render_comparison_sheet(
    *,
    sample_path: str,
    sample_index: int,
    draw_index: int,
    clip_tensor: torch.Tensor,
    clip_indices: list[int],
    strategy_names: list[str],
    mask_groups: list[str],
    mask_cfgs_by_strategy: list[list[dict[str, Any]]],
    masks_enc_by_strategy: list[list[torch.Tensor]],
    masks_pred_by_strategy: list[list[torch.Tensor]],
    grid_h: int,
    grid_w: int,
    temporal_dim: int,
    expected_hw: tuple[int, int],
    frame_display_height: int,
    max_frames: int,
    mask_view: str,
) -> tuple[Image.Image, list[str]]:
    clip_np = np.asarray(clip_tensor.detach().cpu(), dtype=np.float32)
    if clip_np.ndim != 4:
        raise ValueError(f"clip must be [C,T,H,W], got shape={clip_np.shape}")
    channels, num_frames, height, width = [int(v) for v in clip_np.shape]
    frame_ids = _select_frame_ids(num_frames=num_frames, max_frames=max_frames)

    header_lines = [
        f"sample_index={sample_index} draw_index={draw_index}",
        f"path={sample_path}",
        f"clip_shape=[C={channels}, T={num_frames}, H={height}, W={width}]",
        f"clip_indices={clip_indices}",
        (
            f"mask_grid=[T={temporal_dim}, H={grid_h}, W={grid_w}] "
            f"expected_input_hw={expected_hw} actual_input_hw=({height}, {width})"
        ),
        f"strategies={list(zip(strategy_names, mask_groups))}",
    ]
    rows: list[Image.Image] = [
        _make_text_panel(header_lines, width=1000),
    ]

    activity_panels: list[Image.Image] = []
    for frame_idx in frame_ids:
        frame_rgb = _voxel_to_activity_rgb(clip_np[:, frame_idx])
        window_idx = clip_indices[frame_idx] if frame_idx < len(clip_indices) else -1
        label = f"t={frame_idx} win={window_idx}"
        activity_panels.append(_annotate_panel(Image.fromarray(frame_rgb, mode="RGB"), label))
    rows.append(
        _make_panel_grid(
            title=f"Activity | {Path(sample_path).name}",
            panels=[activity_panels],
            row_labels=["input"],
            col_labels=[f"t={frame_idx}" for frame_idx in frame_ids],
            frame_display_height=frame_display_height,
        )
    )

    num_patterns = max(len(cfgs) for cfgs in mask_cfgs_by_strategy)
    summary_lines = list(header_lines)
    total_tokens = int(temporal_dim) * int(grid_h) * int(grid_w)

    for pattern_idx in range(num_patterns):
        pred_volumes: list[np.ndarray | None] = []
        enc_volumes: list[np.ndarray | None] = []
        cfg_texts: list[str] = []
        for cfgs, masks_enc, masks_pred in zip(
            mask_cfgs_by_strategy,
            masks_enc_by_strategy,
            masks_pred_by_strategy,
        ):
            if pattern_idx >= len(cfgs):
                pred_volumes.append(None)
                enc_volumes.append(None)
                cfg_texts.append("<missing>")
                continue
            pred_volume = _mask_volume_from_batch(
                masks_pred[pattern_idx],
                temporal_dim=temporal_dim,
                grid_h=grid_h,
                grid_w=grid_w,
            )
            enc_volume = _mask_volume_from_batch(
                masks_enc[pattern_idx],
                temporal_dim=temporal_dim,
                grid_h=grid_h,
                grid_w=grid_w,
            )
            pred_volumes.append(pred_volume)
            enc_volumes.append(enc_volume)
            cfg_texts.append(_format_cfg_short(cfgs[pattern_idx]))

        for strategy_name, cfg_text, pred_volume, enc_volume in zip(
            strategy_names,
            cfg_texts,
            pred_volumes,
            enc_volumes,
        ):
            if pred_volume is None or enc_volume is None:
                summary_lines.append(
                    f"pattern={pattern_idx} strategy={strategy_name} missing=true"
                )
                continue
            summary_lines.append(
                (
                    f"pattern={pattern_idx} strategy={strategy_name} cfg={cfg_text} "
                    f"pred={int(pred_volume.sum())}/{total_tokens} ({pred_volume.mean():.2%}) "
                    f"context={int(enc_volume.sum())}/{total_tokens} ({enc_volume.mean():.2%})"
                )
            )

        if mask_view in {"predictor", "both"}:
            rows.append(
                _render_mask_grid(
                    title=f"Pattern {pattern_idx} Predictor Mask | red=predicted/hidden",
                    clip_np=clip_np,
                    clip_indices=clip_indices,
                    frame_ids=frame_ids,
                    temporal_dim=temporal_dim,
                    grid_h=grid_h,
                    grid_w=grid_w,
                    strategy_names=strategy_names,
                    volumes=pred_volumes,
                    color=PRED_COLOR,
                    frame_display_height=frame_display_height,
                )
            )
        if mask_view in {"context", "both"}:
            rows.append(
                _render_mask_grid(
                    title=f"Pattern {pattern_idx} Context Mask | green=encoder visible",
                    clip_np=clip_np,
                    clip_indices=clip_indices,
                    frame_ids=frame_ids,
                    temporal_dim=temporal_dim,
                    grid_h=grid_h,
                    grid_w=grid_w,
                    strategy_names=strategy_names,
                    volumes=enc_volumes,
                    color=CONTEXT_COLOR,
                    frame_display_height=frame_display_height,
                )
            )
        summary_lines.append("")

    return _stack_images(rows), summary_lines


def _render_mask_grid(
    *,
    title: str,
    clip_np: np.ndarray,
    clip_indices: list[int],
    frame_ids: list[int],
    temporal_dim: int,
    grid_h: int,
    grid_w: int,
    strategy_names: list[str],
    volumes: list[np.ndarray | None],
    color: tuple[int, int, int],
    frame_display_height: int,
) -> Image.Image:
    _, num_frames, _, _ = clip_np.shape
    panels: list[list[Image.Image]] = []
    row_labels: list[str] = []
    for frame_idx in frame_ids:
        patch_t = _frame_to_temporal_index(
            frame_idx,
            num_frames=num_frames,
            temporal_dim=temporal_dim,
        )
        row: list[Image.Image] = []
        for strategy_name, volume in zip(strategy_names, volumes):
            base_rgb = _voxel_to_activity_rgb(clip_np[:, frame_idx])
            if volume is None:
                panel = Image.new("RGB", (base_rgb.shape[1], base_rgb.shape[0]), color=(240, 240, 240))
                row.append(_annotate_panel(panel, f"{strategy_name}: missing"))
                continue
            overlay = _overlay_patch_mask(base_rgb, volume[patch_t], color=color)
            image = _draw_patch_grid(
                Image.fromarray(overlay, mode="RGB"),
                grid_h=grid_h,
                grid_w=grid_w,
            )
            window_idx = clip_indices[frame_idx] if frame_idx < len(clip_indices) else -1
            label = f"{strategy_name} | pt={patch_t} win={window_idx}"
            row.append(_annotate_panel(image, label))
        panels.append(row)
        row_labels.append(f"t={frame_idx}")

    return _make_panel_grid(
        title=title,
        panels=panels,
        row_labels=row_labels,
        col_labels=strategy_names,
        frame_display_height=frame_display_height,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Compare Random / Adaptive Area / Strategic JEPA masks on the same event sample."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/mask_ablation_debug"),
        help="Directory where PNGs and summary.txt are written.",
    )
    parser.add_argument(
        "--mask-groups",
        nargs="+",
        default=DEFAULT_MASK_GROUPS,
        help="Hydra mask config groups to compare.",
    )
    parser.add_argument(
        "--strategy-names",
        nargs="+",
        default=DEFAULT_STRATEGY_NAMES,
        help="Display names aligned with --mask-groups.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="How many dataset items to visualize when --sample-indices is not set.",
    )
    parser.add_argument(
        "--sample-indices",
        nargs="+",
        type=int,
        default=None,
        help="Explicit dataset indices to visualize.",
    )
    parser.add_argument(
        "--sampling",
        choices=["uniform", "random", "first"],
        default="first",
        help="How to choose dataset indices when --sample-indices is omitted.",
    )
    parser.add_argument(
        "--clip-id",
        type=int,
        default=0,
        help="Which clip to render when data.num_clips > 1.",
    )
    parser.add_argument(
        "--num-draws",
        type=int,
        default=1,
        help="How many times to resample the same dataset index.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=4,
        help="Maximum number of frames to render per clip.",
    )
    parser.add_argument(
        "--frame-height",
        type=int,
        default=150,
        help="Display height for each rendered frame tile.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for dataset sampling and augmentation.",
    )
    parser.add_argument(
        "--mask-seed",
        type=int,
        default=None,
        help="Seed for mask generation. Defaults to --seed.",
    )
    parser.add_argument(
        "--mask-view",
        choices=["predictor", "context", "both"],
        default="both",
        help="Which mask side to render.",
    )
    parser.add_argument(
        "--branch",
        choices=["video", "image"],
        default="video",
        help="Use the main training branch or optional image branch config.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra overrides applied on top of scripts/train/conf/config.yaml. mask=... overrides are ignored.",
    )
    return parser.parse_args()


def main() -> None:
    cli_args = _parse_args()
    mask_groups = [str(v) for v in cli_args.mask_groups]
    strategy_names = [str(v) for v in cli_args.strategy_names]
    if len(mask_groups) != len(strategy_names):
        raise ValueError("--mask-groups and --strategy-names must have the same length")

    common_overrides = _strip_mask_overrides([str(v) for v in cli_args.overrides])
    args_by_strategy, mask_cfgs_by_strategy = _load_strategy_configs(
        common_overrides=common_overrides,
        mask_groups=mask_groups,
    )

    # Data/model geometry must be shared across compared masks.
    base_args = args_by_strategy[0]
    cfg_data, _, crop_size, tubelet_size = _resolve_branch_settings(
        base_args,
        branch=str(cli_args.branch),
    )
    cfg_data_aug = dict(base_args.get("data_aug", {}))
    dataset_fpcs = [int(v) for v in _ensure_list(cfg_data.get("dataset_fpcs", [8]))]
    patch_size = int(cfg_data.get("patch_size", 16))
    if crop_size[0] % patch_size != 0 or crop_size[1] % patch_size != 0:
        raise ValueError(
            f"crop_size={crop_size} must be divisible by patch_size={patch_size}"
        )
    grid_h = crop_size[0] // patch_size
    grid_w = crop_size[1] // patch_size

    output_dir = cli_args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_payload = {
        "common_overrides": common_overrides,
        "mask_groups": mask_groups,
        "strategy_names": strategy_names,
        "base_config": base_args,
    }
    (output_dir / "resolved_config.yaml").write_text(
        OmegaConf.to_yaml(OmegaConf.create(resolved_payload), resolve=True),
        encoding="utf-8",
    )

    _seed_everything(int(cli_args.seed))
    dataset = _build_dataset(
        cfg_data=cfg_data,
        cfg_data_aug=cfg_data_aug,
        crop_size=crop_size,
    )
    sample_indices = _pick_dataset_indices(
        len(dataset),
        num_samples=int(cli_args.num_samples),
        sampling=str(cli_args.sampling),
        explicit_indices=cli_args.sample_indices,
    )

    summary_lines = [
        f"seed={cli_args.seed}",
        f"mask_seed={cli_args.mask_seed if cli_args.mask_seed is not None else cli_args.seed}",
        f"branch={cli_args.branch}",
        f"dataset_len={len(dataset)}",
        f"sample_indices={sample_indices}",
        f"crop_size={crop_size}",
        f"patch_size={patch_size}",
        f"tubelet_size={tubelet_size}",
        f"dataset_fpcs={dataset_fpcs}",
        f"mask_groups={mask_groups}",
        "",
    ]
    written_images: list[Path] = []

    mask_seed = int(cli_args.seed if cli_args.mask_seed is None else cli_args.mask_seed)
    for sample_index in sample_indices:
        sample_path = dataset.samples[int(sample_index)]
        for draw_index in range(int(cli_args.num_draws)):
            # Reseed before loading so each draw is reproducible while still changing across draws.
            _seed_everything(int(cli_args.seed) + int(draw_index))
            loaded = dataset[int(sample_index)]

            udata_ref = None
            masks_enc_by_strategy: list[list[torch.Tensor]] = []
            masks_pred_by_strategy: list[list[torch.Tensor]] = []
            for strategy_offset, cfg_mask in enumerate(mask_cfgs_by_strategy):
                udata, masks_enc, masks_pred = _collate_masks_for_strategy(
                    loaded_sample=loaded,
                    cfg_mask=cfg_mask,
                    dataset_fpcs=dataset_fpcs,
                    crop_size=crop_size,
                    patch_size=patch_size,
                    tubelet_size=int(tubelet_size),
                    mask_seed=mask_seed + int(draw_index),
                )
                if strategy_offset == 0:
                    udata_ref = udata
                masks_enc_by_strategy.append(masks_enc)
                masks_pred_by_strategy.append(masks_pred)

            if udata_ref is None:
                raise RuntimeError("No strategy collations were generated")
            num_clips = len(udata_ref[0])
            if cli_args.clip_id < 0 or cli_args.clip_id >= num_clips:
                raise ValueError(
                    f"clip_id={cli_args.clip_id} is out of range for num_clips={num_clips}"
                )

            clip_tensor = torch.as_tensor(udata_ref[0][cli_args.clip_id][0]).cpu()
            clip_indices = (
                torch.as_tensor(udata_ref[2][cli_args.clip_id][0])
                .reshape(-1)
                .cpu()
                .tolist()
            )
            temporal_dim = max(1, int(clip_tensor.shape[1]) // int(tubelet_size))

            sheet, sample_summary = _render_comparison_sheet(
                sample_path=sample_path,
                sample_index=int(sample_index),
                draw_index=int(draw_index),
                clip_tensor=clip_tensor,
                clip_indices=clip_indices,
                strategy_names=strategy_names,
                mask_groups=mask_groups,
                mask_cfgs_by_strategy=mask_cfgs_by_strategy,
                masks_enc_by_strategy=masks_enc_by_strategy,
                masks_pred_by_strategy=masks_pred_by_strategy,
                grid_h=grid_h,
                grid_w=grid_w,
                temporal_dim=temporal_dim,
                expected_hw=crop_size,
                frame_display_height=int(cli_args.frame_height),
                max_frames=int(cli_args.max_frames),
                mask_view=str(cli_args.mask_view),
            )

            file_stem = Path(sample_path).stem.replace(" ", "_")
            out_path = output_dir / (
                f"mask_ablation_sample_{int(sample_index):06d}_draw_{draw_index:02d}_{file_stem}.png"
            )
            sheet.save(out_path)
            written_images.append(out_path)
            summary_lines.extend(sample_summary)
            summary_lines.append(f"written={out_path}")
            summary_lines.append("")

    summary_path = output_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    if len(written_images) > 1:
        _write_contact_sheet(
            written_images,
            output_dir / "contact_sheet.png",
            columns=min(2, len(written_images)),
        )

    print(f"Wrote {len(written_images)} mask ablation comparison image(s) to {output_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
