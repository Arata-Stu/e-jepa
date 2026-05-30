from __future__ import annotations

import torch
import torch.nn.functional as F


def _normalize01(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().float()
    finite = torch.isfinite(x)
    if not bool(finite.any()):
        return torch.zeros_like(x)
    x = torch.where(finite, x, torch.zeros_like(x))
    x_min = x[finite].min()
    x_max = x[finite].max()
    return (x - x_min) / (x_max - x_min).clamp_min(1.0e-6)


def _temporal_montage(frames: torch.Tensor, max_slices: int = 8) -> torch.Tensor:
    # frames: [T,H,W] -> [3, T*H+(T-1), W]
    frames = frames.detach().float()
    if frames.ndim == 2:
        frames = frames.unsqueeze(0)
    frames = frames[: max(1, int(max_slices))]
    frames = _normalize01(frames)
    if frames.shape[0] <= 1:
        image = frames[0]
    else:
        gap = torch.zeros((1, frames.shape[2]), device=frames.device)
        rows = []
        for i, frame in enumerate(frames):
            if i > 0:
                rows.append(gap)
            rows.append(frame)
        image = torch.cat(rows, dim=0)
    return image.unsqueeze(0).repeat(3, 1, 1).cpu()


def _patch_activity(
    clip: torch.Tensor,
    *,
    patch_size: int,
    tubelet_size: int,
    max_slices: int,
) -> torch.Tensor:
    # clip: [C,T,H,W]. Activity is aligned with the ViT tubelet grid.
    _, t, h, w = clip.shape
    u = int(tubelet_size)
    p = int(patch_size)
    tp = max(1, t // u)
    usable_t = tp * u
    activity = clip[:, :usable_t].detach().float().abs().mean(dim=0)
    activity = activity.reshape(tp, u, h, w).mean(dim=1)
    activity = F.avg_pool2d(
        activity.unsqueeze(1),
        kernel_size=p,
        stride=p,
    ).squeeze(1)
    return _temporal_montage(activity, max_slices=max_slices)


def _frame_activity(clip: torch.Tensor, *, max_slices: int) -> torch.Tensor:
    # clip: [C,T,H,W]
    activity = clip.detach().float().abs().mean(dim=0)
    return _temporal_montage(activity, max_slices=max_slices)


def _gather_tokens(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    index = mask.long().unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
    return torch.gather(tokens, dim=1, index=index)


def _token_values_to_grid(
    token_indices: torch.Tensor,
    values: torch.Tensor,
    *,
    t_patches: int,
    h_patches: int,
    w_patches: int,
) -> torch.Tensor:
    grid = torch.zeros(
        (t_patches * h_patches * w_patches,),
        device=values.device,
        dtype=values.dtype,
    )
    indices = token_indices.long().clamp(0, grid.numel() - 1)
    grid[indices] = values
    return grid.reshape(t_patches, h_patches, w_patches)


def _latent_error_grid(
    pred: torch.Tensor,
    target_tokens: torch.Tensor,
    mask: torch.Tensor,
    *,
    t_patches: int,
    h_patches: int,
    w_patches: int,
    loss_exp: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred = pred.detach().float()
    target_tokens = target_tokens.detach().float()
    mask = mask.detach()
    per_token = torch.abs(pred[0] - target_tokens[0]).pow(float(loss_exp)).mean(dim=-1)
    if float(loss_exp) != 0.0:
        per_token = per_token / float(loss_exp)
    loss_grid = _token_values_to_grid(
        mask[0],
        per_token,
        t_patches=t_patches,
        h_patches=h_patches,
        w_patches=w_patches,
    )
    mask_grid = _token_values_to_grid(
        mask[0],
        torch.ones_like(per_token),
        t_patches=t_patches,
        h_patches=h_patches,
        w_patches=w_patches,
    )
    return loss_grid, mask_grid


def make_jepa_debug_images(
    *,
    clips: list[torch.Tensor],
    z_pred,
    z_context,
    target_tokens,
    masks_pred,
    masks_enc,
    patch_size: int,
    tubelet_size: int,
    loss_exp: float,
    max_slices: int = 8,
) -> dict[str, torch.Tensor]:
    images: dict[str, torch.Tensor] = {}
    if len(clips) == 0 or clips[0].shape[0] == 0:
        return images

    clip = clips[0][0]
    _, t, h, w = clip.shape
    t_patches = t // int(tubelet_size)
    h_patches = h // int(patch_size)
    w_patches = w // int(patch_size)
    images["debug/input_activity_patch"] = _patch_activity(
        clip,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        max_slices=max_slices,
    )

    if len(z_pred) > 0 and len(z_pred[0]) > 0:
        pred = z_pred[0][0]
        pred_mask = masks_pred[0][0]
        pred_target = _gather_tokens(target_tokens[0], pred_mask)
        pred_loss_grid, pred_mask_grid = _latent_error_grid(
            pred,
            pred_target,
            pred_mask,
            t_patches=t_patches,
            h_patches=h_patches,
            w_patches=w_patches,
            loss_exp=loss_exp,
        )
        images["debug/jepa_pred_latent_loss"] = _temporal_montage(
            pred_loss_grid,
            max_slices=max_slices,
        )
        images["debug/jepa_pred_mask"] = _temporal_montage(
            pred_mask_grid,
            max_slices=max_slices,
        )

    if (
        z_context is not None
        and len(z_context) > 0
        and len(z_context[0]) > 0
        and z_context[0][0] is not None
    ):
        context = z_context[0][0]
        context_mask = masks_enc[0][0]
        context_target = _gather_tokens(target_tokens[0], context_mask)
        context_loss_grid, context_mask_grid = _latent_error_grid(
            context,
            context_target,
            context_mask,
            t_patches=t_patches,
            h_patches=h_patches,
            w_patches=w_patches,
            loss_exp=loss_exp,
        )
        images["debug/jepa_context_latent_loss"] = _temporal_montage(
            context_loss_grid,
            max_slices=max_slices,
        )
        images["debug/jepa_context_mask"] = _temporal_montage(
            context_mask_grid,
            max_slices=max_slices,
        )

    return images


def patchify_video(
    x: torch.Tensor,
    *,
    patch_size: int,
    tubelet_size: int,
) -> torch.Tensor:
    b, c, t, h, w = x.shape
    p = int(patch_size)
    u = int(tubelet_size)
    tp = t // u
    hp = h // p
    wp = w // p
    x = x.reshape(b, c, tp, u, hp, p, wp, p)
    x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
    return x.reshape(b, tp * hp * wp, u * p * p * c)


def unpatchify_video(
    patches: torch.Tensor,
    *,
    channels: int,
    frames: int,
    height: int,
    width: int,
    patch_size: int,
    tubelet_size: int,
) -> torch.Tensor:
    b = patches.shape[0]
    c = int(channels)
    p = int(patch_size)
    u = int(tubelet_size)
    tp = frames // u
    hp = height // p
    wp = width // p
    x = patches.reshape(b, tp, hp, wp, u, p, p, c)
    x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
    return x.reshape(b, c, frames, height, width)


def make_mae_debug_images(
    *,
    clips: torch.Tensor,
    pred: torch.Tensor,
    mask: torch.Tensor,
    patch_size: int,
    tubelet_size: int,
    loss_type: str,
    norm_pix_loss: bool,
    max_slices: int = 8,
) -> dict[str, torch.Tensor]:
    images: dict[str, torch.Tensor] = {}
    if clips.shape[0] == 0:
        return images

    b, c, t, h, w = clips.shape
    target = patchify_video(
        clips.detach().float(),
        patch_size=patch_size,
        tubelet_size=tubelet_size,
    )
    loss_target = target
    if norm_pix_loss:
        mean = loss_target.mean(dim=-1, keepdim=True)
        var = loss_target.var(dim=-1, keepdim=True, unbiased=False)
        loss_target = (loss_target - mean) / torch.sqrt(var + 1.0e-6)

    pred_float = pred.detach().float()
    if str(loss_type).lower() == "l1":
        patch_loss = torch.abs(pred_float - loss_target).mean(dim=-1)
    else:
        patch_loss = (pred_float - loss_target).pow(2).mean(dim=-1)

    tp = t // int(tubelet_size)
    hp = h // int(patch_size)
    wp = w // int(patch_size)

    recon = unpatchify_video(
        pred_float,
        channels=c,
        frames=t,
        height=h,
        width=w,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
    )
    error = torch.abs(recon - clips.detach().float())

    images["debug/mae_input_activity"] = _frame_activity(
        clips[0],
        max_slices=max_slices,
    )
    images["debug/mae_reconstruction_activity"] = _frame_activity(
        recon[0],
        max_slices=max_slices,
    )
    images["debug/mae_abs_error_activity"] = _frame_activity(
        error[0],
        max_slices=max_slices,
    )
    images["debug/mae_patch_loss"] = _temporal_montage(
        patch_loss[0].reshape(tp, hp, wp),
        max_slices=max_slices,
    )
    images["debug/mae_mask"] = _temporal_montage(
        mask.detach().float()[0].reshape(tp, hp, wp),
        max_slices=max_slices,
    )
    return images
