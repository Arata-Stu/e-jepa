# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import sys

import torch
import torch.nn.functional as F
import src.models.predictor as vit_pred
import src.models.vision_transformer as video_vit
from src.models.wrappers import MultiSeqWrapper, PredictorMultiSeqWrapper
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.schedulers import (
    CosineWDSchedule,
    LinearDecaySchedule,
    WarmupCosineSchedule,
)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def _align_state_dict_keys(model_state, loaded_state):
    """Return a copy of loaded_state aligned to model_state key prefixes."""
    if len(model_state) == 0 or len(loaded_state) == 0:
        return dict(loaded_state)
    model_first = next(iter(model_state.keys()))
    loaded_first = next(iter(loaded_state.keys()))
    model_has_module = model_first.startswith("module.")
    loaded_has_module = loaded_first.startswith("module.")
    if model_has_module == loaded_has_module:
        return dict(loaded_state)

    aligned = {}
    if model_has_module and not loaded_has_module:
        for key, value in loaded_state.items():
            aligned[f"module.{key}"] = value
    else:
        for key, value in loaded_state.items():
            if key.startswith("module."):
                aligned[key[len("module.") :]] = value
            else:
                aligned[key] = value
    return aligned


def normalize_and_concat(tensor, embed_dim):
    """Split tensor into 4 chunks of size embed_dim along the last axis,
    apply LayerNorm to each chunk, then concatenate back."""
    chunks = [
        F.layer_norm(tensor[:, :, i * embed_dim : (i + 1) * embed_dim], (embed_dim,))
        for i in range(4)
    ]
    return torch.cat(chunks, dim=2)


def normalize_nested(nested, embed_dim):
    """Apply normalize_and_concat recursively over nested lists."""
    return [
        [[normalize_and_concat(z, embed_dim) for z in inner] for inner in outer]
        for outer in nested
    ]


def load_checkpoint(
    r_path,
    encoder,
    predictor,
    target_encoder,
    opt,
    scaler,
    is_anneal=False,
):
    logger.info(f"Loading {r_path}")
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))

    epoch = 0
    if not is_anneal:
        epoch = checkpoint["epoch"]

    pretrained_dict = _align_state_dict_keys(
        encoder.state_dict(), checkpoint["encoder"]
    )
    for k, v in encoder.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f'key "{k}" could not be found in loaded state dict')
        elif pretrained_dict[k].shape != v.shape:
            logger.info(
                f'key "{k}" is of different shape in model and loaded state dict'
            )
            pretrained_dict[k] = v
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    logger.info(f"loaded pretrained encoder from epoch {epoch} with msg: {msg}")

    pretrained_dict = _align_state_dict_keys(
        predictor.state_dict(), checkpoint["predictor"]
    )
    for k, v in predictor.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f'key "{k}" could not be found in loaded state dict')
        elif pretrained_dict[k].shape != v.shape:
            logger.info(
                f'key "{k}" is of different shape in model and loaded state dict'
            )
            pretrained_dict[k] = v
    msg = predictor.load_state_dict(pretrained_dict, strict=False)
    logger.info(f"loaded pretrained predictor from epoch {epoch} with msg: {msg}")

    if target_encoder is not None:
        pretrained_dict = _align_state_dict_keys(
            target_encoder.state_dict(), checkpoint["target_encoder"]
        )
        for k, v in target_encoder.state_dict().items():
            if k not in pretrained_dict:
                logger.info(f'key "{k}" could not be found in loaded state dict')
            elif pretrained_dict[k].shape != v.shape:
                logger.info(
                    f'key "{k}" is of different shape in model and loaded state dict'
                )
                pretrained_dict[k] = v
        msg = target_encoder.load_state_dict(pretrained_dict, strict=False)
        logger.info(
            f"loaded pretrained target encoder from epoch {epoch} with msg: {msg}"
        )

    try:
        opt.load_state_dict(checkpoint["opt"])
    except ValueError:
        print("[warn] Optimizer groups mismatch; reinitializing optimizer.")
    if scaler is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    logger.info(f"loaded optimizers from epoch {epoch}")
    logger.info(f"read-path: {r_path}")
    del checkpoint

    return (
        encoder,
        predictor,
        target_encoder,
        opt,
        scaler,
        epoch,
    )


def init_video_model(
    device,
    in_chans=3,
    patch_size=16,
    max_num_frames=16,
    tubelet_size=2,
    model_name="vit_base",
    crop_size=224,
    pred_depth=6,
    pred_num_heads=None,
    pred_embed_dim=384,
    uniform_power=False,
    use_mask_tokens=False,
    num_mask_tokens=2,
    zero_init_mask_tokens=True,
    use_sdpa=False,
    use_rope=False,
    use_silu=False,
    use_pred_silu=False,
    wide_silu=False,
    is_causal=False,
    pred_is_causal=False,
    use_activation_checkpointing=False,
    return_all_tokens=False,
    chop_last_n_tokens=0,
    init_type="default",
    img_temporal_dim_size=None,
    n_registers=0,
    n_registers_predictor=0,
    has_cls_first=False,
    interpolate_rope=False,
    modality_embedding=False,
):
    encoder = video_vit.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        in_chans=in_chans,
        num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        wide_silu=wide_silu,
        use_activation_checkpointing=use_activation_checkpointing,
        is_causal=is_causal,
        use_rope=use_rope,
        init_type=init_type,
        img_temporal_dim_size=img_temporal_dim_size,
        n_registers=n_registers,
        has_cls_first=has_cls_first,
        interpolate_rope=interpolate_rope,
        modality_embedding=modality_embedding,
    )
    encoder = MultiSeqWrapper(encoder)
    predictor = vit_pred.__dict__["vit_predictor"](
        img_size=crop_size,
        use_mask_tokens=use_mask_tokens,
        patch_size=patch_size,
        num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        embed_dim=encoder.backbone.embed_dim,
        predictor_embed_dim=pred_embed_dim,
        depth=pred_depth,
        num_heads=(
            encoder.backbone.num_heads if pred_num_heads is None else pred_num_heads
        ),
        uniform_power=uniform_power,
        num_mask_tokens=num_mask_tokens,
        zero_init_mask_tokens=zero_init_mask_tokens,
        use_rope=use_rope,
        use_sdpa=use_sdpa,
        is_causal=pred_is_causal,
        use_silu=use_pred_silu,
        wide_silu=wide_silu,
        use_activation_checkpointing=use_activation_checkpointing,
        return_all_tokens=return_all_tokens,
        chop_last_n_tokens=chop_last_n_tokens,
        n_registers=n_registers_predictor,
        has_cls_first=has_cls_first,
        interpolate_rope=interpolate_rope,
        modality_embedding=modality_embedding,
        img_temporal_dim_size=img_temporal_dim_size,
    )
    predictor = PredictorMultiSeqWrapper(predictor)

    encoder.to(device)
    predictor.to(device)
    logger.info(encoder)
    logger.info(predictor)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Encoder number of parameters: {count_parameters(encoder)}")
    logger.info(f"Predictor number of parameters: {count_parameters(predictor)}")

    return encoder, predictor


def init_opt(
    is_anneal,
    encoder,
    predictor,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    use_radamw=False,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    mixed_precision=False,
    ipe_scale=1.25,
    betas=(0.9, 0.999),
    eps=1e-8,
    zero_init_bias_wd=True,
):
    param_groups = [
        {
            "params": (
                p
                for n, p in encoder.named_parameters()
                if ("bias" not in n) and (len(p.shape) != 1)
            )
        },
        {
            "params": (
                p
                for n, p in predictor.named_parameters()
                if ("bias" not in n) and (len(p.shape) != 1)
            )
        },
        {
            "params": (
                p
                for n, p in encoder.named_parameters()
                if ("bias" in n) or (len(p.shape) == 1)
            ),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
        },
        {
            "params": (
                p
                for n, p in predictor.named_parameters()
                if ("bias" in n) or (len(p.shape) == 1)
            ),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
        },
    ]

    if use_radamw:
        raise NotImplementedError(
            "use_radamw=True is not supported in this repository yet. "
            "Set optimization.use_radamw=false."
        )
    else:
        logger.info("Using AdamW")
        optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps)

    if not is_anneal:
        scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=int(warmup * iterations_per_epoch),
            start_lr=start_lr,
            ref_lr=ref_lr,
            final_lr=final_lr,
            T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
        )
    else:
        scheduler = LinearDecaySchedule(
            optimizer,
            ref_lr=ref_lr,
            final_lr=final_lr,
            T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
        )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
    )

    scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
    return optimizer, scaler, scheduler, wd_scheduler
