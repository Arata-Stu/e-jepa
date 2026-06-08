# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from logging import getLogger
from multiprocessing import Value

import torch
import torch.nn.functional as F

_GLOBAL_SEED = 0
logger = getLogger()


def _iter_clip_tensors(value):
    if torch.is_tensor(value):
        if value.ndim == 5:
            yield value
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_clip_tensors(item)


def _estimate_batch_active_pixel_ratio(collated_batch, threshold=1e-6):
    sample_activity = _estimate_sample_active_pixel_ratio(
        collated_batch,
        threshold=threshold,
    )
    if sample_activity is None:
        return None

    return float(sample_activity.mean().item())


def _estimate_sample_active_pixel_ratio(collated_batch, threshold=1e-6):
    sample_activity = []
    for clip in _iter_clip_tensors(collated_batch):
        # clip: [B, C, T, H, W]. Count a pixel active if any channel/time is active.
        active_hw = clip.detach().abs().amax(dim=(1, 2)) > float(threshold)
        sample_activity.append(active_hw.float().mean(dim=(1, 2)))

    if len(sample_activity) == 0:
        return None

    return torch.stack(sample_activity, dim=0).mean(dim=0)


def _clip_to_patch_activity(
    clip,
    *,
    threshold,
    duration,
    height,
    width,
    temporal_patch_size,
    spatial_patch_size,
):
    if not torch.is_tensor(clip) or clip.ndim != 5:
        return None

    patch_h, patch_w = spatial_patch_size
    required_t = int(duration * temporal_patch_size)
    required_h = int(height * patch_h)
    required_w = int(width * patch_w)
    if (
        clip.shape[2] < required_t
        or clip.shape[3] < required_h
        or clip.shape[4] < required_w
    ):
        return None

    # clip: [B, C, T, H, W]. Collapse channels, then measure active ratio per tubelet patch.
    active = (
        clip.detach()
        .abs()[:, :, :required_t, :required_h, :required_w]
        .amax(dim=1)
        .gt(float(threshold))
        .float()
    )
    active = active.contiguous().view(
        active.shape[0],
        duration,
        temporal_patch_size,
        height,
        patch_h,
        width,
        patch_w,
    )
    return active.mean(dim=(2, 4, 6))


def _estimate_batch_patch_activity(
    collated_batch,
    *,
    threshold,
    duration,
    height,
    width,
    temporal_patch_size,
    spatial_patch_size,
):
    patch_activity = []
    for clip in _iter_clip_tensors(collated_batch):
        activity = _clip_to_patch_activity(
            clip,
            threshold=threshold,
            duration=duration,
            height=height,
            width=width,
            temporal_patch_size=temporal_patch_size,
            spatial_patch_size=spatial_patch_size,
        )
        if activity is not None:
            patch_activity.append(activity)

    if len(patch_activity) == 0:
        return None

    batch_size = min(activity.shape[0] for activity in patch_activity)
    if batch_size <= 0:
        return None

    return torch.stack([activity[:batch_size] for activity in patch_activity], dim=0).mean(dim=0)


def _clip_pair(value, min_value, max_value):
    v0 = min(max(float(value[0]), min_value), max_value)
    v1 = min(max(float(value[1]), min_value), max_value)
    if v1 < v0:
        v1 = v0
    return (v0, v1)


class MaskCollator(object):

    def __init__(
        self,
        cfgs_mask,
        dataset_fpcs,
        crop_size=(224, 224),
        patch_size=(16, 16),
        tubelet_size=2,
    ):
        super(MaskCollator, self).__init__()

        self.mask_generators = dict()
        for fpc in dataset_fpcs:
            self.mask_generators[fpc] = []
            for m in cfgs_mask:
                mask_generator = _MaskGenerator(
                    crop_size=crop_size,
                    num_frames=fpc,
                    spatial_patch_size=patch_size,
                    temporal_patch_size=tubelet_size,
                    spatial_pred_mask_scale=m.get("spatial_scale"),
                    temporal_pred_mask_scale=m.get("temporal_scale"),
                    aspect_ratio=m.get("aspect_ratio"),
                    npred=m.get("num_blocks"),
                    max_context_frames_ratio=m.get("max_temporal_keep", 1.0),
                    max_keep=m.get("max_keep", None),
                    full_complement=m.get("full_complement", False),
                    pred_full_complement=m.get("pred_full_complement", False),
                    inv_block=m.get("inv_block", False),
                    activity_adaptive=m.get("activity_adaptive", False),
                    activity_threshold=m.get("activity_threshold", 1e-6),
                    activity_low=m.get("activity_low", 0.005),
                    activity_high=m.get("activity_high", 0.03),
                    activity_low_scale=m.get("activity_low_scale", 0.6),
                    activity_high_scale=m.get("activity_high_scale", 1.15),
                    activity_min_spatial_scale=m.get(
                        "activity_min_spatial_scale", 0.02
                    ),
                    activity_max_spatial_scale=m.get(
                        "activity_max_spatial_scale", 0.5
                    ),
                    activity_adaptive_scope=m.get("activity_adaptive_scope", "batch"),
                    location_strategy=m.get("location_strategy", "random"),
                    activity_location_floor=m.get("activity_location_floor", 0.0),
                    activity_location_random_prob=m.get(
                        "activity_location_random_prob", 0.0
                    ),
                    activity_location_power=m.get("activity_location_power", 1.0),
                )
                self.mask_generators[fpc].append(mask_generator)

    def step(self):
        for fpc in self.mask_generators:
            for mask_generator in self.mask_generators[fpc]:
                mask_generator.step()

    def __call__(self, batch):

        # Batch: [buffer, label, clip_indices] for video
        # or [buffer, label] for images
        filtered_batches = {fpc: [] for fpc in self.mask_generators}
        for sample in batch:
            # Check if sample is from video dataset (has clip_indices) or image dataset
            if len(sample) >= 3 and isinstance(sample[-1], (list, tuple)):
                # Video sample: sample[-1] is clip_indices, sample[-1][-1] contains frame indices
                try:
                    fpc = len(sample[-1][-1])
                except (TypeError, IndexError):
                    # Fallback: assume single frame if structure is unexpected
                    fpc = 1
            else:
                # Image sample: single frame
                fpc = 1
            if fpc in filtered_batches:
                filtered_batches[fpc] += [sample]

        fpc_collations = []
        for fpc in filtered_batches:
            fpc_batch = filtered_batches[fpc]
            batch_size = len(fpc_batch)
            if batch_size == 0:
                continue
            collated_batch = torch.utils.data.default_collate(fpc_batch)
            collated_masks_pred, collated_masks_enc = [], []
            for i, mask_generator in enumerate(self.mask_generators[fpc]):
                batch_activity = None
                sample_activity = None
                if mask_generator.activity_adaptive:
                    if mask_generator.activity_adaptive_scope == "sample":
                        sample_activity = _estimate_sample_active_pixel_ratio(
                            collated_batch,
                            threshold=mask_generator.activity_threshold,
                        )
                        if sample_activity is not None:
                            batch_activity = float(sample_activity.mean().item())
                    if batch_activity is None:
                        batch_activity = _estimate_batch_active_pixel_ratio(
                            collated_batch,
                            threshold=mask_generator.activity_threshold,
                        )
                patch_activity = None
                if mask_generator.location_strategy == "activity_weighted":
                    patch_activity = _estimate_batch_patch_activity(
                        collated_batch,
                        threshold=mask_generator.activity_threshold,
                        duration=mask_generator.duration,
                        height=mask_generator.height,
                        width=mask_generator.width,
                        temporal_patch_size=mask_generator.temporal_patch_size,
                        spatial_patch_size=mask_generator.spatial_patch_size,
                    )
                masks_enc, masks_pred = mask_generator(
                    batch_size,
                    batch_activity=batch_activity,
                    sample_activity=sample_activity,
                    patch_activity=patch_activity,
                )
                collated_masks_enc.append(masks_enc)
                collated_masks_pred.append(masks_pred)
            fpc_collations += [
                (collated_batch, collated_masks_enc, collated_masks_pred)
            ]

        return fpc_collations


class _MaskGenerator(object):

    def __init__(
        self,
        crop_size=(224, 224),
        num_frames=16,
        spatial_patch_size=(16, 16),
        temporal_patch_size=2,
        spatial_pred_mask_scale=(0.2, 0.8),
        temporal_pred_mask_scale=(1.0, 1.0),
        aspect_ratio=(0.3, 3.0),
        npred=1,
        max_context_frames_ratio=1.0,
        max_keep=None,
        inv_block=False,
        full_complement=False,
        pred_full_complement=False,
        activity_adaptive=False,
        activity_threshold=1e-6,
        activity_low=0.005,
        activity_high=0.03,
        activity_low_scale=0.6,
        activity_high_scale=1.15,
        activity_min_spatial_scale=0.02,
        activity_max_spatial_scale=0.5,
        activity_adaptive_scope="batch",
        location_strategy="random",
        activity_location_floor=0.0,
        activity_location_random_prob=0.0,
        activity_location_power=1.0,
    ):
        super(_MaskGenerator, self).__init__()
        if not isinstance(crop_size, tuple):
            crop_size = (crop_size,) * 2
        if not isinstance(spatial_patch_size, tuple):
            spatial_patch_size = (spatial_patch_size,) * 2
        self.crop_size = crop_size
        self.height, self.width = [
            crop_size[i] // spatial_patch_size[i] for i in (0, 1)
        ]
        self.duration = num_frames // temporal_patch_size
        self.full_complement = full_complement
        self.pred_full_complement = pred_full_complement

        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size

        self.aspect_ratio = aspect_ratio
        self.spatial_pred_mask_scale = spatial_pred_mask_scale
        self.temporal_pred_mask_scale = temporal_pred_mask_scale
        self.npred = npred
        self.max_context_duration = max(
            1, int(self.duration * max_context_frames_ratio)
        )  # maximum number of time-steps (frames) spanned by context mask
        self.max_keep = max_keep  # maximum number of patches to keep in context
        self._itr_counter = Value("i", -1)  # collator is shared across worker processes
        self.inv_block = inv_block
        self.activity_adaptive = bool(activity_adaptive)
        self.activity_threshold = float(activity_threshold)
        self.activity_low = float(activity_low)
        self.activity_high = float(activity_high)
        self.activity_low_scale = float(activity_low_scale)
        self.activity_high_scale = float(activity_high_scale)
        self.activity_min_spatial_scale = float(activity_min_spatial_scale)
        self.activity_max_spatial_scale = float(activity_max_spatial_scale)
        self.activity_adaptive_scope = str(activity_adaptive_scope)
        valid_activity_adaptive_scopes = {"batch", "sample"}
        if self.activity_adaptive_scope not in valid_activity_adaptive_scopes:
            raise ValueError(
                f"Unsupported mask activity_adaptive_scope={self.activity_adaptive_scope}. "
                f"Expected one of {sorted(valid_activity_adaptive_scopes)}."
            )
        self.location_strategy = str(location_strategy)
        valid_location_strategies = {"random", "activity_weighted"}
        if self.location_strategy not in valid_location_strategies:
            raise ValueError(
                f"Unsupported mask location_strategy={self.location_strategy}. "
                f"Expected one of {sorted(valid_location_strategies)}."
            )
        self.activity_location_floor = float(activity_location_floor)
        self.activity_location_random_prob = float(activity_location_random_prob)
        self.activity_location_power = float(activity_location_power)

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(
        self, generator, temporal_scale, spatial_scale, aspect_ratio_scale
    ):
        # -- Sample temporal block mask scale
        _rand = torch.rand(1, generator=generator).item()
        min_t, max_t = temporal_scale
        temporal_mask_scale = min_t + _rand * (max_t - min_t)
        t = max(1, int(self.duration * temporal_mask_scale))

        # -- Sample spatial block mask scale
        _rand = torch.rand(1, generator=generator).item()
        min_s, max_s = spatial_scale
        spatial_mask_scale = min_s + _rand * (max_s - min_s)
        spatial_num_keep = max(
            1,
            int(self.height * self.width * spatial_mask_scale),
        )

        # -- Sample block aspect-ratio
        _rand = torch.rand(1, generator=generator).item()
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)

        # -- Compute block height and width (given scale and aspect-ratio)
        h = int(round(math.sqrt(spatial_num_keep * aspect_ratio)))
        w = int(round(math.sqrt(spatial_num_keep / aspect_ratio)))
        h = min(h, self.height)
        w = min(w, self.width)

        return (t, h, w)

    def _adapt_spatial_scale(self, batch_activity):
        if not self.activity_adaptive or batch_activity is None:
            return self.spatial_pred_mask_scale

        low = self.activity_low
        high = self.activity_high
        if high <= low:
            alpha = 1.0 if batch_activity >= high else 0.0
        else:
            alpha = (float(batch_activity) - low) / (high - low)
            alpha = min(max(alpha, 0.0), 1.0)

        scale = self.activity_low_scale + alpha * (
            self.activity_high_scale - self.activity_low_scale
        )
        spatial_scale = (
            float(self.spatial_pred_mask_scale[0]) * scale,
            float(self.spatial_pred_mask_scale[1]) * scale,
        )
        return _clip_pair(
            spatial_scale,
            min_value=self.activity_min_spatial_scale,
            max_value=self.activity_max_spatial_scale,
        )

    def _sample_random_block_origin(self, b_size):
        t, h, w = b_size
        start = int(torch.randint(0, self.duration - t + 1, (1,)).item())
        top = int(torch.randint(0, self.height - h + 1, (1,)).item())
        left = int(torch.randint(0, self.width - w + 1, (1,)).item())
        return start, top, left

    def _sample_activity_weighted_block_origin(self, b_size, patch_activity):
        if patch_activity is None:
            return self._sample_random_block_origin(b_size)

        if (
            float(self.activity_location_random_prob) > 0.0
            and torch.rand(1).item() < float(self.activity_location_random_prob)
        ):
            return self._sample_random_block_origin(b_size)

        t, h, w = b_size
        if tuple(patch_activity.shape) != (self.duration, self.height, self.width):
            return self._sample_random_block_origin(b_size)

        kernel = torch.ones(
            (1, 1, t, h, w),
            dtype=patch_activity.dtype,
            device=patch_activity.device,
        )
        scores = F.conv3d(
            patch_activity.float().unsqueeze(0).unsqueeze(0),
            kernel.float(),
        ).flatten()
        scores = scores.clamp_min(0.0)
        if self.activity_location_power != 1.0:
            scores = scores.pow(self.activity_location_power)

        if scores.numel() == 0:
            return self._sample_random_block_origin(b_size)

        max_score = scores.max()
        if float(max_score.item()) > 0.0 and self.activity_location_floor > 0:
            scores = scores + max_score * self.activity_location_floor

        total_score = scores.sum()
        if (
            not bool(torch.isfinite(total_score).item())
            or float(total_score.item()) <= 0.0
        ):
            return self._sample_random_block_origin(b_size)

        sampled = int(torch.multinomial(scores, num_samples=1).item())
        spatial_positions = (self.height - h + 1) * (self.width - w + 1)
        start = sampled // spatial_positions
        rem = sampled % spatial_positions
        top = rem // (self.width - w + 1)
        left = rem % (self.width - w + 1)
        return int(start), int(top), int(left)

    def _sample_block_mask(self, b_size, patch_activity=None):
        t, h, w = b_size
        if self.location_strategy == "activity_weighted":
            start, top, left = self._sample_activity_weighted_block_origin(
                b_size,
                patch_activity,
            )
        else:
            start, top, left = self._sample_random_block_origin(b_size)

        mask = torch.ones((self.duration, self.height, self.width), dtype=torch.int32)
        mask[start : start + t, top : top + h, left : left + w] = 0

        # Context mask will only span the first X frames
        # (X=self.max_context_frames)
        if self.max_context_duration < self.duration:
            mask[self.max_context_duration :, :, :] = 0

        # --
        return mask

    def __call__(
        self,
        batch_size,
        batch_activity=None,
        sample_activity=None,
        patch_activity=None,
    ):
        """
        Create encoder and predictor masks when collating imgs into a batch
        # 1. sample pred block size using seed
        # 2. sample several pred block locations for each image (w/o seed)
        # 3. return pred masks and complement (enc mask)
        """
        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        use_sample_adaptive_area = (
            self.activity_adaptive
            and self.activity_adaptive_scope == "sample"
            and sample_activity is not None
        )
        batch_p_size = None
        if not use_sample_adaptive_area:
            batch_spatial_scale = self._adapt_spatial_scale(batch_activity)
            batch_p_size = self._sample_block_size(
                generator=g,
                temporal_scale=self.temporal_pred_mask_scale,
                spatial_scale=batch_spatial_scale,
                aspect_ratio_scale=self.aspect_ratio,
            )

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_enc = min_keep_pred = self.duration * self.height * self.width
        for sample_idx in range(batch_size):
            sample_patch_activity = None
            if patch_activity is not None and sample_idx < patch_activity.shape[0]:
                sample_patch_activity = patch_activity[sample_idx]
            p_size = batch_p_size
            if use_sample_adaptive_area and sample_idx < len(sample_activity):
                sample_spatial_scale = self._adapt_spatial_scale(
                    float(torch.as_tensor(sample_activity[sample_idx]).item())
                )
                p_size = self._sample_block_size(
                    generator=g,
                    temporal_scale=self.temporal_pred_mask_scale,
                    spatial_scale=sample_spatial_scale,
                    aspect_ratio_scale=self.aspect_ratio,
                )
            if p_size is None:
                raise RuntimeError("Failed to sample mask block size")

            empty_context = True
            while empty_context:

                mask_e = torch.ones(
                    (self.duration, self.height, self.width), dtype=torch.int32
                )
                for _ in range(self.npred):
                    mask_e *= self._sample_block_mask(
                        p_size,
                        patch_activity=sample_patch_activity,
                    )
                mask_e = mask_e.flatten()

                mask_p = torch.argwhere(mask_e == 0).squeeze()
                mask_e = torch.nonzero(mask_e).squeeze()

                empty_context = len(mask_e) == 0
                if not empty_context:
                    min_keep_pred = min(min_keep_pred, len(mask_p))
                    min_keep_enc = min(min_keep_enc, len(mask_e))
                    collated_masks_pred.append(mask_p)
                    collated_masks_enc.append(mask_e)

        if self.max_keep is not None:
            min_keep_enc = min(min_keep_enc, self.max_keep)

        collated_masks_enc = [cm[:min_keep_enc] for cm in collated_masks_enc]
        collated_masks_pred = [cm[:min_keep_pred] for cm in collated_masks_pred]
        if self.full_complement:  # predictor mask is just complement of encoder mask
            collated_masks_pred = [
                torch.tensor(
                    sorted(
                        list(
                            set(range(int(self.duration * self.height * self.width)))
                            - set(cm.tolist())
                        )
                    ),
                    dtype=cm.dtype,
                )
                for cm in collated_masks_enc
            ]
        elif self.pred_full_complement:
            collated_masks_enc = [
                torch.tensor(
                    sorted(
                        list(
                            set(range(int(self.duration * self.height * self.width)))
                            - set(cm.tolist())
                        )
                    ),
                    dtype=cm.dtype,
                )
                for cm in collated_masks_pred
            ]

        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)

        if self.inv_block:
            return collated_masks_pred, collated_masks_enc  # predict context from block
        else:
            return collated_masks_enc, collated_masks_pred
