from __future__ import annotations

from copy import deepcopy


# Suggested Stage-1 masks for event pretraining.
STAGE1_EVENT_MASKS = [
    {
        "num_blocks": 8,
        "spatial_scale": (0.15, 0.15),
        "temporal_scale": (1.0, 1.0),
        "aspect_ratio": (0.75, 1.5),
    },
    {
        "num_blocks": 2,
        "spatial_scale": (0.7, 0.7),
        "temporal_scale": (1.0, 1.0),
        "aspect_ratio": (0.75, 1.5),
    },
]


# Suggested image mask used in image branch pretraining.
STAGE1_IMAGE_MASKS = [
    {
        "num_blocks": 10,
        "spatial_scale": (0.15, 0.15),
        "temporal_scale": (1.0, 1.0),
        "aspect_ratio": (0.75, 1.5),
    }
]


def get_stage1_event_masks() -> list[dict]:
    return deepcopy(STAGE1_EVENT_MASKS)


def get_stage1_image_masks() -> list[dict]:
    return deepcopy(STAGE1_IMAGE_MASKS)

