from __future__ import annotations

from .event_dataset import make_eventdataset
from .imagenet1k import make_imagenet1k


def init_data(
    *,
    batch_size,
    transform=None,
    shared_transform=None,
    data="eventdataset",
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    training=True,  # Kept for compatibility.
    drop_last=True,
    subset_file=None,  # Kept for compatibility.
    clip_len=None,
    dataset_fpcs=None,
    frame_sample_rate=None,
    duration=None,  # Kept for compatibility.
    fps=None,  # Kept for compatibility.
    num_clips=1,
    random_clip_sampling=True,
    allow_clip_overlap=False,
    filter_short_videos=False,  # Kept for compatibility.
    filter_long_videos=int(1e9),  # Kept for compatibility.
    datasets_weights=None,
    persistent_workers=False,
    deterministic=True,  # Kept for compatibility.
    log_dir=None,  # Kept for compatibility.
    file_pattern="*.h5",
    recursive=True,
    require_voxels_key=True,
):
    if root_path is None:
        raise ValueError("root_path must be provided")

    name = data.lower()
    if name in {"eventdataset", "eventvoxel", "videodataset"}:
        frames_per_clip = clip_len if clip_len is not None else 8
        frame_step = frame_sample_rate if frame_sample_rate is not None else 1

        dataset, data_loader, dist_sampler = make_eventdataset(
            data_paths=root_path,
            batch_size=batch_size,
            frames_per_clip=frames_per_clip,
            dataset_fpcs=dataset_fpcs,
            frame_step=frame_step,
            num_clips=num_clips,
            random_clip_sampling=random_clip_sampling,
            allow_clip_overlap=allow_clip_overlap,
            transform=transform,
            shared_transform=shared_transform,
            rank=rank,
            world_size=world_size,
            datasets_weights=datasets_weights,
            collator=collator,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_mem=pin_mem,
            persistent_workers=persistent_workers,
            file_pattern=file_pattern,
            recursive=recursive,
            require_voxels_key=require_voxels_key,
        )
    elif name == "imagenet":
        dataset, data_loader, dist_sampler = make_imagenet1k(
            transform=transform,
            batch_size=batch_size,
            collator=collator,
            pin_mem=pin_mem,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            root_path=root_path,
            training=training,
            drop_last=drop_last,
            persistent_workers=persistent_workers,
            subset_file=subset_file,
        )
    else:
        raise ValueError(
            f"Unsupported dataset type: {data}. "
            "Use one of: eventdataset, eventvoxel, videodataset, imagenet."
        )
    return data_loader, dist_sampler
