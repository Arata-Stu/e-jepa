from __future__ import annotations

from .event_dataset import make_eventdataset
from .imagenet1k import make_imagenet1k
from .m3ed_raw import make_m3ed_raw_eventdataset


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
    fps=None,
    num_clips=1,
    random_clip_sampling=True,
    allow_clip_overlap=False,
    filter_short_videos=False,  # Kept for compatibility.
    filter_long_videos=int(1e9),  # Kept for compatibility.
    datasets_weights=None,
    persistent_workers=False,
    prefetch_factor=None,
    max_open_h5_files=32,
    deterministic=True,  # Kept for compatibility.
    log_dir=None,  # Kept for compatibility.
    file_pattern="*.h5",
    recursive=True,
    require_voxels_key=True,
    activity_filter_enabled=False,
    activity_filter_min_clip_mean_active_pixel_ratio=None,
    activity_filter_min_clip_mean_activity_score=None,
    activity_filter_min_clip_active_window_ratio=None,
    activity_filter_active_window_threshold=None,
    representation="voxel_grid",
    input_height=720,
    input_width=1280,
    output_height=720,
    output_width=1280,
    downsample_factor=2,
    t_bins=10,
    split_polarity=True,
    normalize=True,
    use_trilinear=False,
    output_dtype="float16",
    event_image_percentile=99.0,
    window_mode="semantics_middle",
    accum_time_us=50000,
    stride_time_us=None,
    start_time_us=None,
    semantics_ts_source="auto",
    semantics_ts_divisor=1,
    depth_ts_source="auto",
    depth_ts_divisor=1,
    filter_known_semantic_sequences=False,
    virtual_chunk_duration_s=20.0,
    min_windows_per_chunk=1,
    activity_filter_max_trials=8,
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
            fps=fps,
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
            prefetch_factor=prefetch_factor,
            max_open_h5_files=max_open_h5_files,
            file_pattern=file_pattern,
            recursive=recursive,
            require_voxels_key=require_voxels_key,
            activity_filter_enabled=activity_filter_enabled,
            activity_filter_min_clip_mean_active_pixel_ratio=activity_filter_min_clip_mean_active_pixel_ratio,
            activity_filter_min_clip_mean_activity_score=activity_filter_min_clip_mean_activity_score,
            activity_filter_min_clip_active_window_ratio=activity_filter_min_clip_active_window_ratio,
            activity_filter_active_window_threshold=activity_filter_active_window_threshold,
        )
    elif name in {"m3ed_raw", "m3edraw", "m3ed_raw_event"}:
        frames_per_clip = clip_len if clip_len is not None else 16
        frame_step = frame_sample_rate if frame_sample_rate is not None else 1
        dataset, data_loader, dist_sampler = make_m3ed_raw_eventdataset(
            data_paths=root_path,
            batch_size=batch_size,
            frames_per_clip=frames_per_clip,
            dataset_fpcs=dataset_fpcs,
            frame_step=frame_step,
            fps=fps,
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
            prefetch_factor=prefetch_factor,
            max_open_h5_files=max_open_h5_files,
            file_pattern=file_pattern,
            recursive=recursive,
            representation=representation,
            input_height=input_height,
            input_width=input_width,
            output_height=output_height,
            output_width=output_width,
            downsample_factor=downsample_factor,
            t_bins=t_bins,
            split_polarity=split_polarity,
            normalize=normalize,
            use_trilinear=use_trilinear,
            output_dtype=output_dtype,
            event_image_percentile=event_image_percentile,
            window_mode=window_mode,
            accum_time_us=accum_time_us,
            stride_time_us=stride_time_us,
            start_time_us=start_time_us,
            semantics_ts_source=semantics_ts_source,
            semantics_ts_divisor=semantics_ts_divisor,
            depth_ts_source=depth_ts_source,
            depth_ts_divisor=depth_ts_divisor,
            filter_known_semantic_sequences=filter_known_semantic_sequences,
            virtual_chunk_duration_s=virtual_chunk_duration_s,
            min_windows_per_chunk=min_windows_per_chunk,
            activity_filter_enabled=activity_filter_enabled,
            activity_filter_min_clip_mean_active_pixel_ratio=activity_filter_min_clip_mean_active_pixel_ratio,
            activity_filter_min_clip_mean_activity_score=activity_filter_min_clip_mean_activity_score,
            activity_filter_min_clip_active_window_ratio=activity_filter_min_clip_active_window_ratio,
            activity_filter_active_window_threshold=activity_filter_active_window_threshold,
            activity_filter_max_trials=activity_filter_max_trials,
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
            "Use one of: eventdataset, eventvoxel, videodataset, "
            "m3ed_raw, imagenet."
        )
    return data_loader, dist_sampler
