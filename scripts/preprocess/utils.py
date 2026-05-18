from __future__ import annotations

from pathlib import Path
import re
import numpy as np


def _compression_opts(
    compression_level: int = 1,
    shuffle: int = 2,
    compressor_type: int = 5,
) -> tuple[int, int, int, int, int, int, int]:
    level = int(compression_level)
    if level < 0 or level > 9:
        raise ValueError(f"compression_level must be in [0,9], got {compression_level}")
    if int(shuffle) not in (0, 1, 2):
        raise ValueError(f"shuffle must be one of {{0,1,2}}, got {shuffle}")
    return 0, 0, 0, 0, level, int(shuffle), int(compressor_type)


def get_h5_compression_flags(
    compression_level: int = 1,
    shuffle: int = 2,
    compressor_type: int = 5,
) -> dict:
    level = int(compression_level)
    if level < 0 or level > 9:
        raise ValueError(f"compression_level must be in [0,9], got {compression_level}")
    try:
        import hdf5plugin  # noqa: F401

        _ = hdf5plugin
        return dict(
            compression=32001,
            compression_opts=_compression_opts(
                compression_level=level,
                shuffle=shuffle,
                compressor_type=compressor_type,
            ),
        )
    except ImportError:
        return dict(
            compression="gzip",
            compression_opts=level,
        )


def tmp_output_path(output_path: Path, tmp_suffix: str) -> Path:
    return output_path.with_name(f"{output_path.name}{tmp_suffix}")


def tmp_media_output_path(output_path: Path, tmp_suffix: str) -> Path:
    return output_path.with_name(f"{output_path.stem}{tmp_suffix}{output_path.suffix}")


def cleanup_tmp_file(tmp_path: Path, context: str, strict: bool = True) -> bool:
    if not tmp_path.exists():
        return True

    try:
        tmp_path.unlink()
        print(f"[RESUME] removed stale tmp: {tmp_path}")
        return True
    except Exception as exc:
        msg = f"[FAILED] could not remove tmp file ({context}): {tmp_path} ({exc})"
        if strict:
            raise RuntimeError(msg) from exc
        print(msg)
        return False


def normalized_output_suffix(output_suffix: str) -> str:
    suffix = output_suffix
    if not suffix.endswith(".h5"):
        suffix = f"{suffix}.h5"
    return suffix


def normalized_output_subdir(output_subdir: str | None) -> str | None:
    if output_subdir is None:
        return None

    subdir = output_subdir.strip().strip("/\\")
    if len(subdir) == 0:
        return None

    subdir_path = Path(subdir)
    if subdir_path.is_absolute() or any(part in {".", ".."} for part in subdir_path.parts):
        raise ValueError("--output_subdir must be a relative path without '.' or '..'")
    return str(subdir_path)


def ensure_scale_tag_in_filename(filename: str, downsample_factor: int) -> str:
    factor = int(downsample_factor)
    if factor < 1:
        raise ValueError("downsample_factor must be >= 1")

    tag = f"{factor}x"
    file_path = Path(filename)
    stem = file_path.stem
    suffix = file_path.suffix

    match = re.search(r"(^|[_-])(\d+x)(?=($|[_-]))", stem)
    if match is None:
        return f"{stem}_{tag}{suffix}"

    current = match.group(2)
    if current == tag:
        return filename

    start = match.start(2)
    end = match.end(2)
    new_stem = f"{stem[:start]}{tag}{stem[end:]}"
    return f"{new_stem}{suffix}"


def normalize_polarity_to_binary(
    polarity: np.ndarray,
    dtype: str | np.dtype | None = None,
) -> np.ndarray:
    """
    Normalize event polarity to binary convention {0,1}.

    Supports common source conventions:
    - signed {-1,+1}
    - binary {0,1}
    Any value > 0 becomes 1, otherwise 0.
    """
    arr = np.asarray(polarity)
    bin_arr = arr > 0
    out_dtype = np.uint8 if dtype is None else dtype
    return bin_arr.astype(out_dtype, copy=False)


class RgbMp4Writer:
    def __init__(
        self,
        output_path: Path,
        *,
        fps: float,
        width: int,
        height: int,
    ):
        try:
            import cv2
        except Exception as exc:
            raise RuntimeError("OpenCV is required for MP4 export. Install via `pip install opencv-python`.") from exc

        self._cv2 = cv2
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = cv2.VideoWriter(
            str(self.output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(max(fps, 1e-6)),
            (int(width), int(height)),
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"failed to open VideoWriter for {self.output_path}")

    def write_rgb(self, frame_rgb: np.ndarray) -> None:
        frame = np.asarray(frame_rgb)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"frame_rgb must be HxWx3, got shape={frame.shape}")
        if np.issubdtype(frame.dtype, np.floating):
            frame = np.clip(frame, 0.0, 1.0)
            frame = np.round(frame * 255.0).astype(np.uint8)
        else:
            frame = np.clip(frame, 0, 255).astype(np.uint8, copy=False)
        self._writer.write(self._cv2.cvtColor(frame, self._cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        writer = getattr(self, "_writer", None)
        if writer is not None:
            writer.release()
            self._writer = None
