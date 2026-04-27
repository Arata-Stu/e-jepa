from __future__ import annotations

from pathlib import Path
import re


def _compression_opts() -> tuple[int, int, int, int, int, int, int]:
    compression_level = 1  # {0, ..., 9}
    shuffle = 2  # {0: none, 1: byte, 2: bit}
    compressor_type = 5  # BLOSC_ZSTD
    return 0, 0, 0, 0, compression_level, shuffle, compressor_type


def get_h5_compression_flags() -> dict:
    try:
        import hdf5plugin  # noqa: F401

        _ = hdf5plugin
        return dict(
            compression=32001,
            compression_opts=_compression_opts(),
        )
    except ImportError:
        return dict(
            compression="gzip",
            compression_opts=1,
        )


def tmp_output_path(output_path: Path, tmp_suffix: str) -> Path:
    return output_path.with_name(f"{output_path.name}{tmp_suffix}")


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
