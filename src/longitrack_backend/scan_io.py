from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from . import _env

_env.prepare_longiseg_env()

# Registration and segmentation each prepare every scan, so whichever decompresses one
# first hands its volume to the other. Only the pair in flight is kept.
_HANDOVER: dict[tuple, tuple[np.ndarray, dict]] = {}
_LOCK = threading.Lock()


def _signature(path: Path) -> tuple:
    stat = path.stat()
    return str(path.resolve()), stat.st_mtime_ns, stat.st_size


def read_scan(image: str | Path) -> tuple[np.ndarray, dict]:
    """Read a scan as (c, z, y, x) plus its geometry, reusing a just-read volume."""
    from longiseg.imageio.simpleitk_reader_writer import SimpleITKIO

    path = Path(image).expanduser()
    key = _signature(path)
    with _LOCK:
        waiting = _HANDOVER.get(key)
    if waiting is not None:
        return waiting

    data, properties = SimpleITKIO().read_images([str(path)])
    with _LOCK:
        while len(_HANDOVER) >= 2:
            _HANDOVER.pop(next(iter(_HANDOVER)))
        _HANDOVER[key] = (data, properties)
    return data, properties


def forget(image: str | Path) -> None:
    with _LOCK:
        _HANDOVER.pop(_signature(Path(image).expanduser()), None)


def itk_float_image(data: np.ndarray, properties: dict):
    """Rebuild the float ITK image `itk.imread(path, itk.F)` would have returned."""
    import itk

    geometry = properties["sitk_stuff"]
    image = itk.image_from_array(np.ascontiguousarray(data[0], dtype=np.float32))
    image.SetOrigin(tuple(float(value) for value in geometry["origin"]))
    image.SetSpacing(tuple(float(value) for value in geometry["spacing"]))
    image.SetDirection(itk.matrix_from_array(np.asarray(geometry["direction"], dtype=float).reshape(3, 3)))
    return image
