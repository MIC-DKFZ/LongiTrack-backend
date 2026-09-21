# Synthetic baseline/follow-up pair used by the offline tests.
from __future__ import annotations

from pathlib import Path

import numpy as np

from longitrack_backend._env import CACHE_DIR

SHAPE = (64, 160, 160)
SPACING_ZYX = (3.0, 1.0, 1.0)
BASELINE_CENTER = (30, 70, 82)
BASELINE_RADIUS = 6.0
FOLLOWUP_SHIFT = (2, 7, -5)
FOLLOWUP_RADIUS = 9.0


def _volume(center: tuple[int, int, int], radius: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    grid = np.stack(np.meshgrid(*[np.arange(s, dtype=np.float32) for s in SHAPE], indexing="ij"))
    physical = grid * np.asarray(SPACING_ZYX, dtype=np.float32)[:, None, None, None]

    volume = np.full(SHAPE, -900.0, dtype=np.float32)  # air

    # an elliptical body cross-section
    body = (
        ((physical[1] - SHAPE[1] * SPACING_ZYX[1] / 2) / (SHAPE[1] * SPACING_ZYX[1] * 0.40)) ** 2
        + ((physical[2] - SHAPE[2] * SPACING_ZYX[2] / 2) / (SHAPE[2] * SPACING_ZYX[2] * 0.32)) ** 2
    ) < 1.0
    volume[body] = 30.0

    # a few bright ribs, so a registration has some structure to work with
    for z in range(4, SHAPE[0], 8):
        ring = body & (np.abs(z - grid[0]) < 1.0)
        edge = ring & ~np.roll(ring, 4, axis=2) | ring & ~np.roll(ring, -4, axis=2)
        volume[edge] = 900.0

    center_physical = np.asarray(center, dtype=np.float32) * np.asarray(SPACING_ZYX, dtype=np.float32)
    distance = np.sqrt(((physical - center_physical[:, None, None, None]) ** 2).sum(axis=0))
    volume[distance <= radius] = 120.0
    volume[(distance > radius) & (distance <= radius + 1.5)] = 75.0

    return (volume + rng.normal(0.0, 12.0, SHAPE)).astype(np.float32)


def write_sample_pair(folder: str | Path | None = None) -> tuple[Path, Path]:
    import SimpleITK as sitk

    folder = Path(folder) if folder is not None else CACHE_DIR / "sample_data"
    folder.mkdir(parents=True, exist_ok=True)

    followup_center = tuple(c + s for c, s in zip(BASELINE_CENTER, FOLLOWUP_SHIFT, strict=True))
    specs = (
        ("longitrack_sample_baseline.nii.gz", BASELINE_CENTER, BASELINE_RADIUS, 0),
        ("longitrack_sample_followup.nii.gz", followup_center, FOLLOWUP_RADIUS, 1),
    )

    paths = []
    for name, center, radius, seed in specs:
        path = folder / name
        if not path.is_file():
            image = sitk.GetImageFromArray(_volume(center, radius, seed))
            image.SetSpacing(tuple(reversed(SPACING_ZYX)))
            sitk.WriteImage(image, str(path), useCompression=True)
        paths.append(path)
    return paths[0], paths[1]


