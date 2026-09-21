from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# Three coordinate systems, and mixing them up fails silently:
#   index         (z, y, x) into the array as SimpleITKIO reads it. What a viewer shows,
#                 what a Points layer stores, and the interface of this package.
#   xyz           ITK voxel order. LongiSeg's tracking json and PairRegistration.propagate.
#   preprocessed  after transpose_forward, cropping and resampling to the plan's spacing.
#                 What the network sees.


def index_to_xyz(index_zyx: Sequence[float]) -> list[float]:
    return [float(c) for c in reversed(list(index_zyx))]


def xyz_to_index(point_xyz: Sequence[float]) -> list[float]:
    return [float(c) for c in reversed(list(point_xyz))]


def snap_to_volume(point: Sequence[float], shape: Sequence[int]) -> list[int]:
    if len(point) != len(shape):
        raise ValueError(f"Point {list(point)} and shape {list(shape)} have different dimensionality.")
    return [int(np.clip(round(float(c)), 0, int(s) - 1)) for c, s in zip(point, shape, strict=True)]


def crop_bbox_lower_bounds(properties: dict, dim: int) -> list[int]:
    bbox = properties.get("bbox_used_for_cropping")
    if bbox is None:
        return [0] * dim
    return [0 if bounds[0] is None else int(bounds[0]) for bounds in bbox]


def index_to_preprocessed(
    index_zyx: Sequence[float],
    properties: dict,
    transpose_forward: Sequence[int],
    target_spacing: Sequence[float],
    preprocessed_shape: Sequence[int],
) -> list[int]:
    # same order of operations as the preprocessor: transpose, crop, resample. Matches
    # tracking_inference.predict_patient for the identity transpose and no-op crop that
    # LongiSegTrackingPreprocessor produces, but rounds instead of truncating.
    transposed = [float(index_zyx[axis]) for axis in transpose_forward]
    spacing = [float(properties["spacing"][axis]) for axis in transpose_forward]
    offsets = crop_bbox_lower_bounds(properties, len(transposed))
    cropped = [coordinate - offset for coordinate, offset in zip(transposed, offsets, strict=True)]
    resampled = [coordinate * spacing[d] / float(target_spacing[d]) for d, coordinate in enumerate(cropped)]
    return snap_to_volume(resampled, preprocessed_shape)


def preprocessed_to_index(
    point: Sequence[float],
    properties: dict,
    transpose_forward: Sequence[int],
    target_spacing: Sequence[float],
    original_shape: Sequence[int],
) -> list[int]:
    spacing = [float(properties["spacing"][axis]) for axis in transpose_forward]
    offsets = crop_bbox_lower_bounds(properties, len(point))
    transposed = [
        float(coordinate) * float(target_spacing[d]) / spacing[d] + offsets[d] for d, coordinate in enumerate(point)
    ]
    index = [0.0] * len(transposed)
    for d, axis in enumerate(transpose_forward):
        index[axis] = transposed[d]
    return snap_to_volume(index, original_shape)
