import pytest

from longitrack_backend.geometry import (
    index_to_preprocessed,
    index_to_xyz,
    snap_to_volume,
    xyz_to_index,
)

IDENTITY = [0, 1, 2]


def properties(spacing=(3.0, 0.7, 0.7), bbox=None, shape_before=None):
    return {
        "spacing": list(spacing),
        "bbox_used_for_cropping": bbox if bbox is not None else [[None, None]] * 3,
        "shape_before_cropping": list(shape_before) if shape_before else None,
    }


def test_xyz_roundtrip():
    assert index_to_xyz([1, 2, 3]) == [3.0, 2.0, 1.0]
    assert xyz_to_index(index_to_xyz([1, 2, 3])) == [1.0, 2.0, 3.0]


def test_snap_clips_and_rounds():
    assert snap_to_volume([-4.0, 2.6, 99.0], (10, 10, 10)) == [0, 3, 9]
    with pytest.raises(ValueError):
        snap_to_volume([1.0, 2.0], (10, 10, 10))


def test_index_to_preprocessed_matches_the_reference_formula():
    spacing = (5.0, 0.5, 0.5)
    target = [2.5, 1.0, 1.0]
    point = [10, 40, 60]
    expected = [int(p * s / t) for p, s, t in zip(point, spacing, target, strict=True)]
    got = index_to_preprocessed(point, properties(spacing), IDENTITY, target, (200, 200, 200))
    assert got == expected


def test_index_to_preprocessed_snaps_into_the_volume():
    got = index_to_preprocessed([0, 0, 500], properties(), IDENTITY, [3.0, 0.7, 0.7], (64, 128, 128))
    assert got == [0, 0, 127]

