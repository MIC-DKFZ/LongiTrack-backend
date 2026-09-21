import os

import numpy as np
import pytest
from synthetic import (
    BASELINE_CENTER,
    BASELINE_RADIUS,
    FOLLOWUP_RADIUS,
    FOLLOWUP_SHIFT,
    write_sample_pair,
)

pytestmark = pytest.mark.skipif(
    not os.environ.get("LONGITRACK_MODEL_DIR"),
    reason="set LONGITRACK_MODEL_DIR to a LongiSeg tracking model folder to run the pipeline tests",
)


def sphere_volume_ml(radius_mm: float) -> float:
    return 4.0 / 3.0 * np.pi * radius_mm**3 / 1000.0


@pytest.fixture(scope="module")
def engine():
    import torch

    from longitrack_backend.inference import TrackingEngine
    from longitrack_backend.model import resolve_model_folder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return TrackingEngine(resolve_model_folder(None, folds=(0,)), folds=(0,), device=device)


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    return write_sample_pair(tmp_path_factory.mktemp("pipeline"))


def test_track_finds_both_synthetic_lesions(engine, pair):
    baseline, followup = pair
    baseline_point = list(BASELINE_CENTER)
    followup_point = [c + s for c, s in zip(BASELINE_CENTER, FOLLOWUP_SHIFT, strict=True)]

    result = engine.track(baseline, baseline_point, followup, followup_point)

    shape, _ = engine.image_geometry(baseline)
    assert result.baseline.mask.shape == shape
    assert result.followup.mask.shape == shape

    # a coordinate bug shows up here as an empty or wildly misplaced mask
    assert result.baseline.volume_ml == pytest.approx(sphere_volume_ml(BASELINE_RADIUS), rel=0.5)
    assert result.followup.volume_ml == pytest.approx(sphere_volume_ml(FOLLOWUP_RADIUS), rel=0.5)
    assert result.volume_change_ml > 0

    # and the masks have to sit where the spheres are
    for mask, centre in ((result.baseline.mask, BASELINE_CENTER), (result.followup.mask, followup_point)):
        centroid = np.argwhere(mask).mean(axis=0)
        assert np.allclose(centroid, centre, atol=4)


def test_preprocessing_is_cached(engine, pair):
    baseline, followup = pair
    engine.preprocess(baseline)
    before = len(engine._cache)
    engine.preprocess(baseline)
    assert len(engine._cache) == before


def test_baseline_only(engine, pair):
    baseline, _ = pair
    mask = engine.segment_baseline(baseline, list(BASELINE_CENTER))
    assert not mask.empty
    assert mask.mask.shape == engine.image_geometry(baseline)[0]


def test_baseline_and_followup_are_independent(engine, pair):
    # the baseline segmentation is not an input to the follow-up prediction
    baseline, followup = pair
    baseline_point = list(BASELINE_CENTER)
    followup_point = [c + s for c, s in zip(BASELINE_CENTER, FOLLOWUP_SHIFT, strict=True)]

    together = engine.track(baseline, baseline_point, followup, followup_point, segment_baseline=True)
    alone = engine.track(baseline, baseline_point, followup, followup_point, segment_baseline=False)
    reference = engine.segment_baseline(baseline, baseline_point)

    assert np.array_equal(together.followup.mask, alone.followup.mask)
    assert np.array_equal(together.baseline.mask, reference.mask)


def test_restore_patch_matches_the_full_volume_reference(engine, pair):
    # _restore_patch resamples only the small predicted patch, as an optimization over
    # _restore's full-volume resample (used here purely as the ground truth it must
    # reproduce exactly). A patch that is a small, off-center fraction of the full
    # preprocessed volume exercises the boundary math that a patch covering the whole
    # tiny synthetic volume (as the other pipeline tests happen to, at this resolution)
    # would never catch.
    import torch

    _, followup = pair
    fu_image = engine.preprocess(followup)
    full_shape = fu_image.preprocessed_shape

    rng = np.random.default_rng(0)
    for trial in range(3):
        size = [int(rng.integers(3, max(4, s // 2))) for s in full_shape]
        lower = tuple(int(rng.integers(0, s - sz)) for s, sz in zip(full_shape, size, strict=True))
        upper = tuple(lo + sz for lo, sz in zip(lower, size, strict=True))

        # a smooth confidence field, like a real prediction's, rather than uniform noise:
        # noise puts roughly half of all voxels at an exact float32 tie between the two
        # classes, where two independently-implemented interpolations (grid_sample here vs
        # F.interpolate in the reference) can legitimately round a hair differently and flip
        # the argmax -- not a bug, just not a realistic probability field to demand bit-exactness on.
        grids = torch.meshgrid(*(torch.arange(dim, dtype=torch.float32) for dim in size), indexing="ij")
        centre = [rng.uniform(0, dim) for dim in size]
        distance = sum((axis - c) ** 2 for axis, c in zip(grids, centre, strict=True)).sqrt()
        logit = (5.0 - distance) * 2.0
        probabilities = torch.softmax(torch.stack([-logit, logit]), dim=0)

        local, bounds = engine._restore_patch(probabilities, lower, upper, fu_image)
        patch_mask = np.zeros(fu_image.original_shape, dtype=local.dtype)
        patch_mask[tuple(slice(s, e) for s, e in bounds)] = local

        full_probabilities = torch.zeros((2, *full_shape), dtype=probabilities.dtype)
        full_probabilities[0] = 1.0
        full_probabilities[(slice(None), *(slice(a, b) for a, b in zip(lower, upper, strict=True)))] = probabilities
        full_mask = engine._restore(full_probabilities, fu_image)

        assert np.array_equal(patch_mask, full_mask), f"trial {trial}: lower={lower} upper={upper}"
