"""_FieldPairRegistration must agree with the ITK transform it replaces, exactly."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
itk = pytest.importorskip("itk")
itk_wrapper = pytest.importorskip("icon_registration.itk_wrapper")

from longitrack_backend.registration import _FieldPairRegistration  # noqa: E402

SHAPE = (8, 8, 8)


def _image(origin, spacing):
    image = itk.image_from_array(np.zeros((12, 10, 9), dtype=np.float32))
    image.SetOrigin(origin)
    image.SetSpacing(spacing)
    return image


def _identity_and_phi(seed: int):
    # the identity map icon_registration builds, plus a smooth non-trivial deformation
    axes = [torch.linspace(0.0, 1.0, size, dtype=torch.float32) for size in SHAPE]
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"))[None]
    generator = torch.Generator().manual_seed(seed)
    phi = grid + 0.02 * torch.rand(grid.shape, generator=generator, dtype=torch.float32)
    return grid, phi


def test_field_mapping_matches_the_itk_transform_it_replaces():
    identity, phi = _identity_and_phi(0)
    baseline = _image((10.0, -5.0, 2.0), (1.1, 0.9, 3.0))
    followup = _image((7.0, -3.0, 5.0), (1.0, 1.0, 2.5))

    direct = _FieldPairRegistration(baseline, followup, phi, identity, 1.5)
    reference_transform = itk_wrapper.create_itk_transform(phi, identity, followup, baseline)

    from longiseg.tracking.registration import ITKPairRegistration

    reference = ITKPairRegistration(baseline, followup, None, reference_transform, "transform", 1.5)

    size = np.asarray(baseline.GetLargestPossibleRegion().GetSize(), dtype=float) - 1.0
    rng = np.random.default_rng(1)
    compared = 0
    for _ in range(40):
        point = rng.uniform(0, size)
        expected = reference.propagate(point)
        assert direct.propagate(point) == pytest.approx(expected, abs=1e-9)
        compared += 1
    assert compared == 40


def test_a_point_outside_the_image_is_still_refused():
    identity, phi = _identity_and_phi(2)
    baseline, followup = _image((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), _image((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    direct = _FieldPairRegistration(baseline, followup, phi, identity, 1.5)

    with pytest.raises(ValueError, match="outside the image"):
        direct.propagate([-1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="3 coordinates"):
        direct.propagate([0.0, 0.0])
