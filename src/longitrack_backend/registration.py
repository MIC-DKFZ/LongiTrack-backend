from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import _env

_env.prepare_longiseg_env()

from .geometry import index_to_xyz, snap_to_volume, xyz_to_index  # noqa: E402
from .scan_io import itk_float_image, read_scan  # noqa: E402

DEFAULT_IO_ITERATIONS = 50
UNIGRADICON_INPUT_SHAPE = (175, 175, 175)
Progress = Callable[[str], None]


@dataclass
class PointPropagation:
    baseline_index: list[int]
    followup_index: list[int] | None
    out_of_bounds: bool = False
    error: str | None = None
    extras: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.followup_index is not None


def _signature(path: str | Path) -> tuple[str, float, int]:
    stat = Path(path).stat()
    return str(Path(path).resolve()), stat.st_mtime, stat.st_size


def _resampling_affine(image, shape) -> tuple[np.ndarray, np.ndarray]:
    """`itk_wrapper.resampling_transform` as a plain affine: y = matrix @ x + offset.

    The grid scaling composed with the image direction, centered on the two geometric
    centres.
    """
    size = np.asarray(image.GetLargestPossibleRegion().GetSize(), dtype=np.float64)
    spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
    origin = np.asarray(image.GetOrigin(), dtype=np.float64)
    direction = np.asarray(image.GetDirection(), dtype=np.float64)
    grid = np.asarray(shape, dtype=np.float64)
    matrix = direction @ np.diag(spacing * size / grid)
    image_center = origin + direction @ (spacing * (size - 1.0) / 2.0)
    return matrix, image_center - matrix @ ((grid - 1.0) / 2.0)


class _FieldPairRegistration:
    """Map points through uniGradICON's displacement field without materializing it.

    The same arithmetic `create_itk_transform` does -- the two resampling affines and
    trilinear interpolation of the field -- for the handful of points the viewer asks
    about. A point outside the field falls back to the real ITK transform.
    """

    def __init__(self, baseline_image, followup_image, phi, identity_map, marker_sigma_vox: float) -> None:
        import torch

        self.bl_image, self.fu_image = baseline_image, followup_image
        self.mapping_method = "transform"
        self.marker_sigma_vox = marker_sigma_vox
        self._phi, self._identity_map = phi, identity_map
        self._shape = list(identity_map.shape[2:])
        scale = torch.tensor(self._shape, device=phi.device, dtype=phi.dtype)
        for _ in self._shape:
            scale = scale[:, None]
        # create_itk_transform's own `disp *= scale - 1`, kept on the device it is already on
        self._displacement = (phi - identity_map)[0] * (scale - 1)
        reversed_shape = list(reversed(self._shape))
        # the ITK composite applies these back to front: baseline physical -> network
        # index, displace, network index -> follow-up physical
        to_baseline, baseline_offset = _resampling_affine(baseline_image, reversed_shape)
        self._to_network = (np.linalg.inv(to_baseline), baseline_offset)
        self._to_followup = _resampling_affine(followup_image, reversed_shape)
        self._itk = None

    def _itk_registration(self):
        import icon_registration.itk_wrapper as itk_wrapper
        from longiseg.tracking.registration import ITKPairRegistration

        if self._itk is None:
            transform = itk_wrapper.create_itk_transform(
                self._phi, self._identity_map, self.fu_image, self.bl_image
            )
            self._itk = ITKPairRegistration(
                self.bl_image, self.fu_image, None, transform, "transform", self.marker_sigma_vox
            )
        return self._itk

    def _displace(self, point: np.ndarray) -> np.ndarray | None:
        """ITK's linear vector interpolation at one continuous index, or None if outside."""
        index = point[::-1]  # the field image has unit spacing at the origin, so physical == index
        if np.any(index < 0) or np.any(index > np.asarray(self._shape) - 1):
            return None
        lower = np.floor(index).astype(int)
        fraction = index - lower
        upper = [min(value + 1, size - 1) for value, size in zip(lower, self._shape, strict=True)]
        corners = self._displacement[
            :, lower[0] : upper[0] + 1, lower[1] : upper[1] + 1, lower[2] : upper[2] + 1
        ]
        block = corners.detach().cpu().numpy().astype(np.float64)  # 8 vectors, not 175^3
        weights = np.zeros(block.shape[1:], dtype=np.float64)
        for i in range(block.shape[1]):
            for j in range(block.shape[2]):
                for k in range(block.shape[3]):
                    weights[i, j, k] = (
                        ((1.0 - fraction[0]) if i == 0 else fraction[0])
                        * ((1.0 - fraction[1]) if j == 0 else fraction[1])
                        * ((1.0 - fraction[2]) if k == 0 else fraction[2])
                    )
        return (block * weights).sum(axis=(1, 2, 3))[::-1]  # components back to (x, y, z)

    def propagate(self, bl_point_xyz: Sequence[float]) -> list[float]:
        from longiseg.tracking.registration import ITKPairRegistration

        bl_point_xyz = np.asarray(bl_point_xyz, dtype=np.float64)
        if bl_point_xyz.shape != (3,):
            raise ValueError(f"Expected a point with 3 coordinates, got {bl_point_xyz.tolist()}.")
        size_xyz = np.asarray(self.bl_image.GetLargestPossibleRegion().GetSize(), dtype=np.float64) - 1.0
        if np.any(bl_point_xyz < 0.0) or np.any(bl_point_xyz > size_xyz):
            raise ValueError(
                f"Baseline point {bl_point_xyz.tolist()} lies outside the image, which covers "
                f"[0, {size_xyz.tolist()}]."
            )

        physical = ITKPairRegistration._voxel_to_physical(self.bl_image, bl_point_xyz)
        inverse, baseline_offset = self._to_network
        network = inverse @ (physical - baseline_offset)
        displacement = self._displace(network)
        if displacement is None:
            return self._itk_registration().propagate(bl_point_xyz)
        matrix, offset = self._to_followup
        followup = matrix @ (network + displacement) + offset
        return [float(c) for c in ITKPairRegistration._physical_to_voxel(self.fu_image, followup)]


class RegistrationService:
    def __init__(
        self,
        model: str = "unigradicon",
        modality: str = "ct",
        mapping_method: str = "transform",
        io_iterations: int = DEFAULT_IO_ITERATIONS,
    ) -> None:
        self.model = model
        self.modality = modality
        self.mapping_method = mapping_method
        self.io_iterations = io_iterations
        self._propagator = None
        self._registration = None
        self._cache_key: tuple | None = None
        self._warmed = False
        # Native ITK geometry and the corresponding normalized/resized CUDA tensor.
        self._prepared_scans: dict[tuple[str, float, int], tuple[object, object]] = {}

    def preload_scan(self, image: str | Path, device: str = "cuda", progress: Progress | None = None) -> None:
        """Prepare the fixed UniGradICON input on CUDA without loading its weights."""
        import torch
        import torch.nn.functional as F

        key = _signature(image)
        if key in self._prepared_scans:
            return
        if progress:
            progress(f"Preparing {Path(image).name} for registration on {device}...")
        started = time.perf_counter()
        # the segmentation side reads the same file; scan_io hands the volume over
        raw, properties = read_scan(image)
        itk_image = itk_float_image(raw, properties)
        # uniGradICON's CT preprocess (clamp [-1000, 1000], scale to [0, 1]), done on the device
        tensor = torch.from_numpy(np.ascontiguousarray(raw[0])).to(device=device)[None, None].float()
        tensor = tensor.clamp_(-1000, 1000).add_(1000).mul_(1 / 2000)
        tensor = F.interpolate(tensor, size=UNIGRADICON_INPUT_SHAPE, mode="trilinear", align_corners=False)
        self._prepared_scans[key] = (itk_image, tensor)
        if progress:
            progress(f"  registration input ready ({time.perf_counter() - started:.1f} s).")

    def preload_scans(self, images: Sequence[str | Path], device: str = "cuda",
                      progress: Progress | None = None) -> None:
        """Prepare several scans, decompressing them concurrently."""
        from concurrent.futures import ThreadPoolExecutor

        unique = list(dict.fromkeys(str(Path(image).expanduser()) for image in images))
        paths = [path for path in unique if _signature(path) not in self._prepared_scans]
        if len(paths) < 2:
            for path in paths:
                self.preload_scan(path, device=device, progress=progress)
            return
        if progress:
            progress(f"Preparing {len(paths)} scans for registration on {device}...")
        # reading is CPU/disk work and releases the GIL, so it overlaps; the CUDA side stays serial
        with ThreadPoolExecutor(max_workers=len(paths), thread_name_prefix="longitrack-read") as pool:
            list(pool.map(read_scan, paths))
        for path in paths:
            self.preload_scan(path, device=device, progress=progress)

    def _get_propagator(self, refinement_steps: int | None):
        from longiseg.tracking.registration import UniGradIconPropagator

        if self._propagator is None:
            self._propagator = UniGradIconPropagator(
                model=self.model,
                io_iterations=self.io_iterations,
                mapping_method=self.mapping_method,
                modality=self.modality,
            )
        self._propagator.io_iterations = refinement_steps
        return self._propagator

    def warm_up(self, progress: Progress | None = None) -> None:
        """Load UniGradICON and exercise its fixed-shape no-refinement CUDA path."""
        if self._warmed:
            return
        import icon_registration.itk_wrapper as itk_wrapper
        import torch
        from icon_registration import config

        if progress:
            progress("Loading and warming the uniGradICON registration network...")
        propagator = self._get_propagator(self.io_iterations)
        if propagator._net is None:
            propagator._net = propagator._build_net()
        network = propagator._net.to(config.device)
        dummy = torch.zeros((1, 1, *UNIGRADICON_INPUT_SHAPE), device=config.device)
        with torch.inference_mode():
            network(dummy, dummy)
            phi_ab = network.phi_AB(network.identity_map)
            phi_ba = network.phi_BA(network.identity_map)
        del phi_ab, phi_ba

        # a first backward pass here, so the slider's refinement steps never pay for it
        if progress:
            progress("  warming the refinement path...")
        # zeros would make LNCC degenerate; the content is irrelevant either way
        noise = torch.rand((1, 1, *UNIGRADICON_INPUT_SHAPE), device=config.device)
        try:
            itk_wrapper.finetune_execute(
                network, noise, noise.roll(8, dims=2), 1, itk_wrapper.DEFAULT_FINETUNE_LEARNING_RATE
            )
        except Exception as error:  # noqa: BLE001 - a warm-up must never break start-up
            if progress:
                progress(f"  could not warm the refinement path ({error}); the first refined run pays it instead.")
        network.zero_grad(set_to_none=True)

        if torch.cuda.is_available():
            torch.cuda.synchronize(config.device)
        del dummy, noise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._warmed = True

    def _register_prepared(self, propagator, baseline_key: tuple, followup_key: tuple):
        """UniGradICON's register_pair tail, using already-prepared CUDA inputs."""
        import icon_registration.itk_wrapper as itk_wrapper
        import torch
        from icon_registration import config
        from longiseg.tracking.registration import ITKPairRegistration

        baseline_image, baseline_tensor = self._prepared_scans[baseline_key]
        followup_image, followup_tensor = self._prepared_scans[followup_key]
        # upstream's lazy build, mirrored before touching the cached CUDA inputs
        if propagator._net is None:
            propagator._net = propagator._build_net()
        network = propagator._net.to(config.device)
        baseline_tensor = baseline_tensor.to(config.device)
        followup_tensor = followup_tensor.to(config.device)
        if propagator.io_iterations is None:
            with torch.no_grad():
                network(baseline_tensor, followup_tensor)
        else:
            itk_wrapper.finetune_execute(
                network, baseline_tensor, followup_tensor, propagator.io_iterations,
                itk_wrapper.DEFAULT_FINETUNE_LEARNING_RATE,
            )
        phi_fu_to_bl = network.phi_AB(network.identity_map)
        phi_bl_to_fu = network.phi_BA(network.identity_map)
        # marker_warp is the only mapping that still needs a real ITK transform
        if propagator.mapping_method != "marker_warp":
            # Sampling the displacement field directly, instead of handing ITK a
            # materialized copy of it, is the same arithmetic for a great deal less work.
            return _FieldPairRegistration(
                baseline_image, followup_image, phi_bl_to_fu, network.identity_map,
                propagator.marker_sigma_vox,
            )
        phi_fu_to_bl_itk = itk_wrapper.create_itk_transform(
            phi_fu_to_bl, network.identity_map, baseline_image, followup_image
        )
        phi_bl_to_fu_itk = itk_wrapper.create_itk_transform(
            phi_bl_to_fu, network.identity_map, followup_image, baseline_image
        )
        return ITKPairRegistration(
            baseline_image, followup_image, phi_fu_to_bl_itk, phi_bl_to_fu_itk, propagator.mapping_method,
            propagator.marker_sigma_vox,
        )

    def register(
        self,
        baseline_image: str | Path,
        followup_image: str | Path,
        refinement_steps: int | None = None,
        progress: Progress | None = None,
    ):
        baseline_key, followup_key = _signature(baseline_image), _signature(followup_image)
        pair_key = (baseline_key, followup_key, self.model, self.mapping_method)
        key = (*pair_key, refinement_steps)
        if key == self._cache_key and self._registration is not None:
            if progress:
                progress("Reusing the cached registration.")
            return self._registration

        if progress:
            mode = "no refinement" if refinement_steps is None else f"{refinement_steps} refinement steps"
            progress(f"Registering {Path(baseline_image).name} -> {Path(followup_image).name} ({mode})...")

        propagator = self._get_propagator(refinement_steps)
        started = time.perf_counter()
        if baseline_key in self._prepared_scans and followup_key in self._prepared_scans:
            self._registration = self._register_prepared(propagator, baseline_key, followup_key)
        else:
            # A direct API caller may bypass viewer preload; retain upstream behavior
            # in that case rather than making registration dependent on the GUI.
            self._registration = propagator.register(str(baseline_image), str(followup_image))
        self._cache_key = key
        if progress:
            progress(f"Registration done in {time.perf_counter() - started:.1f} s.")
        return self._registration

    def invalidate(self) -> None:
        # Only the cached registration result is dropped; the network itself never needs rebuilding.
        self._registration = None
        self._cache_key = None

    def clear_prepared_scans(self) -> None:
        """Drop every preloaded (fixed-size, normalized) CUDA input tensor.

        This is the registration side of "clear everything": it frees per-scan GPU
        memory without touching the network weights, so the model stays initialized.
        """
        self._prepared_scans.clear()

    def propagate(
        self,
        baseline_image: str | Path,
        followup_image: str | Path,
        baseline_points: Sequence[Sequence[float]],
        followup_shape: Sequence[int],
        refinement_steps: int | None = None,
        progress: Progress | None = None,
    ) -> list[PointPropagation]:
        registration = self.register(
            baseline_image, followup_image, refinement_steps=refinement_steps, progress=progress
        )

        results: list[PointPropagation] = []
        for number, point in enumerate(baseline_points, start=1):
            baseline_index = [int(round(float(c))) for c in point]
            try:
                followup_xyz = registration.propagate(index_to_xyz(baseline_index))
            except Exception as error:
                # one bad point must not take the others down with it
                if progress:
                    progress(f"  point {number}: propagation failed ({error})")
                results.append(PointPropagation(baseline_index, None, error=str(error)))
                continue

            raw_index = xyz_to_index(followup_xyz)
            snapped = snap_to_volume(raw_index, followup_shape)
            out_of_bounds = any(abs(a - b) > 0.5 for a, b in zip(raw_index, snapped, strict=True))
            if progress:
                suffix = "  (clipped to the volume)" if out_of_bounds else ""
                progress(f"  point {number}: {baseline_index} -> {snapped}{suffix}")
            results.append(
                PointPropagation(
                    baseline_index,
                    snapped,
                    out_of_bounds=out_of_bounds,
                    extras={"unclipped": [float(c) for c in raw_index]},
                )
            )
        return results
