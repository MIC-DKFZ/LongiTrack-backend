from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import _env

_env.prepare_longiseg_env()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from .geometry import crop_bbox_lower_bounds, index_to_preprocessed, snap_to_volume  # noqa: E402
from .model import CHECKPOINT_NAME, available_folds, locate_model_folder  # noqa: E402
from .scan_io import read_scan  # noqa: E402

Progress = Callable[[str], None]
_PROMPT_KERNEL_CACHE: dict[tuple[str, torch.dtype, float], torch.Tensor] = {}


def _log(progress: Progress | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _prompt_on_device(
    point: Sequence[int], shape: Sequence[int], sigma: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Build the tracking prompt on CUDA, preserving SciPy edge behavior exactly."""
    radius = int(4 * sigma + 0.5)  # scipy.ndimage.gaussian_filter default truncate=4
    # away from edges this equals SciPy's filtered impulse divided by its centre
    if all(radius <= coord < length - radius for coord, length in zip(point, shape, strict=True)):
        key = (str(device), dtype, float(sigma))
        kernel = _PROMPT_KERNEL_CACHE.get(key)
        if kernel is None:
            coordinates = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
            zz, yy, xx = torch.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
            kernel = torch.exp(-(zz.square() + yy.square() + xx.square()) / (2 * sigma * sigma))
            _PROMPT_KERNEL_CACHE[key] = kernel
        prompt = torch.zeros(tuple(shape), device=device, dtype=dtype)
        prompt[tuple(slice(coord - radius, coord + radius + 1) for coord in point)] = kernel
        return prompt

    from longiseg.training.dataloading.utils import generated_sparse_to_dense_point_rescaled_gauss

    return torch.from_numpy(generated_sparse_to_dense_point_rescaled_gauss(list(point), tuple(shape), sigma=sigma)).to(
        device=device, dtype=dtype
    )


def _predict_one_patch_exact(predictor, data: torch.Tensor) -> torch.Tensor:
    """The one-tile case of nnU-Net prediction, without sliding-window machinery.

    This deliberately retains the upstream Gaussian float16 multiply/add/divide
    arithmetic. A direct comparison against ``predict_sliding_window_return_logits``
    on LongiTrack-L produced bitwise-identical logits.
    """
    from longiseg.inference.sliding_window_prediction import compute_gaussian

    predictor.network.eval()
    # The upstream sliding-window entry point is inference-only as well.
    with torch.inference_mode(), torch.autocast(predictor.device.type, enabled=predictor.device.type == "cuda"):
        prediction = predictor._internal_maybe_mirror_and_predict(data[None])[0]
    gaussian = compute_gaussian(
        tuple(predictor.configuration_manager.patch_size), sigma_scale=1 / 8, value_scaling_factor=10,
        device=predictor.device,
    )
    logits = torch.zeros_like(prediction)
    counts = torch.zeros_like(prediction[0])
    logits += prediction * gaussian
    counts += gaussian
    logits /= counts
    return logits


def _predict_cached_patch(
    bl_data: torch.Tensor,
    bl_point: Sequence[int],
    fu_data: torch.Tensor,
    fu_point: Sequence[int],
    predictor,
    patch_size: Sequence[int],
    device: torch.device,
    sigma: float,
) -> torch.Tensor:
    """LongiSeg's point-patch prediction path with cached CUDA inputs.

    The upstream helper constructs prompt maps on CPU, which makes ``torch.cat``
    fail when the cached scan tensors live on CUDA. Keeping this small adapter here
    avoids modifying site-packages and preserves its patch geometry, fold loop and
    predictor calls exactly.
    """
    from longiseg.tracking.patch_extraction import compute_paired_patch_bboxes, crop_bbox_to_shape

    bl_shape, fu_shape = bl_data.shape[1:], fu_data.shape[1:]
    dim = len(fu_shape)
    fu_lbs, fu_ubs, bl_lbs, bl_ubs = compute_paired_patch_bboxes(fu_shape, fu_point, bl_shape, bl_point, patch_size)
    valid_fu_lbs, valid_fu_ubs, fu_padding = crop_bbox_to_shape(fu_lbs, fu_ubs, fu_shape)
    valid_bl_lbs, valid_bl_ubs, bl_padding = crop_bbox_to_shape(bl_lbs, bl_ubs, bl_shape)

    fu_data_patch = fu_data[
        (slice(0, fu_data.shape[0]), *[slice(i, j) for i, j in zip(valid_fu_lbs, valid_fu_ubs, strict=True)])
    ]
    bl_data_patch = bl_data[
        (slice(0, bl_data.shape[0]), *[slice(i, j) for i, j in zip(valid_bl_lbs, valid_bl_ubs, strict=True)])
    ]
    bl_seg_patch = torch.zeros((1, *bl_data_patch.shape[1:]), dtype=bl_data_patch.dtype, device=bl_data_patch.device)

    fu_local = [fu_point[d] - valid_fu_lbs[d] for d in range(dim)]
    bl_local = [bl_point[d] - valid_bl_lbs[d] for d in range(dim)]
    fu_prompt = _prompt_on_device(fu_local, fu_data_patch.shape[1:], sigma, fu_data_patch.device, fu_data_patch.dtype)
    bl_prompt = _prompt_on_device(bl_local, bl_data_patch.shape[1:], sigma, bl_data_patch.device, bl_data_patch.dtype)

    def pad_spec(padding: Sequence[tuple[int, int]]) -> tuple[int, ...]:
        return tuple(value for pair in reversed(padding) for value in pair)

    fu_pad, bl_pad = pad_spec(fu_padding), pad_spec(bl_padding)
    fu_data_patch = F.pad(fu_data_patch, fu_pad, mode="constant", value=0)
    bl_data_patch = F.pad(bl_data_patch, bl_pad, mode="constant", value=0)
    bl_seg_patch = F.pad(bl_seg_patch, bl_pad, mode="constant", value=0)
    fu_prompt = F.pad(fu_prompt.unsqueeze(0), fu_pad, mode="constant", value=0)
    bl_prompt = F.pad(bl_prompt.unsqueeze(0), bl_pad, mode="constant", value=0)
    # With sufficient VRAM the scans are resident on CUDA; otherwise the same
    # operation transfers just this cropped five-channel patch from the CPU cache.
    data = torch.cat((fu_data_patch, bl_data_patch, bl_seg_patch, fu_prompt, bl_prompt), dim=0).to(device)

    network = predictor.network
    if len(predictor.list_of_parameters) == 1:
        # The generic predictor reloads its CPU checkpoint state dict before every call so it can ensemble folds.
        if not getattr(predictor, "_longitrack_single_fold_on_device", False):
            network.to(device)
            predictor._longitrack_single_fold_on_device = True
        predicted_patch = _predict_one_patch_exact(predictor, data)
    else:
        predicted_patch = None
        for params in predictor.list_of_parameters:
            network.load_state_dict(params)
            logits = _predict_one_patch_exact(predictor, data)
            predicted_patch = logits if predicted_patch is None else predicted_patch + logits
    predicted_patch = torch.softmax(predicted_patch, dim=0)
    crop = (
        slice(None),
        *[slice(fu_padding[d][0], predicted_patch.shape[d + 1] - fu_padding[d][1]) for d in range(dim)],
    )
    predicted_patch = predicted_patch[crop]

    return predicted_patch, tuple(valid_fu_lbs), tuple(valid_fu_ubs)


@dataclass
class PreprocessedImage:
    path: Path
    data: torch.Tensor  # (c, z', y', x') after transpose/crop/resample
    seg: torch.Tensor | None
    properties: dict
    original_shape: tuple[int, ...]  # (z, y, x) as read from disk
    spacing_zyx: tuple[float, ...]

    @property
    def preprocessed_shape(self) -> tuple[int, ...]:
        return tuple(self.data.shape[1:])


@dataclass
class LesionMask:
    mask: np.ndarray  # uint8, local patch in original index (z, y, x) order
    point: list[int]  # (z, y, x) prompt that produced it
    spacing_zyx: tuple[float, ...]
    scan: Path | None = None
    properties: dict = field(default_factory=dict)  # carries sitk_stuff, so the mask can be written out
    bounds: tuple[tuple[int, int], ...] | None = None
    original_shape: tuple[int, ...] | None = None

    def full_mask(self) -> np.ndarray:
        """Materialize the scan-sized mask only for an explicit export."""
        if self.bounds is None or self.original_shape is None:
            return self.mask
        full = np.zeros(self.original_shape, dtype=self.mask.dtype)
        full[tuple(slice(start, stop) for start, stop in self.bounds)] = self.mask
        return full

    @property
    def voxels(self) -> int:
        return int(self.mask.sum())

    @property
    def volume_ml(self) -> float:
        return float(self.voxels * np.prod(self.spacing_zyx) / 1000.0)

    @property
    def empty(self) -> bool:
        return self.voxels == 0


@dataclass
class TrackingResult:
    followup: LesionMask
    baseline: LesionMask | None = None
    seconds: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def volume_change_ml(self) -> float | None:
        if self.baseline is None:
            return None
        return self.followup.volume_ml - self.baseline.volume_ml

    def summary(self) -> str:
        parts = [f"follow-up: {self.followup.volume_ml:.2f} ml ({self.followup.voxels} voxels)"]
        if self.baseline is not None:
            parts.insert(0, f"baseline: {self.baseline.volume_ml:.2f} ml ({self.baseline.voxels} voxels)")
            change = self.volume_change_ml
            relative = ""
            if self.baseline.volume_ml > 0:
                relative = f", {100.0 * change / self.baseline.volume_ml:+.1f}%"
            parts.append(f"change: {change:+.2f} ml{relative}")
        return " | ".join(parts)


class TrackingEngine:
    def __init__(
        self,
        model_folder: str | Path,
        folds: Sequence[int | str] | None = None,
        device: str | torch.device = "cuda",
        disable_tta: bool = True,
        tile_step_size: float = 0.5,
        use_gaussian: bool = True,
        checkpoint_name: str = CHECKPOINT_NAME,
        sigma: float = 1.0,
        cache_size: int = 4,
        gpu_cache_bytes: int | None = None,
        compile: bool = False,
    ) -> None:
        # opt-in torch.compile of the network (see longiseg's LongiSeg_compile env var).
        self.compile = compile
        self.model_folder = locate_model_folder(model_folder)
        # no folds given means every fold the folder actually ships
        self.folds = tuple(folds) if folds else tuple(available_folds(self.model_folder))
        if not self.folds:
            raise RuntimeError(f"{self.model_folder} contains no fold_*/{CHECKPOINT_NAME}.")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch reports no CUDA device. Pick 'cpu' instead.")
        self.disable_tta = disable_tta
        self.tile_step_size = tile_step_size
        self.use_gaussian = use_gaussian
        self.checkpoint_name = checkpoint_name
        self.sigma = sigma
        self.cache_size = cache_size
        self._requested_gpu_cache_bytes = gpu_cache_bytes
        self.gpu_cache_bytes = gpu_cache_bytes if self.device.type == "cuda" else None

        self._predictor = None
        self._cache: dict[tuple, PreprocessedImage] = {}
        self._cache_bytes = 0
        # the widget touches the engine from both the GUI thread and its workers
        self._state_lock = threading.RLock()

    @property
    def is_loaded(self) -> bool:
        return self._predictor is not None

    def load(self, progress: Progress | None = None):
        with self._state_lock:
            if self._predictor is not None:
                return self._predictor
            return self._load(progress)

    def _load(self, progress: Progress | None):
        from longiseg.inference.predict_from_raw_data_longi import LongiSegPredictor

        _log(progress, f"Loading {self.model_folder.name} (folds {list(self.folds)}) on {self.device}...")
        predictor = LongiSegPredictor(
            tile_step_size=self.tile_step_size,
            use_gaussian=self.use_gaussian,
            use_mirroring=not self.disable_tta,
            perform_everything_on_device=True,
            device=self.device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        previous_compile_env = os.environ.get("LongiSeg_compile")
        if self.compile:
            os.environ["LongiSeg_compile"] = "true"
        try:
            predictor.initialize_from_trained_model_folder(
                str(self.model_folder), use_folds=self.folds, checkpoint_name=self.checkpoint_name
            )
        finally:
            # don't leak this into other engines/processes sharing the interpreter
            if previous_compile_env is None:
                os.environ.pop("LongiSeg_compile", None)
            else:
                os.environ["LongiSeg_compile"] = previous_compile_env
        self._configure_resampling(predictor)
        if len(predictor.list_of_parameters) == 1:
            # Interactive tracking uses a single fold.
            predictor.network.to(self.device)
            predictor._longitrack_single_fold_on_device = True
        self._predictor = predictor
        self.refresh_gpu_cache_limit()
        _log(progress, f"Model ready (patch size {list(predictor.configuration_manager.patch_size)}).")
        return predictor

    def refresh_gpu_cache_limit(self) -> None:
        """Budget scan tensors from VRAM free after resident model weights are loaded."""
        if self.device.type != "cuda":
            return
        free_bytes, _ = torch.cuda.mem_get_info(self.device)
        # Keep room for the two model forwards, TTA intermediates, and restoration.
        # Model parameters are not cache entries and are therefore never evicted.
        safe_bytes = max(0, free_bytes - 4 * 1_024 * 1_024 * 1_024)
        if self._requested_gpu_cache_bytes is not None:
            safe_bytes = min(safe_bytes, self._requested_gpu_cache_bytes)
        self.gpu_cache_bytes = safe_bytes

    def _configure_resampling(self, predictor) -> None:
        # the torch resampling implementation, as tracking_inference.predict sets it up
        configuration_manager = predictor.configuration_manager
        for name, is_seg, memory_efficient in (
            ("data", False, False),
            ("seg", True, True),
            ("probabilities", False, False),
        ):
            configuration_manager.configuration[f"resampling_fn_{name}"] = "resample_torch_fornnunet"
            configuration_manager.configuration[f"resampling_fn_{name}_kwargs"] = {
                "is_seg": is_seg,
                "force_separate_z": False,
                "memefficient_seg_resampling": memory_efficient,
                # resampling probabilities on the GPU; the rest of the preprocessor is NumPy
                "device": self.device if name == "probabilities" else torch.device("cpu"),
            }
            # the properties are lru_cached; drop anything resolved before we got here
            prop = getattr(type(configuration_manager), f"resampling_fn_{name}", None)
            if prop is not None and hasattr(prop.fget, "cache_clear"):
                prop.fget.cache_clear()

    @property
    def predictor(self):
        return self.load()

    @property
    def label_names(self) -> dict:
        return self.predictor.dataset_json.get("labels", {})

    @property
    def file_ending(self) -> str:
        return self.predictor.dataset_json.get("file_ending", ".nii.gz")

    @property
    def num_image_channels(self) -> int:
        dataset_json = self.predictor.dataset_json
        return len(dataset_json.get("channel_names") or dataset_json["modality"])

    def warm_up(self, progress: Progress | None = None) -> None:
        """Pay one-time CUDA kernel setup by running a dummy prediction through _predict.

        A bare ``predictor.network(dummy)`` forward warms only the network module, not the
        sliding-window slicer and Gaussian importance map the real call builds, so the
        warm-up goes through the same _predict path a lesion does.
        """
        predictor = self.load(progress)
        patch = tuple(int(value) for value in predictor.configuration_manager.patch_size)
        channels = self.num_image_channels
        dummy = torch.zeros((channels, *patch), dtype=torch.float32, device=self.device)
        centre = [size // 2 for size in patch]
        _log(progress, "Warming CUDA with one dummy prediction through the real code path...")
        started = time.perf_counter()
        probabilities = self._predict(
            PreprocessedImage(
                path=Path("warm-up"), data=dummy, seg=None, properties={}, original_shape=patch,
                spacing_zyx=(1.0, 1.0, 1.0),
            ),
            centre,
            PreprocessedImage(
                path=Path("warm-up"), data=dummy, seg=None, properties={}, original_shape=patch,
                spacing_zyx=(1.0, 1.0, 1.0),
            ),
            centre,
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        del probabilities, dummy
        _log(progress, f"Segmentation model ready ({time.perf_counter() - started:.1f} s warmup).")

    @staticmethod
    def _signature(path: Path) -> tuple:
        stat = path.stat()
        return (str(path.resolve()), stat.st_mtime_ns, stat.st_size)

    @staticmethod
    def _image_nbytes(image: PreprocessedImage) -> int:
        tensors = (image.data, image.seg)
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors if tensor is not None)

    def _remember(self, key: tuple, image: PreprocessedImage) -> None:
        image_bytes = self._image_nbytes(image) if image.data.device.type == "cuda" else 0
        with self._state_lock:
            # Retain every recently loaded preprocessed scan even if VRAM is full.
            while (
                image.data.device.type == "cuda"
                and self.gpu_cache_bytes is not None
                and self._cache_bytes + image_bytes > self.gpu_cache_bytes
            ):
                resident_key = next(
                    (candidate for candidate, cached in self._cache.items() if cached.data.device.type == "cuda"),
                    None,
                )
                if resident_key is None:
                    image.data = image.data.cpu()
                    if image.seg is not None:
                        image.seg = image.seg.cpu()
                    image_bytes = 0
                    break
                resident = self._cache[resident_key]
                resident_bytes = self._image_nbytes(resident)
                resident.data = resident.data.cpu()
                if resident.seg is not None:
                    resident.seg = resident.seg.cpu()
                self._cache_bytes -= resident_bytes

            while len(self._cache) >= self.cache_size:
                _, evicted = self._cache.popitem(last=False)
                if evicted.data.device.type == "cuda":
                    self._cache_bytes -= self._image_nbytes(evicted)
            self._cache[key] = image
            self._cache_bytes += image_bytes

    def _preprocess_to_cuda(self, raw: np.ndarray, properties: dict, predictor) -> tuple[torch.Tensor, dict]:
        """Run LongiSeg tracking preprocessing while retaining its resample on CUDA.

        ``LongiSegTrackingPreprocessor.run_case_npy`` accepts and returns NumPy, so
        configuring its torch resampler for CUDA still copies the full result back to
        CPU. This is its inference-relevant sequence, kept deliberately small: no
        training-label work or file-oriented pipeline is needed for an interactive
        scan. Normalization remains the upstream implementation; the final data
        resampling is the same LongiSeg torch routine but receives a CUDA tensor.
        """
        from longiseg.preprocessing.resampling.default_resampling import compute_new_shape
        from longiseg.preprocessing.resampling.resample_torch import resample_torch_fornnunet

        plans_manager = predictor.plans_manager
        configuration_manager = predictor.configuration_manager
        preprocessor = configuration_manager.preprocessor_class(verbose=False)

        data = raw.astype(np.float32)
        forward = tuple(int(axis) for axis in plans_manager.transpose_forward)
        data = data.transpose((0, *(axis + 1 for axis in forward)))
        original_spacing = [properties["spacing"][axis] for axis in forward]

        # Tracking inference has no input segmentation.
        seg = np.zeros((1, *data.shape[1:]), dtype=np.int8)
        properties["shape_before_cropping"] = data.shape[1:]
        properties["bbox_used_for_cropping"] = [[None, None] for _ in data.shape[1:]]
        properties["shape_after_cropping_and_before_resampling"] = data.shape[1:]

        target_spacing = list(configuration_manager.spacing)
        if len(target_spacing) < len(data.shape[1:]):
            target_spacing = [original_spacing[0], *target_spacing]
        new_shape = compute_new_shape(data.shape[1:], original_spacing, target_spacing)
        data = preprocessor._normalize(
            data, seg, configuration_manager, plans_manager.foreground_intensity_properties_per_channel
        )

        # CUDA input is what separates this from the upstream NumPy entry point
        data_cuda = torch.from_numpy(data).to(self.device)
        data_cuda = resample_torch_fornnunet(
            data_cuda,
            new_shape,
            original_spacing,
            target_spacing,
            is_seg=False,
            num_threads=4,
            device=self.device,
            memefficient_seg_resampling=False,
            force_separate_z=False,
        )
        return data_cuda, properties

    def preprocess(self, image: str | Path, progress: Progress | None = None) -> PreprocessedImage:
        path = Path(image).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"{path} does not exist. The tracking pipeline reads images from disk.")

        key = self._signature(path)
        with self._state_lock:
            cached = self._cache.pop(key, None)
            if cached is not None:
                self._cache[key] = cached  # refresh recency
                return cached

        predictor = self.load(progress)
        _log(progress, f"Preprocessing {path.name}...")
        started = time.perf_counter()
        raw, properties = read_scan(path)
        original_shape = tuple(raw.shape[1:])
        data_tensor, properties = self._preprocess_to_cuda(raw, properties, predictor)
        del raw
        # A loaded scan belongs to the backend cache, not the client.
        preprocessed = PreprocessedImage(
            path=path,
            data=data_tensor,
            seg=None,
            properties=properties,
            original_shape=original_shape,
            spacing_zyx=tuple(float(s) for s in properties["spacing"]),
        )

        self._remember(key, preprocessed)
        _log(
            progress,
            f"  {original_shape} @ {np.round(preprocessed.spacing_zyx, 3).tolist()} mm "
            f"-> {preprocessed.preprocessed_shape} ({time.perf_counter() - started:.1f} s)",
        )
        return preprocessed

    def preload(self, image: str | Path, progress: Progress | None = None) -> PreprocessedImage:
        """Prepare a scan once and retain its model-ready tensor in the backend cache."""
        return self.preprocess(image, progress)

    def preload_many(self, images: Sequence[str | Path], progress: Progress | None = None) -> list[PreprocessedImage]:
        """Preprocess independent scans concurrently; model inference remains serialized."""
        unique = list(dict.fromkeys(str(Path(image).expanduser()) for image in images))
        if len(unique) < 2:
            return [self.preload(image, progress) for image in unique]
        _log(progress, f"Preprocessing {len(unique)} scans concurrently...")
        with ThreadPoolExecutor(max_workers=min(2, len(unique)), thread_name_prefix="longitrack-preprocess") as pool:
            # Progress writes protocol frames to the backend client. Keep those writes
            # on this thread so concurrent workers cannot interleave frames.
            prepared = list(pool.map(lambda image: self.preload(image), unique))
        for image in prepared:
            _log(progress, f"  prepared {image.path.name}")
        return prepared

    def release_scan(self, image: str | Path) -> None:
        path = Path(image).expanduser()
        with self._state_lock:
            for key in [key for key in self._cache if key[0] == str(path.resolve())]:
                evicted = self._cache.pop(key)
                if evicted.data.device.type == "cuda":
                    self._cache_bytes -= self._image_nbytes(evicted)

    def clear_cache(self) -> None:
        with self._state_lock:
            self._cache.clear()
            self._cache_bytes = 0

    def image_geometry(self, image: str | Path) -> tuple[tuple[int, ...], tuple[float, ...]]:
        from longiseg.imageio.simpleitk_reader_writer import SimpleITKIO

        data, properties = SimpleITKIO().read_images([str(Path(image).expanduser())])
        return tuple(data.shape[1:]), tuple(float(s) for s in properties["spacing"])

    def to_preprocessed_point(self, point_zyx: Sequence[float], image: PreprocessedImage) -> list[int]:
        return index_to_preprocessed(
            point_zyx,
            image.properties,
            self.predictor.plans_manager.transpose_forward,
            self.predictor.configuration_manager.spacing,
            image.preprocessed_shape,
        )

    def _restore(
        self, probabilities: torch.Tensor, image: PreprocessedImage, progress: Progress | None = None
    ) -> np.ndarray:
        from longiseg.inference.export_prediction import (
            convert_predicted_logits_to_segmentation_with_correct_shape,
        )

        # resamples probabilities back to the original volume and argmaxes them
        _log(progress, "  restoring to original resolution...")
        started = time.perf_counter()
        predictor = self.predictor
        # the converter hands the probabilities to numpy, as nnUNet's export path does
        mask = convert_predicted_logits_to_segmentation_with_correct_shape(
            probabilities.cpu(),
            predictor.plans_manager,
            predictor.configuration_manager,
            predictor.label_manager,
            image.properties,
        )
        _log(progress, f"  restored to {image.original_shape} ({time.perf_counter() - started:.1f} s)")
        return mask

    # trilinear sampling needs one neighbour on each side of every output voxel
    _RESTORE_PATCH_MARGIN = 2

    @staticmethod
    def _scale_bounds(
        lower: Sequence[int], upper: Sequence[int], cropped_shape: Sequence[int], source_shape: Sequence[int]
    ) -> tuple[list[int], list[int]]:
        target_lower = [
            max(0, int(np.floor(start * target / source)))
            for start, target, source in zip(lower, cropped_shape, source_shape, strict=True)
        ]
        target_upper = [
            min(target, int(np.ceil(stop * target / source)))
            for stop, target, source in zip(upper, cropped_shape, source_shape, strict=True)
        ]
        return target_lower, target_upper

    def _restore_patch(
        self,
        probabilities: torch.Tensor,
        lower: Sequence[int],
        upper: Sequence[int],
        image: PreprocessedImage,
    ) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
        """Resample only the predicted patch back into the scan's index space.

        The full nnU-Net exporter is intentionally not used here: its API requires a
        scan-sized probability volume even though every voxel outside this patch is
        known background. Export calls ``LesionMask.full_mask`` later if dense data is
        actually requested.
        """
        predictor = self.predictor
        source_shape = image.preprocessed_shape
        cropped_shape = tuple(int(v) for v in image.properties["shape_after_cropping_and_before_resampling"])
        margin = self._RESTORE_PATCH_MARGIN
        padded_lower = [max(0, start - margin) for start in lower]
        padded_upper = [min(size, stop + margin) for stop, size in zip(upper, source_shape, strict=True)]

        padded = probabilities.new_zeros((2, *(pu - pl for pl, pu in zip(padded_lower, padded_upper, strict=True))))
        padded[0] = 1.0  # background everywhere, matching how the patch prediction fills the full volume
        offset = [start - pl for start, pl in zip(lower, padded_lower, strict=True)]
        inner = tuple(
            slice(o, o + (stop - start)) for o, start, stop in zip(offset, lower, upper, strict=True)
        )
        padded[(slice(None), *inner)] = probabilities

        target_lower, target_upper = self._scale_bounds(lower, upper, cropped_shape, source_shape)
        target_shape = tuple(stop - start for start, stop in zip(target_lower, target_upper, strict=True))
        if any(size <= 0 for size in target_shape):
            raise RuntimeError(f"Invalid restored patch shape {target_shape}.")

        # F.interpolate derives its coordinate scale from the local shape, so pad first
        padded_shape = padded.shape[1:]
        axes = []
        for axis in range(3):
            scale = source_shape[axis] / cropped_shape[axis]
            index = torch.arange(target_lower[axis], target_upper[axis], device=probabilities.device,
                                  dtype=probabilities.dtype)
            source_local = (index + 0.5) * scale - 0.5 - padded_lower[axis]
            # align_corners=False grid_sample normalization; inverse of its own unnormalize
            # formula, so this coordinate reads back exactly `source_local` inside `padded`
            axes.append((2.0 * source_local + 1.0) / padded_shape[axis] - 1.0)
        grid_z, grid_y, grid_x = torch.meshgrid(*axes, indexing="ij")
        grid = torch.stack((grid_x, grid_y, grid_z), dim=-1).unsqueeze(0)
        # border padding matches interpolate's own clamp-to-edge handling at the true scan
        # boundary; the margin above means it is otherwise never actually exercised
        resized = F.grid_sample(padded.unsqueeze(0), grid, mode="bilinear", align_corners=False,
                                 padding_mode="border")[0]
        mask = resized.argmax(0).to(dtype=torch.uint8)
        # Everything outside the lesion is background, so only its own box has to leave
        # the GPU and cross the socket. Done here, on the device,
        # the voxels that are kept are bit-identical to the full-patch result.
        filled = mask.nonzero()
        if filled.numel():
            lower = filled.amin(dim=0)
            upper = filled.amax(dim=0) + 1
            mask = mask[tuple(slice(int(a), int(b)) for a, b in zip(lower, upper, strict=True))]
            target_lower = [start + int(offset) for start, offset in zip(target_lower, lower, strict=True)]
            target_upper = [start + int(offset) for start, offset in zip(target_lower, upper - lower, strict=True)]
        local = mask.cpu().numpy()

        # The preprocessor works in transpose-forward order. Convert both the local
        # array and its placement back into the z/y/x index order clients display.
        forward = tuple(int(axis) for axis in predictor.plans_manager.transpose_forward)
        inverse = tuple(int(axis) for axis in np.argsort(forward))
        local = local.transpose(inverse)
        crop_lower = crop_bbox_lower_bounds(image.properties, len(forward))
        transposed_bounds = [
            (crop_lower[d] + target_lower[d], crop_lower[d] + target_upper[d]) for d in range(len(forward))
        ]
        bounds = [None] * len(forward)
        for dimension, original_axis in enumerate(forward):
            bounds[original_axis] = transposed_bounds[dimension]
        return local, tuple(bounds)

    def _predict(
        self,
        baseline: PreprocessedImage,
        baseline_point: Sequence[int],
        followup: PreprocessedImage,
        followup_point: Sequence[int],
    ) -> tuple[torch.Tensor, tuple[int, ...], tuple[int, ...]]:
        predictor = self.predictor
        try:
            probabilities = _predict_cached_patch(
                baseline.data,
                list(baseline_point),
                followup.data,
                list(followup_point),
                predictor,
                predictor.configuration_manager.patch_size,
                self.device,
                self.sigma,
            )
        except torch.cuda.OutOfMemoryError as error:
            torch.cuda.empty_cache()
            raise RuntimeError(
                "Ran out of GPU memory. These scans are large; try device='cpu', or free up the GPU and retry."
            ) from error
        # Extract the populated patch before returning. The caller deliberately does
        # not build the all-background remainder of the scan for interactive use.
        return probabilities

    def _segment(
        self,
        baseline: PreprocessedImage,
        baseline_point: Sequence[float],
        followup: PreprocessedImage,
        followup_point: Sequence[float],
        label: str,
        progress: Progress | None = None,
    ) -> LesionMask:
        baseline_pre = self.to_preprocessed_point(baseline_point, baseline)
        followup_pre = self.to_preprocessed_point(followup_point, followup)
        _log(progress, f"[{label}] predicting at {list(followup_pre)} in {followup.path.name}...")

        started = time.perf_counter()
        probabilities, lower, upper = self._predict(baseline, baseline_pre, followup, followup_pre)
        _log(progress, f"  predicted ({time.perf_counter() - started:.2f} s)")
        restore_started = time.perf_counter()
        mask, bounds = self._restore_patch(probabilities, lower, upper, followup)
        del probabilities
        _log(progress, f"  restored ({time.perf_counter() - restore_started:.3f} s)")

        result = LesionMask(
            mask=mask,
            point=snap_to_volume(followup_point, followup.original_shape),
            spacing_zyx=followup.spacing_zyx,
            scan=followup.path,
            properties=followup.properties,
            bounds=bounds,
            original_shape=followup.original_shape,
        )
        _log(progress, f"[{label}] {result.volume_ml:.2f} ml ({result.voxels} voxels)")
        return result

    def segment_baseline(
        self,
        baseline_image: str | Path,
        baseline_point: Sequence[float],
        disable_tta: bool = False,
        progress: Progress | None = None,
    ) -> LesionMask:
        # the network only works on a pair: the baseline scan is handed in as both
        image = self.preprocess(baseline_image, progress)
        predictor = self.predictor
        previous_mirroring = predictor.use_mirroring
        predictor.use_mirroring = not disable_tta
        try:
            return self._segment(image, baseline_point, image, baseline_point, "baseline", progress)
        finally:
            predictor.use_mirroring = previous_mirroring

    def track(
        self,
        baseline_image: str | Path,
        baseline_point: Sequence[float],
        followup_image: str | Path,
        followup_point: Sequence[float],
        segment_baseline: bool = True,
        disable_tta: bool = False,
        progress: Progress | None = None,
    ) -> TrackingResult:
        started = time.perf_counter()
        self.load(progress)

        predictor = self.predictor
        previous_mirroring = predictor.use_mirroring
        predictor.use_mirroring = not disable_tta
        try:
            baseline = self.preprocess(baseline_image, progress)
            followup = self.preprocess(followup_image, progress)

            # the baseline segmentation is not an input to the follow-up prediction, so the order is free.
            baseline_mask = None
            if segment_baseline:
                # LongiSeg predicts its second timepoint.
                baseline_mask = self._segment(followup, followup_point, baseline, baseline_point, "baseline", progress)
            followup_mask = self._segment(baseline, baseline_point, followup, followup_point, "follow-up", progress)
        finally:
            predictor.use_mirroring = previous_mirroring

        notes = []
        if baseline_mask is not None and baseline_mask.empty:
            notes.append("The baseline segmentation came out empty; check that the point is on the lesion.")
        if followup_mask.empty:
            notes.append("The follow-up segmentation is empty -- the lesion may have resolved, or the point is off.")

        result = TrackingResult(
            followup=followup_mask,
            baseline=baseline_mask,
            seconds=time.perf_counter() - started,
            notes=notes,
        )
        _log(progress, f"Done in {result.seconds:.1f} s. {result.summary()}")
        for note in notes:
            _log(progress, f"  note: {note}")
        return result
