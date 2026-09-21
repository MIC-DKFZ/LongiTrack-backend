from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_REPO_ID = os.environ.get("LONGITRACK_PANTRACK_REPO", "mrokuss/PanTrack")
# five annotated lesions, and one of the smaller multi-lesion downloads
DEFAULT_PATIENT = os.environ.get("LONGITRACK_PANTRACK_PATIENT", "PanTrack_029")
DEFAULT_PAIR_INDEX = int(os.environ.get("LONGITRACK_PANTRACK_PAIR", "2"))
FILE_ENDING = ".nii.gz"

Progress = Callable[[str], None]


class PanTrackError(RuntimeError):
    pass


@dataclass
class ScanPair:
    patient: str
    pair_index: int
    baseline: str
    followup: str
    lesions: dict = field(default_factory=dict)
    baseline_image: Path | None = None
    followup_image: Path | None = None

    @property
    def label(self) -> str:
        return f"{self.patient} [{self.pair_index}]  {self.baseline} -> {self.followup}"


def _log(progress: Progress | None, message: str) -> None:
    if progress is not None:
        progress(message)


def image_file(identifier: str, channel: int = 0) -> str:
    return f"images/{identifier}_{channel:04d}{FILE_ENDING}"


def _fetch(repo_id: str, filename: str, revision: str | None, token: str | None) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id, filename, repo_type="dataset", revision=revision, token=token))


def load_tracking(repo_id: str = DEFAULT_REPO_ID, revision: str | None = None, token: str | None = None) -> dict:
    import json

    try:
        path = _fetch(repo_id, "tracking.json", revision, token)
    except Exception as error:
        raise PanTrackError(
            f"Could not reach the {repo_id} dataset on the Hugging Face Hub: {error}\n"
            f"PanTrack is public; if you are behind a proxy set HF_ENDPOINT or HTTPS_PROXY, or open your own scans "
            f"with 'Open...' instead."
        ) from error
    return json.loads(path.read_text())


def _is_pair_spec(scan_dict: dict) -> bool:
    return "img_bl" in scan_dict and "img_fu" in scan_dict


def _scan_pair_of(patient: str, scan_dict: dict) -> tuple[str, str]:
    # vendored from longiseg.tracking.tracking_file: a patient's tracking entry is either
    # a single {lesion: info} dict or a list of them, one per baseline/follow-up pair
    if _is_pair_spec(scan_dict):
        return scan_dict["img_bl"], scan_dict["img_fu"]
    images = {(info.get("img_bl"), info.get("img_fu")) for info in scan_dict.values()}
    if len(images) != 1 or None in next(iter(images)):
        raise PanTrackError(
            f"Patient {patient} has an entry whose lesions do not all share one img_bl/img_fu pair: "
            f"{sorted(images)}."
        )
    return next(iter(images))


def list_pairs(
    repo_id: str = DEFAULT_REPO_ID, revision: str | None = None, token: str | None = None, tracking: dict | None = None
) -> list[ScanPair]:
    tracking = tracking if tracking is not None else load_tracking(repo_id, revision, token)
    pairs = []
    for patient, patient_tracking in tracking.items():
        entries = patient_tracking if isinstance(patient_tracking, list) else [patient_tracking]
        for pair_index, scan_dict in enumerate(entries):
            baseline, followup = _scan_pair_of(patient, scan_dict)
            pairs.append(ScanPair(patient, pair_index, baseline, followup, dict(scan_dict)))
    return pairs


def find_pair(pairs: Sequence[ScanPair], patient: str | None = None, pair_index: int | None = None) -> ScanPair:
    patient = patient or DEFAULT_PATIENT
    pair_index = DEFAULT_PAIR_INDEX if pair_index is None else pair_index
    for pair in pairs:
        if pair.patient == patient and pair.pair_index == pair_index:
            return pair
    available = sorted({p.patient for p in pairs})
    raise PanTrackError(
        f"{patient} pair {pair_index} is not in the dataset. {len(available)} patients are available, "
        f"for example {', '.join(available[:5])}."
    )


def pair_size_mb(pair: ScanPair, repo_id: str = DEFAULT_REPO_ID, token: str | None = None) -> float | None:
    from huggingface_hub import HfApi

    try:
        info = HfApi(token=token).dataset_info(repo_id, files_metadata=True)
    except Exception:
        return None
    sizes = {sibling.rfilename: (sibling.size or 0) for sibling in info.siblings}
    total = sizes.get(image_file(pair.baseline), 0) + sizes.get(image_file(pair.followup), 0)
    return total / 1e6 if total else None


def download_pair(
    patient: str | None = None,
    pair_index: int | None = None,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str | None = None,
    token: str | None = None,
    pairs: Sequence[ScanPair] | None = None,
    progress: Progress | None = None,
) -> ScanPair:
    pairs = pairs if pairs is not None else list_pairs(repo_id, revision, token)
    pair = find_pair(pairs, patient, pair_index)

    size = pair_size_mb(pair, repo_id, token)
    estimate = f" (about {size:.0f} MB, cached afterwards)" if size else ""
    _log(progress, f"Fetching {pair.patient} {pair.baseline} -> {pair.followup}{estimate}...")

    pair.baseline_image = _fetch(repo_id, image_file(pair.baseline), revision, token)
    _log(progress, f"  {pair.baseline_image.name}")
    pair.followup_image = _fetch(repo_id, image_file(pair.followup), revision, token)
    _log(progress, f"  {pair.followup_image.name}")

    # dataset metadata, not a prompt: where to click is the user's call
    _log(progress, f"Ready. The dataset annotates {len(pair.lesions)} lesion(s) in this pair: {sorted(pair.lesions)}.")
    return pair
