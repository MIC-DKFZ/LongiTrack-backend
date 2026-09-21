from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import _env

_env.prepare_longiseg_env()

from .inference import TrackingResult  # noqa: E402

Progress = Callable[[str], None]

META_FILENAME = "inference_meta.json"


@dataclass
class TrackedLesion:
    lesion: int
    baseline_point: Sequence[float]  # (z, y, x) in the baseline scan
    followup_point: Sequence[float]  # (z, y, x) in the follow-up scan
    result: TrackingResult
    propagated_point: Sequence[float] | None = None


_CHANNEL_SUFFIX = re.compile(r"_\d{4}$")


def _scan_name(path: Path | None, file_ending: str) -> str:
    # nnU-Net names images {identifier}_{channel:04d}{ending}; keep the identifier
    if path is None:
        return "scan"
    name = path.name
    name = name[: -len(file_ending)] if name.endswith(file_ending) else path.stem
    return _CHANNEL_SUFFIX.sub("", name)


def _write(mask: np.ndarray, path: Path, properties: dict, progress: Progress | None) -> Path:
    from longiseg.imageio.simpleitk_reader_writer import SimpleITKIO

    SimpleITKIO().write_seg(mask, str(path), properties)
    if progress:
        progress(f"  {path.name}")
    return path


def _point_fields(item: TrackedLesion) -> dict:
    return {
        "bl_point": [float(value) for value in item.baseline_point],
        "fu_point_corrected": [float(value) for value in item.followup_point],
        "fu_point_propagated": (
            [float(value) for value in item.propagated_point] if item.propagated_point is not None else None
        ),
    }


class ExportCollisionError(FileExistsError):
    pass


def _load_existing_meta(meta_path: Path) -> dict[tuple, dict]:
    # keyed the same way `cases` below is, so a case from a previous export call (a
    # different pair, or the same one re-exported) merges instead of getting lost
    if not meta_path.exists():
        return {}
    try:
        entries = json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(entries, list):
        return {}
    existing = {}
    for entry in entries:
        if isinstance(entry, dict) and "baseline_scan" in entry and "followup_scan" in entry:
            existing[(entry["baseline_scan"], entry["followup_scan"])] = entry
    return existing


def export_tracking(
    folder: str | Path,
    lesions: Sequence[TrackedLesion],
    file_ending: str = ".nii.gz",
    patient: str = "case_000",
    timepoints: Sequence[str] = ("baseline", "followup"),
    progress: Progress | None = None,
) -> list[Path]:
    if not lesions:
        raise ValueError("Nothing to export, run a segmentation first.")

    folder = Path(folder).expanduser()
    folder.mkdir(parents=True, exist_ok=True)

    wanted = set(timepoints)
    unknown = wanted - {"baseline", "followup"}
    if unknown:
        raise ValueError(f"Unknown export timepoint(s): {sorted(unknown)}")
    if not wanted:
        raise ValueError("Choose at least one timepoint to export.")

    # one metadata entry per baseline/follow-up pair
    cases: dict[tuple, dict] = {}
    case_order: list[tuple] = []
    pending_writes: list[tuple[str, np.ndarray, dict]] = []  # (filename, mask, properties)

    for item in lesions:
        baseline_scan = (
            Path(item.result.baseline.scan).resolve() if item.result.baseline and item.result.baseline.scan else None
        )
        followup_scan = Path(item.result.followup.scan).resolve() if item.result.followup.scan else None
        baseline_name = _scan_name(baseline_scan, file_ending)
        followup_name = _scan_name(followup_scan, file_ending)

        case_key = (baseline_scan, followup_scan)
        if case_key not in cases:
            cases[case_key] = {"baseline_scan": baseline_name, "followup_scan": followup_name, "lesions": {}}
            case_order.append(case_key)
        cases[case_key]["lesions"][str(item.lesion)] = _point_fields(item)

        for timepoint, mask, scan_name, suffix in (
            ("baseline", item.result.baseline, baseline_name, "bl"),
            ("followup", item.result.followup, followup_name, "fu"),
        ):
            if timepoint not in wanted or mask is None:
                continue
            filename = f"{scan_name}_{suffix}_{item.lesion}{file_ending}"
            pending_writes.append((filename, mask.full_mask(), mask.properties))

    # checked up front, before anything is written
    collisions = sorted({filename for filename, _mask, _props in pending_writes if (folder / filename).exists()})
    if collisions:
        raise ExportCollisionError(
            f"{len(collisions)} file(s) already exist in {folder} and would be overwritten: "
            f"{', '.join(collisions)}. Export to a different folder, or remove those files first."
        )

    if progress:
        progress(f"Exporting selected masks from {len(lesions)} lesion(s) to {folder}")
    written = [_write(mask, folder / filename, properties, progress) for filename, mask, properties in pending_writes]

    meta_path = folder / META_FILENAME
    merged = _load_existing_meta(meta_path)
    for key in case_order:
        meta_key = (cases[key]["baseline_scan"], cases[key]["followup_scan"])
        if meta_key in merged:
            merged[meta_key]["lesions"].update(cases[key]["lesions"])
        else:
            merged[meta_key] = cases[key]
    meta_path.write_text(json.dumps(list(merged.values()), indent=2) + "\n")
    written.append(meta_path)
    if progress:
        progress(f"  {meta_path.name}")
    return written
