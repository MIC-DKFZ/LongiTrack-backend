from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

DEFAULT_REPO_ID = os.environ.get("LONGITRACK_HF_REPO", "ykirchhoff/LongiTrack")
DEFAULT_REVISION = os.environ.get("LONGITRACK_HF_REVISION") or None
CHECKPOINT_NAME = "checkpoint_final.pth"
REQUIRED_FILES = ("dataset.json", "plans.json")
_FOLD_PATTERN = re.compile(r"^fold_(\d+|all)$")
_VERSION_PATTERN = re.compile(r"(\d+(?:\.\d+)*)\s*$")

Progress = Callable[[str], None]


class ModelNotFoundError(RuntimeError):
    """Raised when no usable LongiSeg tracking model folder could be produced."""


def _log(progress: Progress | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _fold_name(fold: int | str) -> str:
    return f"fold_{fold}"


def allow_patterns(folds: Iterable[int | str] | None, prefix: str = "") -> list[str]:
    base = f"{prefix}/" if prefix else ""
    patterns = [f"{base}{name}" for name in REQUIRED_FILES] + [f"{base}*.md", f"{base}*.json"]
    if folds is None:
        patterns.append(f"{base}fold_*/{CHECKPOINT_NAME}")
    else:
        patterns += [f"{base}{_fold_name(fold)}/{CHECKPOINT_NAME}" for fold in folds]
    return patterns


def available_folds(folder: str | Path) -> list[int | str]:
    folder = Path(folder)
    folds: list[int | str] = []
    for child in sorted(folder.iterdir()) if folder.is_dir() else []:
        match = _FOLD_PATTERN.match(child.name)
        if match and (child / CHECKPOINT_NAME).is_file():
            folds.append(int(match.group(1)) if match.group(1).isdigit() else match.group(1))
    return sorted(folds, key=lambda f: (isinstance(f, str), f))


def _version_key(name: str) -> tuple:
    """Sort key for a folder name's trailing version, e.g. 'LongiTrack_v2.0' -> (2, 0)."""
    match = _VERSION_PATTERN.search(name)
    return tuple(int(part) for part in match.group(1).split(".")) if match else (-1,)


def _is_model_folder(folder: Path) -> bool:
    return all((folder / name).is_file() for name in REQUIRED_FILES) and bool(available_folds(folder))


def available_versions(root: str | Path) -> list[str]:
    """Model folders one level under `root`, newest version first."""
    root = Path(root).expanduser()
    if not root.is_dir():
        return []
    children = [child for child in root.iterdir() if child.is_dir() and _is_model_folder(child)]
    return [child.name for child in sorted(children, key=lambda child: _version_key(child.name), reverse=True)]


def locate_model_folder(root: str | Path, version: str | None = None) -> Path:
    root = Path(root).expanduser()
    if not root.exists():
        raise ModelNotFoundError(f"{root} does not exist.")

    if _is_model_folder(root):
        return root

    names = available_versions(root)
    if not names:
        raise ModelNotFoundError(
            f"{root} does not look like a LongiSeg model folder. Expected {', '.join(REQUIRED_FILES)} and at least "
            f"one fold_*/{CHECKPOINT_NAME}, either directly inside it or one level down."
        )
    if version is None:
        return root / names[0]  # newest first

    matches = [name for name in names if version == name or version in _VERSION_PATTERN.findall(name)]
    if not matches:
        raise ModelNotFoundError(f"Version {version!r} not found under {root}. Available: {', '.join(names)}.")
    return root / matches[0]


def validate_model_folder(
    folder: str | Path, folds: Sequence[int | str] | None = None, version: str | None = None
) -> Path:
    folder = locate_model_folder(folder, version=version)
    present = available_folds(folder)
    if folds is None:
        return folder  # locate_model_folder already refused a folder with no folds
    missing = [fold for fold in folds if fold not in present]
    if missing:
        raise ModelNotFoundError(
            f"{folder} has folds {present} but {missing} were requested. Either download the missing folds or pick "
            f"from the available ones."
        )
    return folder


def _remote_version_folder(repo_id: str, version: str | None, revision: str | None, token: str | None) -> str:
    """Which top-level folder in the repo holds the model, without downloading anything. '' means the repo root."""
    from huggingface_hub import HfApi

    files = set(HfApi(token=token).list_repo_files(repo_id, revision=revision))

    def has_model(prefix: str) -> bool:
        base = f"{prefix}/" if prefix else ""
        if not all(f"{base}{name}" in files for name in REQUIRED_FILES):
            return False
        fold_pattern = re.compile(rf"^{re.escape(base)}fold_(\d+|all)/{re.escape(CHECKPOINT_NAME)}$")
        return any(fold_pattern.match(f) for f in files)

    if has_model(""):
        return ""

    top_level = {f.split("/", 1)[0] for f in files if "/" in f}
    names = sorted((name for name in top_level if has_model(name)), key=_version_key, reverse=True)
    if not names:
        raise ModelNotFoundError(f"{repo_id} does not look like a LongiSeg model repository.")
    if version is None:
        return names[0]  # newest first
    matches = [name for name in names if version == name or version in _VERSION_PATTERN.findall(name)]
    if not matches:
        raise ModelNotFoundError(f"Version {version!r} not found in {repo_id}. Available: {', '.join(names)}.")
    return matches[0]


def download_model(
    repo_id: str = DEFAULT_REPO_ID,
    folds: Sequence[int | str] | None = None,
    version: str | None = None,
    revision: str | None = DEFAULT_REVISION,
    token: str | None = None,
    cache_dir: str | Path | None = None,
    progress: Progress | None = None,
) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:  # pragma: no cover - declared dependency
        raise ModelNotFoundError(
            "huggingface_hub is required to download the model. Install it with `uv sync`, or point the viewer at a "
            "local model folder instead."
        ) from error

    try:
        prefix = _remote_version_folder(repo_id, version, revision, token)
    except ModelNotFoundError:
        raise
    except Exception as error:
        raise ModelNotFoundError(f"Could not inspect {repo_id} on the Hugging Face Hub: {error}") from error

    wanted = "every fold" if folds is None else f"folds {list(folds)}"
    where = f" ({prefix})" if prefix else ""
    _log(progress, f"Fetching {repo_id}{where} ({wanted}) from the Hugging Face Hub...")
    try:
        snapshot = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=allow_patterns(folds, prefix),
            token=token,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )
    except Exception as error:
        raise ModelNotFoundError(
            f"Could not download {repo_id} from the Hugging Face Hub: {error}\n"
            f"If the repository is private, log in with `huggingface-cli login` or set $HF_TOKEN. To work from a local "
            f"checkpoint instead, set $LONGITRACK_MODEL_DIR or select 'Local folder' in the widget."
        ) from error

    folder = validate_model_folder(Path(snapshot) / prefix if prefix else snapshot, folds)
    _log(progress, f"Model ready at {folder}")
    return folder


def _split_version(source: str) -> tuple[str, str | None]:
    """Pull a trailing '@1.0'-style version selector off a path or repo id."""
    if "@" in source:
        base, _, tail = source.rpartition("@")
        if re.fullmatch(r"\d+(?:\.\d+)*", tail):
            return base, tail
    return source, None


# an existing folder wins, then $LONGITRACK_MODEL_DIR, then the argument or
# $LONGITRACK_HF_REPO or DEFAULT_REPO_ID read as a Hub repo id. Either can end in
# '@<version>' to pick a specific model folder when more than one is available.
def resolve_model_folder(
    source: str | Path | None = None,
    folds: Sequence[int | str] | None = None,
    version: str | None = None,
    revision: str | None = DEFAULT_REVISION,
    token: str | None = None,
    progress: Progress | None = None,
) -> Path:
    if source is not None:
        source = str(source).strip()
    if source and version is None:
        source, version = _split_version(source)
    if source:
        candidate = Path(source).expanduser()
        if candidate.exists():
            _log(progress, f"Using local model folder {candidate}")
            return validate_model_folder(candidate, folds, version=version)

    local = os.environ.get("LONGITRACK_MODEL_DIR")
    if not source and local:
        _log(progress, f"Using $LONGITRACK_MODEL_DIR ({local})")
        return validate_model_folder(local, folds, version=version)

    repo_id = source or DEFAULT_REPO_ID
    if "/" not in repo_id:
        raise ModelNotFoundError(
            f"{repo_id!r} is neither an existing folder nor a Hugging Face repo id of the form 'owner/name'."
        )
    return download_model(repo_id, folds=folds, version=version, revision=revision, token=token, progress=progress)


def describe_model(folder: str | Path) -> dict:
    import json

    folder = locate_model_folder(folder)
    dataset = json.loads((folder / "dataset.json").read_text())
    plans = json.loads((folder / "plans.json").read_text())
    configuration = plans["configurations"][next(iter(plans["configurations"]))]
    return {
        "folder": str(folder),
        "dataset": dataset.get("name", "unknown"),
        "labels": dataset.get("labels", {}),
        "file_ending": dataset.get("file_ending", ".nii.gz"),
        "folds": available_folds(folder),
        "patch_size": configuration.get("patch_size"),
        "spacing": configuration.get("spacing"),
    }


MODEL_CARD = """\
---
license: apache-2.0
library_name: longiseg
tags:
  - medical-imaging
  - image-segmentation
  - longitudinal
  - lesion-tracking
  - nnunet
pipeline_tag: image-segmentation
---

# LongiSeg tracking model

Point-prompted longitudinal lesion tracking. Given a baseline scan, a point on a
lesion in it, a follow-up scan and a point on the same lesion there, the network
segments that lesion in the follow-up scan.

Trained with `{trainer}` on `{dataset}`.

## Layout

This repository is a LongiSeg / nnU-Net model folder and can be passed to
`LongiSegPredictor.initialize_from_trained_model_folder` after downloading:

```
dataset.json
plans.json
fold_0/checkpoint_final.pth
```

## Usage

From Python:

```python
from longitrack_backend.inference import TrackingEngine
from longitrack_backend.model import resolve_model_folder

engine = TrackingEngine(resolve_model_folder("{repo_id}"))
result = engine.track(
    baseline_image="baseline.nii.gz",
    baseline_point=(58, 212, 190),      # (z, y, x) voxel index
    followup_image="followup.nii.gz",
    followup_point=(61, 208, 194),
)
```

Batched, from the command line, see `LongiSeg_prepare_tracking` and
`LongiSeg_predict_tracking` in [LongiSeg](https://github.com/MIC-DKFZ/LongiSeg).

## Citation

{citation}
"""


def push_model_folder(
    local_folder: str | Path,
    repo_id: str = DEFAULT_REPO_ID,
    path_in_repo: str | None = None,
    private: bool = True,
    token: str | None = None,
    revision: str | None = None,
    commit_message: str = "Upload LongiSeg tracking model",
    write_model_card: bool = True,
    progress: Progress | None = None,
) -> str:
    import json

    from huggingface_hub import HfApi

    folder = locate_model_folder(local_folder)
    folds = available_folds(folder)
    if not folds:
        raise ModelNotFoundError(f"{folder} contains no fold_*/{CHECKPOINT_NAME}, nothing to upload.")

    api = HfApi(token=token)
    _log(progress, f"Creating/updating {repo_id}...")
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

    if write_model_card and not (folder / "README.md").is_file():
        dataset = json.loads((folder / "dataset.json").read_text())
        card = MODEL_CARD.format(
            trainer=folder.name.split("__")[0],
            dataset=dataset.get("name", "an unpublished dataset"),
            repo_id=repo_id,
            citation=dataset.get("citation", "See the LongiSeg repository."),
        )
        (folder / "README.md").write_text(card)
        _log(progress, "Wrote a model card.")

    destination = f" to {path_in_repo}" if path_in_repo else ""
    _log(progress, f"Uploading folds {folds} from {folder}{destination}...")
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(folder),
        path_in_repo=path_in_repo or ".",
        repo_type="model",
        revision=revision,
        commit_message=commit_message,
        allow_patterns=allow_patterns(folds) + ["README.md"],
    )
    url = f"https://huggingface.co/{repo_id}"
    _log(progress, f"Done: {url}")
    return url
