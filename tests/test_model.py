import json

import pytest

from longitrack_backend.model import (
    CHECKPOINT_NAME,
    ModelNotFoundError,
    available_folds,
    locate_model_folder,
    resolve_model_folder,
)

PLANS = {
    "configurations": {
        "3d_fullres": {"patch_size": [112, 224, 224], "spacing": [3.0, 0.85, 0.85]},
    }
}
DATASET = {"name": "Demo", "labels": {"background": 0, "tumor": 1}, "file_ending": ".nii.gz"}


def make_model(root, folds=(0,), name="LongiSegTrainerTracking__Plans__3d_fullres"):
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "plans.json").write_text(json.dumps(PLANS))
    (folder / "dataset.json").write_text(json.dumps(DATASET))
    for fold in folds:
        fold_dir = folder / f"fold_{fold}"
        fold_dir.mkdir()
        (fold_dir / CHECKPOINT_NAME).write_bytes(b"not really a checkpoint")
    return folder


def test_locate_accepts_the_folder_itself(tmp_path):
    folder = make_model(tmp_path)
    assert locate_model_folder(folder) == folder


def test_locate_looks_one_level_down(tmp_path):
    folder = make_model(tmp_path)
    assert locate_model_folder(tmp_path) == folder


def test_locate_rejects_an_unrelated_folder(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ModelNotFoundError, match="does not look like"):
        locate_model_folder(tmp_path / "empty")


def test_available_folds_ignores_folds_without_a_checkpoint(tmp_path):
    folder = make_model(tmp_path, folds=(0, 2))
    (folder / "fold_1").mkdir()
    assert available_folds(folder) == [0, 2]


def test_resolve_prefers_an_existing_local_folder(tmp_path):
    folder = make_model(tmp_path)
    assert resolve_model_folder(str(folder), folds=(0,)) == folder


def test_locate_picks_the_newest_version_by_default(tmp_path):
    make_model(tmp_path, name="LongiTrack_v1.0")
    v2 = make_model(tmp_path, name="LongiTrack_v2.0")
    assert locate_model_folder(tmp_path) == v2


def test_locate_can_pick_an_explicit_version(tmp_path):
    v1 = make_model(tmp_path, name="LongiTrack_v1.0")
    make_model(tmp_path, name="LongiTrack_v2.0")
    assert locate_model_folder(tmp_path, version="1.0") == v1


def test_locate_rejects_an_unknown_version(tmp_path):
    make_model(tmp_path, name="LongiTrack_v1.0")
    with pytest.raises(ModelNotFoundError, match="Available: LongiTrack_v1.0"):
        locate_model_folder(tmp_path, version="9.9")


def test_resolve_model_folder_parses_an_at_version_suffix(tmp_path):
    v1 = make_model(tmp_path, name="LongiTrack_v1.0")
    make_model(tmp_path, name="LongiTrack_v2.0")
    assert resolve_model_folder(f"{tmp_path}@1.0") == v1

