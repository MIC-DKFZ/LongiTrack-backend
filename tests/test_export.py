import json
from pathlib import Path

import numpy as np
import pytest

from longitrack_backend.export import ExportCollisionError, TrackedLesion, export_tracking
from longitrack_backend.inference import LesionMask, TrackingResult

SHAPE = (8, 16, 16)
SPACING_ZYX = (3.0, 1.0, 1.0)
PROPERTIES = {
    "sitk_stuff": {
        "spacing": (1.0, 1.0, 3.0),
        "origin": (10.0, -5.0, 2.0),
        "direction": (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    },
    "spacing": list(SPACING_ZYX),
}


def mask(centre, scan):
    array = np.zeros(SHAPE, dtype=np.uint8)
    array[centre[0], centre[1] - 1 : centre[1] + 2, centre[2] - 1 : centre[2] + 2] = 1
    return LesionMask(array, list(centre), SPACING_ZYX, scan=Path(scan), properties=PROPERTIES)


def lesion(number, baseline_centre, followup_centre):
    result = TrackingResult(
        followup=mask(followup_centre, "fu_scan.nii.gz"),
        baseline=mask(baseline_centre, "bl_scan.nii.gz"),
    )
    return TrackedLesion(number, list(baseline_centre), list(followup_centre), result, list(followup_centre))


def test_export_writes_one_mask_per_lesion_per_timepoint(tmp_path):
    written = export_tracking(tmp_path, [lesion(1, (4, 8, 8), (5, 9, 7))])
    names = sorted(p.name for p in written)
    assert names[-1] == "inference_meta.json"
    assert any(name.endswith("_bl_1.nii.gz") for name in names)
    assert any(name.endswith("_fu_1.nii.gz") for name in names)
    assert all(p.is_file() for p in written)


def test_inference_meta_is_a_list_of_pairs_with_their_lesions(tmp_path):
    written = export_tracking(tmp_path, [lesion(1, (4, 8, 8), (5, 9, 7))])
    meta = json.loads((tmp_path / "inference_meta.json").read_text())

    assert isinstance(meta, list)
    assert len(meta) == 1
    case = meta[0]
    assert case["baseline_scan"] == "bl_scan"
    assert case["followup_scan"] == "fu_scan"
    assert set(case["lesions"]) == {"1"}
    point = case["lesions"]["1"]
    assert point["bl_point"] == [4.0, 8.0, 8.0]
    assert point["fu_point_corrected"] == [5.0, 9.0, 7.0]
    assert point["fu_point_propagated"] == [5.0, 9.0, 7.0]
    assert set(point) == {"bl_point", "fu_point_corrected", "fu_point_propagated"}
    assert written[-1].name == "inference_meta.json"



def test_a_later_export_call_merges_into_the_metadata_file(tmp_path):
    # exporting more pairs (or lesions) into the same folder over the course of a session
    # must not lose an earlier call's metadata -- each pair's own entry is updated, not
    # the whole file replaced, so annotating several scan pairs one after another and
    # exporting each keeps every one of them in inference_meta.json
    export_tracking(tmp_path, [lesion(1, (4, 8, 8), (5, 9, 7))], patient="patient_007")
    export_tracking(tmp_path, [lesion(2, (4, 8, 8), (5, 9, 7))], patient="patient_007", timepoints=["followup"])
    meta = json.loads((tmp_path / "inference_meta.json").read_text())

    assert len(meta) == 1
    assert set(meta[0]["lesions"]) == {"1", "2"}


def test_a_different_pairs_export_becomes_its_own_metadata_entry(tmp_path):
    export_tracking(tmp_path, [lesion(1, (4, 8, 8), (5, 9, 7))])
    other = lesion(1, (2, 4, 4), (2, 5, 5))
    other.result.baseline.scan = Path("other_bl_scan.nii.gz")
    other.result.followup.scan = Path("other_fu_scan.nii.gz")
    export_tracking(tmp_path, [other])

    meta = json.loads((tmp_path / "inference_meta.json").read_text())
    assert len(meta) == 2
    assert {case["baseline_scan"] for case in meta} == {"bl_scan", "other_bl_scan"}


def test_exporting_the_same_lesion_twice_refuses_to_overwrite(tmp_path):
    export_tracking(tmp_path, [lesion(1, (4, 8, 8), (5, 9, 7))])
    meta_before = (tmp_path / "inference_meta.json").read_text()

    with pytest.raises(ExportCollisionError, match="already exist"):
        export_tracking(tmp_path, [lesion(1, (1, 2, 2), (1, 3, 3))])

    # nothing from the failed call was written, not even a partial change to the metadata
    assert (tmp_path / "inference_meta.json").read_text() == meta_before


def test_channel_suffix_is_dropped_from_the_identifier(tmp_path):
    # images come in as {identifier}_0000.nii.gz; LongiSeg keys everything on {identifier}
    item = lesion(1, (4, 8, 8), (5, 9, 7))
    item.result.baseline.scan = Path("PanTrack_003_20211006_0000.nii.gz")
    item.result.followup.scan = Path("PanTrack_003_20211011_0000.nii.gz")

    written = export_tracking(tmp_path, [item])
    names = sorted(p.name for p in written)
    assert any(name.startswith("PanTrack_003_20211006_bl_") for name in names)
    assert any(name.startswith("PanTrack_003_20211011_fu_") for name in names)
    meta = json.loads((tmp_path / "inference_meta.json").read_text())
    assert meta[0]["baseline_scan"] == "PanTrack_003_20211006"
    assert meta[0]["followup_scan"] == "PanTrack_003_20211011"


def test_export_refuses_an_empty_run(tmp_path):
    with pytest.raises(ValueError, match="Nothing to export"):
        export_tracking(tmp_path, [])

