

from longitrack_backend.pantrack import (
    find_pair,
    list_pairs,
)

TRACKING = {
    "PanTrack_001": [
        {
            "51": {
                "img_bl": "PanTrack_001_20220111",
                "img_fu": "PanTrack_001_20220711",
                "bl_point": [235.7, 186.0, 664.7],
                "fu_point": [278.7, 165.0, 629.3],
                "fu_point_prop": [230.6, 175.4, 652.7],
                "merged_lesions": [51],
            }
        }
    ],
    "PanTrack_002": [
        {"81": {"img_bl": "a", "img_fu": "b", "bl_point": [1.0, 2.0, 3.0], "fu_point": [4.0, 5.0, 6.0]}},
        {"81": {"img_bl": "b", "img_fu": "c", "bl_point": [7.0, 8.0, 9.0], "fu_point": [10.0, 11.0, 12.0]}},
    ],
}



def test_list_pairs_expands_multi_timepoint_patients():
    pairs = list_pairs(tracking=TRACKING)
    assert [(p.patient, p.pair_index) for p in pairs] == [
        ("PanTrack_001", 0),
        ("PanTrack_002", 0),
        ("PanTrack_002", 1),
    ]
    assert pairs[2].baseline == "b" and pairs[2].followup == "c"


def test_find_pair_picks_the_requested_timepoint():
    pairs = list_pairs(tracking=TRACKING)
    assert find_pair(pairs, "PanTrack_002", 1).baseline == "b"


