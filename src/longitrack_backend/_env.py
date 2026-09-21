from __future__ import annotations

import os
from pathlib import Path

CACHE_DIR = Path(os.environ.get("LONGITRACK_CACHE", Path.home() / ".cache" / "longitrack")).expanduser()


def prepare_longiseg_env() -> None:
    # longiseg.paths warns whenever these are unset, regardless of any legacy nnUNet_*
    # fallback being present. This package never trains or preprocesses a full dataset,
    # so the real value never matters -- always point them at our cache, unconditionally.
    for variable, subfolder in (
        ("LongiSeg_raw", "raw"),
        ("LongiSeg_preprocessed", "preprocessed"),
        ("LongiSeg_results", "results"),
    ):
        if os.environ.get(variable):
            continue
        folder = CACHE_DIR / subfolder
        folder.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(folder)

    os.environ.setdefault("LongiSeg_weights", str(CACHE_DIR / "registration_weights"))
