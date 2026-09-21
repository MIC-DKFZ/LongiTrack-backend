from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .model import (
    DEFAULT_REPO_ID,
    ModelNotFoundError,
    describe_model,
    download_model,
    push_model_folder,
    resolve_model_folder,
)
from .pantrack import DEFAULT_PAIR_INDEX, DEFAULT_PATIENT, PanTrackError


def _folds(values) -> tuple:
    return tuple(int(v) if str(v).isdigit() else str(v) for v in values)


def _sample_pair_index(patient: str | None, pair_index: int | None) -> int | None:
    # the curated pair index belongs to the curated patient; asking for another patient
    # without an index means that patient's first pair, not index DEFAULT_PAIR_INDEX
    if pair_index is None and patient is not None:
        return 0
    return pair_index


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="longitrack-model",
        description="Get the LongiSeg tracking model onto or off this machine.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="download the model from the Hugging Face Hub")
    download.add_argument("-r", "--repo-id", default=DEFAULT_REPO_ID, help=f"default: {DEFAULT_REPO_ID}")
    download.add_argument("--folds", nargs="+", default=["0"], help="folds to fetch. Default: 0")
    download.add_argument("--version", default=None, help="which model folder to use. Default: the newest")
    download.add_argument("--revision", default=None, help="branch, tag or commit")
    download.add_argument("--token", default=None, help="Hugging Face token for private repositories")

    info = subparsers.add_parser("info", help="describe a model folder or repository")
    info.add_argument("source", nargs="?", default=None, help="local folder or repo id. Default: the resolved model")
    info.add_argument("--folds", nargs="+", default=["0"])
    info.add_argument("--version", default=None, help="which model folder to use. Default: the newest")

    sample = subparsers.add_parser("sample", help="download a PanTrack scan pair")
    sample.add_argument(
        "-p", "--patient", default=None, help=f"patient id. Default: {DEFAULT_PATIENT}, the curated pair"
    )
    sample.add_argument(
        "-i",
        "--pair-index",
        type=int,
        default=None,
        help=f"which consecutive pair. Default: 0 with -p, otherwise {DEFAULT_PAIR_INDEX} for the curated pair",
    )
    sample.add_argument("--list", action="store_true", help="only list the available pairs")

    upload = subparsers.add_parser("upload", help="upload a LongiSeg model folder to the Hugging Face Hub")
    upload.add_argument("folder", type=Path, help="the trainer output folder, or a directory containing it")
    upload.add_argument("-r", "--repo-id", default=DEFAULT_REPO_ID, help=f"default: {DEFAULT_REPO_ID}")
    upload.add_argument(
        "--version", default=None, help="publish as a 'LongiTrack_v<version>' folder, alongside older ones"
    )
    upload.add_argument("--public", action="store_true", help="create the repository as public")
    upload.add_argument("--token", default=None, help="Hugging Face token. Falls back to a cached login or $HF_TOKEN")
    upload.add_argument("--revision", default=None, help="branch to push to")
    upload.add_argument("-m", "--message", default="Upload LongiSeg tracking model")
    upload.add_argument("--no-model-card", action="store_true", help="do not write a README.md into the folder")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    log = lambda message: print(message, flush=True)  # noqa: E731

    try:
        if args.command == "download":
            folder = download_model(
                args.repo_id,
                folds=_folds(args.folds),
                version=args.version,
                revision=args.revision,
                token=args.token,
                progress=log,
            )
            print(folder)
        elif args.command == "info":
            folder = resolve_model_folder(args.source, folds=_folds(args.folds), version=args.version, progress=log)
            print(json.dumps(describe_model(folder), indent=2))
        elif args.command == "sample":
            from .pantrack import download_pair, list_pairs

            pairs = list_pairs()
            if args.list:
                for pair in pairs:
                    print(f"{pair.label}   lesions {sorted(pair.lesions)}")
                return 0
            pair = download_pair(
                patient=args.patient,
                pair_index=_sample_pair_index(args.patient, args.pair_index),
                pairs=pairs,
                progress=log,
            )
            print(pair.baseline_image)
            print(pair.followup_image)
        elif args.command == "upload":
            url = push_model_folder(
                args.folder,
                repo_id=args.repo_id,
                path_in_repo=f"LongiTrack_v{args.version}" if args.version else None,
                private=not args.public,
                token=args.token,
                revision=args.revision,
                commit_message=args.message,
                write_model_card=not args.no_model_card,
                progress=log,
            )
            print(url)
    except (ModelNotFoundError, PanTrackError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
