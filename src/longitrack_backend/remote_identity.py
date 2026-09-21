"""Create and provision Ed25519 identities for LongiTrack remote backends."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shlex
import subprocess
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from .auth_config import REMOTE_AUTHORIZED_KEYS_DIR

IDENTITY_DIR = Path.home() / ".config" / "longitrack" / "remote"
OWNER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
def _validate_name(name: str) -> None:
    if Path(name).name != name or name in {"", ".", ".."}:
        raise ValueError("--name must be a simple file name, not a path")


def _validate_owner(owner: str) -> None:
    if not OWNER_PATTERN.fullmatch(owner):
        raise ValueError("--owner may contain only letters, numbers, dots, underscores, and hyphens")


def create_identity(name: str, *, force: bool = False) -> tuple[Path, Path, str]:
    """Create a private identity locally and return its paths and fingerprint."""
    _validate_name(name)
    private_path = IDENTITY_DIR / name
    public_path = private_path.with_suffix(private_path.suffix + ".pub")
    if (private_path.exists() or public_path.exists()) and not force:
        raise FileExistsError(f"{private_path} already exists; use --force only if it is safe to replace it")
    IDENTITY_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(IDENTITY_DIR, 0o700)

    key = Ed25519PrivateKey.generate()
    private_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    os.chmod(private_path, 0o600)
    public = key.public_key()
    public_bytes = public.public_bytes(serialization.Encoding.OpenSSH, serialization.PublicFormat.OpenSSH)
    public_path.write_bytes(public_bytes + b"\n")
    os.chmod(public_path, 0o644)
    raw = public.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return private_path, public_path, hashlib.sha256(raw).hexdigest()


def push_identity(ssh_destination: str, name: str, owner: str) -> tuple[Path, str]:
    """Install a local public key under its owner/fingerprint server-side name."""
    _validate_name(name)
    _validate_owner(owner)
    public_path = IDENTITY_DIR / f"{name}.pub"
    public_key = public_path.read_text(encoding="utf-8").strip()
    if not public_key.startswith("ssh-ed25519 "):
        raise ValueError(f"{public_path} must contain one OpenSSH Ed25519 public key")
    key = serialization.load_ssh_public_key(public_key.encode("utf-8"))
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError(f"{public_path} is not an Ed25519 public key")
    raw = key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    fingerprint = hashlib.sha256(raw).hexdigest()
    remote_dir = shlex.quote(REMOTE_AUTHORIZED_KEYS_DIR)
    remote_file = f'{remote_dir}/{owner}--{fingerprint}.pub'
    displayed_remote_file = f"{REMOTE_AUTHORIZED_KEYS_DIR}/{owner}--{fingerprint}.pub"
    # /srv is administrator-owned. The setgid directory assigns this file to
    # the shared group; clients only need permission to create their own key.
    command = (
        f"umask 007; test -d {remote_dir} || {{ "
        f"echo 'Authorized-key directory is not configured: {REMOTE_AUTHORIZED_KEYS_DIR}' >&2; exit 1; }}; "
        f"test -w {remote_dir} || {{ "
        f"echo 'No write permission for authorized-key directory: {REMOTE_AUTHORIZED_KEYS_DIR}' >&2; exit 1; }}; "
        f"printf '%s\\n' {shlex.quote(public_key)} > {remote_file}; chmod 640 {remote_file}"
    )
    subprocess.run(["ssh", ssh_destination, command], check=True)  # noqa: S603 - explicit SSH destination
    return public_path, displayed_remote_file


def _parser_create(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--name", required=True, help="local identity name, for example 'gpu-cluster'")
    parser.add_argument("--force", action="store_true", help="replace an existing local identity")


def _parser_push(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ssh", required=True, help="SSH destination, e.g. admin@gpu-server")
    parser.add_argument("--name", required=True, help="local identity name to push")
    parser.add_argument("--owner", required=True, help="authorized person or account, recorded on the server")


def create_main() -> None:
    parser = argparse.ArgumentParser(description="Create an app-specific Ed25519 identity for a LongiTrack backend.")
    _parser_create(parser)
    args = parser.parse_args()
    try:
        private_path, public_path, fingerprint = create_identity(args.name, force=args.force)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(f"Private key: {private_path}")
    print(f"Public key:  {public_path}")
    print(f"Fingerprint: sha256:{fingerprint}")


def push_main() -> None:
    parser = argparse.ArgumentParser(description="Push one LongiTrack public identity to a backend server over SSH.")
    _parser_push(parser)
    args = parser.parse_args()
    try:
        public_path, remote_file = push_identity(args.ssh, args.name, args.owner)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Authorized {public_path} on {args.ssh}:{remote_file}")


def create_and_push_main() -> None:
    parser = argparse.ArgumentParser(description="Create a LongiTrack identity locally and authorize it on a server.")
    _parser_create(parser)
    parser.add_argument("--ssh", required=True, help="SSH destination, e.g. admin@gpu-server")
    parser.add_argument("--owner", required=True, help="authorized person or account, recorded on the server")
    args = parser.parse_args()
    try:
        private_path, public_path, fingerprint = create_identity(args.name, force=args.force)
        _, remote_file = push_identity(args.ssh, args.name, args.owner)
    except (OSError, ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(f"Private key: {private_path}")
    print(f"Public key:  {public_path}")
    print(f"Fingerprint: sha256:{fingerprint}")
    print(f"Authorized on {args.ssh}:{remote_file}")
