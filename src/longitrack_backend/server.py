from __future__ import annotations

import argparse
import base64
import grp
import hashlib
import json
import os
import pwd
import queue
import re
import secrets
import shutil
import socket
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from . import protocol
from ._env import CACHE_DIR
from .auth_config import AUTHORIZED_KEYS_DIR, AUTHORIZED_KEYS_GROUP

REMOTE_PORT = 8765
SESSION_IDLE_SECONDS = 60 * 60

# this module never imports a GUI toolkit, directly or indirectly


class _GpuWorker:
    """The one thread every CUDA-touching request runs on.

    A thread's first CUDA forward pass pays a large per-thread initialization, and every
    new thread pays it again -- while the server answers each request on a fresh connection
    thread. Inference is serialized on the single device anyway. Non-GPU work (connection
    I/O, uploads, disk reads, export) deliberately stays off this thread.
    """

    def __init__(self) -> None:
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._loop, name="longitrack-gpu", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while True:
            call, done = self._queue.get()
            try:
                result, error = call(), None
            except BaseException as caught:  # noqa: BLE001 - handed back to the caller intact
                result, error = None, caught
            done.put((result, error))

    def run(self, call):
        """Run `call` on the GPU thread and return its result (or re-raise its error)."""
        if threading.current_thread() is self._thread:
            return call()  # a handler that re-enters must not deadlock on itself
        done: queue.Queue = queue.Queue(maxsize=1)
        self._queue.put((call, done))
        result, error = done.get()
        if error is not None:
            raise error
        return result


class _State:
    # everything mutable the backend owns, in one place.
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.operation_lock = threading.Lock()
        self.gpu = _GpuWorker()
        self.engine = None  # inference.TrackingEngine, built on "initialize"
        self.registration = None  # registration.RegistrationService
        self.device: str | None = None
        # per (baseline, follow-up, baseline point, follow-up point) TrackingResult
        self.results: dict = {}
        self.lesions: dict[str, TrackedLesionRecord] = {}
        self.scan_cache_dir = CACHE_DIR / "scans"
        self.model_cache_dir = CACHE_DIR / "models"
        self.session_cache_dir = CACHE_DIR / "sessions"
        # Sessions cannot survive a backend restart: remove their uploaded inputs.
        shutil.rmtree(self.session_cache_dir, ignore_errors=True)
        self.session_cache_dir.mkdir(parents=True, exist_ok=True)
        self.sessions: dict[str, Path] = {}
        self.session_last_used: dict[str, float] = {}


class TrackedLesionRecord:
    __slots__ = ("lesion", "baseline_point", "followup_point", "propagated_point", "result")

    def __init__(self, lesion, baseline_point, followup_point, propagated_point, result):
        self.lesion = lesion
        self.baseline_point = baseline_point
        self.followup_point = followup_point
        self.propagated_point = propagated_point
        self.result = result


def _pick_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_device(requested: str | None, progress) -> str:
    device = requested or _pick_device()
    if device == "cuda":
        import torch

        if not torch.cuda.is_available():
            progress("No CUDA GPU detected -- falling back to CPU. This will be much slower.")
            device = "cpu"
    return device


def handle_initialize(state: _State, request: dict, progress) -> dict:
    from .inference import TrackingEngine
    from .model import describe_model, resolve_model_folder
    from .registration import RegistrationService

    device = _resolve_device(request.get("device"), progress)
    location = request.get("location")
    folder = resolve_model_folder(location, folds=None, progress=progress)

    with state.lock:
        if state.registration is None:
            state.registration = RegistrationService()
        registration = state.registration
        state.device = device

    # Order matters, and not in the direction it looks.
    registration.warm_up(progress=progress)

    engine = TrackingEngine(folder, folds=None, device=device, disable_tta=False)
    engine.warm_up(progress=progress)
    engine.refresh_gpu_cache_limit()

    with state.lock:
        state.engine = engine
        state.results.clear()
        state.lesions.clear()

    info = describe_model(folder)
    info["device"] = device
    return info


def handle_health(state: _State, request: dict, progress) -> dict:
    """Torch-free probe used before a GUI commits to a remote endpoint."""
    return {"protocol": 1, "service": "longitrack-backend"}


def _session_directory(state: _State, request: dict) -> Path:
    _prune_expired_sessions(state)
    session_id = str(request.get("session_id", ""))
    if not re.fullmatch(r"[0-9a-f]{32}", session_id) or session_id not in state.sessions:
        raise ValueError("Unknown or expired backend session.")
    state.session_last_used[session_id] = time.monotonic()
    return state.sessions[session_id]


def _prune_expired_sessions(state: _State) -> None:
    deadline = time.monotonic() - SESSION_IDLE_SECONDS
    for session_id, last_used in list(state.session_last_used.items()):
        if last_used < deadline:
            folder = state.sessions.pop(session_id, None)
            state.session_last_used.pop(session_id, None)
            if folder is not None:
                shutil.rmtree(folder, ignore_errors=True)


def handle_open_session(state: _State, request: dict, progress) -> dict:
    _prune_expired_sessions(state)
    session_id = str(request.get("session_id", ""))
    if not re.fullmatch(r"[0-9a-f]{32}", session_id):
        raise ValueError("Invalid backend session identifier.")
    folder = state.session_cache_dir / session_id
    folder.mkdir(parents=True, exist_ok=True)
    state.sessions[session_id] = folder
    state.session_last_used[session_id] = time.monotonic()
    return {"session_id": session_id}


def handle_close_session(state: _State, request: dict, progress) -> dict:
    session_id = str(request.get("session_id", ""))
    folder = state.sessions.pop(session_id, None)
    state.session_last_used.pop(session_id, None)
    if folder is not None:
        shutil.rmtree(folder, ignore_errors=True)
    return {}


def _uploaded_scan_path(state: _State, request: dict) -> Path:
    digest = str(request.get("sha256", ""))
    suffix = str(request.get("suffix", ""))
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("Invalid uploaded scan SHA-256.")
    allowed_suffix_characters = ".abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    if not suffix or len(suffix) > 16 or any(char not in allowed_suffix_characters for char in suffix):
        raise ValueError("Invalid uploaded scan file suffix.")
    return _session_directory(state, request) / "uploads" / f"{digest}{suffix}"


def handle_scan_exists(state: _State, request: dict, progress) -> dict:
    path = _uploaded_scan_path(state, request)
    expected_size = int(request.get("size", -1))
    exists = path.is_file() and path.stat().st_size == expected_size
    return {"available": exists, "path": str(path) if exists else None}


def handle_materialize_model(state: _State, request: dict, progress) -> dict:
    """Build a model directory from files already uploaded to backend storage."""
    files = request.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("A model upload must contain at least one file.")
    session_root = _session_directory(state, request)
    cache_root = (session_root / "uploads").resolve()
    entries: list[tuple[Path, Path]] = []
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("Invalid model upload entry.")
        relative = Path(str(item.get("relative", "")))
        source = Path(str(item.get("path", "")))
        if relative.is_absolute() or relative == Path(".") or ".." in relative.parts:
            raise ValueError("Invalid relative model file path.")
        if source.parent.resolve() != cache_root or not source.is_file():
            raise ValueError("Model upload refers to a file outside backend upload storage.")
        entries.append((relative, source))
    manifest = [(str(relative), source.name) for relative, source in sorted(entries)]
    identifier = hashlib.sha256(json.dumps(manifest, separators=(",", ":")).encode("utf-8")).hexdigest()
    destination = session_root / "models" / identifier
    if destination.is_dir():
        return {"folder": str(destination), "reused": True}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".model-", dir=destination.parent))
    try:
        for relative, source in entries:
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, target)
        os.replace(temporary, destination)
    except Exception:
        import shutil

        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {"folder": str(destination), "reused": False}


def handle_upload_scan(state: _State, request: dict, sock: socket.socket, progress) -> dict:
    path = _uploaded_scan_path(state, request)
    expected_size = int(request.get("size", -1))
    if expected_size < 0:
        raise ValueError("Invalid uploaded scan size.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=".upload-", delete=False) as temporary:
            temporary_name = temporary.name
            protocol.recv_file(sock, temporary, expected_size)
        with open(temporary_name, "rb") as temporary:
            actual_digest = hashlib.file_digest(temporary, "sha256").hexdigest()
        if actual_digest != request["sha256"]:
            raise ValueError("Uploaded scan checksum does not match the client source.")
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return {"path": str(path), "uploaded": True}


def handle_load_scan(state: _State, request: dict, progress) -> dict:
    with state.lock:
        engine = state.engine
    if engine is None:
        raise RuntimeError("Initialize the model before preparing a scan.")
    image = engine.preload(request["path"], progress=progress)
    return {
        "path": str(image.path),
        "original_shape": list(image.original_shape),
        "preprocessed_shape": list(image.preprocessed_shape),
        "spacing_zyx": list(image.spacing_zyx),
        "device": str(image.data.device),
    }


def handle_load_scans(state: _State, request: dict, progress) -> dict:
    with state.lock:
        engine = state.engine
    if engine is None:
        raise RuntimeError("Initialize the model before preparing scans.")
    images = engine.preload_many(request["paths"], progress=progress)
    return {
        "scans": [
            {
                "path": str(image.path),
                "original_shape": list(image.original_shape),
                "preprocessed_shape": list(image.preprocessed_shape),
                "spacing_zyx": list(image.spacing_zyx),
                "device": str(image.data.device),
            }
            for image in images
        ]
    }


def handle_preload_registration_scans(state: _State, request: dict, progress) -> dict:
    from .registration import RegistrationService

    device = _resolve_device(request.get("device"), progress)
    with state.lock:
        if state.registration is None:
            state.registration = RegistrationService()
        registration = state.registration
        state.device = device
    registration.preload_scans(request["paths"], device=device, progress=progress)
    return {}


def handle_warm_up_registration(state: _State, request: dict, progress) -> dict:
    """Pay UniGradICON one-time CUDA work before any user action."""
    from .registration import RegistrationService

    device = _resolve_device(request.get("device"), progress)
    with state.lock:
        if state.registration is None:
            state.registration = RegistrationService()
        registration = state.registration
        state.device = device
    registration.warm_up(progress=progress)
    return {"device": device}


def handle_release_scan(state: _State, request: dict, progress) -> dict:
    with state.lock:
        engine = state.engine
    if engine is not None:
        engine.release_scan(request["path"])
    return {}


def handle_propagate(state: _State, request: dict, progress) -> dict:
    from .registration import RegistrationService

    with state.lock:
        registration = state.registration
        if registration is None:
            # Registration is deliberately independent from LongiSeg. It can be used
            # before any segmentation model has been selected or initialized.
            registration = RegistrationService()
            state.registration = registration

    propagations = registration.propagate(
        request["baseline_path"],
        request["followup_path"],
        request["points"],
        tuple(request["followup_shape"]),
        refinement_steps=request.get("refinement_steps"),
        progress=progress,
    )
    return {
        "propagations": [
            {
                "baseline_index": p.baseline_index,
                "followup_index": p.followup_index,
                "out_of_bounds": p.out_of_bounds,
                "error": p.error,
            }
            for p in propagations
        ]
    }


def handle_invalidate_registration(state: _State, request: dict, progress) -> dict:
    with state.lock:
        registration = state.registration
        engine = state.engine
    if registration is not None:
        registration.invalidate()
    if request.get("clear_tracking", False):
        # A quality-mode change invalidates both rendered masks and the private
        # server-side records that export uses to materialize full masks.
        with state.lock:
            state.results.clear()
            state.lesions.clear()
    if request.get("clear_scans", False):
        # Clear everything: drop every cached scan tensor on the GPU
        if engine is not None:
            engine.clear_cache()
        if registration is not None:
            registration.clear_prepared_scans()
    return {}


def _mask_payload(mask) -> dict | None:
    if mask is None:
        return None
    bounds = getattr(mask, "bounds", None)
    original_shape = getattr(mask, "original_shape", None)
    return {
        "mask": mask.mask,
        "point": mask.point,
        "spacing_zyx": list(mask.spacing_zyx),
        "scan": str(mask.scan) if mask.scan is not None else None,
        "properties": mask.properties,
        "bounds": [list(item) for item in bounds] if bounds is not None else None,
        "original_shape": list(original_shape) if original_shape is not None else None,
        "volume_ml": mask.volume_ml,
        "voxels": mask.voxels,
    }


def handle_track(state: _State, request: dict, progress) -> dict:
    with state.lock:
        engine = state.engine
    if engine is None:
        raise RuntimeError("Initialize the model before segmenting.")

    baseline_path = request["baseline_path"]
    followup_path = request["followup_path"]
    baseline_point = request["baseline_point"]
    followup_point = request["followup_point"]
    key = (
        engine._signature(Path(baseline_path)),
        engine._signature(Path(followup_path)),
        tuple(baseline_point),
        tuple(followup_point),
        bool(request.get("segment_baseline", True)),
        bool(request.get("disable_tta", False)),
    )

    with state.lock:
        cached = state.results.get(key)
    if cached is not None:
        progress("Unchanged since last time -- keeping the previous result.")
        result = cached
        reused = True
    else:
        result = engine.track(
            baseline_path,
            baseline_point,
            followup_path,
            followup_point,
            segment_baseline=bool(request.get("segment_baseline", True)),
            disable_tta=bool(request.get("disable_tta", False)),
            progress=progress,
        )
        with state.lock:
            state.results[key] = result
        reused = False

    lesion_id = request.get("lesion_id")
    if lesion_id is not None:
        with state.lock:
            state.lesions[lesion_id] = TrackedLesionRecord(
                lesion=request.get("lesion_number", 1),
                baseline_point=baseline_point,
                followup_point=followup_point,
                propagated_point=request.get("propagated_point"),
                result=result,
            )

    return {
        "reused": reused,
        "seconds": result.seconds,
        "notes": result.notes,
        "followup": _mask_payload(result.followup),
        "baseline": _mask_payload(result.baseline),
    }


def handle_export(state: _State, request: dict, progress) -> dict:
    from .export import TrackedLesion, export_tracking

    with state.lock:
        engine = state.engine
        records = [state.lesions[lesion_id] for lesion_id in request["lesion_ids"] if lesion_id in state.lesions]
    if not records:
        raise ValueError("Nothing to export, run a segmentation first.")

    lesions = [
        TrackedLesion(
            lesion=record.lesion,
            baseline_point=record.baseline_point,
            followup_point=record.followup_point,
            result=record.result,
            propagated_point=record.propagated_point,
        )
        for record in records
    ]
    file_ending = engine.file_ending if engine is not None else ".nii.gz"
    written = export_tracking(
        request["folder"], lesions, file_ending=file_ending, patient=request.get("patient", "case_000"),
        timepoints=request.get("timepoints", ("baseline", "followup")),
        progress=progress,
    )
    return {"written": [str(path) for path in written]}


# requests whose handler touches torch/CUDA: they all run on the one GPU thread
_GPU_REQUESTS = frozenset(
    {
        "initialize",
        "load_scan",
        "load_scans",
        "preload_registration_scans",
        "warm_up_registration",
        "release_scan",
        "propagate",
        "invalidate_registration",
        "track",
    }
)

_HANDLERS = {
    "health": handle_health,
    "open_session": handle_open_session,
    "close_session": handle_close_session,
    "scan_exists": handle_scan_exists,
    "materialize_model": handle_materialize_model,
    "initialize": handle_initialize,
    "load_scan": handle_load_scan,
    "load_scans": handle_load_scans,
    "preload_registration_scans": handle_preload_registration_scans,
    "warm_up_registration": handle_warm_up_registration,
    "release_scan": handle_release_scan,
    "propagate": handle_propagate,
    "invalidate_registration": handle_invalidate_registration,
    "track": handle_track,
    "export": handle_export,
}


class _ConnectionHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        sock: socket.socket = self.request
        state: _State = self.server.state  # type: ignore[attr-defined]
        authorized_path = getattr(self.server, "authorized_keys_path", None)
        if authorized_path is None:
            authorized_keys = getattr(self.server, "authorized_keys", {})
            authenticated = not authorized_keys
        else:
            # Reload on every connection so provisioning and revocation take effect
            # immediately; the directory normally contains only a handful of keys.
            try:
                authorized_keys = load_authorized_keys(authorized_path)
            except ValueError as error:
                print(f"Could not load authorized keys: {error}", file=sys.stderr, flush=True)
                authorized_keys = {}
            authenticated = False
        challenge = secrets.token_bytes(32) if not authenticated else None
        try:
            while True:
                try:
                    request = protocol.recv_json(sock)
                except protocol.ConnectionClosed:
                    return
                job_id = request.get("id", "")
                request_type = request.get("type")
                if request_type == "health":
                    protocol.send_json(
                        sock,
                        {
                            "type": "result",
                            "id": job_id,
                            "protocol": 1,
                            "service": "longitrack-backend",
                            "authentication_required": not authenticated,
                            "authentication": "ed25519" if not authenticated else None,
                            "challenge": base64.b64encode(challenge).decode("ascii") if not authenticated else None,
                        },
                    )
                    continue
                if not authenticated:
                    if request_type != "authenticate":
                        protocol.send_json(sock, {"type": "error", "id": job_id, "message": "authenticate first"})
                        continue
                    key = authorized_keys.get(str(request.get("key_id", "")))
                    try:
                        signature = base64.b64decode(str(request.get("signature", "")), validate=True)
                        if key is None:
                            raise InvalidSignature
                        key.verify(signature, challenge)
                    except (InvalidSignature, ValueError):
                        protocol.send_json(sock, {"type": "error", "id": job_id, "message": "authentication failed"})
                        return
                    authenticated = True
                    peer = getattr(self, "client_address", ("unknown", 0))
                    print(
                        f"AUTHENTICATED {peer[0]}:{peer[1]} key={request['key_id']}",
                        file=sys.stderr,
                        flush=True,
                    )
                    protocol.send_json(sock, {"type": "result", "id": job_id, "authenticated": True})
                    continue
                self._dispatch(sock, state, request)
        except Exception:  # noqa: BLE001 - the connection is going away either way
            traceback.print_exc(file=sys.stderr)

    def _dispatch(self, sock: socket.socket, state: _State, request: dict) -> None:
        job_id = request.get("id", "")
        request_type = request.get("type")

        if request_type == "shutdown":
            protocol.send_json(sock, {"type": "result", "id": job_id})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return

        def progress(message: str) -> None:
            try:
                protocol.send_json(sock, {"type": "progress", "id": job_id, "message": message})
            except OSError:
                pass  # the client walked away (e.g. it cancelled); the job still finishes

        if request_type == "upload_scan":
            try:
                fields = handle_upload_scan(state, request, sock, progress)
            except Exception as error:  # noqa: BLE001 - return upload failures to the client
                traceback.print_exc(file=sys.stderr)
                try:
                    protocol.send_json(sock, {"type": "error", "id": job_id, "message": str(error)})
                except OSError:
                    pass
            else:
                protocol.send_result(sock, job_id, fields)
            return

        handler = _HANDLERS.get(request_type)
        if handler is None:
            protocol.send_json(sock, {"type": "error", "id": job_id, "message": f"unknown request {request_type!r}"})
            return

        try:
            if request_type not in {"health", "open_session", "close_session"}:
                _session_directory(state, request)
            if request_type in {"health", "scan_exists", "open_session", "close_session", "materialize_model"}:
                fields = handler(state, request, progress)
            elif request_type in _GPU_REQUESTS:
                # the operation lock still serializes against non-GPU handlers that read the same state (export); the
                with state.operation_lock:
                    fields = state.gpu.run(lambda: handler(state, request, progress))
            else:
                with state.operation_lock:
                    fields = handler(state, request, progress)
        except Exception as error:  # noqa: BLE001 - reported to the client, not swallowed
            traceback.print_exc(file=sys.stderr)
            try:
                protocol.send_json(sock, {"type": "error", "id": job_id, "message": str(error)})
            except OSError:
                pass
            return
        try:
            protocol.send_result(sock, job_id, fields)
        except OSError:
            pass


class _ConcurrentBackendServer(socketserver.ThreadingMixIn):
    daemon_threads = True


class UnixBackendServer(_ConcurrentBackendServer, socketserver.UnixStreamServer):
    # Uploads are independent CPU/disk I/O; model operations retain operation_lock.
    allow_reuse_address = True

    def __init__(self, socket_path: str) -> None:
        self.state = _State()
        super().__init__(socket_path, _ConnectionHandler)


def _fingerprint(public_key: Ed25519PublicKey) -> str:
    raw = public_key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return hashlib.sha256(raw).hexdigest()


def load_authorized_keys(path: str | Path) -> dict[str, Ed25519PublicKey]:
    keys: dict[str, Ed25519PublicKey] = {}
    folder = Path(path)
    if not folder.is_dir():
        raise ValueError(f"Authorized-key directory does not exist: {folder}")
    for key_path in sorted(folder.glob("*.pub")):
        key = serialization.load_ssh_public_key(key_path.read_bytes().strip())
        if not isinstance(key, Ed25519PublicKey):
            raise ValueError(f"Only ssh-ed25519 public keys are supported: {key_path}")
        keys[_fingerprint(key)] = key
    if not keys:
        raise ValueError(f"The authorized-key directory contains no .pub Ed25519 public keys: {folder}")
    return keys


class TcpBackendServer(_ConcurrentBackendServer, socketserver.TCPServer):
    # TCP retains the same serialized request semantics. Authentication is performed
    # in _ConnectionHandler with an Ed25519 allowlist.
    allow_reuse_address = True

    def __init__(
        self, host: str, port: int, authorized_keys: dict | None = None, authorized_keys_path: Path | None = None,
    ) -> None:
        self.state = _State()
        self.authorized_keys = authorized_keys or {}
        self.authorized_keys_path = authorized_keys_path
        super().__init__((host, port), _ConnectionHandler)
        self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


def run_unix(socket_path: str) -> None:
    import os

    if os.path.exists(socket_path):
        os.unlink(socket_path)
    server = UnixBackendServer(socket_path)
    try:
        print(f"READY {socket_path}", flush=True)
        server.serve_forever()
    finally:
        server.server_close()
        shutil.rmtree(server.state.session_cache_dir, ignore_errors=True)
        if os.path.exists(socket_path):
            os.unlink(socket_path)


def _advertised_ipv4_addresses() -> list[str]:
    """Return local non-loopback IPv4 candidates a VPN/LAN client can use."""
    addresses: set[str] = set()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            # UDP connect chooses the default outbound interface without sending data.
            probe.connect(("192.0.2.1", 9))
            address = probe.getsockname()[0]
            if not address.startswith("127."):
                addresses.add(address)
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET, socket.SOCK_STREAM):
            address = info[4][0]
            if not address.startswith("127."):
                addresses.add(address)
    except OSError:
        pass
    return sorted(addresses)


def run_tcp(
    host: str = "0.0.0.0",
    port: int = REMOTE_PORT,
    authorized_keys: dict | None = None,
    authorized_keys_path: Path | None = None,
) -> None:
    server = TcpBackendServer(host, port, authorized_keys=authorized_keys, authorized_keys_path=authorized_keys_path)
    try:
        bound_port = server.server_address[1]
        print(f"READY {host}:{bound_port}", flush=True)
        for address in _advertised_ipv4_addresses():
            print(f"CONNECT {address}:{bound_port}", flush=True)
        server.serve_forever()
    finally:
        server.server_close()
        shutil.rmtree(server.state.session_cache_dir, ignore_errors=True)


def _current_username() -> str:
    return pwd.getpwuid(os.getuid()).pw_name


def ensure_authorized_keys_directory() -> None:
    """Create the shared authorization directory on first TCP-server start.

    `/srv` needs administrator privileges, so `sudo` deliberately owns the one-time
    setup. It prompts in the terminal and is never invoked again once the directory
    exists. The backend itself continues to run as the original, unprivileged user.
    """
    if AUTHORIZED_KEYS_DIR.exists():
        if not AUTHORIZED_KEYS_DIR.is_dir():
            raise ValueError(f"Authorized-key path is not a directory: {AUTHORIZED_KEYS_DIR}")
        return
    if not sys.stdin.isatty():
        raise ValueError(
            f"Authorized-key directory does not exist: {AUTHORIZED_KEYS_DIR}. "
            "Start the server interactively once so it can perform the sudo setup."
        )
    username = _current_username()
    try:
        grp.getgrnam(AUTHORIZED_KEYS_GROUP)
    except KeyError:
        group_exists = False
    else:
        group_exists = True
    print(f"Setting up {AUTHORIZED_KEYS_DIR} for shared remote authorization.", flush=True)
    try:
        subprocess.run(["sudo", "-v"], check=True)  # prompts for the user's password
        if not group_exists:
            subprocess.run(["sudo", "groupadd", "--system", AUTHORIZED_KEYS_GROUP], check=True)
        subprocess.run(["sudo", "usermod", "-aG", AUTHORIZED_KEYS_GROUP, username], check=True)
        subprocess.run(
            [
                "sudo",
                "install",
                "-d",
                "-o",
                username,
                "-g",
                AUTHORIZED_KEYS_GROUP,
                "-m",
                "3770",
                str(AUTHORIZED_KEYS_DIR),
            ],
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise ValueError(f"Could not set up {AUTHORIZED_KEYS_DIR} with sudo: {error}") from error
    print(
        f"Created {AUTHORIZED_KEYS_DIR}. Reconnect SSH clients after this group change, then push an identity.",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", help="Unix domain socket path to listen on (local, fastest)")
    parser.add_argument("--confirm-tcp-listener", choices=("yes",), help="explicit non-interactive acknowledgement")
    args = parser.parse_args()
    if args.socket:
        run_unix(args.socket)
    else:
        host, port = "0.0.0.0", REMOTE_PORT
        ensure_authorized_keys_directory()
        authorized_path = AUTHORIZED_KEYS_DIR
        # The server starts safely with an empty allowlist. _ConnectionHandler
        # reloads this directory for each connection, so added keys work at once.
        authorized_keys = {}
        warning = (
            f"Starting a potentially insecure LongiTrack server session on {host}:{port}. "
            "It is protected by Ed25519 keys only; TCP traffic is not encrypted."
        )
        print(f"WARNING: {warning}", file=sys.stderr)
        if args.confirm_tcp_listener != "yes":
            try:
                approved = input("Do you want to start the server (y/N): ").strip().lower()
            except EOFError:
                approved = ""
            if approved not in {"yes", "y"}:
                parser.error("TCP listener was not approved")
        run_tcp(host, port, authorized_keys=authorized_keys, authorized_keys_path=authorized_path)


if __name__ == "__main__":
    main()
