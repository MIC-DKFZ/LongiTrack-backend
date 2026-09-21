from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from longitrack_backend import protocol as proto
from longitrack_backend import server
from longitrack_backend.inference import _predict_cached_patch


def _result() -> SimpleNamespace:
    mask = SimpleNamespace(
        mask=np.zeros((2, 3, 4), dtype=np.uint8),
        point=[0, 0, 0],
        spacing_zyx=(1.0, 1.0, 1.0),
        scan=None,
        properties={},
        volume_ml=0.0,
        voxels=0,
    )
    return SimpleNamespace(followup=mask, baseline=None, seconds=0.0, notes=[])


def test_backend_servers_do_not_run_model_requests_concurrently():
    # File uploads may run concurrently with a model request, but the server protects
    # all model/registration state through this single operation lock.
    import socketserver

    assert issubclass(server.UnixBackendServer, socketserver.ThreadingMixIn)
    assert issubclass(server.TcpBackendServer, socketserver.ThreadingMixIn)
    assert hasattr(server._State(), "operation_lock")


def test_clear_scans_drops_the_engine_and_registration_caches_not_the_model():
    class FakeEngine:
        def __init__(self):
            self.cache_cleared = False

        def clear_cache(self):
            self.cache_cleared = True

    class FakeRegistration:
        def __init__(self):
            self.prepared_scans_cleared = False

        def invalidate(self):
            pass

        def clear_prepared_scans(self):
            self.prepared_scans_cleared = True

    state = server._State()
    engine, registration = FakeEngine(), FakeRegistration()
    state.engine = engine
    state.registration = registration

    server.handle_invalidate_registration(state, {"clear_scans": True}, lambda _message: None)

    assert engine.cache_cleared
    assert registration.prepared_scans_cleared
    # the model/checkpoint itself is untouched: nothing here resets state.engine to None
    assert state.engine is engine


def test_track_cache_includes_baseline_option_and_file_signature(tmp_path):
    image = tmp_path / "image.nii.gz"
    image.write_bytes(b"image")

    class Engine:
        def __init__(self):
            self.calls = 0

        @staticmethod
        def _signature(path):
            stat = path.stat()
            return str(path), stat.st_mtime_ns, stat.st_size

        def track(self, *_args, **_kwargs):
            self.calls += 1
            return _result()

    state = server._State()
    state.engine = Engine()
    request = {
        "baseline_path": str(image),
        "followup_path": str(image),
        "baseline_point": [0, 0, 0],
        "followup_point": [0, 0, 0],
        "segment_baseline": False,
    }
    server.handle_track(state, request, lambda _message: None)
    request["segment_baseline"] = True
    server.handle_track(state, request, lambda _message: None)

    assert state.engine.calls == 2


def test_cached_patch_adapter_keeps_prediction_on_its_input_device():
    class Predictor:
        def __init__(self):
            self.list_of_parameters = [{}]
            self.network = torch.nn.Identity()
            self.device = torch.device("cpu")
            self.configuration_manager = SimpleNamespace(patch_size=(4, 4, 4))

        def _internal_maybe_mirror_and_predict(self, data):
            return torch.zeros((1, 2, *data.shape[2:]), dtype=torch.float32, device=data.device)

    prediction, lower, upper = _predict_cached_patch(
        torch.zeros((1, 5, 5, 5)),
        [2, 2, 2],
        torch.zeros((1, 5, 5, 5)),
        [2, 2, 2],
        Predictor(),
        [4, 4, 4],
        torch.device("cpu"),
        1.0,
    )

    assert prediction.shape == (2, 4, 4, 4)
    assert prediction.device.type == "cpu"
    assert lower == (0, 0, 0)
    assert upper == (4, 4, 4)


def test_resolve_device_falls_back_to_cpu_and_logs_why(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    messages = []
    assert server._resolve_device("cuda", messages.append) == "cpu"
    assert any("cpu" in m.lower() for m in messages)


# ------------------------------------------------------------------------ protocol -
def test_rle_round_trips_a_lesion_shaped_mask():
    mask = np.zeros((64, 160, 160), dtype=np.uint8)
    mask[20:30, 60:90, 70:100] = 1
    arrays: list = []
    packed = proto._pack({"m": mask}, arrays)

    assert packed["m"]["encoding"] == "rle"
    assert sum(a.nbytes for a in arrays) < mask.nbytes // 100  # a compact lesion compresses hugely

    restored = proto._unpack(packed, [a.tobytes() for a in arrays])["m"]
    assert np.array_equal(restored, mask)
    assert restored.dtype == mask.dtype


# ------------------------------------------------- GPU work stays on one thread -
def test_every_cuda_touching_request_is_routed_to_the_gpu_thread():
    # A thread's first forward pass pays a large per-thread CUDA setup cost, and it
    # explodes once both model stacks are resident. The server answers every request on
    # a fresh connection thread, so anything that touches torch has to be handed to the
    # one long-lived GPU thread instead -- dropping a request type from this set
    # silently reintroduces that cost per call.
    assert server._GPU_REQUESTS == {
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


def test_the_gpu_worker_runs_everything_on_one_thread_that_is_not_the_caller():
    import threading

    worker = server._GpuWorker()
    threads = {worker.run(lambda: threading.current_thread()) for _ in range(5)}

    assert len(threads) == 1, "every call must land on the same thread"
    assert threads.pop() is not threading.current_thread()


def test_first_remote_start_bootstraps_the_authorized_key_directory(monkeypatch, tmp_path):
    target = tmp_path / "srv" / "longitrack" / "authorized_keys"
    commands = []
    monkeypatch.setattr(server, "AUTHORIZED_KEYS_DIR", target)
    monkeypatch.setattr(server, "AUTHORIZED_KEYS_GROUP", "longitrack")
    monkeypatch.setattr(server, "_current_username", lambda: "alice")
    monkeypatch.setattr(server.grp, "getgrnam", lambda _name: (_ for _ in ()).throw(KeyError))
    monkeypatch.setattr(server.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(server.subprocess, "run", lambda command, check: commands.append((command, check)))

    server.ensure_authorized_keys_directory()

    assert commands == [
        (["sudo", "-v"], True),
        (["sudo", "groupadd", "--system", "longitrack"], True),
        (["sudo", "usermod", "-aG", "longitrack", "alice"], True),
        (["sudo", "install", "-d", "-o", "alice", "-g", "longitrack", "-m", "3770", str(target)], True),
    ]


def test_local_socket_start_skips_authorized_key_bootstrap(monkeypatch):
    called = []
    monkeypatch.setattr(server, "ensure_authorized_keys_directory", lambda: called.append(True))
    monkeypatch.setattr(server, "run_unix", lambda _path: None)
    monkeypatch.setattr(server.sys, "argv", ["longitrack-backend", "--socket", "/tmp/test.sock"])

    server.main()

    assert called == []


def test_empty_authorized_key_directory_does_not_prevent_server_start(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(server, "ensure_authorized_keys_directory", lambda: None)
    monkeypatch.setattr(server, "AUTHORIZED_KEYS_DIR", tmp_path)
    monkeypatch.setattr(server, "run_tcp", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(server.sys, "argv", ["longitrack-backend", "--confirm-tcp-listener", "yes"])

    server.main()

    assert calls[0][1]["authorized_keys"] == {}
    assert calls[0][1]["authorized_keys_path"] == tmp_path
