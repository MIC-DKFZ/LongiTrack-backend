from __future__ import annotations

import json
import socket
import struct
from collections.abc import Callable
from typing import Any, BinaryIO

# One frame: 4-byte big-endian length, then that many bytes of UTF-8 JSON. Array data
# follows its JSON header as raw binary frames (same length prefix, no envelope), so a
# mask is a memcpy rather than base64. Frame kinds are told apart by the schema,
# not by a tag. A socket, not shared memory, so local and remote use the same code.

_LENGTH = struct.Struct(">I")
MAX_FRAME_BYTES = 4 * 1024 * 1024 * 1024  # 4 GiB ceiling, sanity bound not a real limit


class ProtocolError(RuntimeError):
    pass


class ConnectionClosed(ProtocolError):
    pass


def send_frame(sock: socket.socket, data: bytes) -> None:
    if len(data) > MAX_FRAME_BYTES:
        raise ProtocolError(f"frame of {len(data)} bytes exceeds the {MAX_FRAME_BYTES} byte limit")
    sock.sendall(_LENGTH.pack(len(data)))
    sock.sendall(data)


_CHUNK = 4 * 1024 * 1024


def _recv_exact(sock: socket.socket, n: int) -> bytearray:
    buffer = bytearray(n)
    view = memoryview(buffer)
    received = 0
    while received < n:
        count = sock.recv_into(view[received:], min(n - received, _CHUNK))
        if count == 0:
            raise ConnectionClosed("connection closed while reading a frame")
        received += count
    return buffer


def recv_frame(sock: socket.socket) -> bytearray:
    (length,) = _LENGTH.unpack(_recv_exact(sock, 4))
    if length > MAX_FRAME_BYTES:
        raise ProtocolError(f"peer announced a {length} byte frame, exceeding the {MAX_FRAME_BYTES} byte limit")
    return _recv_exact(sock, length)


def send_json(sock: socket.socket, message: dict) -> None:
    send_frame(sock, json.dumps(message).encode("utf-8"))


def recv_json(sock: socket.socket) -> dict:
    return json.loads(recv_frame(sock).decode("utf-8"))


def send_array(sock: socket.socket, array) -> None:
    send_frame(sock, array.tobytes())


def send_file(
    sock: socket.socket, source: BinaryIO, size: int, progress: Callable[[int], None] | None = None,
) -> None:
    """Stream an already-announced file payload without buffering it in RAM."""
    sent = 0
    while sent < size:
        chunk = source.read(min(_CHUNK, size - sent))
        if not chunk:
            raise ProtocolError(f"source ended after {sent} bytes; expected {size}")
        sock.sendall(chunk)
        sent += len(chunk)
        if progress is not None:
            progress(sent)


def recv_file(
    sock: socket.socket, destination: BinaryIO, size: int, progress: Callable[[int], None] | None = None,
) -> None:
    """Receive an already-announced file payload directly into a file."""
    received = 0
    buffer = bytearray(min(_CHUNK, max(1, size)))
    view = memoryview(buffer)
    while received < size:
        count = sock.recv_into(view[:min(len(view), size - received)])
        if count == 0:
            raise ConnectionClosed(f"connection closed after {received} bytes; expected {size}")
        destination.write(view[:count])
        received += count
        if progress is not None:
            progress(received)


# ---------------------------------------------------------------------- requests --
# every request is {"type": ..., "id": <job id, str>, ...fields}; every response line
# for that id is one of:
#   {"type": "progress", "id": ..., "message": ...}      -- zero or more
#   {"type": "result", "id": ..., ...fields}              -- exactly one, terminal
#   {"type": "error", "id": ..., "message": ...}          -- exactly one, terminal
# "result" may be followed by one or more raw binary frames when it says so (each
# array-bearing field names itself in the JSON as {"array": true, "shape": ..., "dtype": ...}
# and the binary frames follow in the SAME order those fields appear, depth-first).

REQUEST_TYPES = frozenset({
    "health",
    "authenticate",
    "open_session",
    "close_session",
    "scan_exists",
    "upload_scan",
    "materialize_model",
    "initialize",
    "load_scan",
    "load_scans",
    "preload_registration_scans",
    "warm_up_registration",
    "release_scan",
    "propagate",
    "track",
    "invalidate_registration",
    "export",
    "cancel",
    "shutdown",
})


def make_request(type_: str, job_id: str, **fields: Any) -> dict:
    if type_ not in REQUEST_TYPES:
        raise ProtocolError(f"unknown request type {type_!r}")
    return {"type": type_, "id": job_id, **fields}


# --------------------------------------------------------- embedding ndarrays in results
# _pack replaces every array, however deeply nested, with a JSON-safe marker and returns
# the arrays in depth-first order; _unpack reverses it given the frames in that order.
# A mask is mostly background, so runs of (value, length) are sent when they come out
# smaller than the raw array, and raw otherwise.

_LENGTH_DTYPE = "uint32"  # a single run over a ~4 billion voxel volume is not a real case


def _rle_encode(flat) -> tuple:
    import numpy as np

    if flat.size == 0:
        return flat, np.zeros(0, dtype=_LENGTH_DTYPE)
    change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    boundaries = np.empty(change.size + 2, dtype=np.int64)
    boundaries[0] = 0
    boundaries[1:-1] = change
    boundaries[-1] = flat.size
    lengths = np.diff(boundaries)
    if lengths.max(initial=0) > np.iinfo(_LENGTH_DTYPE).max:
        raise ProtocolError("a single run exceeds the RLE length dtype -- this should not happen for real masks")
    return flat[boundaries[:-1]], lengths.astype(_LENGTH_DTYPE)


def _rle_decode(values, lengths, shape):
    import numpy as np

    return np.repeat(values, lengths).reshape(shape)


def _pack(value, arrays: list):
    import numpy as np

    if isinstance(value, np.ndarray):
        raw_bytes = value.nbytes
        values, lengths = _rle_encode(value.reshape(-1))
        rle_bytes = values.nbytes + lengths.nbytes
        if rle_bytes < raw_bytes:
            arrays.append(values)
            arrays.append(lengths)
            return {
                "__array__": True,
                "encoding": "rle",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "lengths_dtype": _LENGTH_DTYPE,
            }
        arrays.append(value)
        return {"__array__": True, "encoding": "raw", "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {key: _pack(item, arrays) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_pack(item, arrays) for item in value]
    return value


def _unpack(value, frames: list):
    if isinstance(value, dict):
        if value.get("__array__"):
            shape = tuple(value["shape"])
            if value["encoding"] == "rle":
                values = recv_array_from_frame(frames.pop(0), (-1,), value["dtype"], reshape=False)
                lengths = recv_array_from_frame(frames.pop(0), (-1,), value["lengths_dtype"], reshape=False)
                return _rle_decode(values, lengths, shape)
            return recv_array_from_frame(frames.pop(0), shape, value["dtype"])
        return {key: _unpack(item, frames) for key, item in value.items()}
    if isinstance(value, list):
        return [_unpack(item, frames) for item in value]
    return value


def recv_array_from_frame(raw: bytes, shape: tuple[int, ...], dtype: str, reshape: bool = True):
    import numpy as np

    array = np.frombuffer(raw, dtype=np.dtype(dtype)).copy()  # frombuffer aliases `raw`; own the memory
    if not reshape:
        return array  # a flat RLE component (values or lengths): its own length is the only shape it has
    expected = 1
    for dim in shape:
        expected *= dim
    if array.size != expected:
        raise ProtocolError(f"expected {expected} elements for shape {shape}, got {array.size}")
    return array.reshape(shape)


def send_result(sock: socket.socket, job_id: str, fields: dict, message_type: str = "result") -> None:
    arrays: list = []
    packed = _pack(fields, arrays)
    send_json(sock, {"type": message_type, "id": job_id, **packed})
    for array in arrays:
        send_array(sock, array)


def recv_result(sock: socket.socket, message: dict) -> dict:
    # one binary frame per __array__ marker in the envelope, in order
    body = {key: value for key, value in message.items() if key not in ("type", "id")}
    marker_count = _count_markers(body)
    frames = [recv_frame(sock) for _ in range(marker_count)]
    return _unpack(body, frames)


def _count_markers(value) -> int:
    if isinstance(value, dict):
        if value.get("__array__"):
            return 2 if value["encoding"] == "rle" else 1
        return sum(_count_markers(item) for item in value.values())
    if isinstance(value, list):
        return sum(_count_markers(item) for item in value)
    return 0
