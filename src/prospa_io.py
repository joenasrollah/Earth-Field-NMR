"""Minimal reader for Magritek Prospa (.par, .1d, .2d, .txt) data files."""

from pathlib import Path
import struct
import numpy as np


def _parse_value(raw: str):
    raw = raw.strip()
    if raw.startswith('"') and raw.endswith('"'):
        return raw[1:-1]
    if raw.startswith("[") and raw.endswith("]"):
        items = [x.strip() for x in raw[1:-1].split(",") if x.strip()]
        return [_parse_value(x) for x in items]
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def read_par(path: str | Path) -> dict:
    """Parse a Prospa acqu.par file into a dict."""
    params = {}
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        params[key.strip()] = _parse_value(val)
    return params


def read_prospa_binary(path: str | Path) -> np.ndarray:
    """Read a Prospa SORPATAD 1D real or 2D/3D complex binary file."""
    raw = Path(path).read_bytes()
    if len(raw) < 32 or raw[:8] != b"SORPATAD":
        raise ValueError(f"Not a valid Prospa SORPATAD file: {path}")

    _, data_type, nx, ny, nz, nq = struct.unpack("<6I", raw[8:32])
    payload = raw[32:]
    n_elems = nx * ny * nz * nq

    if data_type == 500:  # real float32
        expected = n_elems * 4
        if len(payload) != expected:
            raise ValueError(f"{path}: expected {expected} bytes, got {len(payload)}")
        arr = np.frombuffer(payload, dtype="<f4").astype(np.float64)
    elif data_type == 501:  # complex64 (interleaved float32 real, imag)
        expected = n_elems * 8
        if len(payload) != expected:
            raise ValueError(f"{path}: expected {expected} bytes, got {len(payload)}")
        arr = np.frombuffer(payload, dtype="<c8").astype(np.complex128)
    else:
        raise ValueError(f"{path}: unsupported Prospa dataType {data_type}")

    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{path}: contains NaN or Inf values")

    shape = tuple(d for d in (nz, ny, nx) if d > 1)
    return arr.reshape(shape) if shape else arr


def read_t1_txt(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read two-column (delay_ms, signal) output from a T1Bp run."""
    data = np.loadtxt(path, dtype=np.float64)
    return data[:, 0], data[:, 1]
