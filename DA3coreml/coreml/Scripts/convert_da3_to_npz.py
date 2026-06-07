#!/usr/bin/env python3
"""
Convert DA3CoreML `.da3` files into a NumPy `.npz` archive.

Why this exists:
- The upstream DA3 codebase commonly exports inference results as `results.npz`.
- This repo uses `.da3` as a fast, Swift-friendly binary container for large tensors.
- This script makes it easy to consume `.da3` outputs in Python without re-running inference.

Output keys:
- depth: (H, W) float32
- conf: (H, W) float32            (if present)
- rays: (C, H, W) float32         (if present; C inferred from blob size)
- ray_conf: (H, W) float32        (if present)
- width, height, min_depth, max_depth, inference_time_s, timestamp_ms: scalars

The `.da3` format is documented in `README.md` ("File formats: why .da3").
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import zlib
from dataclasses import dataclass
from typing import Any

import numpy as np


MAGIC = b"DA3C"


@dataclass(frozen=True)
class DA3Header:
    version: int
    flags: int
    width: int
    height: int
    min_depth: float
    max_depth: float
    inference_time_s: float
    timestamp_ms: int

    @property
    def has_rays(self) -> bool:
        return bool(self.flags & 0x1)

    @property
    def has_confidence(self) -> bool:
        return bool(self.flags & 0x2)

    @property
    def is_compressed(self) -> bool:
        return bool(self.flags & 0x4)


def _read_exact(f, n: int) -> bytes:
    b = f.read(n)
    if len(b) != n:
        raise ValueError(f"Unexpected EOF (wanted {n} bytes, got {len(b)})")
    return b


def _read_u16(f) -> int:
    return struct.unpack("<H", _read_exact(f, 2))[0]


def _read_u32(f) -> int:
    return struct.unpack("<I", _read_exact(f, 4))[0]


def _read_u64(f) -> int:
    return struct.unpack("<Q", _read_exact(f, 8))[0]


def _read_f32(f) -> float:
    return struct.unpack("<f", _read_exact(f, 4))[0]


def _read_blob(f, *, compressed: bool) -> bytes:
    raw_size = _read_u32(f)
    stored_size = _read_u32(f)
    data = _read_exact(f, stored_size)
    if compressed:
        # NOTE: DA3CoreML uses `NSData.compressed(using: .zlib)`, which produces a **raw DEFLATE**
        # stream (no zlib/gzip wrapper). Python's zlib expects a wrapper by default, so we use
        # `wbits=-MAX_WBITS`.
        #
        # We still keep a small fallback for robustness in case a future Swift implementation
        # switches to a wrapped stream.
        try:
            out = zlib.decompress(data, -zlib.MAX_WBITS)
        except zlib.error:
            out = zlib.decompress(data)
        if len(out) != raw_size:
            raise ValueError(f"Decompressed size mismatch (expected {raw_size}, got {len(out)})")
        return out
    if stored_size != raw_size:
        raise ValueError(f"Uncompressed blob size mismatch (raw={raw_size}, stored={stored_size})")
    return data


def read_da3(path: str) -> tuple[DA3Header, dict[str, np.ndarray]]:
    with open(path, "rb") as f:
        magic = _read_exact(f, 4)
        if magic != MAGIC:
            raise ValueError(f"Bad magic: {magic!r} (expected {MAGIC!r})")

        version = _read_u16(f)
        flags = _read_u16(f)

        width = _read_u32(f)
        height = _read_u32(f)

        min_depth = _read_f32(f)
        max_depth = _read_f32(f)
        inference_time_s = _read_f32(f)
        timestamp_ms = _read_u64(f)

        # Reserved
        _ = _read_exact(f, 32)

        hdr = DA3Header(
            version=version,
            flags=flags,
            width=width,
            height=height,
            min_depth=min_depth,
            max_depth=max_depth,
            inference_time_s=inference_time_s,
            timestamp_ms=timestamp_ms,
        )

        blobs: dict[str, np.ndarray] = {}

        # Depth (required)
        depth_bytes = _read_blob(f, compressed=hdr.is_compressed)
        depth = np.frombuffer(depth_bytes, dtype="<f4")
        if depth.size != width * height:
            raise ValueError(f"Depth element count mismatch (expected {width*height}, got {depth.size})")
        blobs["depth"] = depth.reshape((height, width))

        # Depth confidence (optional)
        if hdr.has_confidence:
            conf_bytes = _read_blob(f, compressed=hdr.is_compressed)
            conf = np.frombuffer(conf_bytes, dtype="<f4")
            if conf.size != width * height:
                raise ValueError(f"Conf element count mismatch (expected {width*height}, got {conf.size})")
            blobs["conf"] = conf.reshape((height, width))

        # Rays (+ ray_conf) (optional)
        if hdr.has_rays:
            rays_bytes = _read_blob(f, compressed=hdr.is_compressed)
            rays = np.frombuffer(rays_bytes, dtype="<f4")
            denom = width * height
            if denom <= 0 or rays.size % denom != 0:
                raise ValueError(f"Rays element count mismatch (got {rays.size}, width*height={denom})")
            channels = rays.size // denom
            blobs["rays"] = rays.reshape((channels, height, width))

            if hdr.has_confidence:
                ray_conf_bytes = _read_blob(f, compressed=hdr.is_compressed)
                ray_conf = np.frombuffer(ray_conf_bytes, dtype="<f4")
                if ray_conf.size != width * height:
                    raise ValueError(
                        f"Ray conf element count mismatch (expected {width*height}, got {ray_conf.size})"
                    )
                blobs["ray_conf"] = ray_conf.reshape((height, width))

        return hdr, blobs


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert DA3CoreML .da3 to .npz")
    ap.add_argument("input", help="Path to a .da3 file")
    ap.add_argument("--out", help="Output .npz path (default: <input>.npz)")
    ap.add_argument(
        "--include-meta-json",
        action="store_true",
        help="If `<input>_meta.json` exists, include it as `meta_json` (UTF-8 string).",
    )
    args = ap.parse_args()

    in_path = args.input
    if not os.path.isfile(in_path):
        raise SystemExit(f"Input not found: {in_path}")
    if not in_path.lower().endswith(".da3"):
        raise SystemExit("Input must be a .da3 file")

    out_path = args.out or (os.path.splitext(in_path)[0] + ".npz")

    hdr, blobs = read_da3(in_path)

    # Add scalars / metadata as 0-d arrays (np.savez supports scalars).
    save_dict: dict[str, Any] = dict(blobs)
    save_dict.update(
        {
            "width": np.int32(hdr.width),
            "height": np.int32(hdr.height),
            "min_depth": np.float32(hdr.min_depth),
            "max_depth": np.float32(hdr.max_depth),
            "inference_time_s": np.float32(hdr.inference_time_s),
            "timestamp_ms": np.int64(hdr.timestamp_ms),
        }
    )

    if args.include_meta_json:
        meta_path = os.path.splitext(in_path)[0] + "_meta.json"
        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            save_dict["meta_json"] = np.string_(json.dumps(meta, ensure_ascii=False))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(out_path, **save_dict)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
