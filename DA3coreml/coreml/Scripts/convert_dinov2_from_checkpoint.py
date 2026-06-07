#!/usr/bin/env python3
"""
Convert the **official** Depth-Anything-3 (DA3) DINOv2 backbone from a DA3 checkpoint to CoreML.

Why this exists
---------------
DA3 checkpoints do **not** use a vanilla HuggingFace DINOv2. The backbone includes DA3-specific
behavior (e.g. RoPE starting at `rope_start`, camera-token injection at `alt_start`, QK-norm,
and `cat_token=True` feature concatenation). If you export a generic DINOv2 backbone, the
DualDPT head will still run, but depth will be incorrect because the feature tensors don't
match what the head was trained on.

This converter imports the upstream `depth_anything_3` implementation from `../src` and traces
the real backbone used by the official checkpoints.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import coremltools as ct
import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors


def _load_state_dict(path: str) -> dict[str, Any]:
    if path.endswith(".safetensors"):
        return load_safetensors(path)
    ckpt = torch.load(path, map_location="cpu")
    return ckpt.get("state_dict", ckpt.get("model", ckpt))


def _guess_backbone_prefix(sd: dict[str, Any]) -> str:
    candidates = [
        "model.backbone.pretrained.",
        "backbone.pretrained.",
        "module.backbone.pretrained.",
        "net.pretrained.",
    ]
    for prefix in candidates:
        if any(k.startswith(prefix) for k in sd.keys()):
            return prefix

    # Fallback: infer prefix from a known suffix.
    suffix = "blocks.0.attn.qkv.weight"
    for k in sd.keys():
        if k.endswith(suffix) and suffix in k:
            return k[: -len(suffix)]
    raise RuntimeError("Could not infer backbone prefix from checkpoint keys.")


def _extract_prefixed(sd: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {k[len(prefix) :]: v for (k, v) in sd.items() if k.startswith(prefix)}


def _load_da3_net_config(checkpoint_path: str, size: str) -> dict[str, Any]:
    ckpt_dir = Path(checkpoint_path).resolve().parent
    cfg_path = ckpt_dir / "config.json"
    if cfg_path.exists():
        with cfg_path.open("r") as f:
            raw = json.load(f)
        net_cfg = raw.get("config", {}).get("net", None)
        if isinstance(net_cfg, dict):
            return net_cfg

    # Fallback defaults (only used if config.json is missing).
    defaults: dict[str, dict[str, Any]] = {
        "small": dict(
            name="vits",
            out_layers=[2, 5, 8, 11],
            alt_start=4,
            qknorm_start=4,
            rope_start=4,
            cat_token=True,
        ),
        "base": dict(
            name="vitb",
            out_layers=[2, 5, 8, 11],
            alt_start=4,
            qknorm_start=4,
            rope_start=4,
            cat_token=True,
        ),
        "large": dict(
            name="vitl",
            out_layers=[5, 11, 17, 23],
            alt_start=8,
            qknorm_start=8,
            rope_start=8,
            cat_token=True,
        ),
        "giant": dict(
            name="vitg",
            out_layers=[19, 27, 33, 39],
            alt_start=13,
            qknorm_start=13,
            rope_start=13,
            cat_token=True,
        ),
    }
    return defaults[size]


class _DA3DinoV2BackboneCoreMLFriendly(torch.nn.Module):
    """
    CoreML-friendly, monocular-only wrapper for the official DA3 DinoVisionTransformer.

    Motivation:
      - The upstream implementation uses `torch.cartesian_prod` (not convertible) and an
        in-place `x[:, :, 0] = cam_token` assignment (coremltools currently fails to
        convert this in-place update).

    This wrapper re-implements the essential DA3 logic in a conversion-friendly way:
      - Fixed single-view (S=1) forward
      - Out-of-place camera-token injection at `alt_start`
      - Proper `cat_token=True` concatenation using the correct local_x update pattern
      - Same output normalization behavior as `get_intermediate_layers`
      - Returns 4 patch-feature tensors: (B, 1369, 2*embed_dim)
    """

    def __init__(self, vit: torch.nn.Module, out_layers: list[int]):
        super().__init__()
        self.vit = vit
        self.out_layers = set(int(x) for x in out_layers)

    def forward(self, pixel_values: torch.Tensor):
        x = pixel_values.unsqueeze(1)  # (B, 1, 3, H, W)
        B, S, _, H, W = x.shape

        x = self.vit.prepare_tokens_with_masks(x)  # (B, S, N, C)
        pos, pos_nodiff = self.vit._prepare_rope(B, S, H, W, x.device)

        outputs = []
        local_x = x

        for i, blk in enumerate(self.vit.blocks):
            if i < self.vit.rope_start or self.vit.rope is None:
                g_pos, l_pos = None, None
            else:
                g_pos = pos_nodiff
                l_pos = pos

            # Camera-token injection at alt_start (out-of-place to avoid in-place assign)
            if self.vit.alt_start != -1 and i == self.vit.alt_start:
                # Monocular export only (S=1): avoid zero-sized tensors (S-1==0) because they
                # tend to break downstream shape inference in coremltools.
                cam_token = self.vit.camera_token[:, :1].expand(B, S, -1).to(x.dtype)  # (B, 1, C)
                cam_token = cam_token.unsqueeze(2)  # (B, S, 1, C)
                x = torch.cat([cam_token, x[:, :, 1:, :]], dim=2)
                # In upstream, this was an in-place write so `local_x` sees it too (they share storage).
                local_x = torch.cat([cam_token, local_x[:, :, 1:, :]], dim=2)

            # Alternating local/global attention pattern (for S=1, shapes are identical but local_x updates differ)
            if self.vit.alt_start != -1 and i >= self.vit.alt_start and (i % 2 == 1):
                x = self.vit.process_attention(x, blk, "global", pos=g_pos, attn_mask=None)
            else:
                x = self.vit.process_attention(x, blk, "local", pos=l_pos)
                local_x = x

            if i in self.out_layers:
                out_x = torch.cat([local_x, x], dim=-1) if self.vit.cat_token else x
                outputs.append(out_x)

        if len(outputs) != 4:
            raise RuntimeError(f"Expected 4 outputs for out_layers={sorted(self.out_layers)}, got {len(outputs)}")

        # Match DinoVisionTransformer.get_intermediate_layers() normalization behavior.
        if outputs[0].shape[-1] == self.vit.embed_dim:
            outputs = [self.vit.norm(o) for o in outputs]
        elif outputs[0].shape[-1] == (self.vit.embed_dim * 2):
            outputs = [
                torch.cat([o[..., : self.vit.embed_dim], self.vit.norm(o[..., self.vit.embed_dim :])], dim=-1)
                for o in outputs
            ]
        else:
            raise RuntimeError(f"Unexpected output dim: {outputs[0].shape}")

        # Drop special tokens (CLS + register tokens) -> patch tokens only.
        outputs = [o[..., 1 + self.vit.num_register_tokens :, :] for o in outputs]

        # Squeeze view dimension (S=1).
        return tuple(o.squeeze(1) for o in outputs)


class _FixedPositionGetter(torch.nn.Module):
    """
    CoreML-friendly replacement for `PositionGetter`.

    The upstream implementation uses `torch.cartesian_prod`, which coremltools doesn't
    currently convert. For a fixed input size (518x518 with patch=14), the patch grid is
    always 37x37, so we can precompute and return constant (y,x) positions.

    Notes:
      - This is only intended for the fixed-shape CoreML export in this repo.
      - It assumes B*S == 1 (monocular inference, single view).
    """

    def __init__(self, positions: torch.Tensor):
        super().__init__()
        self.register_buffer("positions", positions, persistent=False)

    def forward(self, batch_size: int, height: int, width: int, device=None) -> torch.Tensor:  # type: ignore[override]
        # The exported CoreML model is fixed-shape and traced with batch_size=1; keep it simple.
        return self.positions


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert DA3 (official) DINOv2 backbone to CoreML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to DA3 checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Output .mlpackage path")
    parser.add_argument("--size", type=str, default="giant", choices=["small", "base", "large", "giant"])
    parser.add_argument("--input-size", type=int, default=518, help="Fixed input image size (default: 518)")
    parser.add_argument("--precision", type=str, default="float16", choices=["float16", "float32"])
    args = parser.parse_args()

    # Import upstream DA3 modules from ../src.
    project_root = Path(__file__).resolve().parents[1]  # .../coreml
    src_root = project_root.parent / "src"
    if not src_root.exists():
        raise RuntimeError(f"Expected DA3 source at {src_root}, but directory does not exist.")
    sys.path.insert(0, str(src_root))
    from depth_anything_3.model.dinov2.dinov2 import DinoV2  # noqa: E402

    net_cfg = _load_da3_net_config(args.checkpoint, args.size)
    name = net_cfg.get("name", "vitg")
    out_layers = net_cfg.get("out_layers", [19, 27, 33, 39])
    alt_start = int(net_cfg.get("alt_start", -1))
    qknorm_start = int(net_cfg.get("qknorm_start", -1))
    rope_start = int(net_cfg.get("rope_start", -1))
    cat_token = bool(net_cfg.get("cat_token", True))
    if not cat_token:
        raise ValueError("This CoreML pipeline assumes cat_token=True (dim_in=2*embed_dim).")

    print("=" * 72)
    print("DA3 backbone conversion (official DINOv2 implementation)")
    print("=" * 72)
    print(f"Checkpoint: {args.checkpoint}")
    print(
        "Resolved net config:",
        f"name={name}",
        f"out_layers={out_layers}",
        f"alt_start={alt_start}",
        f"qknorm_start={qknorm_start}",
        f"rope_start={rope_start}",
        f"cat_token={cat_token}",
    )

    sd = _load_state_dict(args.checkpoint)
    prefix = _guess_backbone_prefix(sd)
    backbone_sd = _extract_prefixed(sd, prefix)
    print(f"Backbone prefix: {prefix}")
    print(f"Backbone keys: {len(backbone_sd)}")

    # Build backbone and load weights.
    backbone = DinoV2(
        name=name,
        out_layers=out_layers,
        alt_start=alt_start,
        qknorm_start=qknorm_start,
        rope_start=rope_start,
        cat_token=cat_token,
    )
    missing, unexpected = backbone.pretrained.load_state_dict(backbone_sd, strict=False)
    print(f"Loaded weights: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print(f"  Missing (first 10): {missing[:10]}")
    if unexpected:
        print(f"  Unexpected (first 10): {unexpected[:10]}")
    backbone.eval()

    # Patch in a CoreML-friendly PositionGetter (avoids torch.cartesian_prod).
    # For 518x518 and patch=14 => 37x37 patches.
    ph = args.input_size // 14
    pw = args.input_size // 14
    yy, xx = np.meshgrid(np.arange(ph, dtype=np.int64), np.arange(pw, dtype=np.int64), indexing="ij")
    positions = np.stack([yy, xx], axis=-1).reshape(1, ph * pw, 2)  # (1, 1369, 2)
    backbone.pretrained.position_getter = _FixedPositionGetter(torch.from_numpy(positions))

    model = _DA3DinoV2BackboneCoreMLFriendly(backbone.pretrained, out_layers=out_layers).eval()

    example = torch.randn(1, 3, args.input_size, args.input_size)
    with torch.no_grad():
        out = model(example)
    print("Output shapes:", [tuple(t.shape) for t in out])

    traced = torch.jit.trace(model, example)
    precision = ct.precision.FLOAT16 if args.precision == "float16" else ct.precision.FLOAT32

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="pixel_values", shape=example.shape, dtype=np.float32)],
        outputs=[
            ct.TensorType(name="features_layer5"),
            ct.TensorType(name="features_layer7"),
            ct.TensorType(name="features_layer9"),
            ct.TensorType(name="features_layer11"),
        ],
        convert_to="mlprogram",
        compute_precision=precision,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS14,
    )

    embed_dim = int(backbone.pretrained.embed_dim)
    mlmodel.user_defined_metadata["patch_size"] = "14"
    mlmodel.user_defined_metadata["embed_dim"] = str(embed_dim)
    mlmodel.user_defined_metadata["output_dim"] = str(embed_dim * 2)
    mlmodel.user_defined_metadata["output_layers"] = ",".join(map(str, out_layers))
    mlmodel.user_defined_metadata["alt_start"] = str(alt_start)
    mlmodel.user_defined_metadata["qknorm_start"] = str(qknorm_start)
    mlmodel.user_defined_metadata["rope_start"] = str(rope_start)
    mlmodel.user_defined_metadata["cat_token"] = "true"
    mlmodel.author = "DA3CoreML"
    mlmodel.short_description = f"DA3 DINOv2-{args.size} backbone (official) for Depth-Anything-3 (cat_token=True)"

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(out_path))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
