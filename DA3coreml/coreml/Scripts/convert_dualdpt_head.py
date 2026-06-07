#!/usr/bin/env python3
"""
Convert the *official* Depth-Anything-3 DualDPT head to CoreML.

Why this script exists:
- The existing CoreML DualDPT conversion path in this repo uses a standalone re-implementation
  with a bilinear-upsample "fix". That path can produce NaNs for `rays` / `ray_confidence`
  when converted to float16, which breaks downstream camera/ray usage.
- This script instead converts the upstream DA3 `DualDPT` implementation (with the same
  architecture/ops as training), and wraps it into a CoreML-friendly signature:
    inputs:  features_layer{5,7,9,11}  float32  [1, 1369, dim_in]
    outputs: depth, depth_confidence   float16  [1, 1, 518, 518]
             rays, ray_confidence      float16  [1, 6, 296, 296] and [1, 1, 296, 296]

Notes:
- DA3's PyTorch head returns rays at the *aux* resolution (296×296 for 518 input, patch=14).
  DA3CoreML will resize rays to the requested target size in postprocess.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import coremltools as ct
from safetensors.torch import load_file as load_safetensors


def _add_da3_src_to_path() -> None:
    # Repo layout: <repo>/DA3coreml/coreml/Scripts/this_file.py
    # DA3 python src: <repo>/DA3coreml/src
    da3_src = Path(__file__).resolve().parent.parent.parent / "src"
    sys.path.insert(0, str(da3_src))


def _load_head_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    if checkpoint_path.suffix == ".safetensors":
        state_dict = load_safetensors(str(checkpoint_path))
    else:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))

    head_weights: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k.startswith("model.head."):
            head_weights[k[len("model.head.") :]] = v
    return head_weights


class DualDPTOfficialWrapper(torch.nn.Module):
    def __init__(self, head: torch.nn.Module, *, H: int, W: int, patch_start_idx: int) -> None:
        super().__init__()
        self.head = head
        self.H = H
        self.W = W
        self.patch_start_idx = patch_start_idx

    def forward(
        self,
        features_layer5: torch.Tensor,
        features_layer7: torch.Tensor,
        features_layer9: torch.Tensor,
        features_layer11: torch.Tensor,
    ):
        # Input tensors are patch tokens only: [B, N, C]
        # DA3's DualDPT expects a list of 4 items where each item is indexable and `[0]`
        # is a token tensor of shape [B, S, N, C]. We set S=1.
        f5 = features_layer5.unsqueeze(1)
        f7 = features_layer7.unsqueeze(1)
        f9 = features_layer9.unsqueeze(1)
        f11 = features_layer11.unsqueeze(1)

        feats = [(f5,), (f7,), (f9,), (f11,)]
        out = self.head(feats, self.H, self.W, self.patch_start_idx, chunk_size=None)

        # DA3 returns:
        #   depth:      [B, S, H, W]
        #   depth_conf: [B, S, H, W]
        #   ray:        [B, S, h, w, 6]
        #   ray_conf:   [B, S, h, w]
        depth = out["depth"][:, 0].unsqueeze(1)          # [B, 1, H, W]
        depth_conf = out["depth_conf"][:, 0].unsqueeze(1)  # [B, 1, H, W]
        ray = out["ray"][:, 0].permute(0, 3, 1, 2)       # [B, 6, h, w]
        ray_conf = out["ray_conf"][:, 0].unsqueeze(1)    # [B, 1, h, w]

        return depth, depth_conf, ray, ray_conf


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert DA3 official DualDPT head to CoreML")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to DA3 checkpoint (.safetensors or .pth)")
    parser.add_argument("--output", required=True, type=str, help="Output .mlpackage path")
    parser.add_argument("--size", choices=["small", "base", "large", "giant"], default="giant")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--precision", choices=["float16", "float32"], default="float16")
    parser.add_argument(
        "--conf-activation",
        choices=["expp1", "exp", "softplus", "linear"],
        default="expp1",
        help=(
            "Confidence activation inside the exported head. "
            "Use 'linear' to export *logits* and apply a stable activation in Swift/MPSGraph."
        ),
    )
    args = parser.parse_args()

    _add_da3_src_to_path()
    from depth_anything_3.model.dualdpt import DualDPT  # noqa: E402

    size_to_dim_in = {
        "small": 768,
        "base": 1536,
        "large": 2048,
        "giant": 3072,
    }
    dim_in = size_to_dim_in[args.size]
    H = W = args.input_size

    # NOTE: DA3's default `pos_embed=True` path uses `torch.meshgrid` inside `create_uv_grid`,
    # which coremltools currently fails to convert (expects 1D tensors). We disable pos-embed
    # here to keep the conversion path fully supported.
    head = DualDPT(
        dim_in=dim_in,
        patch_size=args.patch_size,
        output_dim=2,
        features=256,
        pos_embed=False,
        conf_activation=args.conf_activation,
    )
    head_weights = _load_head_state_dict(Path(args.checkpoint))
    missing, unexpected = head.load_state_dict(head_weights, strict=False)
    print(f"Loaded head weights: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  Missing (first 10):", missing[:10])
    if unexpected:
        print("  Unexpected (first 10):", unexpected[:10])
    head.eval()

    # Patch tokens only: 37x37 = 1369 for 518 input + patch=14.
    num_patches = (args.input_size // args.patch_size) ** 2
    feat_shape = (1, num_patches, dim_in)

    wrapper = DualDPTOfficialWrapper(head, H=H, W=W, patch_start_idx=0)
    wrapper.eval()

    # Trace with example inputs
    ex = tuple(torch.randn(*feat_shape) for _ in range(4))
    traced = torch.jit.trace(wrapper, ex)

    precision = ct.precision.FLOAT16 if args.precision == "float16" else ct.precision.FLOAT32
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="features_layer5", shape=feat_shape, dtype=np.float32),
            ct.TensorType(name="features_layer7", shape=feat_shape, dtype=np.float32),
            ct.TensorType(name="features_layer9", shape=feat_shape, dtype=np.float32),
            ct.TensorType(name="features_layer11", shape=feat_shape, dtype=np.float32),
        ],
        outputs=[
            ct.TensorType(name="depth"),
            ct.TensorType(name="depth_confidence"),
            ct.TensorType(name="rays"),
            ct.TensorType(name="ray_confidence"),
        ],
        convert_to="mlprogram",
        compute_precision=precision,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS14,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Persist key export settings so Swift can auto-configure without hardcoding assumptions.
    mlmodel.user_defined_metadata["size"] = args.size
    mlmodel.user_defined_metadata["input_size"] = str(args.input_size)
    mlmodel.user_defined_metadata["patch_size"] = str(args.patch_size)
    mlmodel.user_defined_metadata["precision"] = args.precision
    mlmodel.user_defined_metadata["conf_activation"] = args.conf_activation
    mlmodel.author = "DA3CoreML"
    mlmodel.short_description = f"DualDPT head ({args.size}) - official DA3 implementation"
    mlmodel.version = "1.0"
    mlmodel.save(str(out_path))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
