#!/usr/bin/env python3
"""
Validate an *official* DA3 DualDPT head CoreML export against the PyTorch reference.

This is a *head-only* check:
- We feed random (float32) backbone feature tensors to both PyTorch and CoreML.
- We compare shapes, finite-ness, and max absolute error for:
  depth, depth_confidence, rays, ray_confidence

Why this matters:
- fp16 CoreML exports with `conf_activation="expp1"` can overflow `exp()` and produce NaN/Inf
  (often in `ray_confidence`). Exporting with `conf_activation="linear"` (logits) avoids this.
- This script lets you systematically verify a given export without running the full backbone.
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
    # Repo layout: <repo>/coreml/Scripts/this_file.py
    # DA3 python src: <repo>/src
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
        f5 = features_layer5.unsqueeze(1)
        f7 = features_layer7.unsqueeze(1)
        f9 = features_layer9.unsqueeze(1)
        f11 = features_layer11.unsqueeze(1)

        feats = [(f5,), (f7,), (f9,), (f11,)]
        out = self.head(feats, self.H, self.W, self.patch_start_idx, chunk_size=None)

        depth = out["depth"][:, 0].unsqueeze(1)  # [B,1,H,W]
        depth_conf = out["depth_conf"][:, 0].unsqueeze(1)
        ray = out["ray"][:, 0].permute(0, 3, 1, 2)  # [B,6,h,w]
        ray_conf = out["ray_conf"][:, 0].unsqueeze(1)
        return depth, depth_conf, ray, ray_conf


def _np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def _stats(name: str, arr: np.ndarray) -> str:
    finite = np.isfinite(arr)
    if finite.any():
        mn = float(arr[finite].min())
        mx = float(arr[finite].max())
    else:
        mn = float("nan")
        mx = float("nan")
    return f"{name}: shape={list(arr.shape)} finite={finite.mean()*100:.2f}% min={mn:.6g} max={mx:.6g}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate DA3 official DualDPT head CoreML export vs PyTorch")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to DA3 checkpoint (.safetensors or .pth)")
    parser.add_argument("--coreml", required=True, type=str, help="Path to CoreML model (.mlpackage or .mlmodelc)")
    parser.add_argument("--size", choices=["small", "base", "large", "giant"], default="giant")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument(
        "--conf-activation",
        choices=["expp1", "exp", "softplus", "linear"],
        default="expp1",
        help="Must match the PyTorch head construction used for export.",
    )
    parser.add_argument("--trials", type=int, default=3, help="Number of random trials")
    parser.add_argument("--seed", type=int, default=123, help="Base RNG seed")
    parser.add_argument("--compute-units", choices=["all", "cpu"], default="cpu", help="CoreML compute units")
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
    num_patches = (args.input_size // args.patch_size) ** 2
    feat_shape = (1, num_patches, dim_in)

    head = DualDPT(
        dim_in=dim_in,
        patch_size=args.patch_size,
        output_dim=2,
        features=256,
        pos_embed=False,
        conf_activation=args.conf_activation,
    )
    head_weights = _load_head_state_dict(Path(args.checkpoint))
    head.load_state_dict(head_weights, strict=False)
    head.eval()

    wrapper = DualDPTOfficialWrapper(head, H=H, W=W, patch_start_idx=0).eval()

    compute_units = ct.ComputeUnit.ALL if args.compute_units == "all" else ct.ComputeUnit.CPU_ONLY
    mlmodel = ct.models.MLModel(args.coreml, compute_units=compute_units)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("Validating DualDPT head export:")
    print(f"- checkpoint: {args.checkpoint}")
    print(f"- coreml:     {args.coreml} (compute_units={args.compute_units})")
    print(f"- size: {args.size} dim_in={dim_in} num_patches={num_patches} input={H}x{W} patch={args.patch_size}")
    print(f"- conf_activation: {args.conf_activation}")

    for t in range(args.trials):
        seed = args.seed + t
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Random float32 features (match CoreML signature).
        ex = [torch.randn(*feat_shape, dtype=torch.float32) for _ in range(4)]

        with torch.no_grad():
            pt_depth, pt_dconf, pt_rays, pt_rconf = wrapper(*ex)

        # CoreML expects numpy float32 inputs.
        inputs = {
            "features_layer5": _np(ex[0]).astype(np.float32),
            "features_layer7": _np(ex[1]).astype(np.float32),
            "features_layer9": _np(ex[2]).astype(np.float32),
            "features_layer11": _np(ex[3]).astype(np.float32),
        }
        out = mlmodel.predict(inputs)

        cm_depth = np.array(out["depth"])
        cm_dconf = np.array(out["depth_confidence"])
        cm_rays = np.array(out["rays"])
        cm_rconf = np.array(out["ray_confidence"])

        # Compare.
        pt = {
            "depth": _np(pt_depth),
            "depth_confidence": _np(pt_dconf),
            "rays": _np(pt_rays),
            "ray_confidence": _np(pt_rconf),
        }
        cm = {
            "depth": cm_depth,
            "depth_confidence": cm_dconf,
            "rays": cm_rays,
            "ray_confidence": cm_rconf,
        }

        print(f"\nTrial {t+1}/{args.trials} (seed={seed})")
        for k in ["depth", "depth_confidence", "rays", "ray_confidence"]:
            a = pt[k].astype(np.float32)
            b = cm[k].astype(np.float32)
            if a.shape != b.shape:
                print(f"- {k}: SHAPE MISMATCH pt={list(a.shape)} coreml={list(b.shape)}")
                continue
            finite = np.isfinite(a) & np.isfinite(b)
            if not finite.all():
                print(f"- {k}: NON-FINITE detected (pt={np.isfinite(a).mean()*100:.2f}%, coreml={np.isfinite(b).mean()*100:.2f}%)")
            diff = np.abs(a - b)
            max_err = float(np.nanmax(diff))
            mean_err = float(np.nanmean(diff))
            print(f"- {k}: max_abs_err={max_err:.6g} mean_abs_err={mean_err:.6g}")
            if k in ("depth_confidence", "ray_confidence"):
                print(f"  { _stats('pt', a) }")
                print(f"  { _stats('coreml', b) }")


if __name__ == "__main__":
    main()

