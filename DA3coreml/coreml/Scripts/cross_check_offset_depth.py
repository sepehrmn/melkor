#!/usr/bin/env python3
"""
Cross-check DA3 GSHead `offset_depth` statistics in the original PyTorch implementation.

This is useful when diagnosing "giant planes / huge scale" feed-forward 3DGS outputs:
if `offset_depth` outliers exist in upstream PyTorch, they will also exist in CoreML
unless you scale/disable/prune `offset_depth`.

This script intentionally uses a *square resize to input_size×input_size* (default 518×518)
to match this repo's CoreML CLI preprocessing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file as load_safetensors


def _add_src_to_path() -> None:
    # This repo keeps the upstream DA3 Python sources in ../src/
    repo_root = Path(__file__).resolve().parent.parent.parent
    src = repo_root / "src"
    sys.path.insert(0, str(src))


def _preprocess_square_imagenet(image_path: str, input_size: int) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    img = img.resize((input_size, input_size), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0  # H W C in [0,1]
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1 3 H W

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    return (x - mean) / std


def _pick_device(requested: str) -> torch.device:
    req = requested.lower()
    if req == "mps":
        if not torch.backends.mps.is_available():
            raise SystemExit("Requested --device mps, but torch.backends.mps.is_available() is false")
        return torch.device("mps")
    if req == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("Requested --device cuda, but torch.cuda.is_available() is false")
        return torch.device("cuda")
    if req == "cpu":
        return torch.device("cpu")

    # auto
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _stats(t: torch.Tensor) -> dict[str, float]:
    t = t.detach()
    t = t[torch.isfinite(t)]
    if t.numel() == 0:
        return {"min": float("nan"), "max": float("nan"), "mean": float("nan"), "p99": float("nan")}
    return {
        "min": float(t.min().item()),
        "max": float(t.max().item()),
        "mean": float(t.mean().item()),
        "p99": float(torch.quantile(t, 0.99).item()),
    }

def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach()
    b = b.detach()
    mask = torch.isfinite(a) & torch.isfinite(b)
    a = a[mask]
    b = b[mask]
    if a.numel() < 2:
        return float("nan")
    a = a.float()
    b = b.float()
    a0 = a - a.mean()
    b0 = b - b.mean()
    denom = (a0.std() * b0.std()).clamp_min(1e-9)
    return float(((a0 * b0).mean() / denom).item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Check DA3 PyTorch GSHead offset_depth stats")
    parser.add_argument("--checkpoint", required=True, help="Path to DA3 checkpoint (.safetensors)")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--device", default="auto", help="auto|mps|cuda|cpu (default: auto)")
    args = parser.parse_args()

    _add_src_to_path()
    from depth_anything_3.model.dinov2.dinov2 import DinoV2
    from depth_anything_3.model.dualdpt import DualDPT
    from depth_anything_3.model.gsdpt import GSDPT

    device = _pick_device(args.device)
    print("device:", device)

    sd = load_safetensors(args.checkpoint)

    # Backbone (DA3 giant = DinoV2 vitg + cat_token=True)
    backbone = DinoV2(
        name="vitg",
        out_layers=[19, 27, 33, 39],
        alt_start=13,
        qknorm_start=13,
        rope_start=13,
        cat_token=True,
    )
    backbone_sd = {k.replace("model.backbone.", ""): v for k, v in sd.items() if k.startswith("model.backbone.")}
    missing, unexpected = backbone.load_state_dict(backbone_sd, strict=False)
    print(f"backbone load: missing={len(missing)} unexpected={len(unexpected)}")
    backbone.eval().to(device)

    # Depth head (DualDPT)
    head = DualDPT(dim_in=3072, patch_size=args.patch_size, output_dim=2, features=256)
    head_sd = {k.replace("model.head.", ""): v for k, v in sd.items() if k.startswith("model.head.")}
    missing, unexpected = head.load_state_dict(head_sd, strict=False)
    print(f"head load: missing={len(missing)} unexpected={len(unexpected)}")
    head.eval().to(device)

    # GSHead (GSDPT)
    gshead = GSDPT(
        dim_in=3072,
        patch_size=args.patch_size,
        output_dim=38,
        activation="linear",
        conf_activation="sigmoid",
        features=256,
        out_channels=(256, 512, 1024, 1024),
        pos_embed=True,
        down_ratio=1,
        conf_dim=1,
        norm_type="idt",
        fusion_block_inplace=False,
    )
    gs_sd = {k.replace("model.gs_head.", ""): v for k, v in sd.items() if k.startswith("model.gs_head.")}
    missing, unexpected = gshead.load_state_dict(gs_sd, strict=False)
    print(f"gshead load: missing={len(missing)} unexpected={len(unexpected)}")
    gshead.eval().to(device)

    x = _preprocess_square_imagenet(args.image, args.input_size).to(device)
    imgs = x.unsqueeze(1)  # 1 1 3 H W

    with torch.no_grad():
        feats, _ = backbone(imgs)
        depth_outs = head(feats, args.input_size, args.input_size, patch_start_idx=0)
        depth = depth_outs["depth"][0, 0]  # H W
        gs_outs = gshead(feats=feats, H=args.input_size, W=args.input_size, patch_start_idx=0, images=imgs)
        raw_gs = gs_outs["raw_gs"][0, 0]  # H W 37
        offset_depth = raw_gs[..., -1]
        conf = gs_outs["raw_gs_conf"][0, 0]

    print("\nPyTorch stats (square resize, ImageNet norm):")
    d_stats = _stats(depth)
    o_stats = _stats(offset_depth)
    print("depth:", d_stats)
    print("offset_depth:", o_stats)
    print("ray_depth = depth + offset_depth:", _stats(depth + offset_depth))
    print("gs_conf(prob):", _stats(conf))

    if np.isfinite(d_stats["max"]) and np.isfinite(o_stats["max"]) and o_stats["max"] > 0:
        suggested = max(1e-6, min(1.0, d_stats["max"] / o_stats["max"]))
        print(f"\nSuggested CoreML CLI knob: --gs-offset-depth-scale {suggested:.6f}  (~depth_max/offset_depth_max)")

    # Show whether outliers correlate with confidence (they often do, which means minConfidence won't remove them).
    pearson = _pearson(offset_depth.flatten(), conf.flatten())
    print(f"\npearson(offset_depth, gs_conf) = {pearson:.4f}")

    # Confidence stats for the top 0.1% offset_depth pixels.
    offd = offset_depth.flatten()
    cf = conf.flatten()
    finite = torch.isfinite(offd) & torch.isfinite(cf)
    offd = offd[finite]
    cf = cf[finite]
    k = max(1, int(0.001 * offd.numel()))
    vals, idx = torch.topk(offd, k)
    cf_top = cf[idx]
    print("top 0.1% offset_depth:", _stats(vals))
    print("gs_conf for top 0.1% offset_depth:", _stats(cf_top))


if __name__ == "__main__":
    main()
