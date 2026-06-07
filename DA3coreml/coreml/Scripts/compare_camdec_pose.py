#!/usr/bin/env python3
"""
Compare DA3 CamDec pose decoding when CamDec is fed:
  A) the **camera token** features (upstream DA3 default)
  B) the **patch token** features (what this CoreML repo currently does), then mean-reduced

This helps answer: "Is our CamDec usage fundamentally wrong, or just a practical approximation?"

Preprocess intentionally matches this repo's CoreML CLI:
  - square resize to input_size×input_size (default 518×518)
  - ImageNet normalization
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file as load_safetensors


def _add_src_to_path() -> None:
    repo_root = Path(__file__).resolve().parent.parent.parent
    src = repo_root / "src"
    sys.path.insert(0, str(src))


def _preprocess_square_imagenet(image_path: str, input_size: int) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    img = img.resize((input_size, input_size), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
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

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _rotation_angle_deg(R_a: torch.Tensor, R_b: torch.Tensor) -> float:
    """
    Compute angle between rotations R_a and R_b (3x3) using trace of relative rotation.
    """
    R_rel = R_b @ R_a.transpose(-1, -2)
    tr = float(torch.trace(R_rel).item())
    # Clamp due to numerical issues.
    cos = (tr - 1.0) / 2.0
    cos = max(-1.0, min(1.0, cos))
    return float(math.degrees(math.acos(cos)))


def _pretty_mat3(K: torch.Tensor) -> str:
    K = K.detach().cpu().float()
    rows = []
    for r in range(3):
        rows.append("[" + " ".join(f"{float(K[r, c]):9.3f}" for c in range(3)) + "]")
    return "\n".join(rows)


def _pretty_mat34(E: torch.Tensor) -> str:
    E = E.detach().cpu().float()
    rows = []
    for r in range(3):
        rows.append("[" + " ".join(f"{float(E[r, c]):9.4f}" for c in range(4)) + "]")
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare DA3 CamDec camera-token vs patch-token feeding")
    parser.add_argument("--checkpoint", required=True, help="Path to DA3 checkpoint (.safetensors)")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--device", default="auto", help="auto|mps|cuda|cpu (default: auto)")
    args = parser.parse_args()

    _add_src_to_path()
    from depth_anything_3.model.dinov2.dinov2 import DinoV2
    from depth_anything_3.model.cam_dec import CameraDec
    from depth_anything_3.model.utils.transform import pose_encoding_to_extri_intri

    device = _pick_device(args.device)
    print("device:", device)

    sd = load_safetensors(args.checkpoint)

    # Backbone (DA3 giant defaults)
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

    # CamDec
    camdec = CameraDec(dim_in=3072)
    cam_sd = {k.replace("model.cam_dec.", ""): v for k, v in sd.items() if k.startswith("model.cam_dec.")}
    missing, unexpected = camdec.load_state_dict(cam_sd, strict=False)
    print(f"camdec load: missing={len(missing)} unexpected={len(unexpected)}")
    camdec.eval().to(device)

    x = _preprocess_square_imagenet(args.image, args.input_size).to(device)
    imgs = x.unsqueeze(1)  # 1 1 3 H W

    with torch.no_grad():
        feats, _ = backbone(imgs)

        # feats is list[ (patch_tokens, camera_token) ] per layer
        patch_tokens = feats[-1][0]  # 1 1 N C
        cam_token = feats[-1][1]  # 1 1 C

        # A) Upstream mode: cam token (B,S,C) treated as (B,N,C) where N=S
        pose_enc_cam = camdec(cam_token)  # 1 1 9

        # B) CoreML mode approximation: patch tokens (B,1,N,C) -> (B,N,C) and mean-reduce pose enc
        pose_enc_patch_all = camdec(patch_tokens.squeeze(1))  # 1 N 9
        t_mean = pose_enc_patch_all[..., :3].mean(dim=1, keepdim=True)  # 1 1 3
        q_mean = pose_enc_patch_all[..., 3:7].mean(dim=1, keepdim=True)  # 1 1 4 (xyzw)
        fov_mean = pose_enc_patch_all[..., 7:9].mean(dim=1, keepdim=True)  # 1 1 2
        # Normalize quaternion to match Swift `CamDecCoreML.decodePose`.
        q_norm = torch.linalg.vector_norm(q_mean, dim=-1, keepdim=True).clamp_min(1e-9)
        q_mean = q_mean / q_norm
        pose_enc_patch = torch.cat([t_mean, q_mean, fov_mean], dim=-1)  # 1 1 9

        # Decode both to (c2w 3x4) + K 3x3
        E_cam, K_cam = pose_encoding_to_extri_intri(pose_enc_cam, (args.input_size, args.input_size))
        E_patch, K_patch = pose_encoding_to_extri_intri(pose_enc_patch, (args.input_size, args.input_size))

        # Remove batch/view dims: [1,1,...] -> [...]
        E_cam = E_cam[0, 0]
        E_patch = E_patch[0, 0]
        K_cam = K_cam[0, 0]
        K_patch = K_patch[0, 0]

    fx_cam, fy_cam = float(K_cam[0, 0].item()), float(K_cam[1, 1].item())
    fx_patch, fy_patch = float(K_patch[0, 0].item()), float(K_patch[1, 1].item())
    cx_cam, cy_cam = float(K_cam[0, 2].item()), float(K_cam[1, 2].item())
    cx_patch, cy_patch = float(K_patch[0, 2].item()), float(K_patch[1, 2].item())

    t_cam = E_cam[:, 3]
    t_patch = E_patch[:, 3]
    t_diff = float(torch.linalg.vector_norm(t_patch - t_cam).item())

    R_cam = E_cam[:, :3]
    R_patch = E_patch[:, :3]
    ang = _rotation_angle_deg(R_cam, R_patch)

    print("\nCamDec decode comparison (square resize, ImageNet norm):")
    print("A) camera token (upstream DA3):")
    print(_pretty_mat3(K_cam))
    print(_pretty_mat34(E_cam))

    print("\nB) patch tokens + mean-reduce (CoreML approximation):")
    print(_pretty_mat3(K_patch))
    print(_pretty_mat34(E_patch))

    print("\nDelta:")
    print(f"  |Δt| = {t_diff:.6f}")
    print(f"  ΔR angle = {ang:.3f}°")
    print(f"  Δfx = {fx_patch - fx_cam:+.3f}, Δfy = {fy_patch - fy_cam:+.3f}")
    print(f"  Δcx = {cx_patch - cx_cam:+.3f}, Δcy = {cy_patch - cy_cam:+.3f}")


if __name__ == "__main__":
    main()

