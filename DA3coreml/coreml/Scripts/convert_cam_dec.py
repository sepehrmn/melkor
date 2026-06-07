#!/usr/bin/env python3
"""
Convert DA3 camera decoder (cam_dec) to CoreML.

Usage:
  python convert_camdec_to_coreml.py --checkpoint Models/DA3-GIANT.safetensors --size base --output camdec_base.mlpackage
"""

import argparse
import torch
import torch.nn as nn
import coremltools as ct
import numpy as np
from pathlib import Path
from safetensors.torch import load_file as load_safetensors


class CameraDec(nn.Module):
    def __init__(self, dim_in=768):
        super().__init__()
        output_dim = dim_in
        self.backbone = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
        )
        self.fc_t = nn.Linear(output_dim, 3)
        self.fc_qvec = nn.Linear(output_dim, 4)
        self.fc_fov = nn.Sequential(nn.Linear(output_dim, 2), nn.ReLU())

    def forward(self, feat):
        B, N, D = feat.shape
        x = feat.reshape(B * N, D)
        x = self.backbone(x)
        t = self.fc_t(x).reshape(B, N, 3)
        q = self.fc_qvec(x).reshape(B, N, 4)
        fov = self.fc_fov(x).reshape(B, N, 2)
        pose_enc = torch.cat([t, q, fov], dim=-1)  # (B, N, 9)
        return pose_enc


def load_state_dict(path):
    if path.endswith('.safetensors'):
        return load_safetensors(path)
    ckpt = torch.load(path, map_location='cpu')
    return ckpt.get('state_dict', ckpt.get('model', ckpt))


def main():
    parser = argparse.ArgumentParser(description="Convert DA3 camera decoder to CoreML")
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default='camdec.mlpackage')
    parser.add_argument('--size', type=str, default='base', choices=['small','base','large','giant'])
    parser.add_argument('--precision', type=str, default='float16', choices=['float16','float32'])
    parser.add_argument('--num-tokens', type=int, default=None, help='Override token count (default: 1024 for base, 1369 for giant)')
    args = parser.parse_args()

    sd = load_state_dict(args.checkpoint)
    cam_weights = {}
    for k,v in sd.items():
        if 'cam_dec' in k or 'camera_head' in k:
            nk = k.split('cam_dec.')[-1].split('camera_head.')[-1]
            cam_weights[nk] = v

    # Auto-detect hidden dim from checkpoint weights (backbone.0.weight shape is [dim, dim])
    dim_in = None
    for key in ['backbone.0.weight', '0.weight']:
        if key in cam_weights:
            dim_in = cam_weights[key].shape[0]
            print(f"Auto-detected hidden dim from '{key}': {dim_in}")
            break

    if dim_in is None:
        # Fallback to size-based defaults
        size_cfg = {
            'small': 384,
            'base': 768,
            'large': 1024,
            'giant': 3072,  # Giant camdec uses 3072, not 1536
        }
        dim_in = size_cfg[args.size]
        print(f"Using default hidden dim for {args.size}: {dim_in}")

    model = CameraDec(dim_in=dim_in)
    missing, unexpected = model.load_state_dict(cam_weights, strict=False)
    print(f"Loaded cam_dec: missing {len(missing)}, unexpected {len(unexpected)}")
    model.eval()

    # Example input: (1, num_tokens, dim_in). We use 1369 tokens (patch14) for giant; 1024 for base (patch16)
    if args.num_tokens is not None:
        num_tokens = args.num_tokens
    else:
        num_tokens = 1369 if args.size == 'giant' else 1024
    print(f"Using num_tokens: {num_tokens}")
    example = torch.randn(1, num_tokens, dim_in)

    traced = torch.jit.trace(model, example)
    precision = ct.precision.FLOAT16 if args.precision == 'float16' else ct.precision.FLOAT32

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name='features', shape=example.shape, dtype=np.float32)],
        outputs=[ct.TensorType(name='pose_enc')],
        convert_to='mlprogram',
        compute_precision=precision,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS14,
    )

    # metadata
    mlmodel.user_defined_metadata['dim_in'] = str(dim_in)
    mlmodel.user_defined_metadata['num_tokens'] = str(num_tokens)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(args.output)
    print(f"Saved cam_dec to {args.output}")


if __name__ == '__main__':
    main()
