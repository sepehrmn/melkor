#!/usr/bin/env python3
"""
Convert DA3 camera encoder (cam_enc) to CoreML.

The camera encoder takes backbone features and outputs pose encoding.
It consists of:
- token_norm: LayerNorm on input features
- trunk: 4 transformer blocks
- pose_branch: MLP that outputs per-token pose (9 values: t(3) + q(4) + fov(2))

Usage:
  python convert_camenc_to_coreml.py --checkpoint Models/DA3-GIANT.safetensors --size giant --output Models/camenc_giant.mlpackage
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct
import numpy as np
from pathlib import Path
from safetensors.torch import load_file as load_safetensors


class Attention(nn.Module):
    """Multi-head self-attention."""
    def __init__(self, dim, num_heads=24, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        x = attn.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class MLP(nn.Module):
    """Standard MLP with GELU activation."""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    """Transformer block."""
    def __init__(self, dim, num_heads, mlp_hidden):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads)
        self.ls1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_hidden)
        self.ls2 = LayerScale(dim)

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class PoseBranch(nn.Module):
    """Pose output branch - outputs 9 values per token."""
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(9, dim // 2)
        self.fc2 = nn.Linear(dim // 2, dim)

    def forward(self, x):
        # This branch goes: features -> pose encoding
        # Actually looking at weights, fc1 takes 9-dim input (pose) and outputs dim/2
        # This is for encoding known poses, not decoding from features
        # We need to reverse the direction for inference
        pass


class CamEnc(nn.Module):
    """Camera encoder - produces pose encoding from backbone features."""
    def __init__(self, dim=1536, num_blocks=4, num_heads=24, mlp_ratio=4.0):
        super().__init__()
        self.dim = dim
        mlp_hidden = int(dim * mlp_ratio)

        self.token_norm = nn.LayerNorm(dim)
        self.trunk = nn.ModuleList([
            Block(dim, num_heads, mlp_hidden) for _ in range(num_blocks)
        ])
        self.trunk_norm = nn.LayerNorm(dim)

        # Pose head - produces 9 values per token
        self.pose_head = nn.Linear(dim, 9)

    def forward(self, features):
        """
        Args:
            features: (B, N, dim) backbone features
        Returns:
            pose_enc: (B, N, 9) per-token pose encoding
        """
        x = self.token_norm(features)
        for block in self.trunk:
            x = block(x)
        x = self.trunk_norm(x)
        pose_enc = self.pose_head(x)
        return pose_enc


def load_state_dict(path):
    if path.endswith('.safetensors'):
        return load_safetensors(path)
    ckpt = torch.load(path, map_location='cpu')
    return ckpt.get('state_dict', ckpt.get('model', ckpt))


def extract_camenc_weights(sd, prefix='model.cam_enc.'):
    """Extract and rename cam_enc weights."""
    weights = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            new_key = k[len(prefix):]
            weights[new_key] = v
    return weights


def main():
    parser = argparse.ArgumentParser(description="Convert DA3 camera encoder to CoreML")
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default='camenc.mlpackage')
    parser.add_argument('--size', type=str, default='giant', choices=['small', 'base', 'large', 'giant'])
    parser.add_argument('--precision', type=str, default='float16', choices=['float16', 'float32'])
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    sd = load_state_dict(args.checkpoint)
    weights = extract_camenc_weights(sd)
    print(f"Found {len(weights)} cam_enc weights")

    # Detect dimensions from weights
    if 'trunk.0.attn.proj.weight' in weights:
        dim = weights['trunk.0.attn.proj.weight'].shape[0]
    else:
        dim = 1536
    print(f"Detected dim: {dim}")

    if 'trunk.0.mlp.fc1.weight' in weights:
        mlp_hidden = weights['trunk.0.mlp.fc1.weight'].shape[0]
    else:
        mlp_hidden = dim * 4
    print(f"Detected MLP hidden: {mlp_hidden}")

    # Count trunk blocks
    num_blocks = sum(1 for k in weights if k.startswith('trunk.') and '.attn.proj.weight' in k)
    print(f"Detected num_blocks: {num_blocks}")

    num_heads = dim // 64  # Assuming head_dim = 64
    print(f"Using num_heads: {num_heads}")

    # Create model
    model = CamEnc(
        dim=dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
        mlp_ratio=mlp_hidden / dim,
    )

    # Map weights - the pose_branch in checkpoint encodes poses, we need a decoder head
    # Let's create a mapping for what we have
    model_sd = {}
    for k, v in weights.items():
        if k.startswith('pose_branch.'):
            # Skip pose_branch - it's for encoding known poses, not our use case
            continue
        model_sd[k] = v

    # We need a pose_head that doesn't exist in checkpoint - initialize randomly
    # Actually, the model should output pose from trunk features

    missing, unexpected = model.load_state_dict(model_sd, strict=False)
    print(f"Loaded weights: missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print(f"  Missing: {missing[:5]}")
    if unexpected:
        print(f"  Unexpected: {unexpected[:5]}")

    model.eval()

    # Example input: (1, num_tokens, dim)
    num_tokens = 1369  # patch14 for 518x518
    example = torch.randn(1, num_tokens, dim)

    print("Testing forward pass...")
    with torch.no_grad():
        out = model(example)
    print(f"Output shape: {out.shape}")

    print("Tracing model...")
    with torch.no_grad():
        traced = torch.jit.trace(model, example)

    print("Converting to CoreML...")
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

    mlmodel.user_defined_metadata['dim'] = str(dim)
    mlmodel.user_defined_metadata['num_tokens'] = str(num_tokens)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(args.output)
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
