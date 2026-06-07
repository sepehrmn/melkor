#!/usr/bin/env python3
"""
Convert DA3 GS (Gaussian Splatting) head to CoreML.

The GS head outputs 38 channels per pixel:
- 2: offset_xy (pixel offset)
- 3: scales (Gaussian scales)
- 4: quaternion (rotation)
- 27: SH coefficients (3 * 9 for sh_degree=2)
- 1: offset_depth
- 1: confidence/opacity

Usage:
  python convert_gshead_to_coreml.py --checkpoint Models/DA3-GIANT.safetensors --size giant --output Models/gshead_giant.mlpackage

Notes:
- Upstream DA3's GS head (`depth_anything_3/model/gsdpt.py`) uses UV positional embeddings by default (`pos_embed=True`).
- The CoreML export should include the same pos-embed behavior for parity with Python/CUDA inference.
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct
import numpy as np
from pathlib import Path
from safetensors.torch import load_file as load_safetensors


def make_sincos_pos_embed(embed_dim: int, pos: torch.Tensor, omega_0: float = 100.0) -> torch.Tensor:
    """
    Match DA3's `make_sincos_pos_embed` (see `depth_anything_3/model/utils/head_utils.py`):
    returns float32 embeddings in the range [-1, 1].
    """
    assert embed_dim % 2 == 0, "embed_dim must be even"
    half = embed_dim // 2
    omega = torch.arange(half, dtype=torch.float32, device=pos.device)
    omega /= float(half)
    omega = 1.0 / (omega_0 ** omega)

    pos = pos.reshape(-1).float()
    out = pos[:, None] * omega[None, :]
    emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)
    return emb.float()


def create_uv_coords(
    width: int,
    height: int,
    aspect_ratio: float = 1.0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Match DA3's `create_uv_grid` spans (see `depth_anything_3/model/utils/head_utils.py`),
    but return 1D coordinate vectors so we can build the full embedding efficiently.
    """
    diag_factor = (aspect_ratio**2 + 1.0) ** 0.5
    span_x = aspect_ratio / diag_factor
    span_y = 1.0 / diag_factor

    left_x = -span_x * (width - 1) / width
    right_x = span_x * (width - 1) / width
    top_y = -span_y * (height - 1) / height
    bottom_y = span_y * (height - 1) / height

    x_coords = torch.linspace(left_x, right_x, steps=width, dtype=dtype, device=device)
    y_coords = torch.linspace(top_y, bottom_y, steps=height, dtype=dtype, device=device)
    return x_coords, y_coords


def build_uv_pos_embed_nchw(
    embed_dim: int,
    height: int,
    width: int,
    *,
    aspect_ratio: float = 1.0,
    ratio: float = 0.1,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Build DA3-style UV sinusoidal positional embedding in NCHW layout:
      (1, C, H, W), where C = embed_dim.

    This is equivalent to DA3's `DPT._add_pos_embed` but computed with separability
    (x and y are independent) to avoid creating a full HxWx2 grid.
    """
    assert embed_dim % 2 == 0, "embed_dim must be even"
    half = embed_dim // 2
    assert half % 2 == 0, "embed_dim/2 must be even"

    x_coords, y_coords = create_uv_coords(
        width=width,
        height=height,
        aspect_ratio=aspect_ratio,
        dtype=torch.float32,
        device=device,
    )

    emb_x = make_sincos_pos_embed(half, x_coords)  # [W, C/2]
    emb_y = make_sincos_pos_embed(half, y_coords)  # [H, C/2]

    # Broadcast to NCHW and concatenate as [x_embed, y_embed] along channels.
    pe_x = emb_x.t().unsqueeze(0).unsqueeze(2).expand(1, half, height, width)  # 1 C/2 H W
    pe_y = emb_y.t().unsqueeze(0).unsqueeze(3).expand(1, half, height, width)  # 1 C/2 H W
    pe = torch.cat([pe_x, pe_y], dim=1) * float(ratio)
    return pe.contiguous()


class ResidualConvUnit(nn.Module):
    """Lightweight residual conv block used within fusion."""
    def __init__(self, features: int):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(features, features, 3, 1, 1, bias=True)
        self.activation = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        return out + x


class FeatureFusionBlock(nn.Module):
    """Top-down fusion block: residual merge + upsample + 1x1 shrink."""
    def __init__(self, features: int, has_residual: bool = True):
        super().__init__()
        self.has_residual = has_residual
        self.resConfUnit1 = ResidualConvUnit(features) if has_residual else None
        self.resConfUnit2 = ResidualConvUnit(features)
        self.out_conv = nn.Conv2d(features, features, 1, 1, 0, bias=True)

    def forward(self, x, res=None, size=None):
        y = x
        if self.has_residual and res is not None and self.resConfUnit1 is not None:
            y = y + self.resConfUnit1(res)
        y = self.resConfUnit2(y)
        if size is not None:
            y = F.interpolate(y, size=size, mode='bilinear', align_corners=True)
        else:
            y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
        y = self.out_conv(y)
        return y


class StandaloneGSHead(nn.Module):
    """
    Standalone implementation of GSDPT (Gaussian Splatting DPT head).

    Outputs 38 channels:
    - 2: offset_xy
    - 3: scales
    - 4: quaternion
    - 27: SH coefficients (3 * 9)
    - 1: offset_depth
    - 1: confidence
    """

    def __init__(
        self,
        dim_in: int = 3072,
        patch_size: int = 14,
        input_size: int = 518,
        features: int = 256,
        out_channels: tuple = (256, 512, 1024, 1024),
        gs_out_dim: int = 38,  # 37 GS params + 1 confidence
        pos_embed: bool = True,
        pos_embed_ratio: float = 0.1,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.dim_in = dim_in
        self.input_size = int(input_size)
        self.pos_embed = pos_embed
        self.pos_embed_ratio = float(pos_embed_ratio)

        # Token pre-norm (identity for GS head based on norm_type="idt")
        self.norm = nn.Identity()

        # Projection layers for each scale
        self.projects = nn.ModuleList([
            nn.Conv2d(dim_in, out_channels[0], kernel_size=1, stride=1, padding=0),
            nn.Conv2d(dim_in, out_channels[1], kernel_size=1, stride=1, padding=0),
            nn.Conv2d(dim_in, out_channels[2], kernel_size=1, stride=1, padding=0),
            nn.Conv2d(dim_in, out_channels[3], kernel_size=1, stride=1, padding=0),
        ])

        # Resize layers for spatial alignment
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),
        ])

        # Scratch layers
        self.scratch = nn.Module()
        self.scratch.layer1_rn = nn.Conv2d(out_channels[0], features, 3, 1, 1, bias=False)
        self.scratch.layer2_rn = nn.Conv2d(out_channels[1], features, 3, 1, 1, bias=False)
        self.scratch.layer3_rn = nn.Conv2d(out_channels[2], features, 3, 1, 1, bias=False)
        self.scratch.layer4_rn = nn.Conv2d(out_channels[3], features, 3, 1, 1, bias=False)

        # Fusion refinenets
        self.scratch.refinenet4 = FeatureFusionBlock(features, has_residual=False)
        self.scratch.refinenet3 = FeatureFusionBlock(features)
        self.scratch.refinenet2 = FeatureFusionBlock(features)
        self.scratch.refinenet1 = FeatureFusionBlock(features)

        # Output convolutions
        head_features_1 = features
        head_features_2 = 32
        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)

        # Images merger (inject RGB into features)
        self.images_merger = nn.Sequential(
            nn.Conv2d(3, head_features_1 // 8, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(head_features_1 // 8, head_features_1 // 4, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(head_features_1 // 4, head_features_1 // 2, 3, 1, 1),
            nn.GELU(),
        )

        # Final output (38 channels)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_features_2, gs_out_dim, kernel_size=1, stride=1, padding=0),
        )

        # DA3 UV positional embeddings (pos_embed=True by default in upstream GSDPT).
        #
        # DA3 applies pos-embed:
        #  - per-stage on the projected patch grid features (ph=pw=37 for 518@patch14)
        #  - once at full output resolution (H=W=518) after image-merger injection
        #
        # The CoreML export assumes a fixed input size (default 518×518), so we precompute
        # the embeddings for those known shapes to avoid tracing/conversion issues.
        if self.pos_embed:
            ph = pw = self.input_size // self.patch_size
            # DA3 uses aspect_ratio = W/H from the input image. For fixed 518×518 this is 1.
            aspect_ratio = 1.0

            stage_pe = []
            for oc in out_channels:
                pe = build_uv_pos_embed_nchw(
                    embed_dim=int(oc),
                    height=ph,
                    width=pw,
                    aspect_ratio=aspect_ratio,
                    ratio=self.pos_embed_ratio,
                    device=torch.device("cpu"),
                ).to(dtype=torch.float16)
                stage_pe.append(pe)

            self.register_buffer("pos_embed_stage0", stage_pe[0], persistent=False)
            self.register_buffer("pos_embed_stage1", stage_pe[1], persistent=False)
            self.register_buffer("pos_embed_stage2", stage_pe[2], persistent=False)
            self.register_buffer("pos_embed_stage3", stage_pe[3], persistent=False)

            pe_out = build_uv_pos_embed_nchw(
                embed_dim=head_features_1 // 2,  # 128 for features=256
                height=self.input_size,
                width=self.input_size,
                aspect_ratio=aspect_ratio,
                ratio=self.pos_embed_ratio,
                device=torch.device("cpu"),
            ).to(dtype=torch.float16)
            self.register_buffer("pos_embed_out", pe_out, persistent=False)

    def forward(self, features, image):
        """
        Forward pass.

        Args:
            features: List of 4 feature tensors (B, N, D)
            image: RGB image tensor (B, 3, H, W)

        Returns:
            gs_params: (B, 38, H, W) Gaussian splatting parameters
        """
        feat0, feat1, feat2, feat3 = features

        B, N, D = feat0.shape
        # For CoreML export we assume fixed square inputs, so patch grid is deterministic.
        H = W = self.input_size // self.patch_size

        resized_feats = []
        for stage_idx, feat in enumerate([feat0, feat1, feat2, feat3]):
            x = self.norm(feat)
            x = x.permute(0, 2, 1).contiguous().reshape(B, D, H, W)
            x = self.projects[stage_idx](x)
            if self.pos_embed:
                pe = getattr(self, f"pos_embed_stage{stage_idx}")
                x = x.float() + pe.to(device=x.device).float()
            x = self.resize_layers[stage_idx](x)
            resized_feats.append(x)

        l1, l2, l3, l4 = resized_feats

        # Apply layer adapters
        l1_rn = self.scratch.layer1_rn(l1)
        l2_rn = self.scratch.layer2_rn(l2)
        l3_rn = self.scratch.layer3_rn(l3)
        l4_rn = self.scratch.layer4_rn(l4)

        # Fusion: 4 -> 3 -> 2 -> 1
        out = self.scratch.refinenet4(l4_rn, size=l3_rn.shape[2:])
        out = self.scratch.refinenet3(out, l3_rn, size=l2_rn.shape[2:])
        out = self.scratch.refinenet2(out, l2_rn, size=l1_rn.shape[2:])
        out = self.scratch.refinenet1(out, l1_rn)

        # Output conv1
        out = self.scratch.output_conv1(out)

        # Upsample to full resolution
        h_out = H * self.patch_size
        w_out = W * self.patch_size
        out = F.interpolate(out, size=(h_out, w_out), mode='bilinear', align_corners=True)

        # Inject image features
        out = out + self.images_merger(image)
        if self.pos_embed:
            out = out.float() + self.pos_embed_out.to(device=out.device).float()

        # Final output
        gs_params = self.scratch.output_conv2(out)

        return gs_params


class GSHeadWrapper(nn.Module):
    """Wrapper that takes separate feature inputs for CoreML."""

    def __init__(self, gshead_model):
        super().__init__()
        self.model = gshead_model

    def forward(self, feat5, feat7, feat9, feat11, image):
        features = [feat5, feat7, feat9, feat11]
        return self.model(features, image)


def load_state_dict(path):
    if path.endswith('.safetensors'):
        return load_safetensors(path)
    ckpt = torch.load(path, map_location='cpu')
    return ckpt.get('state_dict', ckpt.get('model', ckpt))


def main():
    parser = argparse.ArgumentParser(description="Convert DA3 GS head to CoreML")
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default='gshead_giant.mlpackage')
    parser.add_argument('--size', type=str, default='giant', choices=['small', 'base', 'large', 'giant'])
    parser.add_argument('--precision', type=str, default='float16', choices=['float16', 'float32'])
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--patch-size', type=int, default=14)
    parser.add_argument('--pos-embed', dest='pos_embed', action='store_true', help='Enable DA3 UV positional embedding (default)')
    parser.add_argument('--no-pos-embed', dest='pos_embed', action='store_false', help='Disable positional embedding (debug)')
    parser.set_defaults(pos_embed=True)
    parser.add_argument('--pos-embed-ratio', type=float, default=0.1, help='Pos-embed scale ratio (DA3 default: 0.1)')
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    sd = load_state_dict(args.checkpoint)

    # Extract gs_head weights
    gs_weights = {}
    for k, v in sd.items():
        if k.startswith('model.gs_head.'):
            new_key = k[len('model.gs_head.'):]
            gs_weights[new_key] = v

    print(f"Found {len(gs_weights)} gs_head weights")

    # Auto-detect dim from weights
    if 'projects.0.weight' in gs_weights:
        dim_in = gs_weights['projects.0.weight'].shape[1]
        print(f"Auto-detected dim_in: {dim_in}")
    else:
        dim_in = 3072

    # Create model
    model = StandaloneGSHead(
        dim_in=dim_in,
        patch_size=args.patch_size,
        input_size=args.input_size,
        features=256,
        gs_out_dim=38,
        pos_embed=args.pos_embed,
        pos_embed_ratio=args.pos_embed_ratio,
    )

    missing, unexpected = model.load_state_dict(gs_weights, strict=False)
    print(f"Loaded weights: missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print(f"  Missing (first 10): {missing[:10]}")
    if unexpected:
        print(f"  Unexpected (first 10): {unexpected[:10]}")

    model.eval()
    wrapped = GSHeadWrapper(model)
    wrapped.eval()

    # Example inputs
    num_patches = (args.input_size // args.patch_size) ** 2
    feat_shape = (1, num_patches, dim_in)
    img_shape = (1, 3, args.input_size, args.input_size)

    example_feats = tuple(torch.randn(*feat_shape) for _ in range(4))
    example_img = torch.randn(*img_shape)

    print("Testing forward pass...")
    with torch.no_grad():
        out = wrapped(*example_feats, example_img)
    print(f"Output shape: {out.shape}")

    print("Tracing model...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapped, (*example_feats, example_img))

    print("Converting to CoreML...")
    precision = ct.precision.FLOAT16 if args.precision == 'float16' else ct.precision.FLOAT32

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name='features_layer5', shape=feat_shape, dtype=np.float32),
            ct.TensorType(name='features_layer7', shape=feat_shape, dtype=np.float32),
            ct.TensorType(name='features_layer9', shape=feat_shape, dtype=np.float32),
            ct.TensorType(name='features_layer11', shape=feat_shape, dtype=np.float32),
            ct.TensorType(name='image', shape=img_shape, dtype=np.float32),
        ],
        outputs=[ct.TensorType(name='gs_params')],
        convert_to='mlprogram',
        compute_precision=precision,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS14,
    )

    # Metadata
    mlmodel.user_defined_metadata['dim_in'] = str(dim_in)
    mlmodel.user_defined_metadata['gs_out_dim'] = '38'
    mlmodel.user_defined_metadata['patch_size'] = str(args.patch_size)
    mlmodel.author = 'DA3CoreML'
    mlmodel.short_description = f'DA3 GS Head ({args.size}) for Gaussian Splatting'

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(args.output)
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
