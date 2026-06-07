#!/usr/bin/env python3
"""
Convert DualDPT head from Depth-Anything-3 to CoreML format.

This script extracts and converts the DualDPT head (depth-ray decoder)
from a DA3 checkpoint to CoreML mlpackage format.

Usage:
    python convert_dualdpt_to_coreml.py --checkpoint da3_base.pth --output dualdpt_base.mlpackage
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct
from pathlib import Path
import numpy as np
from safetensors.torch import load_file as load_safetensors
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from depth_anything_3.model.dualdpt import DualDPT
    from depth_anything_3.model.dpt import DPTHead
    HAS_DA3 = True
except ImportError:
    HAS_DA3 = False
    print("Warning: DA3 source not found. Using standalone implementation.")


class BilinearUpsampleConv(nn.Module):
    """
    Bilinear upsampling followed by 3x3 convolution.

    This replaces ConvTranspose2d to avoid checkerboard/grid artifacts.
    The conv layer learns to refine the upsampled features.
    """
    def __init__(self, in_channels: int, out_channels: int, scale: int):
        super().__init__()
        self.scale = scale
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=True)
        x = self.conv(x)
        return x


class DualDPTWrapper(nn.Module):
    """
    Wrapper for DualDPT that takes multi-scale features as separate inputs.
    
    The original DualDPT expects features as a list, but CoreML needs
    named tensor inputs. This wrapper accepts 4 separate feature tensors.
    """
    
    def __init__(self, dualdpt_model):
        super().__init__()
        self.model = dualdpt_model
    
    def forward(self, feat5, feat7, feat9, feat11):
        """
        Forward pass with separate feature inputs.
        
        Args:
            feat5: Features from layer 5 - (B, N, D)
            feat7: Features from layer 7 - (B, N, D)
            feat9: Features from layer 9 - (B, N, D)
            feat11: Features from layer 11 - (B, N, D)
            
        Returns:
            depth: Depth prediction (B, 1, H, W)
            depth_conf: Depth confidence (B, 1, H, W)
            rays: Ray directions (B, 3 or 6, H, W)
            ray_conf: Ray confidence (B, 1, H, W)
        """
        features = [feat5, feat7, feat9, feat11]
        
        # Run the model
        outputs = self.model(features)
        
        # Extract outputs
        if isinstance(outputs, dict):
            depth = outputs.get('depth', outputs.get('pred_depth'))
            depth_conf = outputs.get('depth_confidence', torch.ones_like(depth))
            rays = outputs.get('rays', outputs.get('pred_ray'))
            ray_conf = outputs.get('ray_confidence', torch.ones_like(depth))
        elif isinstance(outputs, (list, tuple)):
            depth = outputs[0]
            rays = outputs[1] if len(outputs) > 1 else None
            depth_conf = outputs[2] if len(outputs) > 2 else torch.ones_like(depth)
            ray_conf = outputs[3] if len(outputs) > 3 else torch.ones_like(depth)
        else:
            depth = outputs
            depth_conf = torch.ones_like(depth)
            rays = torch.zeros(depth.shape[0], 3, depth.shape[2], depth.shape[3])
            ray_conf = torch.ones_like(depth)
        
        return depth, depth_conf, rays, ray_conf


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


class StandaloneDualDPT(nn.Module):
    """
    Standalone implementation of DualDPT matching the actual DA3 architecture.

    This implements the full DPT architecture with:
    - LayerNorm on input features
    - 4 projection layers for multi-scale features with out_channels (256, 512, 1024, 1024)
    - Resize layers for spatial alignment
    - Scratch layers (layer1_rn through layer4_rn)
    - 4 refinenets for main depth output
    - 4 refinenets for auxiliary ray output
    - Separate output convs for depth and rays
    """

    def __init__(
        self,
        dim_in: int = 3072,
        patch_size: int = 14,
        features: int = 256,
        out_channels: tuple = (256, 512, 1024, 1024),
        depth_out_dim: int = 2,  # depth + confidence
        ray_out_dim: int = 7,    # 6 ray params + confidence
    ):
        super().__init__()

        self.patch_size = patch_size
        self.dim_in = dim_in

        # Token pre-norm
        self.norm = nn.LayerNorm(dim_in)

        # Projection layers for each scale (from dim_in to out_channels)
        self.projects = nn.ModuleList([
            nn.Conv2d(dim_in, out_channels[0], kernel_size=1, stride=1, padding=0),
            nn.Conv2d(dim_in, out_channels[1], kernel_size=1, stride=1, padding=0),
            nn.Conv2d(dim_in, out_channels[2], kernel_size=1, stride=1, padding=0),
            nn.Conv2d(dim_in, out_channels[3], kernel_size=1, stride=1, padding=0),
        ])

        # Resize layers for spatial alignment (x4, x2, identity, /2)
        # Use BilinearUpsampleConv instead of ConvTranspose2d to avoid checkerboard artifacts.
        # The original checkpoint uses ConvTranspose2d, but bilinear+conv produces smoother results.
        self.resize_layers = nn.ModuleList([
            BilinearUpsampleConv(out_channels[0], out_channels[0], scale=4),
            BilinearUpsampleConv(out_channels[1], out_channels[1], scale=2),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),
        ])

        # Keep original ConvTranspose2d layers for weight loading from checkpoint
        self._orig_resize_0 = nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0)
        self._orig_resize_1 = nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0)

        # Scratch layers (adapters from out_channels to features)
        self.scratch = nn.Module()
        self.scratch.layer1_rn = nn.Conv2d(out_channels[0], features, 3, 1, 1, bias=False)
        self.scratch.layer2_rn = nn.Conv2d(out_channels[1], features, 3, 1, 1, bias=False)
        self.scratch.layer3_rn = nn.Conv2d(out_channels[2], features, 3, 1, 1, bias=False)
        self.scratch.layer4_rn = nn.Conv2d(out_channels[3], features, 3, 1, 1, bias=False)

        # Main fusion refinenets
        self.scratch.refinenet4 = FeatureFusionBlock(features, has_residual=False)
        self.scratch.refinenet3 = FeatureFusionBlock(features)
        self.scratch.refinenet2 = FeatureFusionBlock(features)
        self.scratch.refinenet1 = FeatureFusionBlock(features)

        # Main head output
        head_features_1 = features
        head_features_2 = 32
        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_features_2, depth_out_dim, kernel_size=1, stride=1, padding=0),
        )

        # Auxiliary (ray) fusion refinenets
        self.scratch.refinenet4_aux = FeatureFusionBlock(features, has_residual=False)
        self.scratch.refinenet3_aux = FeatureFusionBlock(features)
        self.scratch.refinenet2_aux = FeatureFusionBlock(features)
        self.scratch.refinenet1_aux = FeatureFusionBlock(features)

        # Auxiliary output (ray head)
        # output_conv1_aux is a ModuleList with 4 levels, each has 5 convs
        self.scratch.output_conv1_aux = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(features, features // 2, 3, 1, 1),
                nn.Conv2d(features // 2, features, 3, 1, 1),
                nn.Conv2d(features, features // 2, 3, 1, 1),
                nn.Conv2d(features // 2, features, 3, 1, 1),
                nn.Conv2d(features, features // 2, 3, 1, 1),
            ) for _ in range(4)
        ])

        # output_conv2_aux is a ModuleList with 4 levels
        # Checkpoint structure differs per level:
        # - Level 0: indices 0 (Conv), 2 (LayerNorm), 5 (Conv) - with Permutes at 1,3 and ReLU at 4
        # - Levels 1-3: indices 0 (Conv), 5 (Conv) only - no LayerNorm, simpler structure
        #
        # IMPORTANT: The forward uses output_conv2_aux[-1] which is level 3 (no LayerNorm)
        # So rays are computed WITHOUT LayerNorm in the actual inference path
        class Permute(nn.Module):
            def __init__(self, dims):
                super().__init__()
                self.dims = dims
            def forward(self, x):
                return x.permute(self.dims)

        # Level 0 with LayerNorm (checkpoint has indices 0, 2, 5)
        level0 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),  # 0
            Permute((0, 2, 3, 1)),  # 1
            nn.LayerNorm(head_features_2),  # 2
            Permute((0, 3, 1, 2)),  # 3
            nn.ReLU(inplace=True),  # 4
            nn.Conv2d(head_features_2, ray_out_dim, kernel_size=1, stride=1, padding=0),  # 5
        )

        # Levels 1-3: checkpoint only has indices 0 and 5 (Conv layers, no LayerNorm)
        # Must match exact indices for weight loading
        class Level1to3(nn.Module):
            def __init__(self):
                super().__init__()
                # Use ModuleDict with string keys to match checkpoint indices "0" and "5"
                self._modules['0'] = nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1)
                self._modules['5'] = nn.Conv2d(head_features_2, ray_out_dim, kernel_size=1, stride=1, padding=0)

            def forward(self, x):
                x = self._modules['0'](x)
                x = F.relu(x)  # ReLU between convs
                x = self._modules['5'](x)
                return x

        levels_1_3 = [Level1to3() for _ in range(3)]
        self.scratch.output_conv2_aux = nn.ModuleList([level0] + levels_1_3)

    def forward(self, features):
        """
        Forward pass.

        Args:
            features: List of 4 feature tensors from backbone layers
                     Each tensor has shape (B, N, D) where N = H*W/patch_size^2

        Returns:
            depth: (B, 1, H, W)
            depth_conf: (B, 1, H, W)
            rays: (B, 6, H, W)
            ray_conf: (B, 1, H, W)
        """
        feat0, feat1, feat2, feat3 = features

        # Get dimensions
        B, N, D = feat0.shape
        H = W = int(np.sqrt(N))

        resized_feats = []
        for stage_idx, feat in enumerate([feat0, feat1, feat2, feat3]):
            x = self.norm(feat)
            # Reshape from (B, N, D) to (B, D, H, W)
            # IMPORTANT: .contiguous() before reshape avoids checkerboard/grid artifacts
            x = x.permute(0, 2, 1).contiguous().reshape(B, D, H, W)
            x = self.projects[stage_idx](x)
            x = self.resize_layers[stage_idx](x)
            resized_feats.append(x)

        l1, l2, l3, l4 = resized_feats

        # Apply layer adapters
        l1_rn = self.scratch.layer1_rn(l1)
        l2_rn = self.scratch.layer2_rn(l2)
        l3_rn = self.scratch.layer3_rn(l3)
        l4_rn = self.scratch.layer4_rn(l4)

        # Main fusion: 4 -> 3 -> 2 -> 1
        out = self.scratch.refinenet4(l4_rn, size=l3_rn.shape[2:])
        aux_out = self.scratch.refinenet4_aux(l4_rn, size=l3_rn.shape[2:])

        out = self.scratch.refinenet3(out, l3_rn, size=l2_rn.shape[2:])
        aux_out = self.scratch.refinenet3_aux(aux_out, l3_rn, size=l2_rn.shape[2:])

        out = self.scratch.refinenet2(out, l2_rn, size=l1_rn.shape[2:])
        aux_out = self.scratch.refinenet2_aux(aux_out, l2_rn, size=l1_rn.shape[2:])

        out = self.scratch.refinenet1(out, l1_rn)
        aux_out = self.scratch.refinenet1_aux(aux_out, l1_rn)

        # Main output conv
        out = self.scratch.output_conv1(out)
        aux_out = self.scratch.output_conv1_aux[-1](aux_out)  # Use last level

        # Upsample to full resolution
        h_out = H * self.patch_size
        w_out = W * self.patch_size
        out = F.interpolate(out, size=(h_out, w_out), mode='bilinear', align_corners=True)
        aux_out = F.interpolate(aux_out, size=(h_out, w_out), mode='bilinear', align_corners=True)

        # Depth output
        depth_logits = self.scratch.output_conv2(out)
        depth = torch.exp(depth_logits[:, :1, :, :])  # exp activation
        depth_conf = torch.exp(depth_logits[:, 1:2, :, :]) + 1  # expp1 activation

        # Ray output
        ray_logits = self.scratch.output_conv2_aux[-1](aux_out)
        rays = ray_logits[:, :6, :, :]  # linear activation
        ray_conf = torch.exp(ray_logits[:, 6:7, :, :]) + 1  # expp1 activation

        return depth, depth_conf, rays, ray_conf


class DualDPTCoreMLExporter:
    """Export DualDPT to CoreML format."""

    def __init__(self, checkpoint_path: str = None, model_size: str = "base", allow_fallback: bool = False, patch_size: int = 14, dim_in_override: int = None):
        self.checkpoint_path = checkpoint_path
        self.model_size = model_size
        self.model = None
        self.allow_fallback = allow_fallback
        self.patch_size_override = patch_size
        self.dim_in_override = dim_in_override
        
        # Size configs
        # Note: dim_in is 2x backbone dim due to cat_token=True in DinoV2
        # The backbone outputs 1536-dim but cat_token concatenates local_x and x
        self.size_configs = {
            "small": {"dim_in": 768, "features": 128},   # 384*2
            "base": {"dim_in": 1536, "features": 256},   # 768*2
            "large": {"dim_in": 2048, "features": 256},  # 1024*2
            "giant": {"dim_in": 3072, "features": 256},  # 1536*2
        }
    
    def load_model(self):
        """Load DualDPT model from checkpoint or create standalone."""
        config = self.size_configs.get(self.model_size, self.size_configs["base"]).copy()

        # Allow override of dim_in for DINOv3 compatibility
        if self.dim_in_override is not None:
            print(f"Using dim_in override: {self.dim_in_override}")
            config["dim_in"] = self.dim_in_override

        if self.checkpoint_path:
            print(f"Loading DualDPT from checkpoint: {self.checkpoint_path}")
            if self.checkpoint_path.endswith('.safetensors'):
                state_dict = load_safetensors(self.checkpoint_path)
            else:
                checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
                state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))

            # Auto-detect dim_in from checkpoint if possible
            for key in ['model.head.norm.weight', 'model.head.projects.0.weight']:
                if key in state_dict:
                    if 'norm.weight' in key:
                        detected_dim = state_dict[key].shape[0]
                    else:
                        detected_dim = state_dict[key].shape[1]  # Conv2d shape is [out, in, H, W]
                    print(f"Auto-detected dim_in from '{key}': {detected_dim}")
                    config["dim_in"] = detected_dim
                    break

            # Choose model impl - always use standalone since it matches checkpoint structure
            model = StandaloneDualDPT(
                dim_in=config["dim_in"],
                patch_size=self.patch_size_override,
                features=config["features"],
                depth_out_dim=2,
                ray_out_dim=7
            )

            # Extract weights with model.head. prefix
            head_weights = {}
            for k, v in state_dict.items():
                if k.startswith('model.head.'):
                    new_key = k[len('model.head.'):]
                    head_weights[new_key] = v

            print(f"Found {len(head_weights)} head weights in checkpoint")

            missing, unexpected = model.load_state_dict(head_weights, strict=False)
            print(f"Loaded checkpoint with missing: {len(missing)}, unexpected: {len(unexpected)}")
            if missing:
                print(f"  Missing (first 10): {missing[:10]}")
            if unexpected:
                print(f"  Unexpected (first 10): {unexpected[:10]}")

            # Transfer ConvTranspose2d weights to BilinearUpsampleConv layers
            # The BilinearUpsampleConv uses bilinear upsampling + 3x3 conv instead of ConvTranspose2d
            # We initialize the conv weights to approximate the ConvTranspose2d behavior
            with torch.no_grad():
                # Load original resize layer weights
                for key in ['resize_layers.0.weight', 'resize_layers.0.bias',
                           'resize_layers.1.weight', 'resize_layers.1.bias']:
                    if key in head_weights:
                        # Store in the hidden ConvTranspose2d layers
                        parts = key.split('.')
                        idx = int(parts[1])
                        param_name = parts[2]
                        if idx == 0:
                            if param_name == 'weight':
                                model._orig_resize_0.weight.copy_(head_weights[key])
                            else:
                                model._orig_resize_0.bias.copy_(head_weights[key])
                        elif idx == 1:
                            if param_name == 'weight':
                                model._orig_resize_1.weight.copy_(head_weights[key])
                            else:
                                model._orig_resize_1.bias.copy_(head_weights[key])

                # Initialize BilinearUpsampleConv conv layers to identity-ish mapping
                # The bilinear upsample does the spatial work, conv refines
                for i, layer in enumerate(model.resize_layers):
                    if isinstance(layer, BilinearUpsampleConv):
                        # Initialize conv to near-identity (preserve upsampled values)
                        nn.init.kaiming_normal_(layer.conv.weight, mode='fan_out', nonlinearity='relu')
                        if layer.conv.bias is not None:
                            nn.init.zeros_(layer.conv.bias)

            print("Replaced ConvTranspose2d with BilinearUpsampleConv to reduce checkerboard artifacts")

            self.model = DualDPTWrapper(model)
        else:
            if not self.allow_fallback:
                raise ValueError(
                    "No checkpoint provided and DA3 sources unavailable. "
                    "Pass --checkpoint to export trained weights or use --allow-fallback "
                    "to generate an untrained CoreML head (not recommended)."
                )

            print(f"Creating standalone DualDPT (size: {self.model_size}) [UNTRAINED FALLBACK]")
            self.model = StandaloneDualDPT(
                dim_in=config["dim_in"],
                features=config["features"]
            )
        
        self.model.eval()
        return self.model
    
    def convert_to_coreml(
        self,
        output_path: str,
        input_size: int = 518,
        compute_precision: str = "float16"
    ):
        """Convert DualDPT to CoreML format."""
        if self.model is None:
            self.load_model()

        config = self.size_configs.get(self.model_size, self.size_configs["base"]).copy()
        if self.dim_in_override is not None:
            config["dim_in"] = self.dim_in_override
        dim_in = config["dim_in"]

        # Ensure model has no gradients for tracing
        for param in self.model.parameters():
            param.requires_grad_(False)
        
        # Calculate feature dimensions
        patch_size = self.patch_size_override or 14
        num_patches = (input_size // patch_size) ** 2
        
        # Create example inputs
        feat_shape = (1, num_patches, dim_in)
        example_inputs = (
            torch.randn(*feat_shape),
            torch.randn(*feat_shape),
            torch.randn(*feat_shape),
            torch.randn(*feat_shape),
        )
        
        # Trace the model
        print("Tracing model...")

        # Create a proper wrapper module for standalone model
        class ForwardModule(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, f5, f7, f9, f11):
                return self.model([f5, f7, f9, f11])

        with torch.no_grad():
            if isinstance(self.model, DualDPTWrapper):
                traced = torch.jit.trace(self.model, example_inputs)
            else:
                # Wrap standalone model in a proper Module
                wrapper = ForwardModule(self.model)
                wrapper.eval()
                traced = torch.jit.trace(wrapper, example_inputs)
        
        # Convert to CoreML
        print("Converting to CoreML...")
        precision = ct.precision.FLOAT16 if compute_precision == "float16" else ct.precision.FLOAT32
        
        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name="features_layer5", shape=feat_shape, dtype=np.float32),
                ct.TensorType(name="features_layer7", shape=feat_shape, dtype=np.float32),
                ct.TensorType(name="features_layer9", shape=feat_shape, dtype=np.float32),
                ct.TensorType(name="features_layer11", shape=feat_shape, dtype=np.float32),
            ],
            # NOTE: CoreML assigns outputs in a different order than Python return tuple.
            # After conversion, CoreML spec shows these shapes:
            #   Output 0: shape [1,1,H,W] - gets name position 0
            #   Output 1: shape [1,6,H,W] - gets name position 1 (this is rays!)
            #   Output 2: shape [1,1,H,W] - gets name position 2 (this is depth_conf!)
            #   Output 3: shape [1,1,H,W] - gets name position 3
            # So we swap names to match the actual tensor content:
            outputs=[
                ct.TensorType(name="depth"),           # pos0: (1,1) - actual depth
                ct.TensorType(name="rays"),            # pos1: (1,6) - actual rays
                ct.TensorType(name="depth_confidence"),# pos2: (1,1) - actual depth_conf
                ct.TensorType(name="ray_confidence"),  # pos3: (1,1) - actual ray_conf
            ],
            convert_to="mlprogram",
            compute_precision=precision,
            compute_units=ct.ComputeUnit.ALL,
            minimum_deployment_target=ct.target.macOS14,
        )
        
        # Add metadata
        mlmodel.author = "DA3CoreML"
        mlmodel.short_description = f"DualDPT head ({self.model_size}) for Depth-Anything-3"
        mlmodel.version = "1.0"
        
        # Save
        print(f"Saving to {output_path}")
        mlmodel.save(output_path)
        
        return mlmodel


def main():
    parser = argparse.ArgumentParser(description="Convert DualDPT to CoreML")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to DA3 checkpoint (optional)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dualdpt.mlpackage",
        help="Output path for CoreML model"
    )
    parser.add_argument(
        "--size",
        type=str,
        choices=["small", "base", "large", "giant"],
        default="base",
        help="Model size"
    )
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Allow exporting an untrained fallback head when no checkpoint is provided"
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=14,
        help="Backbone patch size (use 16 for DINOv3 base/large)."
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=518,
        help="Input image size (default: 518)"
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["float16", "float32"],
        default="float16",
        help="Compute precision"
    )
    parser.add_argument(
        "--dim-in",
        type=int,
        default=None,
        help="Override backbone dim_in (e.g., 2048 for DINOv3-Large, 2560 for DINOv3-H+)"
    )

    args = parser.parse_args()
    
    exporter = DualDPTCoreMLExporter(
        checkpoint_path=args.checkpoint,
        model_size=args.size,
        allow_fallback=args.allow_fallback,
        patch_size=args.patch_size,
        dim_in_override=args.dim_in,
    )
    
    exporter.convert_to_coreml(
        output_path=args.output,
        input_size=args.input_size,
        compute_precision=args.precision
    )
    
    print(f"\nConversion complete! Model saved to: {args.output}")
    print("\nTo use in Swift:")
    print(f'  let head = try DualDPTCoreML(modelPath: "{args.output}")')


if __name__ == "__main__":
    main()
