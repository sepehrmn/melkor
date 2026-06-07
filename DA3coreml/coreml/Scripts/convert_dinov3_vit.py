#!/usr/bin/env python3
"""
Convert DINOv3 Vision Transformer to CoreML format.

This script converts DINOv3 models from HuggingFace to CoreML mlpackage format
with float16 precision for efficient inference on Apple Silicon.

Usage:
    python convert_dinov3_to_coreml.py --model facebook/dinov3-vitb16-pretrain-lvd1689m --output dinov3_vitb.mlpackage
    python convert_dinov3_to_coreml.py --model facebook/dinov3-vit7b16-pretrain-lvd1689m --output dinov3_vit7b.mlpackage
"""

import argparse
import torch
import torch.nn as nn
import coremltools as ct
from pathlib import Path
import numpy as np

# Workaround: coremltools 7.x cannot handle the explicit `scale` parameter in
# torch.nn.functional.scaled_dot_product_attention. We fold the scale into q and
# call the original with scale=None so the converter sees a supported pattern.
import torch.nn.functional as F
_orig_sdpa = F.scaled_dot_product_attention


def _sdpa_no_scale(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    if scale is not None:
        q = q * scale
        scale = None
    return _orig_sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)


F.scaled_dot_product_attention = _sdpa_no_scale

# Try to import from transformers (HuggingFace)
try:
    from transformers import AutoModel, AutoImageProcessor
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers not installed. Install with: pip install transformers")


# Workaround: Replace bicubic interpolation with bilinear for CoreML compatibility
# CoreML doesn't support upsample_bicubic2d, but bilinear is fine
_orig_interpolate = F.interpolate

def _interpolate_bilinear_fallback(*args, **kwargs):
    """Replace bicubic with bilinear for CoreML compatibility."""
    if kwargs.get('mode') == 'bicubic':
        kwargs['mode'] = 'bilinear'
        # bilinear doesn't support antialias
        kwargs.pop('antialias', None)
    return _orig_interpolate(*args, **kwargs)

F.interpolate = _interpolate_bilinear_fallback


class DINOv3Wrapper(nn.Module):
    """
    Wrapper for DINOv2/v3 model that extracts multi-scale features for DPT head.

    Token structure differs between DINOv2 and DINOv3:
    - DINOv2: [CLS] + patch_tokens (1 token to skip)
    - DINOv3: [CLS] + 4 register tokens + patch_tokens (5 tokens to skip)

    For 518x518 input with patch_size=14: 518/14 = 37, 37*37 = 1369 patches

    We extract features from layers [5, 7, 9, 11] for the 4-level DPT pyramid.

    cat_token: If True, concatenates CLS token to each patch token, doubling the
               output dimension (required for DA3 which uses cat_token=True).
    """

    def __init__(self, model, output_layers=[5, 7, 9, 11], model_name="", cat_token=False):
        super().__init__()
        self.model = model
        self.output_layers = output_layers
        self.model_name = model_name.lower()
        self.cat_token = cat_token

        # Determine tokens to skip based on model type
        # DINOv2: skip 1 (CLS only)
        # DINOv3: skip 5 (CLS + 4 registers)
        if 'dinov3' in self.model_name or 'dino_v3' in self.model_name:
            self.skip_tokens = 5
            print(f"  Token skip mode: 5 (DINOv3 - CLS + 4 registers)")
        else:
            # Default to DINOv2 behavior (most common)
            self.skip_tokens = 1
            print(f"  Token skip mode: 1 (DINOv2 - CLS only)")

        if cat_token:
            print(f"  cat_token: True (output dim = 2 * hidden_dim)")

        # Register hooks to capture intermediate features
        self.features = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks on encoder layers to capture intermediate features."""
        def get_hook(layer_idx):
            def hook(module, input, output):
                self.features[layer_idx] = output[0] if isinstance(output, tuple) else output
            return hook
        
        # Access encoder layers - structure varies by model
        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layer'):
            layers = self.model.encoder.layer
        elif hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'blocks'):
            layers = self.model.encoder.blocks
        elif hasattr(self.model, 'blocks'):
            layers = self.model.blocks
        elif hasattr(self.model, 'layer'):
            layers = self.model.layer  # dinov3_vit uses `layer` ModuleList
        else:
            raise ValueError("Unknown model architecture - cannot find encoder layers")
        
        for idx in self.output_layers:
            if idx < len(layers):
                layers[idx].register_forward_hook(get_hook(idx))
    
    def forward(self, pixel_values):
        """
        Forward pass extracting multi-scale features.

        Args:
            pixel_values: Input tensor of shape (B, 3, H, W)

        Returns:
            Tuple of 4 feature tensors from layers [5, 7, 9, 11]
            Each tensor has shape:
              - (B, num_patches, hidden_dim) if cat_token=False
              - (B, num_patches, 2*hidden_dim) if cat_token=True
        """
        self.features = {}

        # Run forward pass
        _ = self.model(pixel_values)

        # Extract features from specified layers
        outputs = []
        for idx in self.output_layers:
            feat = self.features.get(idx)
            if feat is not None:
                # Get CLS token (always at position 0)
                cls_token = feat[:, 0:1, :]  # (B, 1, D)
                # Get patch tokens (skip special tokens)
                patch_tokens = feat[:, self.skip_tokens:, :]  # (B, N, D)

                if self.cat_token:
                    # Broadcast CLS token to match number of patches and concatenate
                    B, N, D = patch_tokens.shape
                    cls_expanded = cls_token.expand(B, N, D)  # (B, N, D)
                    # Concatenate along feature dimension: (B, N, 2*D)
                    patch_tokens = torch.cat([patch_tokens, cls_expanded], dim=-1)

                outputs.append(patch_tokens)

        return tuple(outputs)


class DINOv3CoreMLExporter:
    """Export DINOv3 models to CoreML format with optimizations."""
    
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
        
    def load_model(self):
        """Load DINOv3 model from HuggingFace."""
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers library required. Install with: pip install transformers")
        
        print(f"Loading model: {self.model_name}")
        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        
        self.model.eval()
        self.model.to(self.device)
        
        # Get model config
        config = self.model.config
        print(f"Model config:")
        print(f"  Hidden size: {config.hidden_size}")
        print(f"  Num layers: {config.num_hidden_layers}")
        print(f"  Num heads: {config.num_attention_heads}")
        print(f"  Patch size: {getattr(config, 'patch_size', 16)}")
        
        return self.model
    
    def create_wrapper(self, output_layers=[5, 7, 9, 11], cat_token=False):
        """Create wrapper that extracts multi-scale features."""
        if self.model is None:
            self.load_model()
        return DINOv3Wrapper(self.model, output_layers, model_name=self.model_name, cat_token=cat_token)
    
    def convert_to_coreml(
        self,
        output_path: str,
        input_shape: tuple = (1, 3, 518, 518),
        compute_precision: str = "float16",
        output_layers: list = [5, 7, 9, 11],
        cat_token: bool = False
    ):
        """
        Convert DINOv3 to CoreML mlpackage format.

        Args:
            output_path: Path to save .mlpackage
            input_shape: Input tensor shape (B, C, H, W)
            compute_precision: "float16" or "float32"
            output_layers: Which encoder layers to extract features from
            cat_token: If True, concatenate CLS token to patches (2x output dim)
        """
        wrapper = self.create_wrapper(output_layers, cat_token=cat_token)
        wrapper.eval()
        
        # Create example input
        example_input = torch.randn(*input_shape)
        
        # Trace the model
        print("Tracing model...")
        with torch.no_grad():
            traced_model = torch.jit.trace(wrapper, example_input)
        
        # Convert to CoreML
        print("Converting to CoreML...")
        precision = ct.precision.FLOAT16 if compute_precision == "float16" else ct.precision.FLOAT32
        
        mlmodel = ct.convert(
            traced_model,
            inputs=[
                ct.TensorType(
                    name="pixel_values",
                    shape=input_shape,
                    dtype=np.float32
                )
            ],
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

        # Attach minimal metadata so Swift can auto-configure patch/register tokens
        try:
            patch_size = getattr(self.model.config, "patch_size", None)
            if patch_size is None:
                patch_size = 14  # sensible default for DINOv2
            register_tokens = 4 if "dinov3" in self.model_name.lower() else 0
            if patch_size is not None:
                mlmodel.user_defined_metadata["patch_size"] = str(patch_size)
            mlmodel.user_defined_metadata["register_tokens"] = str(register_tokens)
        except Exception as e:
            print(f"[WARN] Failed to attach metadata: {e}")
        
        # Add metadata
        mlmodel.author = "DA3CoreML"
        mlmodel.short_description = f"DINOv3 Vision Transformer ({self.model_name}) for Depth-Anything-3"
        mlmodel.version = "1.0"
        
        # Save
        print(f"Saving to {output_path}")
        mlmodel.save(output_path)
        
        return mlmodel


def convert_dinov2_fallback(output_path: str, model_size: str = "base"):
    """
    Fallback: Convert DINOv2 to CoreML if DINOv3 is not available.
    
    DINOv2 is used in the original DA3 implementation.
    """
    print(f"Converting DINOv2 {model_size} as fallback...")
    
    # Map size to model name
    size_map = {
        "small": "facebook/dinov2-small",
        "base": "facebook/dinov2-base", 
        "large": "facebook/dinov2-large",
        "giant": "facebook/dinov2-giant",
    }
    
    model_name = size_map.get(model_size, size_map["base"])
    
    exporter = DINOv3CoreMLExporter(model_name)
    exporter.convert_to_coreml(
        output_path=output_path,
        input_shape=(1, 3, 518, 518),
        compute_precision="float16",
        output_layers=[5, 7, 9, 11]
    )


def main():
    parser = argparse.ArgumentParser(description="Convert DINOv3/DINOv2 to CoreML")
    parser.add_argument(
        "--model", 
        type=str, 
        default="facebook/dinov2-base",
        help="HuggingFace model name (e.g., facebook/dinov3-vitb16-pretrain-lvd1689m)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dinov3.mlpackage",
        help="Output path for CoreML model"
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=518,
        help="Input image size (default: 518 for DA3)"
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["float16", "float32"],
        default="float16",
        help="Compute precision"
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="5,7,9,11",
        help="Comma-separated layer indices to extract features from"
    )
    parser.add_argument(
        "--cat-token",
        action="store_true",
        help="Concatenate CLS token to each patch (2x output dim, required for DA3)"
    )

    args = parser.parse_args()

    output_layers = [int(x) for x in args.layers.split(",")]

    exporter = DINOv3CoreMLExporter(args.model)
    exporter.convert_to_coreml(
        output_path=args.output,
        input_shape=(1, 3, args.input_size, args.input_size),
        compute_precision=args.precision,
        output_layers=output_layers,
        cat_token=args.cat_token
    )

    print(f"\nConversion complete! Model saved to: {args.output}")
    print("\nTo use in Swift:")
    print(f'  let model = try DINOv3CoreML(modelPath: "{args.output}")')


if __name__ == "__main__":
    main()
