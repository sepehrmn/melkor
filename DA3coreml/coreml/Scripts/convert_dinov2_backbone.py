#!/usr/bin/env python3
"""
Convert DINOv2 backbone from HuggingFace to CoreML.

This creates a PURE DINOv2 backbone (not the DA3-modified version) which can be
used for depth estimation. This is simpler and more portable than the full DA3 backbone.

DINOv2 Giant specs:
- Hidden dim: 1536
- Patch size: 14
- Num layers: 40
- Num heads: 24
- No cat_token (single output per layer, not doubled)

Usage:
  python convert_dinov2_backbone.py --size giant --output Models/dinov2_giant.mlpackage

  # Or with specific HuggingFace model:
  python convert_dinov2_backbone.py --hf-model facebook/dinov2-giant --output Models/dinov2_giant.mlpackage
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct
import numpy as np
from pathlib import Path


# Model size configurations
DINO_CONFIGS = {
    'small': {
        'hf_name': 'facebook/dinov2-small',
        'embed_dim': 384,
        'depth': 12,
        'num_heads': 6,
        'output_layers': [2, 5, 8, 11],
    },
    'base': {
        'hf_name': 'facebook/dinov2-base',
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
        'output_layers': [2, 5, 8, 11],
    },
    'large': {
        'hf_name': 'facebook/dinov2-large',
        'embed_dim': 1024,
        'depth': 24,
        'num_heads': 16,
        'output_layers': [5, 11, 17, 23],
    },
    'giant': {
        'hf_name': 'facebook/dinov2-giant',
        'embed_dim': 1536,
        'depth': 40,
        'num_heads': 24,
        'output_layers': [9, 19, 29, 39],
    },
}


class DINOv2MultiScaleWrapper(nn.Module):
    """
    Wrapper around HuggingFace DINOv2 to extract multi-scale features.

    Outputs 4 feature tensors from intermediate layers for DPT-style fusion.
    """

    def __init__(self, model, output_layers):
        super().__init__()
        self.model = model
        self.output_layers = output_layers
        self.embed_dim = model.config.hidden_size

    def forward(self, pixel_values):
        """
        Args:
            pixel_values: (B, 3, H, W) normalized image tensor

        Returns:
            Tuple of 4 tensors, each (B, num_patches, hidden_dim)
        """
        # Get intermediate hidden states
        outputs = self.model(
            pixel_values,
            output_hidden_states=True,
            return_dict=True
        )

        hidden_states = outputs.hidden_states  # List of (B, seq_len, hidden_dim)

        # Extract features from specified layers
        # Remove CLS token (first token) for DPT compatibility
        features = []
        for layer_idx in self.output_layers:
            # hidden_states[0] is embedding output, hidden_states[1] is after first block, etc.
            layer_output = hidden_states[layer_idx + 1]  # +1 because index 0 is embedding
            # Remove CLS token
            patch_features = layer_output[:, 1:, :]
            features.append(patch_features)

        return tuple(features)


def main():
    parser = argparse.ArgumentParser(description="Convert DINOv2 to CoreML")
    parser.add_argument('--size', type=str, default='giant',
                        choices=['small', 'base', 'large', 'giant'],
                        help='Model size')
    parser.add_argument('--hf-model', type=str, default=None,
                        help='HuggingFace model name (overrides --size)')
    parser.add_argument('--output', type=str, default='dinov2.mlpackage',
                        help='Output path')
    parser.add_argument('--input-size', type=int, default=518,
                        help='Input image size')
    parser.add_argument('--precision', type=str, default='float16',
                        choices=['float16', 'float32'])
    args = parser.parse_args()

    config = DINO_CONFIGS[args.size]
    hf_model_name = args.hf_model or config['hf_name']

    print("=" * 60)
    print(f"Converting DINOv2 to CoreML")
    print("=" * 60)
    print(f"Model: {hf_model_name}")
    print(f"Size: {args.size}")
    print(f"Input size: {args.input_size}")
    print(f"Output layers: {config['output_layers']}")
    print()

    # Load HuggingFace model
    print("Loading HuggingFace model...")
    from transformers import AutoModel
    hf_model = AutoModel.from_pretrained(hf_model_name, trust_remote_code=True)
    hf_model.eval()

    print(f"Hidden size: {hf_model.config.hidden_size}")
    print(f"Num layers: {hf_model.config.num_hidden_layers}")
    print(f"Patch size: {getattr(hf_model.config, 'patch_size', 14)}")

    # Wrap for multi-scale output
    model = DINOv2MultiScaleWrapper(hf_model, config['output_layers'])
    model.eval()

    # Test forward pass
    print("\nTesting forward pass...")
    example = torch.randn(1, 3, args.input_size, args.input_size)
    with torch.no_grad():
        outputs = model(example)

    print(f"Output shapes:")
    for i, o in enumerate(outputs):
        print(f"  Layer {config['output_layers'][i]}: {o.shape}")

    # Trace for CoreML
    print("\nTracing model...")

    # Custom trace that handles the HF model properly
    class TracableWrapper(nn.Module):
        def __init__(self, wrapper):
            super().__init__()
            self.wrapper = wrapper

        def forward(self, x):
            return self.wrapper(x)

    traceable = TracableWrapper(model)

    with torch.no_grad():
        traced = torch.jit.trace(traceable, example)

    # Convert to CoreML
    print("Converting to CoreML...")
    precision = ct.precision.FLOAT16 if args.precision == 'float16' else ct.precision.FLOAT32

    # Determine output names based on output layers
    output_names = [f'features_layer{i}' for i in range(len(config['output_layers']))]

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name='pixel_values', shape=example.shape, dtype=np.float32)],
        outputs=[ct.TensorType(name=name) for name in output_names],
        convert_to='mlprogram',
        compute_precision=precision,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS14,
    )

    # Add metadata
    mlmodel.user_defined_metadata['model_type'] = 'dinov2'
    mlmodel.user_defined_metadata['size'] = args.size
    mlmodel.user_defined_metadata['patch_size'] = '14'
    mlmodel.user_defined_metadata['embed_dim'] = str(config['embed_dim'])
    mlmodel.user_defined_metadata['depth'] = str(config['depth'])
    mlmodel.user_defined_metadata['output_layers'] = ','.join(map(str, config['output_layers']))
    mlmodel.user_defined_metadata['hf_model'] = hf_model_name
    mlmodel.author = 'DA3CoreML'
    mlmodel.short_description = f'DINOv2-{args.size} backbone for depth estimation'

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(args.output)
    print(f"\nSaved to: {args.output}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    print(f"\nTo use with DPT head, note:")
    print(f"  - Output dim: {config['embed_dim']} (NOT doubled like DA3's cat_token)")
    print(f"  - You may need to adjust the DPT head dim_in accordingly")
    print(f"  - Or use the DA3 backbone converter for cat_token=True behavior")


if __name__ == '__main__':
    main()
