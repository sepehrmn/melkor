#!/usr/bin/env python3
"""
Test DA3 PyTorch implementation directly to verify depth output quality.

This script runs the original DA3 model and compares output with CoreML.
"""

import sys
import argparse
from pathlib import Path

# Add DA3 source to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
# matplotlib is optional
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def load_image(path, size=518):
    """Load and preprocess image for DA3."""
    img = Image.open(path).convert('RGB')
    orig_size = img.size

    # Resize maintaining aspect ratio
    scale = size / max(img.size)
    new_w = int(img.size[0] * scale)
    new_h = int(img.size[1] * scale)
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)

    # Pad to square
    padded = Image.new('RGB', (size, size), (0, 0, 0))
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    padded.paste(img_resized, (pad_x, pad_y))

    # Convert to tensor
    tensor = torch.from_numpy(np.array(padded)).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

    # Normalize with ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std

    return tensor, orig_size, (pad_x, pad_y, new_w, new_h)


def turbo_colormap(value):
    """Turbo colormap for depth visualization."""
    t = np.clip(value, 0, 1)
    r = 0.13572138 + t * (4.6153926 + t * (-42.66032258 + t * (132.13108234 + t * (-152.94239396 + t * 59.28637943))))
    g = 0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604))))
    b = 0.1066733 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771 + t * (-89.90310912 + t * 27.34824973))))
    return np.clip(np.stack([r, g, b], axis=-1), 0, 1)


def test_da3_pytorch(checkpoint_path, image_path, output_dir):
    """Test DA3 PyTorch model directly."""
    print("=" * 60)
    print("Testing DA3 PyTorch Implementation")
    print("=" * 60)

    try:
        from depth_anything_3.model.da3 import DepthAnything3Net
        from depth_anything_3.model.dinov2.dinov2 import DinoV2
        from depth_anything_3.model.dualdpt import DualDPT
        from safetensors.torch import load_file as load_safetensors
        print("DA3 modules loaded successfully")
    except ImportError as e:
        print(f"Error importing DA3: {e}")
        print("Make sure the DA3 source is in ../src/depth_anything_3/")
        return

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    if checkpoint_path.endswith('.safetensors'):
        state_dict = load_safetensors(checkpoint_path)
    else:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt.get('model', ckpt))

    # Detect model size from checkpoint
    for key in state_dict:
        if 'backbone' in key and 'blocks.39' in key:
            print("Detected: GIANT model (40 blocks)")
            model_size = 'giant'
            break
    else:
        model_size = 'base'
        print("Detected: BASE model")

    # Create backbone
    backbone_config = {
        'giant': {'name': 'vitg', 'out_layers': [19, 27, 33, 39]},
        'base': {'name': 'vitb', 'out_layers': [2, 5, 8, 11]},
    }[model_size]

    print(f"\nCreating DinoV2 backbone: {backbone_config['name']}")
    backbone = DinoV2(
        name=backbone_config['name'],
        out_layers=backbone_config['out_layers'],
        cat_token=True,
    )

    # Create DualDPT head
    dim_in = {'giant': 3072, 'base': 1536}[model_size]
    print(f"Creating DualDPT head with dim_in={dim_in}")
    head = DualDPT(
        dim_in=dim_in,
        patch_size=14,
        output_dim=2,
        features=256,
    )

    # Load weights
    # `DinoV2` wraps the actual ViT under `self.pretrained`, so keep that prefix.
    backbone_sd = {k.replace('model.backbone.', ''): v
                   for k, v in state_dict.items()
                   if k.startswith('model.backbone.')}
    head_sd = {k.replace('model.head.', ''): v
               for k, v in state_dict.items()
               if k.startswith('model.head.')}

    print(f"Loading {len(backbone_sd)} backbone weights, {len(head_sd)} head weights")

    missing_b, unexpected_b = backbone.load_state_dict(backbone_sd, strict=False)
    missing_h, unexpected_h = head.load_state_dict(head_sd, strict=False)

    print(f"Backbone: missing={len(missing_b)}, unexpected={len(unexpected_b)}")
    print(f"Head: missing={len(missing_h)}, unexpected={len(unexpected_h)}")

    backbone.eval().to(device)
    head.eval().to(device)

    # Load image
    print(f"\nProcessing image: {image_path}")
    img_tensor, orig_size, (pad_x, pad_y, scaled_w, scaled_h) = load_image(image_path, 518)
    img_tensor = img_tensor.to(device)
    print(f"Original size: {orig_size}, Scaled: ({scaled_w}, {scaled_h})")

    # Run inference
    print("\nRunning backbone...")
    with torch.no_grad():
        # Backbone expects [B, S, C, H, W] format for multi-view
        # For single image, S=1
        img_input = img_tensor.unsqueeze(1)  # (1, 1, 3, 518, 518)
        features, _ = backbone(img_input)

        print(f"Feature shapes: {[f[0].shape for f in features]}")

        # Run head
        print("Running DualDPT head...")
        H, W = 518, 518
        # `get_intermediate_layers` already strips non-patch tokens, so patch_start_idx=0.
        patch_start_idx = 0

        outputs = head(features, H, W, patch_start_idx)

        depth = outputs['depth']
        depth_conf = outputs['depth_conf']
        rays = outputs.get('ray')

        print(f"Depth shape: {depth.shape}")
        print(f"Depth range: [{depth.min().item():.4f}, {depth.max().item():.4f}]")

    # Visualize
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get depth as numpy
    depth_np = depth[0, 0].cpu().numpy()  # Remove batch and view dims
    print(f"Depth numpy shape: {depth_np.shape}")

    # Normalize for visualization
    depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)

    # Apply colormap
    depth_colored = turbo_colormap(depth_norm)
    depth_uint8 = (depth_colored * 255).astype(np.uint8)

    # Save
    output_path = output_dir / "depth_pytorch_direct.png"
    Image.fromarray(depth_uint8).save(output_path)
    print(f"\nSaved depth visualization to: {output_path}")

    # Save raw depth
    np.save(output_dir / "depth_pytorch_direct.npy", depth_np)
    print(f"Saved raw depth to: {output_dir / 'depth_pytorch_direct.npy'}")

    # Also save at original resolution
    depth_tensor = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0)
    depth_upsampled = F.interpolate(
        depth_tensor,
        size=(orig_size[1], orig_size[0]),  # (H, W)
        mode='bilinear',
        align_corners=True
    )[0, 0].numpy()

    depth_up_norm = (depth_upsampled - depth_upsampled.min()) / (depth_upsampled.max() - depth_upsampled.min() + 1e-8)
    depth_up_colored = turbo_colormap(depth_up_norm)
    depth_up_uint8 = (depth_up_colored * 255).astype(np.uint8)

    output_path_full = output_dir / "depth_pytorch_fullres.png"
    Image.fromarray(depth_up_uint8).save(output_path_full)
    print(f"Saved full-res depth to: {output_path_full}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Test DA3 PyTorch implementation")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to DA3 checkpoint')
    parser.add_argument('--image', type=str, required=True, help='Input image path')
    parser.add_argument('--output', type=str, default='output_pytorch_test', help='Output directory')
    args = parser.parse_args()

    test_da3_pytorch(args.checkpoint, args.image, args.output)


if __name__ == '__main__':
    main()
