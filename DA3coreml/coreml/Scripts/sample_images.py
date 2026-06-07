#!/usr/bin/env python3
"""
Sample images from a folder - take every Nth image.

Useful when you have too many images and want to reduce processing time.

Usage:
    # Take every 3rd image
    python sample_images.py /path/to/images --every 3 --output sampled_images/

    # Take every 10th image, copy files
    python sample_images.py /path/to/images --every 10 --output sampled/ --copy

    # Take every 20th image, just list them (dry run)
    python sample_images.py /path/to/images --every 20 --dry-run

    # Take specific count (e.g., 100 evenly spaced images)
    python sample_images.py /path/to/images --count 100 --output sampled/
"""

import argparse
import os
import shutil
from pathlib import Path
from typing import List


def get_image_files(directory: str, extensions: tuple = ('.jpg', '.jpeg', '.png', '.webp', '.heic')) -> List[Path]:
    """Get all image files in directory, sorted by name."""
    path = Path(directory)
    files = []
    for ext in extensions:
        files.extend(path.glob(f'*{ext}'))
        files.extend(path.glob(f'*{ext.upper()}'))
    return sorted(set(files))


def sample_every_nth(files: List[Path], n: int) -> List[Path]:
    """Take every Nth file."""
    return files[::n]


def sample_count(files: List[Path], count: int) -> List[Path]:
    """Take evenly spaced files to get approximately `count` images."""
    if count >= len(files):
        return files
    step = len(files) / count
    indices = [int(i * step) for i in range(count)]
    return [files[i] for i in indices]


def main():
    parser = argparse.ArgumentParser(
        description="Sample images from a folder - take every Nth image",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Take every 3rd image (for 1879 images -> 626 images)
  python sample_images.py /path/to/images --every 3 --output sampled/

  # Take every 10th image (for 1879 images -> 188 images)
  python sample_images.py /path/to/images --every 10 --output sampled/

  # Take every 20th image (for 1879 images -> 94 images)
  python sample_images.py /path/to/images --every 20 --output sampled/

  # Take exactly 100 evenly spaced images
  python sample_images.py /path/to/images --count 100 --output sampled/

  # Just list what would be selected (dry run)
  python sample_images.py /path/to/images --every 5 --dry-run
        """
    )
    parser.add_argument('input_dir', help='Input directory containing images')
    parser.add_argument('--every', type=int, help='Take every Nth image')
    parser.add_argument('--count', type=int, help='Take this many evenly spaced images')
    parser.add_argument('--output', '-o', help='Output directory (default: print to stdout)')
    parser.add_argument('--copy', action='store_true', help='Copy files instead of creating symlinks')
    parser.add_argument('--dry-run', action='store_true', help='Just print what would be done')
    parser.add_argument('--extensions', default='.jpg,.jpeg,.png,.webp,.heic',
                       help='Comma-separated list of extensions (default: .jpg,.jpeg,.png,.webp,.heic)')

    args = parser.parse_args()

    if not args.every and not args.count:
        parser.error("Must specify either --every N or --count N")

    # Parse extensions
    extensions = tuple(ext.strip() for ext in args.extensions.split(','))

    # Get all images
    all_files = get_image_files(args.input_dir, extensions)
    print(f"Found {len(all_files)} images in {args.input_dir}")

    if not all_files:
        print("No images found!")
        return

    # Sample
    if args.every:
        sampled = sample_every_nth(all_files, args.every)
        print(f"Sampling every {args.every}th image -> {len(sampled)} images")
    else:
        sampled = sample_count(all_files, args.count)
        print(f"Sampling {args.count} evenly spaced images -> {len(sampled)} images")

    # Calculate time savings
    time_per_image = 40  # seconds for DA3-Giant
    original_time = len(all_files) * time_per_image / 3600
    sampled_time = len(sampled) * time_per_image / 3600
    print(f"Time estimate: {original_time:.1f}h -> {sampled_time:.1f}h (saved {original_time - sampled_time:.1f}h)")

    if args.dry_run:
        print("\nSampled files (dry run):")
        for f in sampled[:20]:
            print(f"  {f.name}")
        if len(sampled) > 20:
            print(f"  ... and {len(sampled) - 20} more")
        return

    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'Copying' if args.copy else 'Symlinking'} to {output_dir}/")
        for f in sampled:
            dest = output_dir / f.name
            if args.copy:
                shutil.copy2(f, dest)
            else:
                if dest.exists() or dest.is_symlink():
                    dest.unlink()
                # Use a *relative* symlink so the sampled folder stays portable and does not
                # embed absolute machine-specific paths.
                rel_target = os.path.relpath(f, start=output_dir)
                dest.symlink_to(rel_target)

        print(f"Done! {len(sampled)} images in {output_dir}/")
        print(f"\nTo process with DA3-Giant:")
        print(f"  .build/release/da3-coreml infer \\")
        print(f"    --backbone Models/compiled/da3_backbone_giant_official.mlmodelc \\")
        print(f"    --head Models/compiled/dualdpt_giant_da3.mlmodelc \\")
        print(f"    --model-size giant \\")
        print(f"    --output output_sampled \\")
        print(f"    --include-rays \\")
        print(f"    -v \\")
        print(f"    \"{output_dir}\"/*.jpg")
    else:
        # Just print file paths
        print("\nSampled files:")
        for f in sampled:
            print(f)


if __name__ == '__main__':
    main()
