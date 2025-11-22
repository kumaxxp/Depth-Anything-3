#!/usr/bin/env python3
"""
NPZ file viewer for Depth Anything 3 outputs.
"""

import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path


def view_npz(npz_path: str, output_dir: str = None):
    """View contents of NPZ file and optionally save visualizations."""
    
    if not Path(npz_path).exists():
        print(f"❌ File not found: {npz_path}")
        return False
    
    # Load NPZ file
    data = np.load(npz_path)
    
    print("=" * 70)
    print(f"📦 NPZ File: {npz_path}")
    print("=" * 70)
    print()
    
    # Display all keys and their shapes
    print("📋 Contents:")
    print("-" * 70)
    for key in data.files:
        arr = data[key]
        print(f"🔹 {key:20s} | Shape: {str(arr.shape):20s} | Type: {arr.dtype}")
    print()
    
    # Detailed info for each array
    print("📊 Detailed Information:")
    print("-" * 70)
    for key in data.files:
        arr = data[key]
        print(f"\n🔹 {key}:")
        print(f"   Shape:      {arr.shape}")
        print(f"   Data type:  {arr.dtype}")
        
        if arr.size > 0:
            if np.issubdtype(arr.dtype, np.number):
                print(f"   Min value:  {arr.min():.6f}")
                print(f"   Max value:  {arr.max():.6f}")
                print(f"   Mean:       {arr.mean():.6f}")
                print(f"   Std:        {arr.std():.6f}")
            
            # Show sample values for small arrays
            if arr.size <= 20:
                print(f"   Values:     {arr.flatten()}")
    
    print()
    print("=" * 70)
    
    # Visualize depth maps if present
    if 'depth' in data.files and output_dir:
        visualize_depth(data, npz_path, output_dir)
    
    return True


def visualize_depth(data, npz_path: str, output_dir: str):
    """Visualize depth maps from NPZ file."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    depth = data['depth']

    # Support both (N, H, W) and single-frame (H, W)
    if depth.ndim == 3:
        num_images = depth.shape[0]
        iter_depth = (depth[i] for i in range(num_images))
    elif depth.ndim == 2:
        num_images = 1
        iter_depth = (depth,)
    else:
        raise ValueError(f"Unsupported depth shape: {depth.shape}")

    print(f"🎨 Generating depth visualizations for {num_images} images...")

    # Generate visualization for each depth map
    for i, depth_map in enumerate(iter_depth):
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Raw depth map
        im1 = axes[0].imshow(depth_map, cmap='turbo')
        axes[0].set_title(f'Depth Map #{i} (Raw)')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        
        # Inverted depth (for better visualization)
        depth_inv = 1.0 / (depth_map + 1e-6)
        im2 = axes[1].imshow(depth_inv, cmap='turbo')
        axes[1].set_title(f'Depth Map #{i} (Inverted)')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Save
        output_path = output_dir / f"depth_visualization_{i:04d}.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved: {output_path}")
    
    print(f"\n✨ All visualizations saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="View NPZ file contents from Depth Anything 3"
    )
    parser.add_argument("npz_file", nargs='+', help="Path(s) to NPZ file(s)")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate depth visualizations",
    )
    parser.add_argument(
        "--output-dir",
        default="./npz_visualizations",
        help="Output directory for visualizations (default: ./npz_visualizations)",
    )
    
    args = parser.parse_args()
    
    output_dir = args.output_dir if args.visualize else None

    overall_success = True
    for npz_path in args.npz_file:
        # If visualizing and multiple files given, create subfolder per file
        per_file_out = None
        if output_dir:
            base = Path(npz_path).stem
            per_file_out = str(Path(output_dir) / base)
        ok = view_npz(npz_path, per_file_out)
        overall_success = overall_success and ok

    success = overall_success
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
