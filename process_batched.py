#!/usr/bin/env python3
"""
Batched inference script for large datasets.
Processes images in small chunks to avoid OOM errors.
"""

import argparse
import glob
import os
import shutil
import sys
from pathlib import Path

def process_in_batches(
    input_dir: str,
    output_dir: str,
    model_dir: str = "depth-anything/DA3-SMALL",
    batch_size: int = 10,
    process_res: int = 336,
    export_format: str = "mini_npz",
    device: str = "cuda",
):
    """Process images in batches to avoid OOM."""
    
    # Find all images
    patterns = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG"]
    image_files = []
    for pattern in patterns:
        image_files.extend(glob.glob(os.path.join(input_dir, pattern)))
    
    image_files = sorted(image_files)
    total = len(image_files)
    
    if total == 0:
        print(f"❌ No images found in {input_dir}")
        return False
    
    print(f"📊 Found {total} images")
    print(f"🔄 Processing in batches of {batch_size}")
    print(f"📐 Resolution: {process_res}")
    print(f"💾 Export format: {export_format}")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process in batches
    num_batches = (total + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total)
        batch_files = image_files[start_idx:end_idx]
        
        print(f"{'='*60}")
        print(f"📦 Batch {batch_idx + 1}/{num_batches}")
        print(f"   Images {start_idx + 1}-{end_idx} of {total}")
        print(f"{'='*60}")
        
        # Create temporary batch directory
        batch_input_dir = os.path.join(output_dir, f".batch_{batch_idx:04d}_input")
        batch_output_dir = os.path.join(output_dir, f"batch_{batch_idx:04d}")
        
        try:
            # Copy batch images
            os.makedirs(batch_input_dir, exist_ok=True)
            for img_file in batch_files:
                shutil.copy2(img_file, batch_input_dir)
            
            print(f"✅ Copied {len(batch_files)} images to temporary directory")
            
            # Run inference
            # Use python3 -m depth_anything_3.cli to ensure we use the local code
            cmd = (
                f"python3 -m depth_anything_3.cli images \"{batch_input_dir}\" "
                f"--image-extensions \"jpg,jpeg,png\" "
                f"--model-dir \"{model_dir}\" "
                f"--export-format {export_format} "
                f"--export-dir \"{batch_output_dir}\" "
                f"--process-res {process_res} "
                f"--device {device} "
                f"--auto-cleanup"
            )
            
            print(f"🚀 Running: {cmd}")
            print()
            
            ret = os.system(cmd)
            
            if ret != 0:
                print(f"⚠️  Batch {batch_idx + 1} failed with exit code {ret}")
                print(f"   Continuing with next batch...")
            else:
                print(f"✅ Batch {batch_idx + 1} completed successfully")
            
        except Exception as e:
            print(f"❌ Error processing batch {batch_idx + 1}: {e}")
            print(f"   Continuing with next batch...")
        
        finally:
            # Clean up temporary input directory
            if os.path.exists(batch_input_dir):
                shutil.rmtree(batch_input_dir)
                print(f"🧹 Cleaned up temporary directory")
        
        print()
    
    print(f"{'='*60}")
    print(f"✅ All batches completed!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"{'='*60}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Process large image datasets in batches to avoid OOM"
    )
    parser.add_argument("input_dir", help="Directory containing input images")
    parser.add_argument(
        "--output-dir",
        default="workspace/batched_output",
        help="Output directory (default: workspace/batched_output)",
    )
    parser.add_argument(
        "--model-dir",
        default="depth-anything/DA3-SMALL",
        help="Model directory (default: depth-anything/DA3-SMALL)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of images per batch (default: 10)",
    )
    parser.add_argument(
        "--process-res",
        type=int,
        default=336,
        help="Processing resolution (default: 336)",
    )
    parser.add_argument(
        "--export-format",
        default="mini_npz",
        help="Export format (default: mini_npz)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to use (default: cuda)",
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"❌ Input directory not found: {args.input_dir}")
        sys.exit(1)
    
    success = process_in_batches(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        process_res=args.process_res,
        export_format=args.export_format,
        device=args.device,
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
