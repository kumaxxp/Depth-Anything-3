import sys
import os
import torch

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from depth_anything_3.api import DepthAnything3

def main():
    image_path = "/mnt/c/data/da3_input/1115_05_mid/000001_B.jpg"
    output_dir = "single_test_output"
    
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found")
        return

    print(f"Processing {image_path}...")
    
    # Initialize model
    model = DepthAnything3(model_name="da3-small")
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Run inference
    # We export mini_npz, then convert to PLY manually
    model.inference(
        [image_path],
        export_dir=output_dir,
        export_format="mini_npz",
        process_res=518 # Slightly higher res for single image
    )
    
    # Convert to PLY using our reconstruction script
    # We need to import the function from reconstruct_3d.py
    # Since reconstruct_3d.py is in the parent directory, we need to add it to path
    sys.path.append(os.getcwd())
    from reconstruct_3d import process_npz_to_pcd
    import open3d as o3d
    
    # The export structure seems to be export_dir/exports/mini_npz/results.npz
    # Let's try to find it
    npz_file = os.path.join(output_dir, "exports", "mini_npz", "results.npz")
    if not os.path.exists(npz_file):
        # Fallback check
        npz_file = os.path.join(output_dir, "results.npz")

    if os.path.exists(npz_file):
        print(f"Converting {npz_file} to PLY...")
        pcd, _ = process_npz_to_pcd(npz_file, voxel_size=0.01, stride=1, depth_trunc=20.0)
        if pcd is not None:
            output_ply = os.path.join(output_dir, "single_frame.ply")
            o3d.io.write_point_cloud(output_ply, pcd)
            print(f"Saved point cloud to {output_ply}")
    
    print(f"Done. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
