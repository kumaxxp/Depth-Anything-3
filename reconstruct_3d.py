import argparse
import numpy as np
import open3d as o3d
import os
from tqdm import tqdm

def process_npz_to_pcd(npz_path, voxel_size=0.05, depth_scale=1.0, depth_trunc=10.0, conf_thresh=0.8, stride=4, frame_step=1, frame_offset=0):
    """
    Load DA3 npz output and convert to a merged Open3D PointCloud.
    """
    if not os.path.exists(npz_path):
        print(f"Error: {npz_path} not found.")
        return None, None

    print(f"Loading {npz_path}...")

    print(f"Loading {npz_path}...")
    data = np.load(npz_path)
    
    # Extract data
    # Handle both (N, H, W) and (H, W) shapes
    depths = data['depth']
    if depths.ndim == 2:
        depths = depths[np.newaxis, ...]
        
    if 'conf' in data:
        confs = data['conf']
        if confs.ndim == 2:
            confs = confs[np.newaxis, ...]
    else:
        confs = None

    extrinsics = data['extrinsics'] # Shape: (N, 3, 4) or (3, 4)
    if extrinsics.ndim == 2:
        extrinsics = extrinsics[np.newaxis, ...]
        
    intrinsics = data['intrinsics'] # Shape: (N, 3, 3) or (3, 3)
    if intrinsics.ndim == 2:
        intrinsics = intrinsics[np.newaxis, ...]

    num_frames = depths.shape[0]
    height, width = depths.shape[1], depths.shape[2]
    
    merged_pcd = o3d.geometry.PointCloud()

    print(f"Processing {num_frames} frames (step={frame_step}, offset={frame_offset})...")
    for i in tqdm(range(frame_offset, num_frames, frame_step)):
        # 1. Filter by confidence if available
        depth = depths[i]
        if confs is not None:
            mask = confs[i] < conf_thresh # DA3 conf might be uncertainty? Check min/max.
            # Assuming conf is confidence (higher is better) or uncertainty?
            # DA3 output analysis: conf min=1.0, max=3.5. Usually uncertainty map.
            # Let's skip filtering for now or treat as uncertainty (lower is better).
            pass

        # 2. Create Open3D Image from depth
        # Open3D expects depth in uint16 (mm) or float.
        # DA3 depth is likely metric or relative float.
        depth_image = o3d.geometry.Image(depth.astype(np.float32))
        
        # 3. Intrinsic Matrix
        K = intrinsics[i]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, K[0,0], K[1,1], K[0,2], K[1,2])
        
        # 4. Backproject to Point Cloud
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            depth_image, 
            intrinsic, 
            depth_scale=1.0, 
            depth_trunc=depth_trunc,
            stride=stride # Downsample for speed
        )
        
        # 5. Transform to World Coordinates
        # DA3 extrinsics are usually World-to-Camera [R|t]
        # We need Camera-to-World for transforming points: T_c2w = inv(T_w2c)
        extrinsic = np.eye(4)
        extrinsic[:3, :] = extrinsics[i]
        
        try:
            # Inverse to get Camera pose in World
            c2w = np.linalg.inv(extrinsic)
            pcd.transform(c2w)
        except np.linalg.LinAlgError:
            continue

        # 6. Merge
        merged_pcd += pcd

    # 7. Voxel Downsample (to reduce size and merge overlapping points)
    print(f"Downsampling with voxel_size={voxel_size}...")
    merged_pcd = merged_pcd.voxel_down_sample(voxel_size=voxel_size)
    
    # Return last pose for stitching
    last_pose = np.eye(4)
    if num_frames > 0:
        last_pose[:3, :] = extrinsics[-1]
        
    return merged_pcd, last_pose

def main():
    parser = argparse.ArgumentParser(description="Reconstruct 3D Point Cloud from DA3 NPZ")
    parser.add_argument("npz_files", nargs='+', help="Input NPZ files")
    parser.add_argument("--output", default="output.ply", help="Output PLY file")
    parser.add_argument("--voxel-size", type=float, default=0.05, help="Voxel size for downsampling")
    parser.add_argument("--stride", type=int, default=4, help="Stride for depth image downsampling (1=full resolution)")
    parser.add_argument("--max-depth", type=float, default=20.0, help="Max depth to project (points further than this are ignored)")
    parser.add_argument("--frame-step", type=int, default=1, help="Process every Nth frame (use 2 to separate interleaved stereo)")
    parser.add_argument("--frame-offset", type=int, default=0, help="Start processing from this frame index")
    args = parser.parse_args()

    final_pcd = o3d.geometry.PointCloud()
    
    # Global transformation to accumulate batch movements
    global_transform = np.eye(4)
    
    # Keep track of the end of the previous batch to align the next one
    # DA3 batches seem to start near origin. We need to align:
    # Batch N Start -> Batch N-1 End
    # However, we don't have overlap. We only know Batch N starts at 0.
    # We can simply accumulate the relative motion of each batch.
    # T_global_N = T_global_N-1 * T_batch_N-1_end
    
    # Wait, if each batch starts at 0, we need to shift Batch N by the position where Batch N-1 ended.
    # Let's assume the camera movement is continuous.
    # Batch N's first frame should be close to Batch N-1's last frame.
    # Since Batch N resets to 0, we need to add (Batch N-1 End Position) to all points in Batch N.
    
    current_offset = np.eye(4)
    
    for i, f in enumerate(args.npz_files):
        print(f"Processing file {i+1}/{len(args.npz_files)}: {f}")
        pcd, last_pose_in_batch = process_npz_to_pcd(f, voxel_size=args.voxel_size, stride=args.stride, depth_trunc=args.max_depth, frame_step=args.frame_step, frame_offset=args.frame_offset)
        
        if pcd is not None and last_pose_in_batch is not None:
            # Apply current global offset to this batch
            # Note: This is a simplified stitching. It assumes orientation is also reset or consistent.
            # If orientation resets to identity, we need to accumulate rotation too.
            # Based on the log, Frame 0 is always near 0,0,0.
            # So we treat each batch as a relative segment.
            
            # Transform the batch's point cloud by the accumulated offset
            pcd.transform(current_offset)
            final_pcd += pcd
            
            # Update offset for the next batch
            # The next batch starts at 0. We want it to start where this batch ended.
            # So we accumulate the motion of this batch.
            # Motion of this batch = last_pose_in_batch (since it started at 0)
            # But wait, extrinsics are usually World-to-Camera.
            # If Frame 0 is Identity, then Camera is at World Origin.
            # If Frame N is T, then Camera is at T^-1 (or similar depending on definition).
            
            # Let's assume simple translation accumulation for now to fix the "clump" issue.
            # We need to know the coordinate system of DA3.
            # Usually: T_c2w.
            # If Frame 0 is near 0, then last_pose_in_batch is the relative movement.
            
            # We need to accumulate this movement.
            # New Offset = Current Offset * Motion of Batch
            
            # Extract motion from extrinsics (assuming they are [R|t] and we want C2W)
            # If extrinsics are W2C, then C2W = inv(E).
            # In process_npz_to_pcd, we did: c2w = inv(extrinsic).
            # So last_pose_in_batch should be the C2W of the last frame relative to the batch start.
            
            # Calculate the C2W of the last frame
            try:
                # Re-calculate C2W for the last frame to be sure
                # (process_npz_to_pcd returns raw extrinsics? No, it returns pcd)
                # We need the last camera pose C2W.
                
                # Let's parse the last_pose returned (which is raw extrinsics from npz)
                ext_mat = last_pose_in_batch
                c2w_last = np.linalg.inv(ext_mat)
                
                # And the first frame pose (approx Identity)
                # We assume the batch starts at Identity.
                
                # Update global accumulator
                # current_offset = current_offset @ c2w_last
                # This appends the movement of this batch to the global chain.
                current_offset = current_offset @ c2w_last
                
            except np.linalg.LinAlgError:
                pass

    print("Final downsampling...")
    final_pcd = final_pcd.voxel_down_sample(voxel_size=args.voxel_size)
    
    print(f"Saving to {args.output} ({len(final_pcd.points)} points)...")
    o3d.io.write_point_cloud(args.output, final_pcd)
    print("Done.")

if __name__ == "__main__":
    main()
