import argparse
import numpy as np
import os
import matplotlib.pyplot as plt

def inspect_trajectory(npz_files):
    all_translations = []
    batch_starts = []
    
    print(f"Inspecting {len(npz_files)} files...")
    
    global_idx = 0
    for f_idx, npz_path in enumerate(npz_files):
        if not os.path.exists(npz_path):
            continue
            
        data = np.load(npz_path)
        extrinsics = data['extrinsics'] # (N, 3, 4) or (N, 4, 4)
        
        # Handle shape variations
        if extrinsics.ndim == 2:
            extrinsics = extrinsics[np.newaxis, ...]
            
        num_frames = extrinsics.shape[0]
        batch_starts.append(global_idx)
        
        print(f"Batch {f_idx}: {os.path.basename(os.path.dirname(os.path.dirname(npz_path)))} - {num_frames} frames")
        
        for i in range(num_frames):
            # Extrinsics is usually World-to-Camera or Camera-to-World.
            # Let's assume [R|t]. If it's World-to-Camera, Camera pos is -R^T * t.
            # If it's Camera-to-World, Camera pos is t.
            # DA3 usually outputs Camera-to-World for visualization? Let's check raw t.
            
            mat = np.eye(4)
            mat[:3, :] = extrinsics[i]
            
            # Let's just extract the translation part (column 3) for now
            # If this is C2W, this is the position.
            # If this is W2C, this is -R*C.
            # We'll print raw values first.
            t = mat[:3, 3]
            all_translations.append(t)
            
            if i == 0 or i == num_frames - 1:
                print(f"  Frame {i}: t = {t}")
        
        global_idx += num_frames

    all_translations = np.array(all_translations)
    
    # Calculate bounds
    min_xyz = np.min(all_translations, axis=0)
    max_xyz = np.max(all_translations, axis=0)
    print(f"\nTotal Range:")
    print(f"  Min: {min_xyz}")
    print(f"  Max: {max_xyz}")
    print(f"  Span: {max_xyz - min_xyz}")

    # Check for resets
    # Compare last frame of batch N with first frame of batch N+1
    if len(batch_starts) > 1:
        print("\nBatch Transitions (End -> Start):")
        for i in range(len(batch_starts) - 1):
            end_idx = batch_starts[i+1] - 1
            start_idx = batch_starts[i+1]
            
            p_end = all_translations[end_idx]
            p_start = all_translations[start_idx]
            dist = np.linalg.norm(p_end - p_start)
            print(f"  Batch {i}->{i+1}: Dist = {dist:.4f}")
            print(f"    End  : {p_end}")
            print(f"    Start: {p_start}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_files", nargs='+')
    args = parser.parse_args()
    inspect_trajectory(args.npz_files)
