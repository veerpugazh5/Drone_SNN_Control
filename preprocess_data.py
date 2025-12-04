"""
Preprocess HDF5 files to fast-loading tensor format.
Converts all samples once, saves as .pt files for instant loading.
"""
import os
import h5py
import hdf5plugin
import numpy as np
import torch
from pathlib import Path
import glob
from tqdm import tqdm


def process_file(fpath, time_window_ms=200, target_size=(64, 64), 
                 sensor_size=(180, 180), threshold=0.1, windows_per_file=10):
    """Extract multiple windows from one HDF5 file."""
    time_window = time_window_ms / 1000.0
    samples = []
    
    with h5py.File(fpath, 'r') as f:
        xs = f['events']['xs'][:]
        ys = f['events']['ys'][:]
        ts = f['events']['ts'][:]
        ps = f['events']['ps'][:]
        imu_ts = f['angular_rate_imu']['ts'][:]
        imu_z = f['angular_rate_imu']['z'][:]
    
    t_start = ts[0]
    t_end = ts[-1]
    duration = t_end - t_start
    
    # Compute window positions
    margin = time_window
    usable = duration - 2 * margin
    if usable > 0 and windows_per_file > 1:
        step = usable / (windows_per_file - 1)
        offsets = [margin + i * step for i in range(windows_per_file)]
    else:
        offsets = [duration / 2 - time_window / 2]
    
    scale_x = target_size[1] / sensor_size[1]
    scale_y = target_size[0] / sensor_size[0]
    
    for offset in offsets:
        win_start = t_start + offset
        win_end = win_start + time_window
        
        # Get label
        imu_mask = (imu_ts >= win_start) & (imu_ts < win_end)
        yaw = np.mean(imu_z[imu_mask]) if imu_mask.sum() > 0 else np.mean(imu_z)
        
        # Binary label: 0=Straight, 1=Turning
        label = 0 if abs(yaw) < threshold else 1
        
        # Build frame
        mask = (ts >= win_start) & (ts < win_end)
        xs_w, ys_w, ps_w = xs[mask], ys[mask], ps[mask]
        
        frame = np.zeros((2, target_size[0], target_size[1]), dtype=np.float32)
        
        if len(xs_w) > 0:
            xs_s = np.clip((xs_w * scale_x).astype(np.int32), 0, target_size[1] - 1)
            ys_s = np.clip((ys_w * scale_y).astype(np.int32), 0, target_size[0] - 1)
            
            np.add.at(frame[0], (ys_s[ps_w == 0], xs_s[ps_w == 0]), 1)
            np.add.at(frame[1], (ys_s[ps_w == 1], xs_s[ps_w == 1]), 1)
            
            frame = np.log1p(frame)
            if frame.max() > 0:
                frame = frame / frame.max()
        
        samples.append({
            'frame': torch.from_numpy(frame),
            'label': label,
            'yaw': float(yaw),
        })
    
    return samples


def main():
    ROOT = Path("sr_dataset_gt")
    OUT_DIR = Path("preprocessed")
    
    WINDOWS_PER_FILE = 10  # More windows = more data
    TIME_WINDOW_MS = 200
    THRESHOLD = 0.1  # Slightly higher threshold for cleaner separation
    
    OUT_DIR.mkdir(exist_ok=True)
    
    for mode in ['train', 'test']:
        subdir = 'sr_dataset_train' if mode == 'train' else 'sr_dataset_test'
        files = sorted(glob.glob(os.path.join(ROOT, subdir, '*.h5')))
        print(f"\n[{mode}] Processing {len(files)} files...")
        
        all_frames = []
        all_labels = []
        all_yaws = []
        
        for fpath in tqdm(files, desc=mode):
            samples = process_file(
                fpath, 
                time_window_ms=TIME_WINDOW_MS,
                threshold=THRESHOLD,
                windows_per_file=WINDOWS_PER_FILE
            )
            for s in samples:
                all_frames.append(s['frame'])
                all_labels.append(s['label'])
                all_yaws.append(s['yaw'])
        
        # Stack into tensors
        frames_tensor = torch.stack(all_frames)
        labels_tensor = torch.tensor(all_labels, dtype=torch.long)
        yaws_tensor = torch.tensor(all_yaws, dtype=torch.float32)
        
        # Save
        out_path = OUT_DIR / f"{mode}.pt"
        torch.save({
            'frames': frames_tensor,
            'labels': labels_tensor,
            'yaws': yaws_tensor,
        }, out_path)
        
        # Stats
        n_straight = (labels_tensor == 0).sum().item()
        n_turning = (labels_tensor == 1).sum().item()
        print(f"[{mode}] Saved {len(frames_tensor)} samples to {out_path}")
        print(f"[{mode}] Straight={n_straight}, Turning={n_turning}")
        print(f"[{mode}] Yaw range: [{yaws_tensor.min():.3f}, {yaws_tensor.max():.3f}]")


if __name__ == "__main__":
    main()



