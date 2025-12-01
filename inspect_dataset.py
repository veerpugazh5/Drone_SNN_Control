"""
Thorough inspection of the neuromorphic drone dataset.
Outputs statistics about events, timestamps, IMU values, and class distribution.
"""
import h5py
import hdf5plugin
import numpy as np
import glob
import os

ROOT_DIR = r"c:\Users\PRISM LAB\OneDrive - University of Arizona\Documents\Drone\sr_dataset_gt"

def inspect_single_file(fpath):
    """Return stats dict for one HDF5 file."""
    stats = {}
    with h5py.File(fpath, 'r') as f:
        # Events
        ts = f['events']['ts'][:]
        xs = f['events']['xs'][:]
        ys = f['events']['ys'][:]
        ps = f['events']['ps'][:]
        
        stats['num_events'] = len(ts)
        stats['duration_s'] = float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0
        stats['x_min'], stats['x_max'] = int(xs.min()), int(xs.max())
        stats['y_min'], stats['y_max'] = int(ys.min()), int(ys.max())
        stats['polarity_ratio'] = float(ps.sum()) / max(1, len(ps))  # fraction of ON events
        
        # IMU yaw
        imu_ts = f['angular_rate_imu']['ts'][:]
        imu_z = f['angular_rate_imu']['z'][:]
        stats['imu_samples'] = len(imu_z)
        stats['yaw_mean'] = float(np.mean(imu_z))
        stats['yaw_std'] = float(np.std(imu_z))
        stats['yaw_min'] = float(np.min(imu_z))
        stats['yaw_max'] = float(np.max(imu_z))
    return stats


def main():
    for split in ['train', 'test']:
        subdir = 'sr_dataset_train' if split == 'train' else 'sr_dataset_test'
        files = sorted(glob.glob(os.path.join(ROOT_DIR, subdir, '*.h5')))
        print(f"\n=== {split.upper()} split: {len(files)} files ===")
        
        all_stats = []
        for fpath in files:
            try:
                st = inspect_single_file(fpath)
                all_stats.append(st)
            except Exception as e:
                print(f"Error reading {fpath}: {e}")
        
        if not all_stats:
            continue
        
        # Aggregate
        num_events = [s['num_events'] for s in all_stats]
        durations = [s['duration_s'] for s in all_stats]
        yaw_means = [s['yaw_mean'] for s in all_stats]
        yaw_stds = [s['yaw_std'] for s in all_stats]
        
        print(f"Events per file: min={min(num_events)}, max={max(num_events)}, mean={np.mean(num_events):.0f}")
        print(f"Duration (s): min={min(durations):.3f}, max={max(durations):.3f}, mean={np.mean(durations):.3f}")
        print(f"Sensor X range: {all_stats[0]['x_min']}-{all_stats[0]['x_max']}")
        print(f"Sensor Y range: {all_stats[0]['y_min']}-{all_stats[0]['y_max']}")
        print(f"Yaw mean across files: min={min(yaw_means):.4f}, max={max(yaw_means):.4f}, overall mean={np.mean(yaw_means):.4f}")
        print(f"Yaw std across files: min={min(yaw_stds):.4f}, max={max(yaw_stds):.4f}")
        
        # Class distribution at various thresholds
        for thresh in [0.02, 0.05, 0.1, 0.2]:
            left = sum(1 for y in yaw_means if y < -thresh)
            right = sum(1 for y in yaw_means if y > thresh)
            straight = len(yaw_means) - left - right
            print(f"  Threshold {thresh}: Left={left}, Straight={straight}, Right={right}")


if __name__ == "__main__":
    main()



