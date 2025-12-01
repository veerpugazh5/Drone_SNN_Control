"""
Binary classification dataset: Turning vs Straight
"""
import h5py
import hdf5plugin
import numpy as np
import torch
from torch.utils.data import Dataset
import glob
import os


class DroneBinaryDataset(Dataset):
    def __init__(
        self,
        root_dir,
        mode='train',
        time_window_ms=200,
        target_size=(64, 64),
        threshold=0.05,
        windows_per_file=5,
        augment=False,
    ):
        self.time_window = time_window_ms / 1000.0
        self.target_size = target_size
        self.sensor_size = (180, 180)
        self.threshold = threshold
        self.windows_per_file = windows_per_file
        self.augment = augment and mode == 'train'
        
        subdir = 'sr_dataset_train' if mode == 'train' else 'sr_dataset_test'
        files = sorted(glob.glob(os.path.join(root_dir, subdir, '*.h5')))
        print(f"[{mode}] Found {len(files)} files")
        
        self.samples = []
        class_counts = [0, 0]  # [Straight, Turning]
        
        for fpath in files:
            with h5py.File(fpath, 'r') as f:
                ts = f['events']['ts'][:]
                imu_ts = f['angular_rate_imu']['ts'][:]
                imu_z = f['angular_rate_imu']['z'][:]
            
            t_start = ts[0]
            t_end = ts[-1]
            duration = t_end - t_start
            
            # Compute window positions
            margin = self.time_window
            usable = duration - 2 * margin
            if usable > 0 and self.windows_per_file > 1:
                step = usable / (self.windows_per_file - 1)
                offsets = [margin + i * step for i in range(self.windows_per_file)]
            else:
                offsets = [duration / 2 - self.time_window / 2]
            
            for offset in offsets:
                win_start = t_start + offset
                win_end = win_start + self.time_window
                
                mask = (imu_ts >= win_start) & (imu_ts < win_end)
                yaw = np.mean(imu_z[mask]) if mask.sum() > 0 else np.mean(imu_z)
                
                # Binary: 0 = Straight, 1 = Turning
                label = 0 if abs(yaw) < self.threshold else 1
                
                self.samples.append({
                    'path': fpath,
                    'win_start': win_start,
                    'win_end': win_end,
                    'label': label,
                    'flip': False,
                })
                class_counts[label] += 1
                
                # Augment turning samples with flip (turning looks same flipped)
                if self.augment and label == 1:
                    self.samples.append({
                        'path': fpath,
                        'win_start': win_start,
                        'win_end': win_end,
                        'label': 1,
                        'flip': True,
                    })
                    class_counts[1] += 1
        
        self.class_counts = class_counts
        print(f"[{mode}] Samples: {len(self.samples)} | Straight={class_counts[0]}, Turning={class_counts[1]}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        with h5py.File(sample['path'], 'r') as f:
            xs = f['events']['xs'][:]
            ys = f['events']['ys'][:]
            ts = f['events']['ts'][:]
            ps = f['events']['ps'][:]
        
        mask = (ts >= sample['win_start']) & (ts < sample['win_end'])
        xs_w, ys_w, ps_w = xs[mask], ys[mask], ps[mask]
        
        frame = np.zeros((2, self.target_size[0], self.target_size[1]), dtype=np.float32)
        
        if len(xs_w) > 0:
            scale_x = self.target_size[1] / self.sensor_size[1]
            scale_y = self.target_size[0] / self.sensor_size[0]
            
            xs_s = np.clip((xs_w * scale_x).astype(np.int32), 0, self.target_size[1] - 1)
            ys_s = np.clip((ys_w * scale_y).astype(np.int32), 0, self.target_size[0] - 1)
            
            np.add.at(frame[0], (ys_s[ps_w == 0], xs_s[ps_w == 0]), 1)
            np.add.at(frame[1], (ys_s[ps_w == 1], xs_s[ps_w == 1]), 1)
            
            frame = np.log1p(frame)
            if frame.max() > 0:
                frame = frame / frame.max()
        
        if sample['flip']:
            frame = frame[:, :, ::-1].copy()
        
        return torch.from_numpy(frame), torch.tensor(sample['label'], dtype=torch.long)


if __name__ == "__main__":
    root = r"c:\Users\PRISM LAB\OneDrive - University of Arizona\Documents\Drone\sr_dataset_gt"
    train_ds = DroneBinaryDataset(root, mode='train', augment=True, windows_per_file=5)
    test_ds = DroneBinaryDataset(root, mode='test', augment=False, windows_per_file=3)



