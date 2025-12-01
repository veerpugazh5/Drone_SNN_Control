"""
Fast training using preprocessed tensor data.
Full GPU utilization with large batch sizes.
"""
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler
from pathlib import Path

from model_binary import DroneSNNBinary


def load_data(data_dir, augment_train=True):
    """Load preprocessed tensors."""
    data_dir = Path(data_dir)
    
    train_data = torch.load(data_dir / "train.pt")
    test_data = torch.load(data_dir / "test.pt")
    
    train_frames = train_data['frames']
    train_labels = train_data['labels']
    test_frames = test_data['frames']
    test_labels = test_data['labels']
    
    # Augmentation: horizontal flip for turning samples
    if augment_train:
        turning_mask = train_labels == 1
        turning_frames = train_frames[turning_mask]
        turning_labels = train_labels[turning_mask]
        
        # Flip horizontally
        flipped = turning_frames.flip(dims=[3])
        
        train_frames = torch.cat([train_frames, flipped], dim=0)
        train_labels = torch.cat([train_labels, turning_labels], dim=0)
        
        # Also flip some straight samples for balance
        straight_mask = train_labels == 0
        straight_frames = train_frames[straight_mask][:500]  # Take subset
        straight_labels = train_labels[straight_mask][:500]
        flipped_straight = straight_frames.flip(dims=[3])
        
        train_frames = torch.cat([train_frames, flipped_straight], dim=0)
        train_labels = torch.cat([train_labels, straight_labels], dim=0)
    
    print(f"Train: {len(train_frames)} samples | Straight={int((train_labels==0).sum())}, Turning={int((train_labels==1).sum())}")
    print(f"Test: {len(test_frames)} samples | Straight={int((test_labels==0).sum())}, Turning={int((test_labels==1).sum())}")
    
    return train_frames, train_labels, test_frames, test_labels


def train():
    DATA_DIR = r"c:\Users\PRISM LAB\OneDrive - University of Arizona\Documents\Drone\preprocessed"
    BATCH_SIZE = 256  # Large batch for 4090
    EPOCHS = 50
    LR = 3e-3
    NUM_WORKERS = 0  # Data already in memory
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        torch.backends.cudnn.benchmark = True
    
    # Load data
    train_frames, train_labels, test_frames, test_labels = load_data(DATA_DIR, augment_train=True)
    
    # Move to GPU for fastest loading
    train_frames = train_frames.to(device)
    train_labels = train_labels.to(device)
    test_frames = test_frames.to(device)
    test_labels = test_labels.to(device)
    
    train_ds = TensorDataset(train_frames, train_labels)
    test_ds = TensorDataset(test_frames, test_labels)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model
    model = DroneSNNBinary(num_steps=10, beta=0.5, dropout=0.3).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Class weights
    n_straight = (train_labels == 0).sum().float()
    n_turning = (train_labels == 1).sum().float()
    total = n_straight + n_turning
    weights = torch.tensor([total / (2 * n_straight), total / (2 * n_turning)]).to(device)
    print(f"Class weights: Straight={weights[0]:.2f}, Turning={weights[1]:.2f}")
    
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, epochs=EPOCHS, 
        steps_per_epoch=len(train_loader), pct_start=0.1
    )
    scaler = GradScaler()
    
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        train_correct, train_total = 0, 0
        
        t0 = time.time()
        for frames, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(device_type="cuda"):
                outputs = model(frames)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            _, preds = outputs.max(1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        
        train_acc = 100 * train_correct / train_total
        
        # Eval
        model.eval()
        test_correct, test_total = 0, 0
        class_correct = [0, 0]
        class_total = [0, 0]
        
        with torch.no_grad():
            for frames, labels in test_loader:
                outputs = model(frames)
                _, preds = outputs.max(1)
                
                test_correct += (preds == labels).sum().item()
                test_total += labels.size(0)
                
                for c in range(2):
                    mask = labels == c
                    class_total[c] += mask.sum().item()
                    class_correct[c] += ((preds == c) & mask).sum().item()
        
        test_acc = 100 * test_correct / test_total
        elapsed = time.time() - t0
        
        straight_acc = 100 * class_correct[0] / max(1, class_total[0])
        turning_acc = 100 * class_correct[1] / max(1, class_total[1])
        
        balanced_acc = (straight_acc + turning_acc) / 2
        
        print(f"Epoch {epoch+1:02d}/{EPOCHS} [{elapsed:.2f}s] "
              f"Train: {train_acc:.1f}% | Test: {test_acc:.1f}% | "
              f"Straight: {straight_acc:.0f}% | Turning: {turning_acc:.0f}% | "
              f"Balanced: {balanced_acc:.1f}%")
        
        if balanced_acc > best_acc:
            best_acc = balanced_acc
            torch.save(model.state_dict(), "best_snn_fast.pth")
            print(f"  -> Best model saved (Balanced: {best_acc:.1f}%)")
    
    print(f"\nTraining complete. Best balanced accuracy: {best_acc:.1f}%")


if __name__ == "__main__":
    train()



