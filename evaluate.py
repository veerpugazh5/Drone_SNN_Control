"""
Evaluate binary SNN model on test set.
"""
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

from model_binary import DroneSNNBinary


def evaluate():
    DATA_DIR = Path("preprocessed")
    MODEL_PATH = "best_snn_fast.pth"
    BATCH_SIZE = 256
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load preprocessed test data
    test_data = torch.load(DATA_DIR / "test.pt")
    test_frames = test_data['frames']
    test_labels = test_data['labels']
    
    print(f"Test samples: {len(test_frames)}")
    print(f"Class distribution: Straight={(test_labels==0).sum()}, Turning={(test_labels==1).sum()}")
    
    # Move to GPU
    test_frames = test_frames.to(device)
    test_labels = test_labels.to(device)
    
    test_ds = TensorDataset(test_frames, test_labels)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # Load model
    model = DroneSNNBinary(num_steps=10, beta=0.5, dropout=0.2).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Loaded model from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"Warning: {MODEL_PATH} not found, using random weights")
    
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("\nRunning evaluation...")
    with torch.no_grad():
        for frames, labels in test_loader:
            outputs = model(frames)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Metrics
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print("                Predicted")
    print("              Straight  Turning")
    print(f"Straight      {cm[0,0]:6d}  {cm[0,1]:6d}")
    print(f"Turning       {cm[1,0]:6d}  {cm[1,1]:6d}")
    
    print("\nClassification Report:")
    target_names = ['Straight', 'Turning']
    print(classification_report(all_labels, all_preds, target_names=target_names))
    
    # Per-class accuracy
    class_acc = cm.diagonal() / cm.sum(axis=1)
    overall_acc = np.trace(cm) / cm.sum()
    
    print(f"\nOverall Accuracy: {overall_acc*100:.2f}%")
    print(f"Straight Accuracy: {class_acc[0]*100:.2f}%")
    print(f"Turning Accuracy: {class_acc[1]*100:.2f}%")


if __name__ == "__main__":
    evaluate()
