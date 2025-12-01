"""
Export binary SNN model for FPGA simulation.
Uses preprocessed data for fast loading.
"""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import TensorDataset

from model_binary import DroneSNNBinary


def quantize_tensor(tensor, num_bits=8):
    max_abs = float(tensor.abs().max())
    if max_abs == 0.0:
        scale = 1.0
    else:
        scale = (2 ** (num_bits - 1) - 1) / max_abs
    q = torch.round(tensor * scale)
    q = torch.clamp(q, -(2 ** (num_bits - 1)), 2 ** (num_bits - 1) - 1)
    return q.int(), scale


def export_assets(sample_idx, data_dir, params_dir, model_path, device="cpu"):
    """Export model assets for FPGA simulation."""
    data_dir = Path(data_dir)
    test_data = torch.load(data_dir / "test.pt")
    
    test_frames = test_data['frames']
    test_labels = test_data['labels']
    
    frame = test_frames[sample_idx]
    label = test_labels[sample_idx].item()
    
    model = DroneSNNBinary(num_steps=10, beta=0.5, dropout=0.2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    with torch.no_grad():
        frame = frame.unsqueeze(0).to(device)
        logits, hidden_spikes = model(frame, return_hidden=True)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    
    hidden_spikes = hidden_spikes.squeeze(0).cpu()  # (T, 128, 8, 8)
    num_steps = hidden_spikes.shape[0]
    num_features = hidden_spikes[0].numel()
    
    params_dir.mkdir(parents=True, exist_ok=True)
    
    # Export spike stream - write as 32-bit words (256 words per timestep for 8192 bits)
    spike_path = params_dir / "spike_stream.mem"
    words_per_step = (num_features + 31) // 32
    with spike_path.open("w") as f:
        for t in range(num_steps):
            bits = hidden_spikes[t].reshape(-1).int().tolist()
            # Pack into 32-bit words
            for word_idx in range(words_per_step):
                word = 0
                for bit_idx in range(32):
                    bit_pos = word_idx * 32 + bit_idx
                    if bit_pos < len(bits) and bits[bit_pos] > 0:
                        word |= (1 << bit_idx)
                f.write(f"{word:08x}\n")
    
    # Export FC weights/biases (binary: 2 classes)
    model_cpu = model.cpu()
    fc_w = model_cpu.fc.weight.detach()
    fc_b = model_cpu.fc.bias.detach()
    q_w, scale = quantize_tensor(fc_w, num_bits=8)
    q_b = torch.round(fc_b * scale).int()
    
    for cls in range(q_w.shape[0]):
        weight_path = params_dir / f"fc_weights_c{cls}.mem"
        with weight_path.open("w") as f:
            for val in q_w[cls]:
                # Convert signed to unsigned for hex
                val_uint = val.item() & 0xFF
                f.write(f"{val_uint:02x}\n")
    
    bias_path = params_dir / "fc_bias.mem"
    with bias_path.open("w") as f:
        for val in q_b:
            # Convert signed to unsigned for hex
            val_uint = val.item() & 0xFFFF
            f.write(f"{val_uint:04x}\n")
    
    summary = {
        "sample_index": sample_idx,
        "label": int(label),
        "prediction": int(pred),
        "probabilities": probs.squeeze(0).tolist(),
        "scale_factor": float(scale),
        "num_steps": num_steps,
        "num_features": num_features,
        "num_classes": 2,
        "class_names": ["Straight", "Turning"],
        "words_per_step": words_per_step
    }
    summary_path = params_dir / "fpga_export_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Assets exported to {params_dir}")
    print(f"Label: {label} ({'Straight' if label == 0 else 'Turning'}), Prediction: {pred} ({'Straight' if pred == 0 else 'Turning'})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export binary SNN for FPGA simulation.")
    parser.add_argument("--data", type=str,
                        default=r"c:\Users\PRISM LAB\OneDrive - University of Arizona\Documents\Drone\preprocessed",
                        help="Directory with preprocessed .pt files.")
    parser.add_argument("--index", type=int, default=0, help="Sample index from test set.")
    parser.add_argument("--model", type=str, default="best_snn_fast.pth", help="Path to trained model checkpoint.")
    parser.add_argument("--out", type=str, default="fpga/params", help="Output directory for parameter files.")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    export_assets(
        sample_idx=args.index,
        data_dir=args.data,
        params_dir=Path(args.out),
        model_path=args.model,
        device=device,
    )
