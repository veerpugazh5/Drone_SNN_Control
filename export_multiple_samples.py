"""
Export multiple test samples for FPGA multi-sample simulation.
Exports 20 samples with concatenated spike streams.
"""
import json
from pathlib import Path
import torch
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


def export_multiple_samples(num_samples, data_dir, params_dir, model_path, device="cpu"):
    """Export multiple samples for FPGA simulation."""
    data_dir = Path(data_dir)
    test_data = torch.load(data_dir / "test.pt")
    
    test_frames = test_data['frames']
    test_labels = test_data['labels']
    
    model = DroneSNNBinary(num_steps=10, beta=0.5, dropout=0.2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    params_dir = Path(params_dir)
    params_dir.mkdir(parents=True, exist_ok=True)
    
    # Export weights and biases (same for all samples) - temporarily move to CPU
    if device != "cpu":
        model_cpu = model.cpu()
        fc_w = model_cpu.fc.weight.detach()
        fc_b = model_cpu.fc.bias.detach()
        model = model.to(device)  # Move back to device
    else:
        fc_w = model.fc.weight.detach()
        fc_b = model.fc.bias.detach()
    q_w, scale = quantize_tensor(fc_w, num_bits=8)
    q_b = torch.round(fc_b * scale).int()
    
    for cls in range(q_w.shape[0]):
        weight_path = params_dir / f"fc_weights_c{cls}.mem"
        with weight_path.open("w") as f:
            for val in q_w[cls]:
                val_uint = val.item() & 0xFF
                f.write(f"{val_uint:02x}\n")
    
    bias_path = params_dir / "fc_bias.mem"
    with bias_path.open("w") as f:
        for val in q_b:
            val_uint = val.item() & 0xFFFF
            f.write(f"{val_uint:04x}\n")
    
    # Export spike streams for all samples (concatenated)
    num_steps = 10
    num_features = 8192
    words_per_step = (num_features + 31) // 32  # 256 words per step
    words_per_sample = words_per_step * num_steps  # 2560 words per sample
    
    # Find balanced samples (equal Straight and Turning)
    samples_per_class = num_samples // 2
    straight_indices = []
    turning_indices = []
    
    for idx in range(len(test_labels)):
        label = test_labels[idx].item()
        if label == 0 and len(straight_indices) < samples_per_class:
            straight_indices.append(idx)
        elif label == 1 and len(turning_indices) < samples_per_class:
            turning_indices.append(idx)
        if len(straight_indices) >= samples_per_class and len(turning_indices) >= samples_per_class:
            break
    
    # Combine: alternate between classes
    selected_indices = []
    for i in range(samples_per_class):
        selected_indices.append(straight_indices[i])
        selected_indices.append(turning_indices[i])
    
    print(f"Selected {len(straight_indices)} Straight and {len(turning_indices)} Turning samples")
    print(f"Exporting {num_samples} balanced samples...")
    
    # Export individual spike stream files for each sample
    sample_info = []
    
    for export_idx, sample_idx in enumerate(selected_indices):
        frame = test_frames[sample_idx]
        label = test_labels[sample_idx].item()
        
        with torch.no_grad():
            frame_tensor = frame.unsqueeze(0)
            if device != "cpu":
                frame_tensor = frame_tensor.to(device)
            logits, hidden_spikes = model(frame_tensor, return_hidden=True)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
        
        hidden_spikes = hidden_spikes.squeeze(0).cpu()  # (T, 128, 8, 8)
        
        # Write spike stream for this sample to individual file
        spike_file = params_dir / f"spike_stream_{export_idx}.mem"
        with spike_file.open("w") as f:
            for t in range(num_steps):
                bits = hidden_spikes[t].reshape(-1).int().tolist()
                for word_idx in range(words_per_step):
                    word = 0
                    for bit_idx in range(32):
                        bit_pos = word_idx * 32 + bit_idx
                        if bit_pos < len(bits) and bits[bit_pos] > 0:
                            word |= (1 << bit_idx)
                    f.write(f"{word:08x}\n")
        
        sample_info.append({
            "sample_index": sample_idx,
            "export_index": export_idx,
            "label": int(label),
            "prediction": int(pred),
            "probabilities": probs.squeeze(0).tolist(),
            "spike_file": f"spike_stream_{export_idx}.mem"
        })
        
        if (export_idx + 1) % 5 == 0:
            print(f"  Exported {export_idx + 1}/{num_samples} samples...")
    
    # Save summary
    summary = {
        "num_samples": num_samples,
        "words_per_sample": words_per_sample,
        "words_per_step": words_per_step,
        "num_steps": num_steps,
        "num_features": num_features,
        "scale_factor": float(scale),
        "num_classes": 2,
        "class_names": ["Straight", "Turning"],
        "samples": sample_info
    }
    summary_path = params_dir / "multi_sample_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    
    print(f"\nExport complete!")
    print(f"  Total samples: {num_samples}")
    print(f"  Spike stream files: spike_stream_0.mem to spike_stream_{num_samples-1}.mem")
    print(f"  Words per sample: {words_per_sample}")
    print(f"  Summary saved to: {summary_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export multiple samples for FPGA simulation.")
    parser.add_argument("--data", type=str,
                        default="preprocessed",
                        help="Directory with preprocessed .pt files.")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of samples to export.")
    parser.add_argument("--model", type=str, default="best_snn_fast.pth", help="Path to trained model checkpoint.")
    parser.add_argument("--out", type=str, default="fpga/params", help="Output directory for parameter files.")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    export_multiple_samples(
        num_samples=args.num_samples,
        data_dir=args.data,
        params_dir=args.out,
        model_path=args.model,
        device=device,
    )

