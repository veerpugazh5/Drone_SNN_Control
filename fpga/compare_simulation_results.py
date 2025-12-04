"""Compare FPGA simulation results with Python predictions."""
import json
from pathlib import Path

# FPGA predictions from simulation log
fpga_predictions = {
    0: 0, 1: 0, 2: 0, 3: 1, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0,
    10: 0, 11: 1, 12: 0, 13: 0, 14: 0, 15: 1, 16: 0, 17: 0, 18: 0, 19: 1
}

# Load Python predictions
summary_file = Path("params/multi_sample_summary.json")
if not summary_file.exists():
    print(f"ERROR: {summary_file} not found")
    exit(1)

data = json.loads(summary_file.read_text())
samples = data["samples"]

print("="*80)
print("FPGA Simulation vs Python Model Comparison")
print("="*80)
print()

matches = 0
mismatches = 0

for i, sample in enumerate(samples):
    fpga_pred = fpga_predictions[i]
    python_pred = sample["prediction"]
    label = sample["label"]
    match = (fpga_pred == python_pred)
    
    if match:
        matches += 1
        status = "[MATCH]"
    else:
        mismatches += 1
        status = "[MISMATCH]"
    
    fpga_class = "Straight" if fpga_pred == 0 else "Turning"
    python_class = "Straight" if python_pred == 0 else "Turning"
    label_class = "Straight" if label == 0 else "Turning"
    
    print(f"Sample {i:2d}: FPGA={fpga_pred} ({fpga_class:8s}) | "
          f"Python={python_pred} ({python_class:8s}) | "
          f"Label={label} ({label_class:8s}) {status}")

print()
print("="*80)
print("Summary")
print("="*80)
print(f"Total Samples: {len(samples)}")
print(f"Matches: {matches} ({matches*100/len(samples):.1f}%)")
print(f"Mismatches: {mismatches} ({mismatches*100/len(samples):.1f}%)")
print()

if matches == len(samples):
    print("✅ PERFECT MATCH! FPGA simulation matches Python model 100%")
else:
    print(f"⚠️  {mismatches} mismatch(es) found")
    print("   This may indicate quantization differences or implementation issues")

print("="*80)

