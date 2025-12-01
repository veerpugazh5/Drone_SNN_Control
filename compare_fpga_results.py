"""
Compare FPGA simulation results with Python model predictions.
Parses simulation log and compares with exported summary.
"""
import json
import re
from pathlib import Path


def parse_simulation_log(log_path):
    """Parse FPGA simulation log to extract predictions."""
    log_text = log_path.read_text()
    
    # Find all "Sample X: FPGA prediction = Y" lines
    pattern = r"Sample (\d+): FPGA prediction = (\d+)"
    matches = re.findall(pattern, log_text)
    
    fpga_predictions = {}
    for sample_idx_str, pred_str in matches:
        sample_idx = int(sample_idx_str)
        pred = int(pred_str)
        fpga_predictions[sample_idx] = pred
    
    return fpga_predictions


def compare_results(summary_path, fpga_predictions):
    """Compare FPGA predictions with Python model predictions."""
    summary = json.loads(summary_path.read_text())
    
    results = []
    matches = 0
    python_correct = 0
    fpga_correct = 0
    both_correct = 0
    
    for sample_info in summary['samples']:
        sample_idx = sample_info['sample_index']  # Original test set index
        export_idx = sample_info.get('export_index', sample_idx)  # Export order (0-19)
        label = sample_info['label']
        python_pred = sample_info['prediction']
        python_probs = sample_info['probabilities']
        
        # FPGA predictions are keyed by export_index (0-19)
        fpga_pred = fpga_predictions.get(export_idx, -1)
        
        python_match = (python_pred == fpga_pred)
        python_correct_sample = (python_pred == label)
        fpga_correct_sample = (fpga_pred == label)
        both_correct_sample = python_correct_sample and fpga_correct_sample
        
        if python_match:
            matches += 1
        if python_correct_sample:
            python_correct += 1
        if fpga_correct_sample:
            fpga_correct += 1
        if both_correct_sample:
            both_correct += 1
        
        results.append({
            'sample_index': sample_idx,
            'export_index': export_idx,
            'label': label,
            'label_name': 'Straight' if label == 0 else 'Turning',
            'python_pred': python_pred,
            'python_pred_name': 'Straight' if python_pred == 0 else 'Turning',
            'python_prob_straight': python_probs[0] * 100,
            'python_prob_turning': python_probs[1] * 100,
            'fpga_pred': fpga_pred,
            'fpga_pred_name': 'Straight' if fpga_pred == 0 else 'Turning',
            'python_match': python_match,
            'python_correct': python_correct_sample,
            'fpga_correct': fpga_correct_sample,
            'both_correct': both_correct_sample
        })
    
    return {
        'results': results,
        'summary': {
            'total_samples': len(results),
            'python_fpga_matches': matches,
            'match_rate': matches / len(results) * 100 if results else 0,
            'python_accuracy': python_correct / len(results) * 100 if results else 0,
            'fpga_accuracy': fpga_correct / len(results) * 100 if results else 0,
            'both_correct_count': both_correct,
            'both_correct_rate': both_correct / len(results) * 100 if results else 0
        }
    }


def print_comparison_report(comparison):
    """Print human-readable comparison report."""
    print("=" * 80)
    print("FPGA vs Python Model Comparison Report")
    print("=" * 80)
    print()
    
    summary = comparison['summary']
    print("SUMMARY:")
    print(f"  Total Samples: {summary['total_samples']}")
    print(f"  Python-FPGA Matches: {summary['python_fpga_matches']}/{summary['total_samples']} ({summary['match_rate']:.1f}%)")
    print(f"  Python Model Accuracy: {summary['python_accuracy']:.1f}%")
    print(f"  FPGA Model Accuracy: {summary['fpga_accuracy']:.1f}%")
    print(f"  Both Correct: {summary['both_correct_count']}/{summary['total_samples']} ({summary['both_correct_rate']:.1f}%)")
    print()
    
    print("DETAILED RESULTS:")
    print("-" * 90)
    print(f"{'Export':<8} {'Sample':<8} {'Label':<10} {'Python':<15} {'FPGA':<10} {'Match':<8} {'Both Correct':<12}")
    print("-" * 90)
    
    for r in comparison['results']:
        match_symbol = "✓" if r['python_match'] else "✗"
        correct_symbol = "✓" if r['both_correct'] else "✗"
        print(f"{r['export_index']:<8} {r['sample_index']:<8} {r['label_name']:<10} {r['python_pred_name']:<15} "
              f"{r['fpga_pred_name']:<10} {match_symbol:<8} {correct_symbol:<12}")
    
    print("-" * 80)
    print()
    
    # Show mismatches
    mismatches = [r for r in comparison['results'] if not r['python_match']]
    if mismatches:
        print(f"MISMATCHES ({len(mismatches)} samples):")
        print("-" * 80)
        for r in mismatches:
            print(f"  Export {r['export_index']} (Sample {r['sample_index']}): Python={r['python_pred_name']} ({r['python_prob_straight']:.1f}%/{r['python_prob_turning']:.1f}%), "
                  f"FPGA={r['fpga_pred_name']}, Label={r['label_name']}")
        print()
    
    print("=" * 80)


def main():
    summary_path = Path("fpga/params/multi_sample_summary.json")
    log_path = Path("fpga/xsim.log")
    
    if not summary_path.exists():
        print(f"Error: Summary file not found: {summary_path}")
        return
    
    if not log_path.exists():
        print(f"Error: Simulation log not found: {log_path}")
        return
    
    print("Parsing FPGA simulation results...")
    fpga_predictions = parse_simulation_log(log_path)
    print(f"Found {len(fpga_predictions)} FPGA predictions")
    print()
    
    print("Loading Python model predictions...")
    comparison = compare_results(summary_path, fpga_predictions)
    print()
    
    print_comparison_report(comparison)
    
    # Save JSON results
    output_path = Path("fpga/multi_sample_results.json")
    output_path.write_text(json.dumps(comparison, indent=2))
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()

