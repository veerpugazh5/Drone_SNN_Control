"""
Complete Pipeline Runner
========================
One command to run everything: training → evaluation → FPGA export → simulation → analysis

Usage:
    python run_all.py
"""
import sys
import time
import subprocess
from pathlib import Path
import json

# Configuration
DATA_DIR = Path(r"c:\Users\PRISM LAB\OneDrive - University of Arizona\Documents\Drone\preprocessed")
MODEL_PATH = "best_snn_fast.pth"
FPGA_DIR = Path("fpga")
VIVADO_PATH = Path(r"C:\Xilinx\Vivado\2024.2\bin\vivado.bat")
NUM_SAMPLES = 20
CW305_MODE = False  # Set to True for CW305-specific simulation


def print_step(step_num, step_name):
    """Print step header."""
    print(f"\n{'='*80}")
    print(f"STEP {step_num}: {step_name}")
    print(f"{'='*80}")


def step_training():
    """Step 1: Train the SNN model."""
    print_step(1, "SNN Training")
    
    try:
        from train_fast import train
        print("Starting training...")
        train()
        print("✓ Training completed successfully")
        return True
    except Exception as e:
        print(f"✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def step_evaluation():
    """Step 2: Evaluate the trained model."""
    print_step(2, "Model Evaluation")
    
    try:
        from evaluate import evaluate
        print("Evaluating model...")
        evaluate()
        print("✓ Evaluation completed successfully")
        return True
    except Exception as e:
        print(f"✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def step_fpga_export():
    """Step 3: Export samples for FPGA simulation."""
    print_step(3, "FPGA Export")
    
    try:
        from export_multiple_samples import export_multiple_samples
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Exporting {NUM_SAMPLES} samples for FPGA simulation...")
        print(f"Device: {device}")
        
        export_multiple_samples(
            num_samples=NUM_SAMPLES,
            data_dir=str(DATA_DIR),
            params_dir=str(FPGA_DIR / "params"),
            model_path=MODEL_PATH,
            device=device
        )
        print("✓ Export completed successfully")
        return True
    except Exception as e:
        print(f"✗ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def step_fpga_simulation():
    """Step 4: Run FPGA simulation."""
    print_step(4, "FPGA Simulation")
    
    # Choose simulation script based on mode
    if CW305_MODE:
        sim_script = FPGA_DIR / "scripts" / "run_sim_cw305.tcl"
        print("Using CW305 (ChipWhisperer) simulation mode")
        print("Target Device: Xilinx Artix-7 XC7A100T-1FTG256C")
    else:
        sim_script = FPGA_DIR / "scripts" / "run_sim_multi.tcl"
    
    if not sim_script.exists():
        print(f"✗ Simulation script not found: {sim_script}")
        return False
    
    if not VIVADO_PATH.exists():
        print(f"✗ Vivado not found at: {VIVADO_PATH}")
        print("Please update VIVADO_PATH in run_all.py")
        return False
    
    print(f"Running Vivado simulation...")
    print(f"Script: {sim_script}")
    
    try:
        # Use relative path from fpga directory
        script_rel_path = "scripts/run_sim_multi.tcl"
        result = subprocess.run(
            [str(VIVADO_PATH), "-mode", "batch", "-source", script_rel_path],
            cwd=str(FPGA_DIR),
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            print("✓ Simulation completed successfully")
            return True
        else:
            print(f"✗ Simulation failed (return code: {result.returncode})")
            if result.stderr:
                print("Error output:")
                print(result.stderr[-500:])
            return False
    except subprocess.TimeoutExpired:
        print("✗ Simulation timed out")
        return False
    except Exception as e:
        print(f"✗ Simulation error: {e}")
        return False


def step_analysis():
    """Step 5: Analyze and compare results."""
    print_step(5, "Results Analysis")
    
    log_path = FPGA_DIR / "xsim.log"
    summary_path = FPGA_DIR / "params" / "multi_sample_summary.json"
    
    if not log_path.exists():
        print(f"✗ Simulation log not found: {log_path}")
        return False, None
    
    if not summary_path.exists():
        print(f"✗ Summary file not found: {summary_path}")
        return False, None
    
    try:
        from compare_fpga_results import parse_simulation_log, compare_results, print_comparison_report
        
        print("Parsing FPGA simulation results...")
        fpga_predictions = parse_simulation_log(log_path)
        print(f"Found {len(fpga_predictions)} FPGA predictions")
        
        print("Loading Python model predictions...")
        comparison = compare_results(summary_path, fpga_predictions)
        
        print()
        print_comparison_report(comparison)
        
        # Save results
        results_path = FPGA_DIR / "multi_sample_results.json"
        results_path.write_text(json.dumps(comparison, indent=2))
        print(f"\nDetailed results saved to: {results_path}")
        
        print("✓ Analysis completed successfully")
        return True, comparison['summary']
        
    except Exception as e:
        print(f"✗ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def main():
    """Run the complete pipeline."""
    print("="*80)
    print("COMPLETE PIPELINE RUNNER")
    print("="*80)
    print("This script will run:")
    print("  1. SNN Training")
    print("  2. Model Evaluation")
    print("  3. FPGA Export")
    print("  4. FPGA Simulation")
    print("  5. Results Analysis")
    print("="*80)
    
    start_time = time.time()
    results = {}
    
    # Step 1: Training
    if not step_training():
        print("\n✗ Pipeline stopped at training step")
        sys.exit(1)
    
    # Step 2: Evaluation
    if not step_evaluation():
        print("\n⚠ Evaluation failed, but continuing...")
    
    # Step 3: FPGA Export
    if not step_fpga_export():
        print("\n✗ Pipeline stopped at FPGA export step")
        sys.exit(1)
    
    # Step 4: FPGA Simulation
    if not step_fpga_simulation():
        print("\n⚠ Simulation failed, but continuing to analysis...")
    
    # Step 5: Analysis
    success, summary = step_analysis()
    if success and summary:
        results['fpga_summary'] = summary
    
    # Final Summary
    elapsed_time = time.time() - start_time
    print(f"\n{'='*80}")
    print("PIPELINE COMPLETE")
    print(f"{'='*80}")
    print(f"Total execution time: {elapsed_time/60:.1f} minutes")
    print()
    
    if 'fpga_summary' in results:
        summary = results['fpga_summary']
        print("Final Results:")
        print(f"  Python-FPGA Match Rate: {summary['match_rate']:.1f}%")
        print(f"  Python Model Accuracy: {summary['python_accuracy']:.1f}%")
        print(f"  FPGA Model Accuracy: {summary['fpga_accuracy']:.1f}%")
        print()
        
        if summary['match_rate'] == 100.0:
            print("✓ SUCCESS: FPGA matches Python model perfectly!")
        elif summary['match_rate'] >= 95.0:
            print("✓ GOOD: FPGA matches Python model with minor differences")
        else:
            print("⚠ WARNING: Significant differences between FPGA and Python")
    else:
        print("⚠ Could not generate final summary (analysis step may have failed)")
    
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

