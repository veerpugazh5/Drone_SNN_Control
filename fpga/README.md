# FPGA Simulation Assets

This folder contains the minimal RTL and parameter dumps required to replay the spiking fully-connected readout stage of the trained SNN inside Vivado.

## Contents

- `params/` – Quantised weights, biases, and the recorded spike stream exported from the PyTorch model via `export_fpga.py`.
- `rtl/snn_fc_top.sv` – Sequential LIF accumulator that consumes the spike stream and produces class predictions.
- `sim/spike_fc_tb.sv` – Simple testbench that toggles reset/start and prints the predicted class when `done` goes high.
- `scripts/run_sim.tcl` – Batch script for Vivado/XSim (`vivado -mode batch -source scripts/run_sim.tcl`).

## Workflow

1. Run `python export_fpga.py --mode test --index <sample_id>` after training to refresh the `.mem` files under `params/`.
2. Launch Vivado in batch mode from `fpga/` and execute `vivado -mode batch -source scripts/run_sim.tcl`.
3. XSim will compile the RTL/testbench and print the predicted class in the transcript.

> **Note:** Vivado is not currently installed on this machine. Install the AMD Vivado Design Suite (2023.2 or newer) and ensure `vivado` is on your `PATH` before executing the TCL script.




