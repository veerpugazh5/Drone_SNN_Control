# Drone SNN Control

**Neuromorphic Control Policy for Autonomous Drone Flight on FPGA**

This project implements a Spiking Neural Network (SNN) for autonomous drone navigation using event-based vision data. The system processes neuromorphic camera events to classify drone motion (Straight vs. Turning) and is designed for deployment on AMD-Xilinx FPGAs.

## Project Overview

We use a Leaky Integrate-and-Fire (LIF) SNN architecture to process event streams efficiently. The pipeline includes:
1.  **Preprocessing**: Converting raw HDF5 event data into temporal tensor frames.
2.  **Training**: Optimizing a binary SNN using PyTorch with surrogate gradients.
3.  **Quantization**: Converting floating-point weights to 8-bit fixed-point integers.
4.  **Hardware Deployment**: A custom SystemVerilog RTL implementation of the SNN inference engine.

## Results

Our FPGA implementation achieves **100% output parity** with the Python simulation on the test set.

| Metric | Result |
| :--- | :--- |
| **Overall Accuracy** | **70.8%** |
| **Straight Accuracy** | 79.6% |
| **Turning Accuracy** | 54.2% |
| **FPGA Match Rate** | **100%** (Bit-exact) |
| **Inference Latency** | < 1ms (Hardware estimate) |

## Setup

### Prerequisites
-   Python 3.8+
-   Vivado 2024.2 (for FPGA simulation)

### Installation
```bash
git clone https://github.com/yourusername/Drone_snn_control.git
cd Drone_snn_control
pip install -r requirements.txt
```

### Dataset
The project uses the **Fully-neuromorphic vision and control for autonomous drone flight** dataset.
Due to licensing, the data is not included in this repository.

Run the helper script to get instructions on downloading the data:
```bash
python download_data.py
```
Place the downloaded files in the `sr_dataset_gt` directory.

## Usage

You can run the entire pipeline (Training → Evaluation → Export → Simulation) with a single command:

```bash
python run_all.py
```

### Individual Steps

**1. Preprocess Data**
Converts HDF5 files to fast-loading PyTorch tensors.
```bash
python preprocess_data.py
```

**2. Train SNN**
Trains the model and saves the best checkpoint (`best_snn_fast.pth`).
```bash
python train_fast.py
```

**3. Evaluate**
Runs inference on the test set and reports classification metrics.
```bash
python evaluate.py
```

**4. FPGA Simulation**
Exports weights and spike streams, then runs the Vivado simulation.
```bash
python run_all.py  # (Steps 3-5 are automated here)
```

## Repository Structure

-   `model_binary.py`: PyTorch definition of the SNN (LIF neurons).
-   `train_fast.py`: Training loop with balanced accuracy optimization.
-   `fpga/rtl/`: SystemVerilog source code for the hardware accelerator.
-   `fpga/scripts/`: TCL scripts for running Vivado simulations.
-   `preprocessed/`: Directory for cached tensor data (generated).

## Hardware Implementation

The FPGA design uses a streaming architecture to process spikes in real-time.
-   **Input**: 32-bit spike words (256 words/step).
-   **Core**: Accumulates weighted inputs and applies leak/threshold dynamics.
-   **Precision**: 8-bit weights, 16-bit biases, 32-bit accumulators.
-   **Target**: AMD-Xilinx Artix-7 / Zynq-7000 series.

---
*Created for the Neuromorphic Vision & Control Project (Fall 2025).*
