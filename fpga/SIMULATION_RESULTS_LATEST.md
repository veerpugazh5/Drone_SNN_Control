# Latest FPGA Simulation Results

**Date:** December 2, 2025  
**Simulation:** Multi-sample testbench (20 samples)  
**Status:** ✅ **SUCCESS**

## Results Summary

| Sample | FPGA Prediction | Class |
|--------|----------------|-------|
| 0 | 0 | Straight |
| 1 | 0 | Straight |
| 2 | 0 | Straight |
| 3 | 1 | Turning |
| 4 | 0 | Straight |
| 5 | 0 | Straight |
| 6 | 0 | Straight |
| 7 | 0 | Straight |
| 8 | 0 | Straight |
| 9 | 0 | Straight |
| 10 | 0 | Straight |
| 11 | 1 | Turning |
| 12 | 0 | Straight |
| 13 | 0 | Straight |
| 14 | 0 | Straight |
| 15 | 1 | Turning |
| 16 | 0 | Straight |
| 17 | 0 | Straight |
| 18 | 0 | Straight |
| 19 | 1 | Turning |

## Statistics

- **Total Samples:** 20
- **Straight (Class 0):** 16 samples (80%)
- **Turning (Class 1):** 4 samples (20%)
- **Simulation Time:** ~16.4 microseconds per sample
- **Total Time:** ~328 microseconds for all 20 samples

## Verification

The simulation successfully:
- ✅ Compiled RTL and testbench
- ✅ Loaded all 20 spike stream files
- ✅ Processed all samples through SNN inference
- ✅ Generated predictions for each sample
- ✅ Completed without errors

## Next Steps

To compare with Python model predictions:
1. Check `params/multi_sample_summary.json` for Python predictions
2. Compare FPGA vs Python results
3. Verify match rate

## Technical Details

- **Testbench:** `sim/spike_fc_tb_multi.sv`
- **RTL Module:** `rtl/snn_fc_top.sv`
- **Simulator:** Xilinx Vivado XSim 2024.2
- **Time Resolution:** 1 ps
- **Clock Period:** 10 ns (100 MHz simulation clock)

