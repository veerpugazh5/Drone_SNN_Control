`timescale 1ns/1ps

module spike_fc_tb_multi;

    parameter int NUM_SAMPLES = 20;
    
    logic clk = 0;
    logic rst = 1;
    logic [NUM_SAMPLES-1:0] start = 0;
    wire [NUM_SAMPLES-1:0] done;
    wire [0:0] predicted_class [0:NUM_SAMPLES-1];

    // Generate 20 instances, one for each sample
    genvar i;
    generate
        for (i = 0; i < NUM_SAMPLES; i++) begin : gen_dut
            snn_fc_top #(
                .SPIKE_FILE($sformatf("params/spike_stream_%0d.mem", i))
            ) dut (
                .clk(clk),
                .rst(rst),
                .start(start[i]),
                .done(done[i]),
                .predicted_class(predicted_class[i])
            );
        end
    endgenerate

    always #5 clk = ~clk;

    // Results storage
    int fpga_predictions [0:NUM_SAMPLES-1];
    int current_sample = 0;

    initial begin
        $display("======================================================================");
        $display("MULTI-SAMPLE FPGA SIMULATION - Binary SNN Inference");
        $display("======================================================================");
        $display("Processing %0d samples sequentially...", NUM_SAMPLES);
        $display("Class 0 = Straight, Class 1 = Turning");
        $display("----------------------------------------------------------------------");
        
        #50;
        rst = 0;
        #20;
        
        // Process each sample sequentially
        for (current_sample = 0; current_sample < NUM_SAMPLES; current_sample++) begin
            $display("Sample %0d: Starting inference (file: spike_stream_%0d.mem)...", 
                     current_sample, 
                     current_sample);
            
            // Reset all instances
            rst = 1;
            #20;
            rst = 0;
            #20;
            
            // Start only the current sample's instance
            start[current_sample] = 1;
            #10;
            start[current_sample] = 0;
            
            // Wait for completion
            wait(done[current_sample]);
            
            // Store result
            fpga_predictions[current_sample] = predicted_class[current_sample];
            
            $display("Sample %0d: FPGA prediction = %0d (%s)", 
                     current_sample, 
                     predicted_class[current_sample],
                     predicted_class[current_sample] ? "Turning" : "Straight");
            
            #20;  // Small delay between samples
        end
        
        $display("----------------------------------------------------------------------");
        $display("All samples processed. Results:");
        $display("----------------------------------------------------------------------");
        for (int j = 0; j < NUM_SAMPLES; j++) begin
            $display("Sample %0d: FPGA = %0d (%s)", 
                     j,
                     fpga_predictions[j],
                     fpga_predictions[j] ? "Turning" : "Straight");
        end
        $display("======================================================================");
        $display("Simulation complete. Check multi_sample_results.json for comparison.");
        $display("======================================================================");
        
        #20;
        $finish;
    end

endmodule
