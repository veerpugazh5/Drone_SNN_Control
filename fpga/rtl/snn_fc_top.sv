`timescale 1ns/1ps

// Streaming LIF-based fully connected readout for binary classification (Straight vs Turning)
module snn_fc_top #(
    parameter int NUM_CLASSES = 2,
    parameter int NUM_FEATURES = 8192,
    parameter int NUM_STEPS = 10,
    parameter int WORDS_PER_STEP = 256,  // 8192 bits / 32 bits per word
    parameter int WEIGHT_WIDTH = 8,
    parameter int ACC_WIDTH = 32,
    parameter int LEAK_SHIFT = 1,
    parameter logic signed [ACC_WIDTH-1:0] THRESHOLD = 32'sd64,
    parameter string SPIKE_FILE = "params/spike_stream.mem",
    parameter string BIAS_FILE = "params/fc_bias.mem",
    parameter string WEIGHT_FILE0 = "params/fc_weights_c0.mem",
    parameter string WEIGHT_FILE1 = "params/fc_weights_c1.mem"
) (
    input  logic clk,
    input  logic rst,
    input  logic start,
    output logic done,
    output logic [$clog2(NUM_CLASSES)-1:0] predicted_class
);

    typedef enum logic [1:0] {
        S_IDLE,
        S_ACCUM,
        S_LEAK,
        S_DONE
    } state_t;

    state_t state;

    logic [$clog2(NUM_STEPS)-1:0] step_idx;
    logic [$clog2(WORDS_PER_STEP)-1:0] word_idx;
    logic [4:0] bit_idx;
    logic [$clog2(NUM_FEATURES)-1:0] feature_idx;

    // Flat array: total_words = NUM_STEPS * WORDS_PER_STEP
    logic [31:0] spike_words [0:NUM_STEPS*WORDS_PER_STEP-1];
    initial begin
        // Read spike stream from file
        $readmemh(SPIKE_FILE, spike_words);
    end

    logic [ACC_WIDTH-1:0] bias_mem_unsigned [0:NUM_CLASSES-1];
    logic signed [ACC_WIDTH-1:0] bias_mem [0:NUM_CLASSES-1];
    initial begin
        $readmemh(BIAS_FILE, bias_mem_unsigned);
        // Convert to signed (16-bit values)
        for (int i = 0; i < NUM_CLASSES; i++) begin
            bias_mem[i] = $signed(bias_mem_unsigned[i][15:0]);
        end
    end

    logic signed [WEIGHT_WIDTH-1:0] weight_mem [0:NUM_CLASSES-1][0:NUM_FEATURES-1];
    initial begin
        $readmemh(WEIGHT_FILE0, weight_mem[0]);
        $readmemh(WEIGHT_FILE1, weight_mem[1]);
    end

    logic signed [ACC_WIDTH-1:0] membrane [0:NUM_CLASSES-1];
    logic [7:0] spike_counts [0:NUM_CLASSES-1];
    
    logic [31:0] current_word;
    logic current_spike;
    logic [$clog2(NUM_STEPS*WORDS_PER_STEP)-1:0] word_addr;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= S_IDLE;
            step_idx <= '0;
            word_idx <= '0;
            bit_idx <= '0;
            feature_idx <= '0;
            done <= 1'b0;
            predicted_class <= '0;
            for (int c = 0; c < NUM_CLASSES; c++) begin
                membrane[c] <= '0;
                spike_counts[c] <= '0;
            end
        end else begin
            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        step_idx <= '0;
                        word_idx <= '0;
                        bit_idx <= '0;
                        feature_idx <= '0;
                        for (int c = 0; c < NUM_CLASSES; c++) begin
                            membrane[c] <= '0;
                            spike_counts[c] <= '0;
                        end
                        state <= S_ACCUM;
                    end
                end
                S_ACCUM: begin
                    // Calculate address within this sample's spike stream
                    word_addr = step_idx * WORDS_PER_STEP + word_idx;
                    current_word = spike_words[word_addr];
                    current_spike = current_word[bit_idx];
                    
                    if (current_spike && feature_idx < NUM_FEATURES) begin
                        for (int c = 0; c < NUM_CLASSES; c++) begin
                            logic signed [ACC_WIDTH-1:0] weight_val;
                            weight_val = $signed({{(ACC_WIDTH-WEIGHT_WIDTH){weight_mem[c][feature_idx][WEIGHT_WIDTH-1]}}, weight_mem[c][feature_idx]});
                            membrane[c] <= membrane[c] + weight_val;
                        end
                    end
                    
                    bit_idx <= bit_idx + 1;
                    if (bit_idx == 31) begin
                        bit_idx <= '0;
                        word_idx <= word_idx + 1;
                    end
                    
                    // Increment feature index and check if done
                    if (feature_idx < NUM_FEATURES - 1) begin
                        feature_idx <= feature_idx + 1;
                    end else begin
                        // All features processed for this timestep
                        state <= S_LEAK;
                    end
                end
                S_LEAK: begin
                    for (int c = 0; c < NUM_CLASSES; c++) begin
                        logic signed [ACC_WIDTH-1:0] decayed;
                        decayed = membrane[c] - (membrane[c] >>> LEAK_SHIFT);
                        if (decayed >= THRESHOLD) begin
                            membrane[c] <= decayed - THRESHOLD;
                            spike_counts[c] <= spike_counts[c] + 1;
                        end else begin
                            membrane[c] <= decayed;
                        end
                    end
                    step_idx <= step_idx + 1;
                    feature_idx <= '0;
                    if (step_idx == NUM_STEPS - 1) begin
                        state <= S_DONE;
                    end else begin
                        state <= S_ACCUM;
                    end
                end
                S_DONE: begin
                    if (!done) begin
                        int winner;
                        logic signed [ACC_WIDTH-1:0] best_val;
                        winner = 0;
                        best_val = membrane[0] + bias_mem[0];
                        for (int c = 1; c < NUM_CLASSES; c++) begin
                            logic signed [ACC_WIDTH-1:0] candidate;
                            candidate = membrane[c] + bias_mem[c];
                            if (candidate > best_val) begin
                                best_val = candidate;
                                winner = c;
                            end
                        end
                        predicted_class <= winner[$clog2(NUM_CLASSES)-1:0];
                        done <= 1'b1;
                    end
                end
                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
