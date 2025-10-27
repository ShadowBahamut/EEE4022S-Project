// Phase 5 Level 1.1: Processing Element (PE) for Systolic Array
// Single MAC unit with weight-stationary dataflow and data forwarding
//
// Architecture:
//   Weight-stationary: Load weight once, reuse for multiple inputs
//   Data flows: West → East (horizontal propagation)
//   Partial sums: North → South (vertical accumulation)
//
// Operation:
//   1. Load weight via load_weight signal (stays stationary)
//   2. Each cycle: partial_sum_out = partial_sum_in + (weight × data_in)
//   3. Forward data_in to data_out (1 cycle delay)
//
// Timing: 1 cycle latency for all paths (fully pipelined)

module processing_element (
    input wire clk,
    input wire rst,

    // Weight loading (from north, stationary after load)
    input wire signed [7:0] weight_in,      // Weight value to load
    input wire load_weight,                  // Load signal (pulse)

    // Data input (from west, flows to east)
    input wire signed [7:0] data_in,

    // Partial sum input (from north, accumulates to south)
    input wire signed [31:0] partial_sum_in,

    // Data output (to east, 1 cycle delayed)
    output reg signed [7:0] data_out,

    // Partial sum output (to south, accumulated)
    output reg signed [31:0] partial_sum_out
);

    // Stationary weight register
    reg signed [7:0] weight_reg;

    // MAC computation (combinational)
    wire signed [15:0] product;
    wire signed [31:0] mac_result;

    // Multiply: weight × data_in (8-bit × 8-bit = 16-bit)
    assign product = weight_reg * data_in;

    // Accumulate: partial_sum_in + product (sign-extend to 32-bit)
    assign mac_result = partial_sum_in + {{16{product[15]}}, product};

    // Sequential logic
    always @(posedge clk) begin
        if (rst) begin
            weight_reg <= 8'sd0;
            data_out <= 8'sd0;
            partial_sum_out <= 32'sd0;
        end else begin
            // Weight loading (stationary - only update when load_weight asserted)
            if (load_weight) begin
                weight_reg <= weight_in;
                `ifdef DEBUG
                $display("[PE] Loaded weight: %d", $signed(weight_in));
                `endif
            end

            // Data forwarding: West → East (1 cycle delay)
            data_out <= data_in;

            // Partial sum accumulation: North → South
            partial_sum_out <= mac_result;

            // Debug: Show MAC operation (disabled for clean output)
            // `ifdef DEBUG
            // $display("[PE] MAC: weight=%d, data=%d, partial_in=%d → product=%d, partial_out=%d",
            //          $signed(weight_reg), $signed(data_in), $signed(partial_sum_in),
            //          $signed(product), $signed(mac_result));
            // `endif
        end
    end

endmodule
