// Phase 5 Level 2.1: Parameterized Systolic Array Core
// N×N grid of processing elements for parallel matrix multiply
//
// Architecture:
//   - Weight-stationary dataflow
//   - Weights: Preloaded column-by-column, stationary during compute
//   - Data: Flows west → east (horizontal propagation)
//   - Partial sums: Flow north → south (vertical accumulation)
//
// Usage:
//   1. Preload weights: Assert load_weights, feed ARRAY_SIZE columns sequentially
//   2. Compute: Assert data_valid, feed input rows sequentially
//   3. Collect results: Read partial_sum_out (bottom row) when result_valid
//
// Timing: Result valid after ARRAY_SIZE cycles (pipeline latency)
//
// Performance: Processes ARRAY_SIZE × ARRAY_SIZE MACs per cycle during steady state

module systolic_array #(
    parameter ARRAY_SIZE = 8  // Configurable: 4, 8, 12, or 16
) (
    input wire clk,
    input wire rst,

    // Weight loading interface (preload all PEs before compute)
    input wire signed [ARRAY_SIZE-1:0][7:0] weight_col_in,  // One column at a time
    input wire load_weights,                                 // Weight loading mode

    // Input data interface (one row per cycle)
    // CRITICAL FIX: Changed to unpacked to match caller's format
    input wire signed [7:0] data_row_in [0:ARRAY_SIZE-1],   // Input row (unpacked)
    input wire data_valid,                                   // Data valid signal

    // Output control interface
    input wire latch_outputs,                                // Pulse to latch outputs for stable collection

    // Output interface (bottom row PEs)
    output wire signed [ARRAY_SIZE-1:0][31:0] partial_sum_out,
    output reg result_valid                                  // Output valid signal
);

    // PE grid signals
    // Weight connections: North → South (for preloading)
    wire signed [ARRAY_SIZE:0][ARRAY_SIZE-1:0][7:0] weight_vertical;

    // Data connections: West → East
    wire signed [ARRAY_SIZE-1:0][ARRAY_SIZE:0][7:0] data_horizontal;

    // Partial sum connections: North → South
    wire signed [ARRAY_SIZE:0][ARRAY_SIZE-1:0][31:0] partial_sum_vertical;

    // Load weight control for each PE
    reg [ARRAY_SIZE-1:0][ARRAY_SIZE-1:0] pe_load_weight;

    // Weight shift register for column-by-column loading
    reg [ARRAY_SIZE-1:0][7:0] weight_shift_reg [0:ARRAY_SIZE-1];  // [column][row]
    reg [$clog2(ARRAY_SIZE+1)-1:0] weight_col_count;  // Number of columns loaded so far

    // Data valid shift register (tracks pipeline depth)
    reg [ARRAY_SIZE-1:0] data_valid_shift;

    // Output holding registers for stable collection
    reg signed [31:0] output_holding_regs [ARRAY_SIZE-1:0];
    reg outputs_latched;  // State: 0=pass-through, 1=holding latched values

    // Wavefront input delay registers: row i delayed by i cycles
    // Implemented as shift registers that continuously shift
    reg signed [7:0] row_input_delays [1:ARRAY_SIZE-1][0:ARRAY_SIZE-1];

    // Initialize weight shift registers
    integer init_col, init_row;
    initial begin
        for (init_col = 0; init_col < ARRAY_SIZE; init_col = init_col + 1) begin
            for (init_row = 0; init_row < ARRAY_SIZE; init_row = init_row + 1) begin
                weight_shift_reg[init_col][init_row] = 8'sd0;
            end
        end
    end

    // Weight loading with proper timing:
    // Cycle N: Write weight_col_in to shift_reg[col]
    // Cycle N+1: Assert pe_load_weight for shift_reg[col] (data is stable)
    reg load_weights_delayed;
    reg [$clog2(ARRAY_SIZE+1)-1:0] prev_col;

    always @(posedge clk) begin
        if (rst) begin
            weight_col_count <= 0;
            load_weights_delayed <= 1'b0;
            prev_col <= 0;
            // CRITICAL: Reset weight shift registers to clear old weights between passes
            for (init_col = 0; init_col < ARRAY_SIZE; init_col = init_col + 1) begin
                for (init_row = 0; init_row < ARRAY_SIZE; init_row = init_row + 1) begin
                    weight_shift_reg[init_col][init_row] <= 8'sd0;
                end
            end
        end else begin
            // Delay load_weights by 1 cycle
            load_weights_delayed <= load_weights;
            prev_col <= weight_col_count;

            if (load_weights) begin
                // Load current column into shift register
                for (init_row = 0; init_row < ARRAY_SIZE; init_row = init_row + 1) begin
                    weight_shift_reg[weight_col_count][init_row] <= weight_col_in[init_row];
                end

                // Increment column counter (stops at ARRAY_SIZE to prevent wrap-around bug)
                if (weight_col_count < ARRAY_SIZE) begin
                    weight_col_count <= weight_col_count + 1;
                end
            end else begin
                // Reset counter when not loading
                weight_col_count <= 0;
            end
        end
    end

    // Generate load_weight signals for each PE
    // Use prev_col (weight data is stable after 1 cycle delay)
    integer load_row, load_col;
    always @(*) begin
        for (load_row = 0; load_row < ARRAY_SIZE; load_row = load_row + 1) begin
            for (load_col = 0; load_col < ARRAY_SIZE; load_col = load_col + 1) begin
                pe_load_weight[load_row][load_col] = 1'b0;
            end
        end

        // Assert load for previous column (data now stable in shift register)
        if (load_weights_delayed && prev_col < ARRAY_SIZE) begin
            for (load_row = 0; load_row < ARRAY_SIZE; load_row = load_row + 1) begin
                pe_load_weight[load_row][prev_col] = 1'b1;
            end
        end
    end

    // Connect weight inputs (vertical connections for preloading)
    genvar col_idx;
    generate
        for (col_idx = 0; col_idx < ARRAY_SIZE; col_idx = col_idx + 1) begin : gen_weight_col_inputs
            genvar row_idx;
            for (row_idx = 0; row_idx < ARRAY_SIZE; row_idx = row_idx + 1) begin : gen_weight_row_inputs
                assign weight_vertical[row_idx][col_idx] = weight_shift_reg[col_idx][row_idx];
            end
        end
    endgenerate

    // Wavefront delay shift registers
    // Row 0: no delay, connects directly
    assign data_horizontal[0][0] = data_row_in[0];

    // Rows 1+: delayed by hardware shift registers
    integer delay_row, delay_stage;
    always @(posedge clk) begin
        if (rst) begin
            // Clear all delays
            for (delay_row = 1; delay_row < ARRAY_SIZE; delay_row = delay_row + 1) begin
                for (delay_stage = 0; delay_stage < delay_row; delay_stage = delay_stage + 1) begin
                    row_input_delays[delay_row][delay_stage] <= 8'sd0;
                end
            end
        end else begin
            // Always shift, feeding new data or zeros
            for (delay_row = 1; delay_row < ARRAY_SIZE; delay_row = delay_row + 1) begin
                // Stage 0: Input from data_row_in when valid, else 0
                row_input_delays[delay_row][0] <= data_valid ? data_row_in[delay_row] : 8'sd0;

                // Remaining stages: Shift from previous stage
                for (delay_stage = 1; delay_stage < delay_row; delay_stage = delay_stage + 1) begin
                    row_input_delays[delay_row][delay_stage] <= row_input_delays[delay_row][delay_stage-1];
                end
            end

            // DEBUG (Hour 18): Log data_row_in when data_valid asserted (first cycle only)
            if (data_valid && !data_valid_shift[0]) begin
                $display("[SYSTOLIC_ARRAY_INPUT] data_row_in=[%d,%d,%d,%d,%d,%d,%d,%d]",
                        $signed(data_row_in[0]),
                        $signed(data_row_in[1]),
                        $signed(data_row_in[2]),
                        $signed(data_row_in[3]),
                        $signed(data_row_in[4]),
                        $signed(data_row_in[5]),
                        $signed(data_row_in[6]),
                        $signed(data_row_in[7]));
            end
        end
    end

    // Connect delay outputs to PE inputs
    generate
        for (col_idx = 1; col_idx < ARRAY_SIZE; col_idx = col_idx + 1) begin : gen_data_west_delayed
            assign data_horizontal[col_idx][0] = row_input_delays[col_idx][col_idx-1];
        end
    endgenerate

    // Connect partial sum inputs (north edge - zeros for first row)
    generate
        for (col_idx = 0; col_idx < ARRAY_SIZE; col_idx = col_idx + 1) begin : gen_psum_north_inputs
            assign partial_sum_vertical[0][col_idx] = 32'sd0;
        end
    endgenerate

    // Instantiate N×N grid of PEs
    genvar row, col;
    generate
        for (row = 0; row < ARRAY_SIZE; row = row + 1) begin : gen_pe_rows
            for (col = 0; col < ARRAY_SIZE; col = col + 1) begin : gen_pe_cols
                processing_element pe (
                    .clk(clk),
                    .rst(rst),

                    // Weight (from shift register during loading, stationary during compute)
                    .weight_in(weight_vertical[row][col]),
                    .load_weight(pe_load_weight[row][col]),

                    // Data input (from west neighbor or input port)
                    .data_in(data_horizontal[row][col]),

                    // Partial sum input (from north neighbor or zero)
                    .partial_sum_in(partial_sum_vertical[row][col]),

                    // Data output (to east neighbor)
                    .data_out(data_horizontal[row][col+1]),

                    // Partial sum output (to south neighbor or output port)
                    .partial_sum_out(partial_sum_vertical[row+1][col])
                );
            end
        end
    endgenerate

    // Connect outputs (bottom row) with holding register mux
    // When outputs_latched=1, output holding registers (stable)
    // When outputs_latched=0, output direct from array (pass-through)
    generate
        for (col_idx = 0; col_idx < ARRAY_SIZE; col_idx = col_idx + 1) begin : gen_outputs
            assign partial_sum_out[col_idx] = outputs_latched ?
                                              output_holding_regs[col_idx] :
                                              partial_sum_vertical[ARRAY_SIZE][col_idx];
        end
    endgenerate

    // Pipeline valid signal (ARRAY_SIZE cycles for result to propagate)
    always @(posedge clk) begin
        if (rst) begin
            data_valid_shift <= {ARRAY_SIZE{1'b0}};
            result_valid <= 1'b0;
        end else begin
            // Shift register for data_valid
            data_valid_shift <= {data_valid_shift[ARRAY_SIZE-2:0], data_valid};

            // Result valid after ARRAY_SIZE cycles
            result_valid <= data_valid_shift[ARRAY_SIZE-1];
        end
    end

    // Output holding register control FSM
    // State machine: UNLOCKED (pass-through) ↔ LATCHED (holding)
    // Transitions:
    //   - latch_outputs=1 when unlocked → latch and go to LATCHED
    //   - data_valid=1 when latched → release and go to UNLOCKED
    integer latch_idx;
    always @(posedge clk) begin
        if (rst) begin
            // Reset: Unlocked state, clear holding registers
            outputs_latched <= 1'b0;
            for (latch_idx = 0; latch_idx < ARRAY_SIZE; latch_idx = latch_idx + 1) begin
                output_holding_regs[latch_idx] <= 32'sd0;
            end
        end else begin
            // Auto-release: New computation starts (data_valid=1) → unlock
            if (data_valid && outputs_latched) begin
                outputs_latched <= 1'b0;
            end

            // Latch outputs: Capture current outputs into holding registers
            // Only latch if not already latched (prevents accidental overwrite)
            if (latch_outputs && !outputs_latched) begin
                for (latch_idx = 0; latch_idx < ARRAY_SIZE; latch_idx = latch_idx + 1) begin
                    output_holding_regs[latch_idx] <= partial_sum_vertical[ARRAY_SIZE][latch_idx];
                end
                outputs_latched <= 1'b1;
            end
        end
    end

endmodule
