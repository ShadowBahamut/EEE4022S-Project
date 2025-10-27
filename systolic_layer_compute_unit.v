// Phase 5 Level 3: Systolic Array-Based Layer Computation Unit
// Computes entire layer (all neurons) using parallel systolic array
//
// Architecture: Matrix-Vector Multiply y = W × x
//   - W: m×n weight matrix (m neurons, n inputs)
//   - x: n×1 input vector
//   - y: m×1 output vector (all neuron outputs)
//
// Tiling Strategy (8×8 blocks):
//   - Outer loop: Process 8 neurons at a time (m/8 passes)
//   - Inner loop: Process 8 inputs at a time (n/8 blocks per pass)
//   - Each block: 64 parallel MACs (8×8 systolic array)
//   - Accumulate partial results across blocks
//   - Add bias to final accumulated results
//
// Performance: 8× speedup over Phase 4 (64 MACs/cycle vs 8 MACs/cycle)
//
// CRITICAL LESSONS FROM LAYER 2 (applied here):
//   1. Feed ROWS not columns (for data feeding - avoid transpose bug in matrix-matrix)
//   2. Use diagonal collection: results[row][col] = partial_sum_out[col], row = cycle-col
//   3. Preload weights column-by-column
//   4. Results emerge AFTER feeding, no extra wait
//   5. **TRANSPOSE WEIGHTS** for matrix-vector: load W[row][col] into column row (2025-10-20)

module systolic_layer_compute_unit #(
    parameter ARRAY_SIZE = 8,
    parameter INPUT_ZERO_POINT = -128    // Input quantization zero-point (TFLite)
) (
    input wire clk,
    input wire rst,

    // Layer configuration (compatible with layer_compute_unit_bram.v interface)
    input wire [7:0] num_neurons,        // Number of neurons (128, 10, 64)
    input wire [15:0] input_size,        // Input size (784, 128, 64)
    input wire [31:0] layer_base_addr,   // BRAM base address for layer weights
    input wire start,                     // Start layer computation
    input wire clear_buffers,             // Clear input/bias buffers (for new image)
    output reg done,                      // Computation complete

    // Input buffer interface (pre-loaded via DMA)
    input wire input_write_en,
    input wire [15:0] input_write_addr,  // 0-783 for Layer 1
    input wire signed [7:0] input_write_data,

    // Bias buffer interface (pre-loaded)
    input wire bias_write_en,
    input wire [7:0] bias_write_addr,    // 0-127 for Layer 1
    input wire signed [31:0] bias_write_data,

    // Weight BRAM interface - Port B (Phase 6 Layer 1)
    output wire [19:0] bram_addr,        // 20 bits = 1MB address space
    input wire [31:0] bram_data_out,
    output wire bram_en,

    // Weight BRAM interface - Port A (Phase 6 Layer 2: Dual-port reads)
    output wire [19:0] bram_addr_a,      // Port A address
    input wire [31:0] bram_data_a_out,   // Port A data
    output wire bram_en_a,               // Port A enable

    // Output interface
    output reg [7:0] output_addr,        // Neuron index (0-127)
    output reg signed [31:0] output_data,// Neuron output
    output reg output_valid              // Output write strobe
);

    // FSM states
    localparam IDLE              = 4'd0;
    localparam FETCH_WEIGHTS     = 4'd1;
    localparam LOAD_SYSTOLIC_COL = 4'd2;
    localparam WAIT_WEIGHT_LOAD  = 4'd3;
    localparam FEED_DATA         = 4'd4;
    localparam COLLECT_RESULTS   = 4'd5;
    localparam NEXT_BLOCK        = 4'd6;
    localparam ADD_BIAS          = 4'd7;
    localparam WRITE_OUTPUTS     = 4'd8;
    localparam NEXT_PASS         = 4'd9;
    localparam DONE              = 4'd10;
    localparam WAIT_FOR_PROPAGATION = 4'd11;
    localparam WAIT_FETCH_COMPLETE = 4'd12;  // Phase 6 Layer 3 Opt #2

    reg [3:0] state;
    reg [3:0] wait_cycles;

    // Buffers
    reg signed [7:0] input_buffer [0:783];   // Input vector
    reg signed [31:0] bias_buffer [0:127];   // Bias for each neuron

    // Phase 6 Layer 3: Dual-buffer architecture for pipelined fetch + compute
    reg signed [7:0] weight_buffer_a [0:127];  // Buffer A: 8×8 weight block (64 elements)
    reg signed [7:0] weight_buffer_b [0:127];  // Buffer B: 8×8 weight block (64 elements)
    reg active_buffer;                         // 0 = use Buffer A, 1 = use Buffer B
    reg fetch_buffer;                          // 0 = fetch into A, 1 = fetch into B

    // Accumulators for 8 neurons being computed in current pass
    reg signed [31:0] accumulators [0:7];

    // ZERO-POINT CORRECTION: Weight sums for each neuron (Σ(weights))
    // Used to compute correction term: -input_zero_point * Σ(weights)
    // See PHASE5_ZERO_POINT_BUG.md for details
    reg signed [31:0] weight_sums [0:7];

    // ZERO-POINT CORRECTION: Temporary variables for correction computation
    reg signed [31:0] correction;
    reg signed [31:0] corrected_accum;

    // Phase 7 Opt #5: Pre-computed correction pipeline
    // Store pre-computed correction (INPUT_ZERO_POINT * weight_sums[i]) for each neuron
    // Computed during COLLECT_RESULTS, used during WRITE_OUTPUTS
    reg signed [31:0] correction_pipeline [0:7];

    // Control variables
    reg [7:0] neuron_base;         // Base neuron index for current pass (0, 8, 16, ...)
    reg [15:0] input_base;         // Base input index for current block (0, 8, 16, ...)
    reg [7:0] total_passes;        // Total number of 8-neuron passes
    reg [15:0] total_blocks;       // Total number of 8-input blocks per pass
    reg [7:0] current_pass;        // Current pass index
    reg [15:0] current_block;      // Current block index within pass
    reg [3:0] weight_col_idx;      // Column index during weight loading (0-7)
    reg [3:0] data_row_idx;        // Row index during data feeding (0-7)
    reg [4:0] collect_cycle;       // Collection cycle (0-14 for 8×8)
    reg [2:0] output_write_idx;    // Index for sequential output writing (0-7)

    // PROFILING: Cycle counters for each FSM state
    reg [31:0] cycles_idle;
    reg [31:0] cycles_fetch_weights;
    reg [31:0] cycles_load_systolic;
    reg [31:0] cycles_wait_propagation;
    reg [31:0] cycles_feed_data;
    reg [31:0] cycles_collect_results;
    reg [31:0] cycles_next_block;
    reg [31:0] cycles_add_bias;
    reg [31:0] cycles_write_outputs;
    reg [31:0] cycles_next_pass;
    reg [31:0] cycles_done;
    reg [31:0] total_layer_cycles;

    // Systolic array signals
    reg signed [ARRAY_SIZE-1:0][7:0] weight_col_in;
    reg load_weights;
    // CRITICAL FIX: Changed from packed to unpacked to match input_buffer format
    // Packed arrays can cause Verilator simulation artifacts
    reg signed [7:0] data_row_in [0:ARRAY_SIZE-1];
    reg data_valid;
    reg latch_outputs;
    wire signed [ARRAY_SIZE-1:0][31:0] partial_sum_out;
    wire result_valid;

    // Phase 6 Layer 1: Direct BRAM addressing (4-weights-per-read optimization)
    // Phase 6 Layer 2: Dual-port BRAM reads (8-weights-per-dual-read)
    reg [7:0] weight_fetch_row;    // Which row of weights to fetch (neuron index)
    reg [15:0] weight_fetch_col;   // Which column (input index) - increments by 8 (Layer 2)

    // Port B signals (odd words: 1, 3, 5, 7, ...)
    reg [19:0] bram_addr_b_reg;    // Port B address register
    reg bram_en_b_reg;             // Port B enable register
    reg [31:0] bram_data_b_latched; // Latch Port B data (weights 4-7)

    // Port A signals (even words: 0, 2, 4, 6, ...)
    reg [19:0] bram_addr_a_reg;    // Port A address register
    reg bram_en_a_reg;             // Port A enable register
    reg [31:0] bram_data_a_latched; // Latch Port A data (weights 0-3)

    reg [1:0] fetch_state;         // 0=IDLE, 1=REQUEST, 2=WAIT, 3=EXTRACT
    reg wait_extra_cycle;          // Extra wait for BRAM latency

    // Phase 6 Layer 3 Opt #2: Parallel fetch support
    reg fetch_next_block;          // Flag to trigger background fetch of next block
    reg [7:0] prefetch_row;        // Row being prefetched
    reg [15:0] prefetch_col;       // Column being prefetched
    reg [1:0] prefetch_state;      // Prefetch FSM state
    reg prefetch_wait_cycle;       // Wait cycle for prefetch

    // Initialize buffers
    integer i, neuron_idx;
    initial begin
        for (i = 0; i < 784; i = i + 1) begin
            input_buffer[i] = 8'sd0;
        end
        for (i = 0; i < 128; i = i + 1) begin
            bias_buffer[i] = 32'sd0;
            // Phase 6 Layer 3: Initialize both dual buffers
            if (i < 128) begin
                weight_buffer_a[i] = 8'sd0;
                weight_buffer_b[i] = 8'sd0;
            end
        end
        for (i = 0; i < 8; i = i + 1) begin
            accumulators[i] = 32'sd0;
            weight_sums[i] = 32'sd0;  // Initialize weight_sums to prevent X values
        end
    end

    // ===== Input/Bias Buffer Loading =====
    integer clear_idx;
    always @(posedge clk) begin
        if (rst) begin
            // Buffers initialized above
        end else if (clear_buffers) begin
            // DEBUG: DISABLE BUFFER CLEARING to rule it out as cause of bug
            // clear_idx = 0;  // Suppress unused variable warning
            $display("[SYSTOLIC_LAYER] Buffer clear DISABLED (debugging)");
        end else begin
            if (input_write_en && input_write_addr < 784) begin
                input_buffer[input_write_addr] <= input_write_data;
                // DEBUG: Log first 10 AND pixels 200-209 (digit region)
                if (input_write_addr < 10 || (input_write_addr >= 200 && input_write_addr < 210)) begin
                    $display("[DEBUG_INPUT_WRITE] addr=%3d value=%4d", input_write_addr, $signed(input_write_data));
                end
            end
            if (bias_write_en && bias_write_addr < 128) begin
                bias_buffer[bias_write_addr] <= bias_write_data;
            end
        end
    end

    // ===== Phase 6 Layer 1: Direct BRAM Access (no WRR module) =====
    // ===== Phase 6 Layer 2: Dual-Port BRAM Access =====
    // Connect BRAM signals to registers (controlled in FETCH_WEIGHTS state)

    // Port B: Odd words (1, 3, 5, 7, ...)
    assign bram_addr = bram_addr_b_reg;
    assign bram_en = bram_en_b_reg;

    // Port A: Even words (0, 2, 4, 6, ...)
    assign bram_addr_a = bram_addr_a_reg;
    assign bram_en_a = bram_en_a_reg;


    // ===== Systolic Array Instantiation =====
    systolic_array #(
        .ARRAY_SIZE(ARRAY_SIZE)
    ) systolic (
        .clk(clk),
        .rst(rst),
        .weight_col_in(weight_col_in),
        .load_weights(load_weights),
        .data_row_in(data_row_in),
        .data_valid(data_valid),
        .latch_outputs(latch_outputs),
        .partial_sum_out(partial_sum_out),
        .result_valid(result_valid)
    );

    // ===== Main FSM =====
    always @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            done <= 1'b0;
            output_valid <= 1'b0;
            neuron_base <= 8'h0;
            input_base <= 16'h0;
            current_pass <= 8'h0;
            current_block <= 16'h0;
            total_passes <= 8'h0;
            total_blocks <= 16'h0;
            weight_col_idx <= 4'h0;
            data_row_idx <= 4'h0;
            collect_cycle <= 5'h0;
            output_write_idx <= 3'h0;
            wait_cycles <= 4'h0;
            // Phase 6 Layer 1: Direct BRAM addressing signals
            // Phase 6 Layer 2: Dual-port BRAM signals
            weight_fetch_row <= 8'h0;
            weight_fetch_col <= 16'h0;
            fetch_state <= 2'b00;
            bram_addr_a_reg <= 20'h0;
            bram_en_a_reg <= 1'b0;
            bram_addr_b_reg <= 20'h0;
            bram_en_b_reg <= 1'b0;
            wait_extra_cycle <= 1'b0;
            // Phase 6 Layer 3 Opt #2: Parallel fetch signals
            fetch_next_block <= 1'b0;
            prefetch_row <= 8'h0;
            prefetch_col <= 16'h0;
            prefetch_state <= 2'b00;
            prefetch_wait_cycle <= 1'b0;

            // Phase 6 Layer 3: Dual-buffer signals
            active_buffer <= 1'b0;   // Start using Buffer A
            fetch_buffer <= 1'b1;    // Start fetching into Buffer B (opposite of active!)

            // PROFILING: Initialize cycle counters
            cycles_idle <= 32'h0;
            cycles_fetch_weights <= 32'h0;
            cycles_load_systolic <= 32'h0;
            cycles_wait_propagation <= 32'h0;
            cycles_feed_data <= 32'h0;
            cycles_collect_results <= 32'h0;
            cycles_next_block <= 32'h0;
            cycles_add_bias <= 32'h0;
            cycles_write_outputs <= 32'h0;
            cycles_next_pass <= 32'h0;
            cycles_done <= 32'h0;
            total_layer_cycles <= 32'h0;
            load_weights <= 1'b0;
            data_valid <= 1'b0;
            latch_outputs <= 1'b0;

        end else begin
            // Default: clear one-cycle pulses
            load_weights <= 1'b0;
            data_valid <= 1'b0;
            latch_outputs <= 1'b0;
            output_valid <= 1'b0;
            done <= 1'b0;
            // Phase 6 Layer 2: Clear both BRAM port enables by default (set in FETCH_WEIGHTS REQUEST_DATA state)
            bram_en_a_reg <= 1'b0;
            bram_en_b_reg <= 1'b0;

            // PROFILING: Increment total cycle counter
            total_layer_cycles <= total_layer_cycles + 1;

            case (state)
                IDLE: begin
                    // PROFILING: Increment IDLE cycle counter
                    cycles_idle <= cycles_idle + 1;
                    if (start) begin
                        $display("[SYSTOLIC_LAYER] START: num_neurons=%d, input_size=%d",
                                 num_neurons, input_size);

                        // DEBUG: Snapshot of input_buffer at computation start
                        $display("[DEBUG_COMPUTE_START] input_buffer[0:9]=[%d,%d,%d,%d,%d,%d,%d,%d,%d,%d]",
                                 $signed(input_buffer[0]), $signed(input_buffer[1]),
                                 $signed(input_buffer[2]), $signed(input_buffer[3]),
                                 $signed(input_buffer[4]), $signed(input_buffer[5]),
                                 $signed(input_buffer[6]), $signed(input_buffer[7]),
                                 $signed(input_buffer[8]), $signed(input_buffer[9]));
                        $display("[DEBUG_COMPUTE_START] input_buffer[200:209]=[%d,%d,%d,%d,%d,%d,%d,%d,%d,%d]",
                                 $signed(input_buffer[200]), $signed(input_buffer[201]),
                                 $signed(input_buffer[202]), $signed(input_buffer[203]),
                                 $signed(input_buffer[204]), $signed(input_buffer[205]),
                                 $signed(input_buffer[206]), $signed(input_buffer[207]),
                                 $signed(input_buffer[208]), $signed(input_buffer[209]));

                        // Calculate passes and blocks
                        total_passes <= ((num_neurons + ARRAY_SIZE - 1) >> 3);  // Ceiling divide by 8
                        total_blocks <= ((input_size + ARRAY_SIZE - 1) >> 3);   // Ceiling divide by 8

                        $display("[SYSTOLIC_LAYER] total_passes=%d, total_blocks=%d",
                                 ((num_neurons + 7) >> 3), ((input_size + 7) >> 3));

                        // Initialize counters
                        current_pass <= 8'h0;
                        current_block <= 16'h0;
                        neuron_base <= 8'h0;
                        input_base <= 16'h0;

                        // Clear accumulators and weight sums
                        for (i = 0; i < ARRAY_SIZE; i = i + 1) begin
                            accumulators[i] <= 32'sd0;
                            weight_sums[i] <= 32'sd0;  // ZERO-POINT CORRECTION
                            correction_pipeline[i] <= 32'sd0;  // Phase 7 Opt #5: Clear correction pipeline
                        end

                        // Phase 6 Layer 3 Opt #2: Clear weight buffers to prevent stale data
                        // Hypothesis: Stale data from previous image/pass could contaminate computation
                        for (i = 0; i < 64; i = i + 1) begin
                            weight_buffer_a[i] <= 8'sd0;
                            weight_buffer_b[i] <= 8'sd0;
                        end

                        // Start fetching first 8×8 weight block (Phase 6 Layer 1)
                        weight_fetch_row <= neuron_base;  // Start with first neuron in pass
                        fetch_state <= 2'b00;  // IDLE (will initialize weight_fetch_col)
                        state <= FETCH_WEIGHTS;
                    end
                end

                FETCH_WEIGHTS: begin
                    // ===== Phase 6 Layer 1: Direct BRAM Addressing (4-weights-per-read) =====
                    // PROFILING: Increment FETCH_WEIGHTS cycle counter
                    cycles_fetch_weights <= cycles_fetch_weights + 1;

                    // DEBUG: Verify FETCH_WEIGHTS executes
                    if (num_neurons == 128 && current_block == 0 && current_pass == 0) begin
                        $display("[DEBUG_FETCH] In FETCH_WEIGHTS state: fetch_state=%d Block=%d Pass=%d",
                            fetch_state, current_block, current_pass);
                    end

                    // Fetch weights for current 8×8 block
                    // Block covers neurons [neuron_base : neuron_base+7]
                    // and inputs [input_base : input_base+7]
                    //
                    // KEY OPTIMIZATION: Extract ALL 4 bytes from each 32-bit BRAM word!
                    // - 8×8 block = 64 weights
                    // - 64 weights / 4 per word = 16 words to fetch
                    // - 16 words × 2 cycles (addr + latency) = 32 cycles total
                    // - vs WRR: 18,866 cycles (588× faster!)

                    case (fetch_state)
                        2'b00: begin  // IDLE: Initialize fetch for new row
                            weight_fetch_col <= input_base;  // Start at input_base
                            fetch_state <= 2'b01;  // REQUEST_DATA
                        end

                        2'b01: begin  // REQUEST_DATA: Issue dual-port BRAM read (Phase 6 Layer 2)
                            // Calculate base BRAM byte address: layer_base + neuron * input_size + input_col
                            // Convert to word address: byte_addr >> 2
                            reg [19:0] base_byte_addr;
                            reg [19:0] base_word_addr;

                            base_byte_addr = layer_base_addr +
                                            (weight_fetch_row * input_size) +
                                            weight_fetch_col;
                            base_word_addr = base_byte_addr >> 2;  // Convert to word index

                            // Port A: Even word (weights 0-3)
                            bram_addr_a_reg <= base_word_addr;
                            bram_en_a_reg <= 1'b1;

                            // Port B: Odd word (weights 4-7)
                            bram_addr_b_reg <= base_word_addr + 1;
                            bram_en_b_reg <= 1'b1;

                            fetch_state <= 2'b10;  // WAIT_DATA
                        end

                        2'b10: begin  // WAIT_DATA: Wait for BRAM latency (2 cycles for BOTH ports)
                            bram_en_a_reg <= 1'b0;
                            bram_en_b_reg <= 1'b0;
                            // CRITICAL: BRAM has 1-cycle latency (registered output)
                            // Both Port A and Port B have same latency
                            // Cycle N: REQUEST_DATA sets addr, en=1 for BOTH ports
                            // Cycle N+1: WAIT_DATA first time (wait_extra_cycle=0)
                            // Cycle N+2: WAIT_DATA second time (wait_extra_cycle=1), data ready from BOTH ports
                            if (!wait_extra_cycle) begin
                                wait_extra_cycle <= 1'b1;  // Wait one more cycle
                            end else begin
                                bram_data_a_latched <= bram_data_a_out;  // Latch Port A (weights 0-3)
                                bram_data_b_latched <= bram_data_out;    // Latch Port B (weights 4-7)
                                fetch_state <= 2'b11;  // EXTRACT_8_BYTES
                                wait_extra_cycle <= 1'b0;  // Reset for next read
                            end
                        end

                        2'b11: begin  // EXTRACT_8_BYTES: Extract 4 from Port A + 4 from Port B (Phase 6 Layer 2)
                            // Extract 8 weights from 2×32-bit words:
                            // - Port A: weights 0-3 (weight_fetch_col + 0, +1, +2, +3)
                            // - Port B: weights 4-7 (weight_fetch_col + 4, +5, +6, +7)
                            // - weight_buffer layout: [row*8 + col] = W[neuron_base+row][input_base+col]
                            // - Row index: (weight_fetch_row - neuron_base)
                            // - Col index: (weight_fetch_col + offset - input_base)
                            //
                            // Phase 6 Layer 3: Write to fetch_buffer (not active_buffer!)
                            //   fetch_buffer = 0 → write to weight_buffer_a
                            //   fetch_buffer = 1 → write to weight_buffer_b

                            // ZERO-POINT CORRECTION: Accumulate weight sums for ALL blocks
                            // (Phase 6: must accumulate across all blocks, not just block 0)
                            // DEBUG: Trace weight_sum accumulation in FETCH_WEIGHTS
                            if (current_pass == 0 && (weight_fetch_row - neuron_base) == 0 && weight_fetch_col < 16) begin
                                $display("[FETCH_WSUM] Block=%d Row=%d Col=%d old_sum=%d weights=[%d %d %d %d %d %d %d %d] sum_delta=%d",
                                    current_block, weight_fetch_row - neuron_base, weight_fetch_col,
                                    $signed(weight_sums[weight_fetch_row - neuron_base]),
                                    $signed(bram_data_a_latched[7:0]), $signed(bram_data_a_latched[15:8]),
                                    $signed(bram_data_a_latched[23:16]), $signed(bram_data_a_latched[31:24]),
                                    $signed(bram_data_b_latched[7:0]), $signed(bram_data_b_latched[15:8]),
                                    $signed(bram_data_b_latched[23:16]), $signed(bram_data_b_latched[31:24]),
                                    $signed(bram_data_a_latched[7:0]) + $signed(bram_data_a_latched[15:8]) +
                                    $signed(bram_data_a_latched[23:16]) + $signed(bram_data_a_latched[31:24]) +
                                    $signed(bram_data_b_latched[7:0]) + $signed(bram_data_b_latched[15:8]) +
                                    $signed(bram_data_b_latched[23:16]) + $signed(bram_data_b_latched[31:24]));
                            end
                            weight_sums[weight_fetch_row - neuron_base] <=
                                weight_sums[weight_fetch_row - neuron_base] +
                                // Port A weights (0-3)
                                $signed(bram_data_a_latched[7:0]) +
                                $signed(bram_data_a_latched[15:8]) +
                                $signed(bram_data_a_latched[23:16]) +
                                $signed(bram_data_a_latched[31:24]) +
                                // Port B weights (4-7)
                                $signed(bram_data_b_latched[7:0]) +
                                $signed(bram_data_b_latched[15:8]) +
                                $signed(bram_data_b_latched[23:16]) +
                                $signed(bram_data_b_latched[31:24]);

                            // Extract weights 0-3 from Port A (even word)
                            // Phase 6 Layer 3: Write to buffer based on fetch_buffer flag
                            // Weight 0: Port A bits [7:0]
                            if (weight_fetch_col + 0 - input_base < ARRAY_SIZE) begin
                                if (fetch_buffer == 1'b0) begin
                                    weight_buffer_a[(weight_fetch_row - neuron_base) * ARRAY_SIZE +
                                                   (weight_fetch_col + 0 - input_base)] <= $signed(bram_data_a_latched[7:0]);
                                end else begin
                                    weight_buffer_b[(weight_fetch_row - neuron_base) * ARRAY_SIZE +
                                                   (weight_fetch_col + 0 - input_base)] <= $signed(bram_data_a_latched[7:0]);
                                end
                            end

                            // Weight 1: Port A bits [15:8]
                            if (weight_fetch_col + 1 - input_base < ARRAY_SIZE) begin
                                if (fetch_buffer == 1'b0) begin
                                    weight_buffer_a[(weight_fetch_row - neuron_base) * ARRAY_SIZE +
                                                   (weight_fetch_col + 1 - input_base)] <= $signed(bram_data_a_latched[15:8]);
                                end else begin
                                    weight_buffer_b[(weight_fetch_row - neuron_base) * ARRAY_SIZE +
                                                   (weight_fetch_col + 1 - input_base)] <= $signed(bram_data_a_latched[15:8]);
                                end
                            end

                            // Weight 2: Port A bits [23:16]
                            if (weight_fetch_col + 2 - input_base < ARRAY_SIZE) begin
                                if (fetch_buffer == 1'b0) begin
                                    weight_buffer_a[(weight_fetch_row - neuron_base) * ARRAY_SIZE +
                                                   (weight_fetch_col + 2 - input_base)] <= $signed(bram_data_a_latched[23:16]);
                                end else begin
                                    weight_buffer_b[(weight_fetch_row - neuron_base) * ARRAY_SIZE +
                                                   (weight_fetch_col + 2 - input_base)] <= $signed(bram_data_a_latched[23:16]);
                                end
                            end

                            // Weight 3: Port A bits [31:24]
                            if (weight_fetch_col + 3 - input_base < ARRAY_SIZE) begin
                                if (fetch_buffer == 1'b0) begin
                                    weight_buffer_a[(weight_fetch_row - neuron_base) * ARRAY_SIZE +
                                                   (weight_fetch_col + 3 - input_base)] <= $signed(bram_data_a_latched[31:24]);
                                end else begin
                                    weight_buffer_b[(weight_fetch_row - neuron_base) * ARRAY_SIZE +
                                                   (weight_fetch_col + 3 - input_base)] <= $signed(bram_data_a_latched[31:24]);
                                end
                            end

                            // Extract weights 4-7 from Port B (odd word)
                            // Weight 4: Port B bits [7:0]
                            if (weight_fetch_col + 4 - input_base < ARRAY_SIZE) begin
                                if (fetch_buffer == 1'b0) begin
                                    weight_buffer_a[(weight_fetch_row - neuron_base) * ARRAY_SIZE +
                                                   (weight_fetch_col + 4 - input_base)] <= $signed(bram_data_b_latched[7:0]);
                                end else begin
                                    weight_buffer_b[(weight_fetch_row - neuron_base) * ARRAY_SIZE +
                                                   (weight_fetch_col + 4 - input_base)] <= $signed(bram_data_b_latched[7:0]);
                                end
                            end

                            // Weight 5: Port B bits [15:8]
                            if (weight_fetch_col + 5 - input_base < ARRAY_SIZE) begin
                                if (fetch_buffer == 1'b0) begin
                                    weight_buffer_a[(weight_fetch_row - neuron_base) * ARRAY_SIZE +
                                                   (weight_fetch_col + 5 - input_base)] <= $signed(bram_data_b_latched[15:8]);
                                end else begin
                                    weight_buffer_b[(weight_fetch_row - neuron_base) * ARRAY_SIZE +
                                                   (weight_fetch_col + 5 - input_base)] <= $signed(bram_data_b_latched[15:8]);
                                end
                            end

                            // Weight 6: Port B bits [23:16]
                            if (weight_fetch_col + 6 - input_base < ARRAY_SIZE) begin
                                if (fetch_buffer == 1'b0) begin
                                    weight_buffer_a[(weight_fetch_row - neuron_base) * ARRAY_SIZE +
                                                   (weight_fetch_col + 6 - input_base)] <= $signed(bram_data_b_latched[23:16]);
                                end else begin
                                    weight_buffer_b[(weight_fetch_row - neuron_base) * ARRAY_SIZE +
                                                   (weight_fetch_col + 6 - input_base)] <= $signed(bram_data_b_latched[23:16]);
                                end
                            end

                            // Weight 7: Port B bits [31:24]
                            if (weight_fetch_col + 7 - input_base < ARRAY_SIZE) begin
                                if (fetch_buffer == 1'b0) begin
                                    weight_buffer_a[(weight_fetch_row - neuron_base) * ARRAY_SIZE +
                                                   (weight_fetch_col + 7 - input_base)] <= $signed(bram_data_b_latched[31:24]);
                                end else begin
                                    weight_buffer_b[(weight_fetch_row - neuron_base) * ARRAY_SIZE +
                                                   (weight_fetch_col + 7 - input_base)] <= $signed(bram_data_b_latched[31:24]);
                                end
                            end

                            // Move to next dual-read (8 weights at a time)
                            weight_fetch_col <= weight_fetch_col + 8;

                            // Check if current row complete (fetched 8 weights = 1 dual-read for 8×8 array)
                            if (weight_fetch_col + 8 >= input_base + ARRAY_SIZE) begin
                                // Row complete, move to next neuron row
                                if (weight_fetch_row - neuron_base < ARRAY_SIZE - 1 &&
                                    weight_fetch_row + 1 < num_neurons) begin
                                    // Fetch next row
                                    weight_fetch_row <= weight_fetch_row + 1;
                                    fetch_state <= 2'b00;  // IDLE (reinitialize for new row)
                                end else begin
                                    // All 8 rows fetched, load into systolic array
                                    $display("[SYSTOLIC_LAYER] Fetched 8×8 block: neurons[%d:%d], inputs[%d:%d] (Phase 6 Layer 2: 8-weights-per-dual-read)",
                                             neuron_base, neuron_base+ARRAY_SIZE-1, input_base, input_base+ARRAY_SIZE-1);

                                    // DEBUG: Dump first block's weights for verification
                                    // Phase 6 Layer 3: Dump the buffer that was just fetched
                                    if (neuron_base == 0 && input_base == 0) begin
                                        $display("[WEIGHT_BUFFER_DUMP] Block[0,0] - First 8×8 block (fetch_buffer=%d):", fetch_buffer);
                                        if (fetch_buffer == 1'b0) begin
                                            $display("  Buffer A:");
                                            $display("  Row 0: %d %d %d %d %d %d %d %d",
                                                     $signed(weight_buffer_a[0]), $signed(weight_buffer_a[1]), $signed(weight_buffer_a[2]), $signed(weight_buffer_a[3]),
                                                     $signed(weight_buffer_a[4]), $signed(weight_buffer_a[5]), $signed(weight_buffer_a[6]), $signed(weight_buffer_a[7]));
                                            $display("  Row 1: %d %d %d %d %d %d %d %d",
                                                     $signed(weight_buffer_a[8]), $signed(weight_buffer_a[9]), $signed(weight_buffer_a[10]), $signed(weight_buffer_a[11]),
                                                     $signed(weight_buffer_a[12]), $signed(weight_buffer_a[13]), $signed(weight_buffer_a[14]), $signed(weight_buffer_a[15]));
                                        end else begin
                                            $display("  Buffer B:");
                                            $display("  Row 0: %d %d %d %d %d %d %d %d",
                                                     $signed(weight_buffer_b[0]), $signed(weight_buffer_b[1]), $signed(weight_buffer_b[2]), $signed(weight_buffer_b[3]),
                                                     $signed(weight_buffer_b[4]), $signed(weight_buffer_b[5]), $signed(weight_buffer_b[6]), $signed(weight_buffer_b[7]));
                                            $display("  Row 1: %d %d %d %d %d %d %d %d",
                                                     $signed(weight_buffer_b[8]), $signed(weight_buffer_b[9]), $signed(weight_buffer_b[10]), $signed(weight_buffer_b[11]),
                                                     $signed(weight_buffer_b[12]), $signed(weight_buffer_b[13]), $signed(weight_buffer_b[14]), $signed(weight_buffer_b[15]));
                                        end
                                        $display("  Expected (from mnist_raw_weights.h neuron 0, inputs 0-7):");
                                        $display("    Row 0: -9 21 4 -23 23 16 24 19");
                                    end

                                    // Phase 6 Layer 3 Opt #2: Swap buffers before LOAD
                                    // FETCH wrote to fetch_buffer, now make it active
                                    if (current_pass == 0 && current_block < 3) begin
                                        $display("[BUFFER_SWAP] Block=%d Before swap: active=%d fetch=%d", current_block, active_buffer, fetch_buffer);
                                    end
                                    active_buffer <= fetch_buffer;
                                    fetch_buffer <= ~fetch_buffer;
                                    if (current_pass == 0 && current_block < 3) begin
                                        $display("[BUFFER_SWAP] Block=%d After swap: active=%d fetch=%d", current_block, fetch_buffer, ~fetch_buffer);
                                    end

                                    weight_col_idx <= 4'h0;
                                    state <= LOAD_SYSTOLIC_COL;
                                end
                            end else begin
                                // More weights in this row, request next word
                                fetch_state <= 2'b01;  // REQUEST_DATA
                            end
                        end
                    endcase
                end

                LOAD_SYSTOLIC_COL: begin
                    // PROFILING: Increment LOAD_SYSTOLIC_COL cycle counter
                    cycles_load_systolic <= cycles_load_systolic + 1;

                    // Phase 6 Layer 3 Opt #3: Full pipeline - Start prefetch EARLY (during LOAD)
                    // Trigger on first column load if more blocks remain
                    if (weight_col_idx == 0 && current_block + 1 < total_blocks) begin
                        fetch_next_block <= 1'b1;
                        // Set up next block's fetch parameters (Block N+1)
                        prefetch_row <= neuron_base;  // Same neurons, next input block
                        // prefetch_col initialized in prefetch FSM to input_base + ARRAY_SIZE
                        prefetch_state <= 2'b00;  // Start prefetch FSM
                        if (current_pass < 3 && current_block < 2) begin
                            $display("[PREFETCH] Triggering prefetch: Pass=%d Block=%d neuron_base=%d input_base=%d",
                                    current_pass, current_block + 1, neuron_base, input_base + ARRAY_SIZE);
                        end
                        $display("[PREFETCH] Opt #3: Triggering prefetch for Block N+1 during LOAD_SYSTOLIC_COL (full pipeline)");
                    end

                    // Load one column of 8×8 weight block into systolic array
                    // LESSON FROM LAYER 2: Preload column-by-column

                    load_weights <= 1'b1;

                    // Phase 6 Layer 3: Read from active_buffer (NOT fetch_buffer!)
                    // While systolic array computes using active_buffer,
                    // we can simultaneously fetch into fetch_buffer
                    //
                    // Pack column from weight_buffer WITH TRANSPOSE
                    // Layout: weight_buffer[row*8 + col] = W[neuron_base+row][input_base+col]
                    //
                    // CRITICAL FIX (2025-10-20): MUST TRANSPOSE for matrix-vector multiply!
                    //
                    // Systolic array computes: partial_sum_out[col] = W[:,col] · x (column-wise)
                    // Matrix-vector needs: y[row] = W[row,:] · x (row-wise)
                    // Solution: Load W^T so column i contains original row i
                    //
                    // WITHOUT transpose: weight_col_in[i] = weight_buffer[i * 8 + weight_col_idx]
                    //   Loads weight_buffer[0, 8, 16, ...] = W[0,1,2,...][weight_col_idx] (COLUMN)
                    // WITH transpose: weight_col_in[i] = weight_buffer[weight_col_idx * 8 + i]
                    //   Loads weight_buffer[0, 1, 2, ...] = W[weight_col_idx][0,1,2,...] (ROW)
                    //
                    // Unit tests do this in software: weight_col[row] = W[col][row]
                    // Full system must do it in hardware:
                    for (i = 0; i < ARRAY_SIZE; i = i + 1) begin
                        if (active_buffer == 1'b0) begin
                            weight_col_in[i] <= weight_buffer_a[weight_col_idx * 8 + i];  // TRANSPOSE from Buffer A
                        end else begin
                            weight_col_in[i] <= weight_buffer_b[weight_col_idx * 8 + i];  // TRANSPOSE from Buffer B
                        end
                    end

                    // DEBUG: Buffer usage during first few blocks
                    if (current_pass == 0 && current_block < 3 && weight_col_idx == 0) begin
                        $display("[BUFFER_DEBUG] Block=%d Pass=%d Col=%d active_buffer=%d fetch_buffer=%d",
                                current_block, current_pass, weight_col_idx, active_buffer, fetch_buffer);
                        if (active_buffer == 1'b0) begin
                            $display("[BUFFER_DEBUG] Loading from Buffer A: W=[%d,%d,%d,%d,%d,%d,%d,%d]",
                                    $signed(weight_buffer_a[0]), $signed(weight_buffer_a[1]),
                                    $signed(weight_buffer_a[2]), $signed(weight_buffer_a[3]),
                                    $signed(weight_buffer_a[4]), $signed(weight_buffer_a[5]),
                                    $signed(weight_buffer_a[6]), $signed(weight_buffer_a[7]));
                        end else begin
                            $display("[BUFFER_DEBUG] Loading from Buffer B: W=[%d,%d,%d,%d,%d,%d,%d,%d]",
                                    $signed(weight_buffer_b[0]), $signed(weight_buffer_b[1]),
                                    $signed(weight_buffer_b[2]), $signed(weight_buffer_b[3]),
                                    $signed(weight_buffer_b[4]), $signed(weight_buffer_b[5]),
                                    $signed(weight_buffer_b[6]), $signed(weight_buffer_b[7]));
                        end
                    end

                    // DEBUG DISABLED for performance: Load column (8 per block)
                    // $display("[DEBUG] Load column %d: W=[%d,%d,%d,%d,%d,%d,%d,%d]",
                    //         weight_col_idx,
                    //         $signed(weight_buffer[0*8 + weight_col_idx]),
                    //         $signed(weight_buffer[1*8 + weight_col_idx]),
                    //         $signed(weight_buffer[2*8 + weight_col_idx]),
                    //         $signed(weight_buffer[3*8 + weight_col_idx]),
                    //         $signed(weight_buffer[4*8 + weight_col_idx]),
                    //         $signed(weight_buffer[5*8 + weight_col_idx]),
                    //         $signed(weight_buffer[6*8 + weight_col_idx]),
                    //         $signed(weight_buffer[7*8 + weight_col_idx]));

                    if (weight_col_idx == ARRAY_SIZE - 1) begin
                        // All 8 columns loaded
                        weight_col_idx <= 4'h0;
                        state <= WAIT_WEIGHT_LOAD;
                    end else begin
                        weight_col_idx <= weight_col_idx + 1;
                    end
                end

                WAIT_WEIGHT_LOAD: begin
                    // Wait one cycle for weights to latch
                    data_row_idx <= 4'h0;

                    // Pre-load data_row_in with FULL vector (matches test_layer_simple.cpp)
                    // Feed the SAME vector 8 times (each row gets element i due to wavefront delays)
                    for (i = 0; i < ARRAY_SIZE; i = i + 1) begin
                        data_row_in[i] <= input_buffer[input_base + i];
                    end

                    state <= FEED_DATA;
                end

                FEED_DATA: begin
                    // PROFILING: Increment FEED_DATA cycle counter
                    cycles_feed_data <= cycles_feed_data + 1;

                    // FULL VECTOR FEEDING (for matrix-matrix multiply)
                    //
                    // Feed complete input vector each cycle for ARRAY_SIZE cycles
                    // Hardware delays create wavefront:
                    //   - Row 0 sees elements immediately
                    //   - Row i sees elements delayed by i cycles
                    //   - Creates proper temporal alignment for matrix multiply
                    //
                    // Works for matrices with DIFFERENT weight rows (normal case)
                    // NOTE: Does NOT work for identical weight rows (vertical accumulation issue)
                    //       See FINAL_ANALYSIS_IDENTICAL_ROWS.md
                    //
                    // Matches test_layer_simple.cpp approach that passes all Layer 2 tests

                    // Assert data_valid
                    data_valid <= 1'b1;

                    // data_row_in already pre-loaded in WAIT_WEIGHT_LOAD (same vector feeds for all 8 cycles)
                    // This matches test_layer_simple.cpp approach that works in Layer 2

                    // DEBUG: Show what's ACTUALLY in data_row_in (blocks 24-26, cycle 0 only)
                    if (current_pass == 0 && current_block >= 24 && current_block < 27 && data_row_idx == 0) begin
                        $display("[DEBUG_FEED_ACTUAL] block=%d cycle=%d base=%d: data_row_in=[%d,%d,%d,%d,%d,%d,%d,%d]",
                                current_block, data_row_idx, input_base,
                                $signed(data_row_in[0]),
                                $signed(data_row_in[1]),
                                $signed(data_row_in[2]),
                                $signed(data_row_in[3]),
                                $signed(data_row_in[4]),
                                $signed(data_row_in[5]),
                                $signed(data_row_in[6]),
                                $signed(data_row_in[7]));
                    end

                    if (data_row_idx == ARRAY_SIZE - 1) begin
                        // All 8 rows fed
                        data_row_idx <= 4'h0;
                        collect_cycle <= 5'h0;
                        wait_cycles <= 4'h0;
                        state <= WAIT_FOR_PROPAGATION;
                    end else begin
                        data_row_idx <= data_row_idx + 1;
                    end
                    // PROFILING: Increment WAIT_FOR_PROPAGATION cycle counter
                    cycles_wait_propagation <= cycles_wait_propagation + 1;

                end

                WAIT_FOR_PROPAGATION: begin
                    // Wait for results to propagate through systolic array
                    // Need ARRAY_SIZE-1 cycles for data to reach bottom row
                    if (wait_cycles == ARRAY_SIZE - 1) begin
                        state <= COLLECT_RESULTS;
                    end else begin
                        wait_cycles <= wait_cycles + 1;
                    end

                end

                COLLECT_RESULTS: begin
                    // Phase 6 Layer 3 Opt #3: Prefetch now triggered in LOAD_SYSTOLIC_COL (full pipeline)
                    // Old Opt #2 trigger removed (was: trigger at collect_cycle == 0)

                    // CRITICAL FIX: Use 15-cycle diagonal streaming (2*ARRAY_SIZE-1)
                    // Based on test_layer_simple.cpp reference implementation
                    //
                    // Each neuron's result emerges diagonally across columns:
                    // Neuron n collects from columns 0-7 at cycles n, n+1, ..., n+7
                    // Formula: neuron_id = cycle - column
                    //
                    // For neuron 0: collect at cycles 0-7 from columns 0-7
                    // For neuron 1: collect at cycles 1-8 from columns 0-7
                    // ...
                    // For neuron 7: collect at cycles 7-14 from columns 0-7

                    // DEBUG DISABLED for performance: Collect cycle (15 per block)
                    // $display("[COLLECT] Cycle %2d: partial_sum_out = [%d,%d,%d,%d,%d,%d,%d,%d]",
                    //          collect_cycle,
                    //          $signed(partial_sum_out[0]),
                    //          $signed(partial_sum_out[1]),
                    //          $signed(partial_sum_out[2]),
                    //          $signed(partial_sum_out[3]),
                    //          $signed(partial_sum_out[4]),
                    //          $signed(partial_sum_out[5]),
                    //          $signed(partial_sum_out[6]),
                    //          $signed(partial_sum_out[7]));

                    // Diagonal collection for ROW 0 only (matrix-vector multiply)
                    // When feeding same input vector to all rows:
                    // - All output ROWS are identical = result vector y
                    // - Collect ROW 0: at cycle c, partial_sum_out[c] = y[c] (FULL dot product)
                    // - Formula: row = cycle - col, for row 0: cycle = col
                    for (i = 0; i < ARRAY_SIZE; i = i + 1) begin
                        // Collect from column i when cycle == i (row 0 diagonal)
                        if (collect_cycle == i) begin
                            // CRITICAL FIX (Hour 18): ACCUMULATE across blocks! Each block processes 8 inputs.
                            // For 784 inputs, we process 98 blocks and must ADD results from each block.
                            accumulators[i] <= accumulators[i] + partial_sum_out[i];

                            // DEBUG DISABLED for performance: Collected neuron (8 per block)
                            // $display("[SYSTOLIC_LAYER] Cycle %d: Collected neuron %d = %d",
                            //          collect_cycle, i, $signed(partial_sum_out[i]));
                        end
                    end

                    // Phase 6 Layer 3 Opt #1: Fast collection for matrix-vector multiply
                    // For matrix-vector: results emerge at cycles 0-7 (ARRAY_SIZE cycles)
                    // Old (matrix-matrix): 2*ARRAY_SIZE - 1 = 15 cycles (diagonal collection)
                    // New (matrix-vector): ARRAY_SIZE - 1 = 7 cycles (row 0 collection)
                    // Savings: 7 cycles/block × 1,712 blocks = 11,984 cycles (8.4% improvement)
                    if (collect_cycle == ARRAY_SIZE - 1) begin
                        // All partial sums collected (8 cycles for matrix-vector)
                        $display("[SYSTOLIC_LAYER] Block complete: pass=%d, block=%d",
                                 current_pass, current_block);

                        // DEBUG: Layer 2 first pass (neurons 0-7) block-by-block analysis
                        if (current_pass == 0 && num_neurons == 64 && input_size == 128) begin
                            $display("[LAYER2_DEBUG] Pass 0, Block %2d:", current_block);
                            $display("  Input[%3d:%3d] = [%4d,%4d,%4d,%4d,%4d,%4d,%4d,%4d]",
                                    current_block * ARRAY_SIZE,
                                    current_block * ARRAY_SIZE + 7,
                                    $signed(input_buffer[current_block * ARRAY_SIZE + 0]),
                                    $signed(input_buffer[current_block * ARRAY_SIZE + 1]),
                                    $signed(input_buffer[current_block * ARRAY_SIZE + 2]),
                                    $signed(input_buffer[current_block * ARRAY_SIZE + 3]),
                                    $signed(input_buffer[current_block * ARRAY_SIZE + 4]),
                                    $signed(input_buffer[current_block * ARRAY_SIZE + 5]),
                                    $signed(input_buffer[current_block * ARRAY_SIZE + 6]),
                                    $signed(input_buffer[current_block * ARRAY_SIZE + 7]));
                            $display("  Accum[0]=%d (neuron 0 running total after this block)",
                                    $signed(accumulators[0]));
                        end

                        // Phase 7 Opt #4: NEXT_BLOCK logic merged into final collection cycle
                        // Check if prefetch is complete before proceeding
                        if (!fetch_next_block) begin
                            // No prefetch running OR prefetch already complete
                            // Inline NEXT_BLOCK logic: Check if more blocks to process
                            if (current_block + 1 < total_blocks) begin
                                // More input blocks to process
                                current_block <= current_block + 1;
                                input_base <= input_base + ARRAY_SIZE;
                                weight_fetch_row <= neuron_base;
                                fetch_state <= 2'b00;  // IDLE (reinitialize for next prefetch)

                                // Swap buffers: prefetched data becomes active
                                if (current_pass == 0 && current_block < 2) begin
                                    $display("[OPT4_SWAP] Block=%d Before swap: active=%d fetch=%d", current_block + 1, active_buffer, fetch_buffer);
                                end
                                active_buffer <= fetch_buffer;
                                fetch_buffer <= ~fetch_buffer;
                                if (current_pass == 0 && current_block < 2) begin
                                    $display("[OPT4_SWAP] Block=%d After swap: active=%d fetch=%d", current_block + 1, fetch_buffer, ~fetch_buffer);
                                end
                                state <= LOAD_SYSTOLIC_COL;  // Direct transition (skip FETCH_WEIGHTS!)
                                $display("[OPT4] Direct transition COLLECT→LOAD, skipping NEXT_BLOCK state");
                            end else begin
                                // All blocks processed for this pass, add bias
                                // Phase 7 Opt #5: Pre-compute correction pipeline for all 8 neurons
                                // This computation will be used in WRITE_OUTPUTS, eliminating multi-cycle multiply latency
                                for (i = 0; i < ARRAY_SIZE; i = i + 1) begin
                                    correction_pipeline[i] <= INPUT_ZERO_POINT * weight_sums[i];
                                end
                                state <= ADD_BIAS;
                                $display("[OPT5] Last block, pre-computed corrections, transition COLLECT→ADD_BIAS");
                            end
                        end else begin
                            // Prefetch still running (rare case: fetch outlasted collection)
                            // Wait for prefetch to complete
                            state <= WAIT_FETCH_COMPLETE;
                            $display("[PREFETCH] Collection done but fetch still running, waiting...");
                        end
                    end else begin
                        collect_cycle <= collect_cycle + 1;
                    end
                    // PROFILING: Increment COLLECT_RESULTS cycle counter
                    cycles_collect_results <= cycles_collect_results + 1;

                end

                // Phase 7 Opt #4: NEXT_BLOCK state ELIMINATED!
                // Logic merged into COLLECT_RESULTS final cycle (lines 871-895)

                ADD_BIAS: begin
                    // PROFILING: Increment ADD_BIAS cycle counter
                    cycles_add_bias <= cycles_add_bias + 1;

                    // Add bias to accumulated results for 8 neurons
                    // (Biases already in bias_buffer, added during output write)
                    state <= WRITE_OUTPUTS;
                end

                WRITE_OUTPUTS: begin
                    // PROFILING: Increment WRITE_OUTPUTS cycle counter
                    cycles_write_outputs <= cycles_write_outputs + 1;
                    // Write outputs one at a time (sequential, not combinational loop)
                    if (neuron_base + output_write_idx < num_neurons) begin
                        // ZERO-POINT CORRECTION: Apply correction before adding bias
                        // Formula: result = bias + Σ(W*x) - input_zero_point * Σ(W)
                        // Correction term: INPUT_ZERO_POINT * weight_sums[i]
                        // For MNIST: INPUT_ZERO_POINT = -128, so subtracting (-128)*sum adds 128*sum

                        // Phase 7 Opt #5: Use pre-computed correction from pipeline
                        // Correction was computed during COLLECT_RESULTS (last block), eliminating multiply latency here
                        corrected_accum = accumulators[output_write_idx] - correction_pipeline[output_write_idx];

                        // DEBUG: Keep correction variable for logging (will be optimized away in synthesis)
                        correction = correction_pipeline[output_write_idx];

                        // Write output for current neuron
                        output_addr <= neuron_base + output_write_idx;
                        output_data <= corrected_accum + bias_buffer[neuron_base + output_write_idx];
                        output_valid <= 1'b1;

                        // DEBUG: Log ALL neuron outputs for golden model comparison
                        $display("[NEURON_OUTPUT] neuron=%3d accum=%8d weight_sum=%8d correction=%8d corrected=%8d bias=%8d result=%8d",
                                 neuron_base + output_write_idx,
                                 $signed(accumulators[output_write_idx]),
                                 $signed(weight_sums[output_write_idx]),
                                 $signed(correction),
                                 $signed(corrected_accum),
                                 $signed(bias_buffer[neuron_base + output_write_idx]),
                                 $signed(corrected_accum + bias_buffer[neuron_base + output_write_idx]));
                    end

                    // Move to next output or finish
                    if (output_write_idx == ARRAY_SIZE - 1) begin
                        output_write_idx <= 3'h0;
                        state <= NEXT_PASS;
                    end else begin
                        output_write_idx <= output_write_idx + 1;
                        state <= WRITE_OUTPUTS;  // Stay in WRITE_OUTPUTS for next neuron
                    end
                end

                NEXT_PASS: begin
                    // PROFILING: Increment NEXT_PASS cycle counter
                    cycles_next_pass <= cycles_next_pass + 1;

                    // Check if more passes needed
                    if (current_pass + 1 < total_passes) begin
                        // Debug buffer state during pass transition
                        if (current_pass < 3) begin
                            $display("[NEXT_PASS] Pass %d→%d: BEFORE active=%d fetch=%d",
                                    current_pass, current_pass + 1, active_buffer, fetch_buffer);
                        end

                        // More neuron passes needed
                        current_pass <= current_pass + 1;
                        current_block <= 16'h0;
                        neuron_base <= neuron_base + ARRAY_SIZE;
                        input_base <= 16'h0;

                        // Clear accumulators and weight sums for next pass
                        for (i = 0; i < ARRAY_SIZE; i = i + 1) begin
                            accumulators[i] <= 32'sd0;
                            weight_sums[i] <= 32'sd0;  // ZERO-POINT CORRECTION
                            correction_pipeline[i] <= 32'sd0;  // Phase 7 Opt #5: Clear correction pipeline
                        end

                        // Phase 6 Layer 3 Opt #2: Clear weight buffers between passes
                        // Prevents stale data from previous pass contaminating new pass
                        for (i = 0; i < 64; i = i + 1) begin
                            weight_buffer_a[i] <= 8'sd0;
                            weight_buffer_b[i] <= 8'sd0;
                        end

                        weight_fetch_row <= neuron_base + ARRAY_SIZE;
                        fetch_state <= 2'b00;  // IDLE (reinitialize fetch for new pass)
                        state <= FETCH_WEIGHTS;
                    end else begin
                        // All passes complete
                        $display("[SYSTOLIC_LAYER] COMPLETE: All %d neurons computed", num_neurons);

                        // PROFILING: Display cycle breakdown
                        $display("=== PROFILING RESULTS ===");
                        $display("Total cycles: %d", total_layer_cycles);
                        $display("  IDLE:                 %6d cycles (%2d%%)", cycles_idle, (cycles_idle * 100) / (total_layer_cycles + 1));
                        $display("  FETCH_WEIGHTS:        %6d cycles (%2d%%)", cycles_fetch_weights, (cycles_fetch_weights * 100) / (total_layer_cycles + 1));
                        $display("  LOAD_SYSTOLIC:        %6d cycles (%2d%%)", cycles_load_systolic, (cycles_load_systolic * 100) / (total_layer_cycles + 1));
                        $display("  WAIT_PROPAGATION:     %6d cycles (%2d%%)", cycles_wait_propagation, (cycles_wait_propagation * 100) / (total_layer_cycles + 1));
                        $display("  FEED_DATA:            %6d cycles (%2d%%)", cycles_feed_data, (cycles_feed_data * 100) / (total_layer_cycles + 1));
                        $display("  COLLECT_RESULTS:      %6d cycles (%2d%%)", cycles_collect_results, (cycles_collect_results * 100) / (total_layer_cycles + 1));
                        $display("  NEXT_BLOCK:           %6d cycles (%2d%%)", cycles_next_block, (cycles_next_block * 100) / (total_layer_cycles + 1));
                        $display("  ADD_BIAS:             %6d cycles (%2d%%)", cycles_add_bias, (cycles_add_bias * 100) / (total_layer_cycles + 1));
                        $display("  WRITE_OUTPUTS:        %6d cycles (%2d%%)", cycles_write_outputs, (cycles_write_outputs * 100) / (total_layer_cycles + 1));
                        $display("  NEXT_PASS:            %6d cycles (%2d%%)", cycles_next_pass, (cycles_next_pass * 100) / (total_layer_cycles + 1));
                        $display("  DONE:                 %6d cycles (%2d%%)", cycles_done, (cycles_done * 100) / (total_layer_cycles + 1));
                        $display("========================");

                    // PROFILING: Increment DONE cycle counter
                    cycles_done <= cycles_done + 1;

                        done <= 1'b1;
                        state <= DONE;
                    end
                end

                DONE: begin
                    done <= 1'b1;
                    if (!start) begin
                        state <= IDLE;
                    end
                end

                WAIT_FETCH_COMPLETE: begin
                    // Phase 7 Opt #4: Wait for parallel prefetch to finish
                    // Rare case: collection finished in 8 cycles but fetch still running
                    if (!fetch_next_block) begin
                        // Prefetch complete, inline NEXT_BLOCK logic
                        if (current_block + 1 < total_blocks) begin
                            // More input blocks to process
                            current_block <= current_block + 1;
                            input_base <= input_base + ARRAY_SIZE;
                            weight_fetch_row <= neuron_base;
                            fetch_state <= 2'b00;

                            // Swap buffers
                            active_buffer <= fetch_buffer;
                            fetch_buffer <= ~fetch_buffer;
                            state <= LOAD_SYSTOLIC_COL;
                            $display("[OPT4_WAIT] Fetch complete, direct transition WAIT→LOAD");
                        end else begin
                            // All blocks processed
                            // Phase 7 Opt #5: Pre-compute correction pipeline for all 8 neurons
                            for (i = 0; i < ARRAY_SIZE; i = i + 1) begin
                                correction_pipeline[i] <= INPUT_ZERO_POINT * weight_sums[i];
                            end
                            state <= ADD_BIAS;
                            $display("[OPT5_WAIT] Fetch complete, last block, pre-computed corrections, transition WAIT→ADD_BIAS");
                        end
                    end
                    // else stay in this state
                end

                default: begin
                    state <= IDLE;
                end
            endcase

            // ===== Phase 6 Layer 3 Opt #2: Parallel Prefetch FSM =====
            // Runs in PARALLEL with main FSM when fetch_next_block==1
            // Fetches Block N+1 while COLLECT_RESULTS processes Block N
            if (fetch_next_block) begin
                case (prefetch_state)
                    2'b00: begin  // IDLE: Initialize prefetch for new row
                        prefetch_col <= input_base + ARRAY_SIZE;  // Next block's input_base
                        prefetch_state <= 2'b01;  // REQUEST_DATA
                    end

                    2'b01: begin  // REQUEST_DATA: Issue dual-port BRAM read
                        reg [19:0] base_byte_addr_pf;
                        reg [19:0] base_word_addr_pf;

                        base_byte_addr_pf = layer_base_addr +
                                           (prefetch_row * input_size) +
                                           prefetch_col;
                        base_word_addr_pf = base_byte_addr_pf >> 2;

                        // Port A: Even word (weights 0-3)
                        bram_addr_a_reg <= base_word_addr_pf;
                        bram_en_a_reg <= 1'b1;

                        // Port B: Odd word (weights 4-7)
                        bram_addr_b_reg <= base_word_addr_pf + 1;
                        bram_en_b_reg <= 1'b1;

                        prefetch_state <= 2'b10;  // WAIT
                        prefetch_wait_cycle <= 1'b0;
                    end

                    2'b10: begin  // WAIT: Wait for BRAM registered read latency
                        bram_en_a_reg <= 1'b0;
                        bram_en_b_reg <= 1'b0;

                        if (!prefetch_wait_cycle) begin
                            prefetch_wait_cycle <= 1'b1;
                        end else begin
                            // Latch data after 2-cycle wait
                            bram_data_a_latched <= bram_data_a_out;
                            bram_data_b_latched <= bram_data_out;
                            prefetch_state <= 2'b11;  // EXTRACT
                        end
                    end

                    2'b11: begin  // EXTRACT: Extract 8 bytes and write to buffer
                        // ZERO-POINT CORRECTION: Accumulate weight sums
                        // DEBUG: Trace weight_sum accumulation in PREFETCH
                        if (current_pass == 0 && (prefetch_row - neuron_base) == 0 && prefetch_col < input_base + ARRAY_SIZE + 16) begin
                            $display("[PREFETCH_WSUM] Block=%d Row=%d Col=%d old_sum=%d weights=[%d %d %d %d %d %d %d %d] sum_delta=%d",
                                current_block + 1, prefetch_row - neuron_base, prefetch_col,
                                $signed(weight_sums[prefetch_row - neuron_base]),
                                $signed(bram_data_a_latched[7:0]), $signed(bram_data_a_latched[15:8]),
                                $signed(bram_data_a_latched[23:16]), $signed(bram_data_a_latched[31:24]),
                                $signed(bram_data_b_latched[7:0]), $signed(bram_data_b_latched[15:8]),
                                $signed(bram_data_b_latched[23:16]), $signed(bram_data_b_latched[31:24]),
                                $signed(bram_data_a_latched[7:0]) + $signed(bram_data_a_latched[15:8]) +
                                $signed(bram_data_a_latched[23:16]) + $signed(bram_data_a_latched[31:24]) +
                                $signed(bram_data_b_latched[7:0]) + $signed(bram_data_b_latched[15:8]) +
                                $signed(bram_data_b_latched[23:16]) + $signed(bram_data_b_latched[31:24]));
                        end
                        weight_sums[prefetch_row - neuron_base] <=
                            weight_sums[prefetch_row - neuron_base] +
                            // Port A weights (0-3)
                            $signed(bram_data_a_latched[7:0]) +
                            $signed(bram_data_a_latched[15:8]) +
                            $signed(bram_data_a_latched[23:16]) +
                            $signed(bram_data_a_latched[31:24]) +
                            // Port B weights (4-7)
                            $signed(bram_data_b_latched[7:0]) +
                            $signed(bram_data_b_latched[15:8]) +
                            $signed(bram_data_b_latched[23:16]) +
                            $signed(bram_data_b_latched[31:24]);

                        // DEBUG: Prefetch buffer writes for first block
                        if (current_pass == 0 && current_block == 0 && prefetch_row == neuron_base && prefetch_col == input_base + ARRAY_SIZE) begin
                            $display("[PREFETCH_WRITE] Pass=%d Block=%d Row=%d Col=%d input_base=%d fetch_buffer=%d",
                                    current_pass, current_block + 1, prefetch_row - neuron_base, prefetch_col,
                                    input_base, fetch_buffer);
                            $display("[PREFETCH_WRITE] Weights: [%d,%d,%d,%d,%d,%d,%d,%d]",
                                    $signed(bram_data_a_latched[7:0]), $signed(bram_data_a_latched[15:8]),
                                    $signed(bram_data_a_latched[23:16]), $signed(bram_data_a_latched[31:24]),
                                    $signed(bram_data_b_latched[7:0]), $signed(bram_data_b_latched[15:8]),
                                    $signed(bram_data_b_latched[23:16]), $signed(bram_data_b_latched[31:24]));
                            $display("[PREFETCH_WRITE] Index base: (%d - %d) * 8 + (%d - (%d + 8)) = %d",
                                    prefetch_row, neuron_base, prefetch_col, input_base,
                                    (prefetch_row - neuron_base) * ARRAY_SIZE + (prefetch_col - (input_base + ARRAY_SIZE)));
                        end

                        // Extract 4 weights from Port A (even word: weights 0-3)
                        if (prefetch_col + 0 - (input_base + ARRAY_SIZE) < ARRAY_SIZE) begin
                            if (fetch_buffer == 1'b0) begin
                                weight_buffer_a[(prefetch_row - neuron_base) * ARRAY_SIZE +
                                               (prefetch_col + 0 - (input_base + ARRAY_SIZE))] <= $signed(bram_data_a_latched[7:0]);
                            end else begin
                                weight_buffer_b[(prefetch_row - neuron_base) * ARRAY_SIZE +
                                               (prefetch_col + 0 - (input_base + ARRAY_SIZE))] <= $signed(bram_data_a_latched[7:0]);
                            end
                        end
                        if (prefetch_col + 1 - (input_base + ARRAY_SIZE) < ARRAY_SIZE) begin
                            if (fetch_buffer == 1'b0) begin
                                weight_buffer_a[(prefetch_row - neuron_base) * ARRAY_SIZE +
                                               (prefetch_col + 1 - (input_base + ARRAY_SIZE))] <= $signed(bram_data_a_latched[15:8]);
                            end else begin
                                weight_buffer_b[(prefetch_row - neuron_base) * ARRAY_SIZE +
                                               (prefetch_col + 1 - (input_base + ARRAY_SIZE))] <= $signed(bram_data_a_latched[15:8]);
                            end
                        end
                        if (prefetch_col + 2 - (input_base + ARRAY_SIZE) < ARRAY_SIZE) begin
                            if (fetch_buffer == 1'b0) begin
                                weight_buffer_a[(prefetch_row - neuron_base) * ARRAY_SIZE +
                                               (prefetch_col + 2 - (input_base + ARRAY_SIZE))] <= $signed(bram_data_a_latched[23:16]);
                            end else begin
                                weight_buffer_b[(prefetch_row - neuron_base) * ARRAY_SIZE +
                                               (prefetch_col + 2 - (input_base + ARRAY_SIZE))] <= $signed(bram_data_a_latched[23:16]);
                            end
                        end
                        if (prefetch_col + 3 - (input_base + ARRAY_SIZE) < ARRAY_SIZE) begin
                            if (fetch_buffer == 1'b0) begin
                                weight_buffer_a[(prefetch_row - neuron_base) * ARRAY_SIZE +
                                               (prefetch_col + 3 - (input_base + ARRAY_SIZE))] <= $signed(bram_data_a_latched[31:24]);
                            end else begin
                                weight_buffer_b[(prefetch_row - neuron_base) * ARRAY_SIZE +
                                               (prefetch_col + 3 - (input_base + ARRAY_SIZE))] <= $signed(bram_data_a_latched[31:24]);
                            end
                        end

                        // Extract 4 weights from Port B (odd word: weights 4-7)
                        if (prefetch_col + 4 - (input_base + ARRAY_SIZE) < ARRAY_SIZE) begin
                            if (fetch_buffer == 1'b0) begin
                                weight_buffer_a[(prefetch_row - neuron_base) * ARRAY_SIZE +
                                               (prefetch_col + 4 - (input_base + ARRAY_SIZE))] <= $signed(bram_data_b_latched[7:0]);
                            end else begin
                                weight_buffer_b[(prefetch_row - neuron_base) * ARRAY_SIZE +
                                               (prefetch_col + 4 - (input_base + ARRAY_SIZE))] <= $signed(bram_data_b_latched[7:0]);
                            end
                        end
                        if (prefetch_col + 5 - (input_base + ARRAY_SIZE) < ARRAY_SIZE) begin
                            if (fetch_buffer == 1'b0) begin
                                weight_buffer_a[(prefetch_row - neuron_base) * ARRAY_SIZE +
                                               (prefetch_col + 5 - (input_base + ARRAY_SIZE))] <= $signed(bram_data_b_latched[15:8]);
                            end else begin
                                weight_buffer_b[(prefetch_row - neuron_base) * ARRAY_SIZE +
                                               (prefetch_col + 5 - (input_base + ARRAY_SIZE))] <= $signed(bram_data_b_latched[15:8]);
                            end
                        end
                        if (prefetch_col + 6 - (input_base + ARRAY_SIZE) < ARRAY_SIZE) begin
                            if (fetch_buffer == 1'b0) begin
                                weight_buffer_a[(prefetch_row - neuron_base) * ARRAY_SIZE +
                                               (prefetch_col + 6 - (input_base + ARRAY_SIZE))] <= $signed(bram_data_b_latched[23:16]);
                            end else begin
                                weight_buffer_b[(prefetch_row - neuron_base) * ARRAY_SIZE +
                                               (prefetch_col + 6 - (input_base + ARRAY_SIZE))] <= $signed(bram_data_b_latched[23:16]);
                            end
                        end
                        if (prefetch_col + 7 - (input_base + ARRAY_SIZE) < ARRAY_SIZE) begin
                            if (fetch_buffer == 1'b0) begin
                                weight_buffer_a[(prefetch_row - neuron_base) * ARRAY_SIZE +
                                               (prefetch_col + 7 - (input_base + ARRAY_SIZE))] <= $signed(bram_data_b_latched[31:24]);
                            end else begin
                                weight_buffer_b[(prefetch_row - neuron_base) * ARRAY_SIZE +
                                               (prefetch_col + 7 - (input_base + ARRAY_SIZE))] <= $signed(bram_data_b_latched[31:24]);
                            end
                        end

                        // Move to next dual-read (8 weights at a time)
                        prefetch_col <= prefetch_col + 8;

                        // Check if current row complete
                        if (prefetch_col + 8 >= input_base + ARRAY_SIZE + ARRAY_SIZE) begin
                            // Row complete, move to next neuron row
                            if (prefetch_row - neuron_base < ARRAY_SIZE - 1 &&
                                prefetch_row + 1 < num_neurons) begin
                                // Fetch next row
                                prefetch_row <= prefetch_row + 1;
                                prefetch_state <= 2'b00;  // IDLE (reinitialize)
                            end else begin
                                // All 8 rows fetched! Prefetch complete.
                                fetch_next_block <= 1'b0;  // Signal completion
                                $display("[PREFETCH] Block N+1 fetch complete during COLLECT_RESULTS");
                            end
                        end else begin
                            // More weights in this row
                            prefetch_state <= 2'b01;  // REQUEST_DATA
                        end
                    end
                endcase
            end
        end
    end

endmodule
