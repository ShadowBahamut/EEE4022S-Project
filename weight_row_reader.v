// Phase 2 Level 2.1: Weight Row Reader
// Reads a complete neuron's weight row from BRAM using BRAM controller
//
// Features:
// - Configurable neuron selection (row index)
// - Configurable weight count (784 for Layer 1, 128 for Layer 2, 64 for Layer 3)
// - Sequential read of weight row using BRAM controller
// - Start/done handshake protocol
// - Outputs one weight per cycle when ready
//
// Usage:
// 1. Configure layer_base_addr, neuron_index, weight_count
// 2. Assert start for one cycle
// 3. Poll weight_valid and read weights sequentially
// 4. Wait for done signal

module weight_row_reader (
    input wire clk,
    input wire rst,

    // Configuration
    input wire [31:0] layer_base_addr,  // BRAM base address for layer weights
    input wire [31:0] neuron_index,     // Which neuron (row) to read
    input wire [31:0] weight_count,     // Number of weights per neuron
    input wire start,                    // Start weight read (pulse)

    // Status
    output reg done,                     // Read complete
    output reg busy,                     // Read in progress
    output reg weight_valid,             // Weight output valid

    // Weight output
    output reg [7:0] weight_out,         // Current weight (INT8)
    output reg [31:0] weight_index,      // Current weight index (0 to weight_count-1)

    // BRAM interface (connect to BRAM wrapper Port A)
    output wire [19:0] bram_addr,        // BRAM address (20 bits = 1MB max)
    input wire [31:0] bram_data_out,     // BRAM read data
    output wire bram_en                  // BRAM enable
);

    // BRAM controller signals
    wire ctrl_done;
    wire ctrl_busy;
    wire ctrl_addr_valid;
    wire [31:0] ctrl_addr;
    reg ctrl_start;
    reg ctrl_next_addr;

    // Instantiate BRAM controller for sequential addressing
    bram_controller bram_ctrl (
        .clk(clk),
        .rst(rst),
        .base_addr(layer_base_addr + (neuron_index * weight_count)),  // Calculate row start
        .access_count(weight_count),
        .stride(32'h1),              // Sequential (stride=1 byte)
        .mode(1'b0),                 // Sequential mode
        .start(ctrl_start),
        .next_addr(ctrl_next_addr),
        .done(ctrl_done),
        .busy(ctrl_busy),
        .addr_valid(ctrl_addr_valid),
        .current_addr(ctrl_addr)
    );

    // Connect BRAM signals
    // ctrl_addr is BYTE address from bram_controller
    // Pass it directly - cfu_axi.v converts byte→word address for BRAM wrapper
    assign bram_addr = ctrl_addr[19:0];  // Pass byte address (20 bits = 1MB max)
    assign bram_en = ctrl_addr_valid;    // Enable BRAM when address is valid

    // FSM states
    localparam IDLE = 3'd0;
    localparam START_CTRL = 3'd1;
    localparam WAIT_ADDR = 3'd2;    // Wait for bram_controller to register address
    localparam WAIT_DATA = 3'd3;    // Wait for BRAM to output data
    localparam READ_WEIGHTS = 3'd4;
    localparam DONE = 3'd5;

    reg [2:0] state;
    reg [31:0] weights_read;

    // Extract byte from 32-bit BRAM word based on byte offset
    wire [1:0] byte_offset;
    assign byte_offset = ctrl_addr[1:0];

    wire [7:0] byte_from_word;
    assign byte_from_word = (byte_offset == 2'b00) ? bram_data_out[7:0] :
                            (byte_offset == 2'b01) ? bram_data_out[15:8] :
                            (byte_offset == 2'b10) ? bram_data_out[23:16] :
                                                      bram_data_out[31:24];

    always @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            done <= 1'b0;
            busy <= 1'b0;
            weight_valid <= 1'b0;
            weight_out <= 8'h0;
            weight_index <= 32'h0;
            weights_read <= 32'h0;
            ctrl_start <= 1'b0;
            ctrl_next_addr <= 1'b0;

        end else begin
            // Default: clear pulses
            ctrl_start <= 1'b0;
            ctrl_next_addr <= 1'b0;
            done <= 1'b0;
            weight_valid <= 1'b0;  // Clear by default, only set in READ_WEIGHTS

            case (state)
                IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        if (weight_count == 0) begin
                            // Zero-length read - complete immediately
                            done <= 1'b1;
                            state <= IDLE;
                        end else begin
                            // Start BRAM controller
                            busy <= 1'b1;
                            weights_read <= 32'h0;
                            weight_index <= 32'h0;
                            ctrl_start <= 1'b1;
                            state <= START_CTRL;
                        end
                    end
                end

                START_CTRL: begin
                    // Wait one cycle for BRAM controller to start
                    if (ctrl_addr_valid) begin
                        // First address ready, wait for address to propagate to BRAM
                        state <= WAIT_ADDR;
                    end
                end

                WAIT_ADDR: begin
                    // Wait 1 cycle for address to propagate (bram_controller registers current_addr)
                    state <= WAIT_DATA;
                end

                WAIT_DATA: begin
                    // Wait 1 cycle for BRAM read latency
                    // Data is now available
                    state <= READ_WEIGHTS;
                end

                READ_WEIGHTS: begin
                    // BRAM data available (after 1 cycle latency)
                    // Output current weight
                    weight_out <= byte_from_word;
                    weight_valid <= 1'b1;  // Valid for this cycle only
                    weight_index <= weights_read;
                    weights_read <= weights_read + 1;

                    // Debug first 8 weights (disabled for cleaner output)
                    // if (weights_read < 8) begin
                    //     $display("[WRR] READ_WEIGHTS state: weight[%d], ctrl_addr=0x%05x, byte_offset=%d, bram_data=0x%08x, byte=0x%02x",
                    //              weights_read, ctrl_addr, byte_offset, bram_data_out, byte_from_word);
                    // end

                    if (weights_read == weight_count - 1) begin
                        // This is the last weight
                        done <= 1'b1;
                        busy <= 1'b0;
                        state <= IDLE;
                    end else begin
                        // Request next address from controller
                        // Need 2 wait cycles: WAIT_ADDR (for ctrl_addr update) + WAIT_DATA (for BRAM latency)
                        ctrl_next_addr <= 1'b1;
                        state <= WAIT_ADDR;
                    end
                end

                DONE: begin
                    // Should not reach here
                    state <= IDLE;
                end

                default: begin
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule
