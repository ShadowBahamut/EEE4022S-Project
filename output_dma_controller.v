// Phase 4 Level 2.2: Output DMA Controller with AXI
// Writes layer outputs from BRAM activation buffer to external memory
//
// Features:
// - AXI4 burst write interface for efficient memory access
// - Configurable source address (BRAM) and destination address (external memory)
// - Up to 256-beat bursts (1024 bytes per burst)
// - Supports INT32 (intermediate layers) and INT8 (final layer) outputs
// - Performance: ~1.5 cycles/byte (AXI burst efficiency)
//
// Primary use cases:
// - Layer 1→2: 512 bytes (128 INT32 values)
// - Layer 2→3: 256 bytes (64 INT32 values)
// - Layer 3 final: 40 bytes (10 INT32 values)
//
// Architecture:
// - Instantiates axi_write_burst controller (Level 1.2)
// - Streams bytes from BRAM → AXI
// - Simple 3-state FSM (IDLE → TRANSFER → DONE)

module output_dma_controller #(
    parameter ADDR_WIDTH = 32,
    parameter DATA_WIDTH = 8
) (
    input wire clk,
    input wire rst,

    // Control interface
    input wire start,               // Pulse to start DMA transfer
    output reg done,                // Asserted when transfer complete
    output reg busy,                // High during transfer
    output reg error,               // Asserted if AXI error
    input wire [ADDR_WIDTH-1:0] src_addr,      // BRAM source address
    input wire [ADDR_WIDTH-1:0] dst_addr,      // External memory destination address
    input wire [15:0] byte_count,              // Number of bytes to transfer

    // BRAM read interface (byte-oriented)
    output reg [ADDR_WIDTH-1:0] bram_addr,
    output reg bram_read_en,
    input wire [DATA_WIDTH-1:0] bram_read_data,
    input wire bram_read_valid,

    // AXI4 Write Address Channel
    output wire [31:0] awaddr,
    output wire [7:0] awlen,
    output wire [2:0] awsize,
    output wire [1:0] awburst,
    output wire awvalid,
    input wire awready,

    // AXI4 Write Data Channel
    output wire [31:0] wdata,
    output wire [3:0] wstrb,
    output wire wlast,
    output wire wvalid,
    input wire wready,

    // AXI4 Write Response Channel
    input wire [1:0] bresp,
    input wire bvalid,
    output wire bready,

    // Handshake with compute unit
    output reg transfer_complete    // Asserted when data written to external memory
);

    // FSM states (simplified with AXI burst controller)
    localparam IDLE        = 2'b00;
    localparam TRANSFER    = 2'b01;  // Streaming BRAM → AXI
    localparam DONE_STATE  = 2'b10;

    reg [1:0] state;

    // AXI burst controller signals
    reg burst_start;
    wire burst_done;
    wire burst_error;
    reg [7:0] burst_data_in;
    reg burst_data_valid;
    wire burst_data_ready;

    // BRAM read tracking
    reg [ADDR_WIDTH-1:0] current_bram_addr;
    reg [15:0] bytes_remaining;

    // Pipeline buffer for BRAM data (fixes timing mismatch)
    reg [7:0] bram_data_buffer;
    reg buffer_valid;

    // Instantiate AXI write burst controller
    axi_write_burst axi_write (
        .clk(clk),
        .reset(rst),

        // Control
        .start(burst_start),
        .base_addr(dst_addr),
        .num_bytes({16'h0, byte_count}),  // Extend to 32-bit
        .done(burst_done),
        .error(burst_error),

        // Data stream input
        .data_in(burst_data_in),
        .data_valid(burst_data_valid),
        .data_ready(burst_data_ready),

        // AXI4 Write Address Channel
        .awaddr(awaddr),
        .awlen(awlen),
        .awsize(awsize),
        .awburst(awburst),
        .awvalid(awvalid),
        .awready(awready),

        // AXI4 Write Data Channel
        .wdata(wdata),
        .wstrb(wstrb),
        .wlast(wlast),
        .wvalid(wvalid),
        .wready(wready),

        // AXI4 Write Response Channel
        .bresp(bresp),
        .bvalid(bvalid),
        .bready(bready)
    );

    // FSM and datapath
    always @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            done <= 1'b0;
            busy <= 1'b0;
            error <= 1'b0;
            transfer_complete <= 1'b0;
            bram_addr <= {ADDR_WIDTH{1'b0}};
            bram_read_en <= 1'b0;
            burst_start <= 1'b0;
            burst_data_in <= 8'h0;
            burst_data_valid <= 1'b0;
            current_bram_addr <= {ADDR_WIDTH{1'b0}};
            bytes_remaining <= 16'h0;
            bram_data_buffer <= 8'h0;
            buffer_valid <= 1'b0;
        end else begin
            // Default: clear one-shot signals
            burst_start <= 1'b0;
            bram_read_en <= 1'b0;
            burst_data_valid <= 1'b0;
            done <= 1'b0;

            case (state)
                IDLE: begin
                    busy <= 1'b0;
                    error <= 1'b0;
                    transfer_complete <= 1'b0;
                    buffer_valid <= 1'b0;  // Clear buffer on idle

                    if (start) begin
                        if (byte_count == 0) begin
                            // Zero-length transfer
                            done <= 1'b1;
                            transfer_complete <= 1'b1;
                            state <= DONE_STATE;
                        end else begin
                            // Start AXI burst write
                            burst_start <= 1'b1;
                            current_bram_addr <= src_addr;
                            bytes_remaining <= byte_count;
                            busy <= 1'b1;

                            // Pre-set BRAM address so data is ready in TRANSFER state
                            bram_addr <= src_addr;
                            bram_read_en <= 1'b1;

                            state <= TRANSFER;
                        end
                    end
                end

                TRANSFER: begin
                    busy <= 1'b1;
                    bram_read_en <= 1'b1;

                    // Capture BRAM data into buffer when buffer is empty
                    if (!buffer_valid) begin
                        bram_data_buffer <= bram_read_data;
                        buffer_valid <= 1'b1;
                    end

                    // Send buffered data and advance when AXI controller is ready
                    if (burst_data_ready && buffer_valid) begin
                        // Send current buffered byte
                        burst_data_in <= bram_data_buffer;
                        burst_data_valid <= 1'b1;

                        // Advance to next byte and update BRAM address together
                        current_bram_addr <= current_bram_addr + 1;
                        bram_addr <= current_bram_addr + 1;  // Point to next byte
                        buffer_valid <= 1'b0;  // Mark buffer as needing refill
                    end else begin
                        // Keep address at current position
                        bram_addr <= current_bram_addr;
                        burst_data_valid <= 1'b0;
                    end

                    // Check for completion or error
                    if (burst_done) begin
                        bram_read_en <= 1'b0;
                        buffer_valid <= 1'b0;
                        state <= DONE_STATE;
                    end else if (burst_error) begin
                        bram_read_en <= 1'b0;
                        buffer_valid <= 1'b0;
                        error <= 1'b1;
                        state <= DONE_STATE;
                    end
                end

                DONE_STATE: begin
                    busy <= 1'b0;
                    done <= 1'b1;
                    transfer_complete <= !error;  // Only complete if no error

                    if (!start) begin
                        state <= IDLE;
                    end
                end

                default: begin
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule
