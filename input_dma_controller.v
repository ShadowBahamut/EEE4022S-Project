// Phase 4 Level 2.1: Input DMA Controller with AXI
// Loads input activations from external memory to BRAM activation buffer
//
// Features:
// - AXI4 burst read interface for efficient memory access
// - Configurable source address and byte count
// - Up to 256-beat bursts (1024 bytes per burst)
// - Handshake with compute unit (ready/valid protocol)
// - Performance: ~1.5 cycles/byte (AXI burst efficiency)
//
// Primary use case: Load 784-byte MNIST image for Layer 1 input
//
// Architecture:
// - Instantiates axi_read_burst controller (Level 1.1)
// - Streams bytes from AXI → BRAM
// - Simple 3-state FSM (IDLE → TRANSFER → DONE)

module input_dma_controller #(
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
    input wire [ADDR_WIDTH-1:0] src_addr,      // External memory source address
    input wire [ADDR_WIDTH-1:0] dst_addr,      // BRAM destination address
    input wire [15:0] byte_count,              // Number of bytes to transfer

    // AXI4 Read Address Channel
    output wire [31:0] araddr,
    output wire [7:0] arlen,
    output wire [2:0] arsize,
    output wire [1:0] arburst,
    output wire arvalid,
    input wire arready,

    // AXI4 Read Data Channel
    input wire [31:0] rdata,
    input wire [1:0] rresp,
    input wire rlast,
    input wire rvalid,
    output wire rready,

    // BRAM write interface (byte-oriented)
    output reg [ADDR_WIDTH-1:0] bram_addr,
    output reg [DATA_WIDTH-1:0] bram_data,
    output reg bram_we,             // Write enable

    // Handshake with compute unit
    output reg transfer_ready       // Asserted when data ready for compute
);

    // FSM states (simplified with AXI burst controller)
    localparam IDLE        = 3'd0;
    localparam TRANSFER    = 3'd1;  // Streaming AXI → BRAM
    localparam WAIT_FINAL  = 3'd2;  // Wait for final byte write to complete
    localparam DONE_STATE  = 3'd3;

    reg [2:0] state;

    // AXI burst controller signals
    reg burst_start;
    wire burst_done;
    wire burst_error;
    wire [7:0] burst_data_out;
    wire burst_data_valid;
    reg burst_data_ready;

    // BRAM write tracking
    reg [ADDR_WIDTH-1:0] current_bram_addr;
    reg [15:0] bytes_written;

    // Instantiate AXI read burst controller
    axi_read_burst axi_read (
        .clk(clk),
        .reset(rst),

        // Control
        .start(burst_start),
        .base_addr(src_addr),
        .num_bytes({16'h0, byte_count}),  // Extend to 32-bit
        .done(burst_done),
        .error(burst_error),

        // Data stream output
        .data_out(burst_data_out),
        .data_valid(burst_data_valid),
        .data_ready(burst_data_ready),

        // AXI4 Read Address Channel
        .araddr(araddr),
        .arlen(arlen),
        .arsize(arsize),
        .arburst(arburst),
        .arvalid(arvalid),
        .arready(arready),

        // AXI4 Read Data Channel
        .rdata(rdata),
        .rresp(rresp),
        .rlast(rlast),
        .rvalid(rvalid),
        .rready(rready)
    );

    // FSM and datapath
    always @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            done <= 1'b0;
            busy <= 1'b0;
            error <= 1'b0;
            bram_addr <= {ADDR_WIDTH{1'b0}};
            bram_data <= {DATA_WIDTH{1'b0}};
            bram_we <= 1'b0;
            transfer_ready <= 1'b0;
            burst_start <= 1'b0;
            burst_data_ready <= 1'b0;
            current_bram_addr <= {ADDR_WIDTH{1'b0}};
        end else begin
            // Default: clear one-shot signals
            burst_start <= 1'b0;
            bram_we <= 1'b0;
            done <= 1'b0;

            case (state)
                IDLE: begin
                    busy <= 1'b0;
                    error <= 1'b0;
                    transfer_ready <= 1'b0;
                    burst_data_ready <= 1'b0;

                    if (start) begin
                        if (byte_count == 0) begin
                            // Zero-length transfer
                            done <= 1'b1;
                            transfer_ready <= 1'b1;
                            state <= DONE_STATE;
                        end else begin
                            // Start AXI burst read
                            $display("[INPUT_DMA] Starting transfer: src=0x%08x, dst=0x%08x (%d), count=%d",
                                     src_addr, dst_addr, dst_addr, byte_count);
                            burst_start <= 1'b1;
                            current_bram_addr <= dst_addr;
                            bytes_written <= 0;
                            busy <= 1'b1;
                            state <= TRANSFER;
                        end
                    end
                end

                TRANSFER: begin
                    busy <= 1'b1;
                    burst_data_ready <= 1'b1;  // Always ready to accept data

                    // Stream bytes from AXI burst → BRAM
                    if (burst_data_valid && burst_data_ready) begin
                        bram_addr <= current_bram_addr;
                        bram_data <= burst_data_out;
                        bram_we <= 1'b1;
                        current_bram_addr <= current_bram_addr + 1;
                        bytes_written <= bytes_written + 1;
                        // Debug: Layer 2 input loading (first 16 bytes to address 0-15)
                        if (current_bram_addr < 16 && byte_count >= 100 && byte_count <= 150) begin
                            $display("[INPUT_DMA_DEBUG] Layer 2 input? byte=%d addr=%d data=0x%02x (%d)",
                                     bytes_written, current_bram_addr, burst_data_out, $signed(burst_data_out));
                        end
                        // Debug: Neuron 83 weight write tracing (around weight 460-464 and 780-784)
                        if ((current_bram_addr >= 65530 && current_bram_addr <= 65545) ||
                            (current_bram_addr >= 65850 && current_bram_addr <= 65860)) begin
                            $display("[WEIGHT_DMA_WRITE] byte=%d addr=%d (0x%05x) data=0x%02x",
                                     bytes_written, current_bram_addr, current_bram_addr, burst_data_out);
                        end
                    end

                    // Check for completion or error
                    // IMPORTANT: Wait one cycle after burst_done to ensure last write completes
                    if (burst_done) begin
                        state <= WAIT_FINAL;
                    end else if (burst_error) begin
                        error <= 1'b1;
                        state <= DONE_STATE;
                    end
                end

                WAIT_FINAL: begin
                    // Wait one cycle to ensure the last byte write completes
                    // No longer need DRAIN - active modes prevent corruption
                    busy <= 1'b1;
                    burst_data_ready <= 1'b0;
                    state <= DONE_STATE;
                end

                DONE_STATE: begin
                    busy <= 1'b0;
                    done <= 1'b1;
                    transfer_ready <= !error;  // Only ready if no error
                    burst_data_ready <= 1'b0;
                    $display("[INPUT_DMA] Transfer complete: %d bytes written", bytes_written);

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
