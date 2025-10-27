// Phase 2: BRAM Wrapper Module
// Simple dual-port BRAM wrapper for weight and activation storage
//
// Features:
// - Dual-port (Port A and Port B)
// - Single-cycle read latency
// - Byte-enable support for write masking
// - Configurable depth and width
//
// Parameters:
//   ADDR_WIDTH: Address width in bits (default 17 for 128KB)
//   DATA_WIDTH: Data width in bits (default 32 for word access)
//
// This is a behavioral model for Verilator simulation.
// For FPGA synthesis, replace with vendor-specific BRAM primitives.

module bram_wrapper #(
    parameter ADDR_WIDTH = 17,  // 17 bits = 128K addresses
    parameter DATA_WIDTH = 32   // 32-bit data width
) (
    input wire clk,

    // Port A (read/write)
    input wire [ADDR_WIDTH-1:0] addr_a,
    input wire [DATA_WIDTH-1:0] data_in_a,
    input wire [(DATA_WIDTH/8)-1:0] we_a,  // Byte-enable write enable
    input wire en_a,                        // Port enable
    output reg [DATA_WIDTH-1:0] data_out_a,

    // Port B (read/write)
    input wire [ADDR_WIDTH-1:0] addr_b,
    input wire [DATA_WIDTH-1:0] data_in_b,
    input wire [(DATA_WIDTH/8)-1:0] we_b,  // Byte-enable write enable
    input wire en_b,                        // Port enable
    output reg [DATA_WIDTH-1:0] data_out_b
);

    // BRAM storage
    // Note: In simulation, this creates a large array. For synthesis,
    // this would be inferred as BRAM primitives.
    localparam MEM_DEPTH = 1 << ADDR_WIDTH;
    localparam BYTES_PER_WORD = DATA_WIDTH / 8;

    // Organize as byte-addressable memory for byte-enable support
    reg [7:0] mem [0:MEM_DEPTH*BYTES_PER_WORD-1];

    // Port A logic
    always @(posedge clk) begin
        if (en_a) begin
            // Write with byte-enable
            if (we_a[0]) mem[addr_a * BYTES_PER_WORD + 0] <= data_in_a[7:0];
            if (we_a[1]) mem[addr_a * BYTES_PER_WORD + 1] <= data_in_a[15:8];
            if (we_a[2]) mem[addr_a * BYTES_PER_WORD + 2] <= data_in_a[23:16];
            if (we_a[3]) mem[addr_a * BYTES_PER_WORD + 3] <= data_in_a[31:24];

            // Read (always enabled, returns current or updated value)
            data_out_a <= {mem[addr_a * BYTES_PER_WORD + 3],
                          mem[addr_a * BYTES_PER_WORD + 2],
                          mem[addr_a * BYTES_PER_WORD + 1],
                          mem[addr_a * BYTES_PER_WORD + 0]};
        end
    end

    // Port B logic
    always @(posedge clk) begin
        if (en_b) begin
            // Write with byte-enable
            if (we_b[0]) mem[addr_b * BYTES_PER_WORD + 0] <= data_in_b[7:0];
            if (we_b[1]) mem[addr_b * BYTES_PER_WORD + 1] <= data_in_b[15:8];
            if (we_b[2]) mem[addr_b * BYTES_PER_WORD + 2] <= data_in_b[23:16];
            if (we_b[3]) mem[addr_b * BYTES_PER_WORD + 3] <= data_in_b[31:24];

            // Read (always enabled, returns current or updated value)
            data_out_b <= {mem[addr_b * BYTES_PER_WORD + 3],
                          mem[addr_b * BYTES_PER_WORD + 2],
                          mem[addr_b * BYTES_PER_WORD + 1],
                          mem[addr_b * BYTES_PER_WORD + 0]};
        end
    end

    // Initialize memory to zero (for simulation)
    integer i;
    initial begin
        for (i = 0; i < MEM_DEPTH * BYTES_PER_WORD; i = i + 1) begin
            mem[i] = 8'h00;
        end
    end

endmodule
