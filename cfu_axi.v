// Phase 4: AXI-Based CFU with DMA and Async Compute
// Custom Function Unit integrating AXI DMA controllers with async neuron computation
//
// Architecture:
// - Input DMA: External memory → BRAM (weights) or activation buffer (inputs) via AXI
// - Async Compute: Neuron-level computation with Phase 3 async pipeline
// - Output DMA: Results → External memory via AXI
// - BRAM: On-chip weight storage (128KB)
//
// Command Interface (10 instructions):
//   CMD_LOAD_WEIGHTS      = 0  // Load weights: inputs_0=src_addr, inputs_1=num_bytes
//   CMD_LOAD_INPUT        = 1  // Load input: inputs_0=src_addr, inputs_1=num_bytes
//   CMD_SET_NEURON_CONFIG = 2  // Configure neuron: inputs_0=config, inputs_1=bias
//   CMD_START_COMPUTE     = 3  // Start async computation
//   CMD_GET_STATUS        = 4  // Read status register
//   CMD_READ_RESULT       = 5  // Read computation result
//   CMD_WRITE_RESULT      = 6  // Write result via DMA: inputs_0=dst_addr, inputs_1=unused
//   CMD_GET_ERROR         = 7  // Read AXI error status
//   CMD_CLEAR             = 8  // Clear state
//   CMD_RESET             = 9  // Full reset

module cfu_axi (
  // Standard CFU Interface
  input               cmd_valid,
  output              cmd_ready,
  input      [9:0]    cmd_payload_function_id,
  input      [31:0]   cmd_payload_inputs_0,
  input      [31:0]   cmd_payload_inputs_1,
  output reg          rsp_valid,
  input               rsp_ready,
  output reg [31:0]   rsp_payload_outputs_0,
  input               clk,
  input               reset,

  // AXI4 Master Read Interface
  output wire [31:0]  m_axi_araddr,
  output wire [7:0]   m_axi_arlen,
  output wire [2:0]   m_axi_arsize,
  output wire [1:0]   m_axi_arburst,
  output wire         m_axi_arvalid,
  input wire          m_axi_arready,
  input wire  [31:0]  m_axi_rdata,
  input wire  [1:0]   m_axi_rresp,
  input wire          m_axi_rlast,
  input wire          m_axi_rvalid,
  output wire         m_axi_rready,

  // AXI4 Master Write Interface
  output wire [31:0]  m_axi_awaddr,
  output wire [7:0]   m_axi_awlen,
  output wire [2:0]   m_axi_awsize,
  output wire [1:0]   m_axi_awburst,
  output wire         m_axi_awvalid,
  input wire          m_axi_awready,
  output wire [31:0]  m_axi_wdata,
  output wire [3:0]   m_axi_wstrb,
  output wire         m_axi_wlast,
  output wire         m_axi_wvalid,
  input wire          m_axi_wready,
  input wire  [1:0]   m_axi_bresp,
  input wire          m_axi_bvalid,
  output wire         m_axi_bready
);

  // Command function IDs
  localparam CMD_LOAD_WEIGHTS      = 10'd0;
  localparam CMD_LOAD_INPUT        = 10'd1;
  localparam CMD_SET_NEURON_CONFIG = 10'd2;
  localparam CMD_START_COMPUTE     = 10'd3;
  localparam CMD_GET_STATUS        = 10'd4;
  localparam CMD_READ_RESULT       = 10'd5;
  localparam CMD_WRITE_RESULT      = 10'd6;
  localparam CMD_GET_ERROR         = 10'd7;
  localparam CMD_CLEAR             = 10'd8;
  localparam CMD_RESET             = 10'd9;

  // Status codes
  localparam STATUS_IDLE               = 8'h00;
  localparam STATUS_BUSY_LOADING       = 8'h01;
  localparam STATUS_BUSY_COMPUTING     = 8'h02;
  localparam STATUS_DONE               = 8'h03;
  localparam STATUS_BUSY_WRITING       = 8'h04;

  // Configuration registers
  reg [31:0] neuron_config;    // Combined: layer_base[31:16], neuron_idx[15:0]
  reg [31:0] bias;
  reg [15:0] input_size;

  // DMA control signals
  // PHASE 4 DUAL DMA: Separate dedicated controllers eliminate mode switching races

  // Weight DMA: Loads weights from external memory → BRAM
  reg weight_dma_start;
  wire weight_dma_done;
  wire weight_dma_busy;
  wire weight_dma_error;
  reg [31:0] weight_dma_src_addr;
  reg [31:0] weight_dma_dst_addr;  // CRITICAL: Each neuron writes to different BRAM offset!
  reg [15:0] weight_dma_byte_count;

  // Input DMA: Loads inputs from external memory → activation buffer
  reg input_dma_start;
  wire input_dma_done;
  wire input_dma_busy;
  wire input_dma_error;
  reg [31:0] input_dma_src_addr;
  reg [15:0] input_dma_byte_count;

  // Output DMA: Writes results to external memory
  reg output_dma_start;
  wire output_dma_done;
  wire output_dma_busy;
  wire output_dma_error;
  reg [31:0] output_dma_src_addr;
  reg [31:0] output_dma_dst_addr;
  reg [15:0] output_dma_byte_count;

  // Compute unit control
  reg compute_start;
  reg clear_buffers;  // Phase 5: Clear input/bias buffers for new image
  wire compute_done;
  wire [31:0] compute_result;
  reg compute_busy;  // Phase 5: Track if computation is in progress

  // State tracking
  reg [7:0] status;
  reg [7:0] error_flags;
  reg result_ready;
  reg [31:0] result_latch;
  reg operation_done;  // Latches done signals until next command

  // DUAL DMA: No mode logic needed! Each DMA has dedicated routing.

  // Weight DMA → BRAM interface (dedicated)
  wire [19:0] bram_addr_weight;    // 20 bits = 1MB address space
  wire [7:0]  bram_din_weight;
  wire        bram_we_weight;

  // Compute unit reads from BRAM
  // Phase 6 Layer 1: Port B signals (single-port reads)
  wire [19:0] bram_addr_compute;   // Port B address (20 bits = 1MB address space)
  wire [31:0] bram_dout_compute;   // Port B data output

  // Phase 6 Layer 2: Port A signals (dual-port reads)
  wire [19:0] bram_addr_a_compute; // Port A address
  wire [31:0] bram_dout_a_compute; // Port A data output
  wire bram_en_a_compute;          // Port A enable

  // Input DMA → Activation buffer interface (dedicated)
  wire [31:0] input_dma_addr_out;
  wire [7:0]  input_dma_data_out;
  wire        input_dma_we_out;

  // Input buffer write interface (muxed between DMA and layer results)
  reg [15:0] input_write_addr_compute;
  reg [7:0]  input_write_data_compute;
  reg        input_write_en_compute;

  // Phase 5: Inter-layer data transfer signals
  reg        transfer_prev_layer;    // Need to copy previous layer results
  reg        transfer_active;        // Currently transferring
  reg [15:0] transfer_idx;           // Current transfer index

  // AXI bus arbitration signals (internal wires from each DMA)
  wire [31:0] weight_dma_araddr, input_dma_araddr;
  wire [7:0]  weight_dma_arlen, input_dma_arlen;
  wire [2:0]  weight_dma_arsize, input_dma_arsize;
  wire [1:0]  weight_dma_arburst, input_dma_arburst;
  wire        weight_dma_arvalid, input_dma_arvalid;
  wire        weight_dma_arready, input_dma_arready;
  wire        weight_dma_rready, input_dma_rready;
  wire        weight_dma_rvalid, input_dma_rvalid;  // Gated read data valid

  // AXI arbitration select: Use arvalid to catch start-up cycle (busy lags by 1 cycle)
  wire weight_dma_selected;
  assign weight_dma_selected = weight_dma_arvalid || weight_dma_busy;

  // AXI arbitration: Only one DMA active at a time (enforced by sequential commands)
  // Weight DMA has priority (checked first)
  // CRITICAL: Must gate rvalid to each DMA to prevent data stealing!
  assign m_axi_araddr = weight_dma_selected ? weight_dma_araddr : input_dma_araddr;
  assign m_axi_arlen = weight_dma_selected ? weight_dma_arlen : input_dma_arlen;
  assign m_axi_arsize = weight_dma_selected ? weight_dma_arsize : input_dma_arsize;
  assign m_axi_arburst = weight_dma_selected ? weight_dma_arburst : input_dma_arburst;
  assign m_axi_arvalid = weight_dma_arvalid | input_dma_arvalid;
  assign m_axi_rready = weight_dma_rready | input_dma_rready;
  assign weight_dma_arready = weight_dma_selected ? m_axi_arready : 1'b0;
  assign input_dma_arready = !weight_dma_selected ? m_axi_arready : 1'b0;
  // Gate rvalid so each DMA only sees it when selected (prevents data stealing!)
  assign weight_dma_rvalid = weight_dma_selected ? m_axi_rvalid : 1'b0;
  assign input_dma_rvalid = !weight_dma_selected ? m_axi_rvalid : 1'b0;

  // Command ready when not in reset
  // Accept commands immediately (standard CFU protocol)
  assign cmd_ready = !reset;

  // ===== Weight DMA Controller (Dedicated for BRAM loading) =====
  // Loads weights from external memory → BRAM
  // Direct routing: NO mode switching needed!
  input_dma_controller #(
    .ADDR_WIDTH(20),  // 20 bits = 1MB address space (matches bram_addr_weight width)
    .DATA_WIDTH(8)
  ) weight_dma (
    .clk(clk),
    .rst(reset),

    // Control
    .start(weight_dma_start),
    .done(weight_dma_done),
    .busy(weight_dma_busy),
    .error(weight_dma_error),
    .src_addr(weight_dma_src_addr),
    .dst_addr(weight_dma_dst_addr),  // Each neuron at different BRAM offset!
    .byte_count(weight_dma_byte_count),

    // BRAM write interface - Direct to BRAM!
    .bram_addr(bram_addr_weight),
    .bram_data(bram_din_weight),
    .bram_we(bram_we_weight),
    .transfer_ready(),  // Not used

    // AXI4 Read interface (arbitrated - internal wires)
    .araddr(weight_dma_araddr),
    .arlen(weight_dma_arlen),
    .arsize(weight_dma_arsize),
    .arburst(weight_dma_arburst),
    .arvalid(weight_dma_arvalid),
    .arready(weight_dma_arready),
    .rdata(m_axi_rdata),
    .rresp(m_axi_rresp),
    .rlast(m_axi_rlast),
    .rvalid(weight_dma_rvalid),  // Gated rvalid prevents data stealing!
    .rready(weight_dma_rready)
  );

  // ===== Input DMA Controller (Dedicated for activation buffer loading) =====
  // Loads inputs from external memory → compute unit input buffer
  // Direct routing: NO mode switching needed!
  input_dma_controller input_dma (
    .clk(clk),
    .rst(reset),

    // Control
    .start(input_dma_start),
    .done(input_dma_done),
    .busy(input_dma_busy),
    .error(input_dma_error),
    .src_addr(input_dma_src_addr),
    .dst_addr(32'h0),  // Start at offset 0 in activation buffer
    .byte_count(input_dma_byte_count),

    // Output interface - Direct to compute unit!
    .bram_addr(input_dma_addr_out),
    .bram_data(input_dma_data_out),
    .bram_we(input_dma_we_out),
    .transfer_ready(),  // Not used

    // AXI4 Read interface (arbitrated - internal wires)
    .araddr(input_dma_araddr),
    .arlen(input_dma_arlen),
    .arsize(input_dma_arsize),
    .arburst(input_dma_arburst),
    .arvalid(input_dma_arvalid),
    .arready(input_dma_arready),
    .rdata(m_axi_rdata),
    .rresp(m_axi_rresp),
    .rlast(m_axi_rlast),
    .rvalid(input_dma_rvalid),  // Gated rvalid prevents data stealing!
    .rready(input_dma_rready)
  );

  // ===== Output DMA Controller =====
  // Writes INT32 results to external memory
  output_dma_controller output_dma (
    .clk(clk),
    .rst(reset),

    // Control
    .start(output_dma_start),
    .done(output_dma_done),
    .busy(output_dma_busy),
    .error(output_dma_error),
    .src_addr(output_dma_src_addr),    // Not used (data comes from result)
    .dst_addr(output_dma_dst_addr),
    .byte_count(output_dma_byte_count),

    // BRAM read interface (we'll override with result data)
    .bram_addr(),  // Not used
    .bram_read_en(),
    .bram_read_data(result_latch[7:0]),  // Feed result bytes
    .bram_read_valid(1'b1),
    .transfer_complete(),

    // AXI4 Write interface
    .awaddr(m_axi_awaddr),
    .awlen(m_axi_awlen),
    .awsize(m_axi_awsize),
    .awburst(m_axi_awburst),
    .awvalid(m_axi_awvalid),
    .awready(m_axi_awready),
    .wdata(m_axi_wdata),
    .wstrb(m_axi_wstrb),
    .wlast(m_axi_wlast),
    .wvalid(m_axi_wvalid),
    .wready(m_axi_wready),
    .bresp(m_axi_bresp),
    .bvalid(m_axi_bvalid),
    .bready(m_axi_bready)
  );

  // BRAM write byte enable logic (from weight_dma)
  // Map byte address to word address and byte enable
  wire [3:0] bram_we_byte_enable;
  wire [31:0] bram_data_word;
  reg [15:0] bram_write_count;  // 16-bit to handle 784 bytes without overflow!
  initial bram_write_count = 0;

  // DUAL DMA: Direct from weight_dma (no mode checking needed!)
  assign bram_we_byte_enable = bram_we_weight ? (4'b0001 << bram_addr_weight[1:0]) : 4'b0000;
  assign bram_data_word = {24'b0, bram_din_weight} << {bram_addr_weight[1:0], 3'b000};

  // Debug BRAM writes
  always @(posedge clk) begin
    if (bram_we_weight) begin
      // Debug: Per-byte write tracing (disabled for cleaner output)
      // if (bram_write_count < 8) begin
      //   $display("[BRAM_WRITE] byte_addr=%d, word_addr=%d, byte_en=%b, data=0x%02x, data_word=0x%08x",
      //            bram_addr_weight, bram_addr_weight[19:2], bram_we_byte_enable, bram_din_weight, bram_data_word);
      // end
      bram_write_count <= bram_write_count + 1;
    end
    if (weight_dma_done) begin
      `ifdef DEBUG
      $display("[BRAM] Weight load complete: %d bytes written", bram_write_count);
      `endif
      bram_write_count <= 0;
    end
  end

  // ===== BRAM for Weights =====
  // 128KB dual-port BRAM
  bram_wrapper #(
    .ADDR_WIDTH(20),  // 1MB = 2^20 bytes (increased for large weight matrices)
    .DATA_WIDTH(32)   // 32-bit word access
  ) weight_bram (
    .clk(clk),

    // Port A: Weight DMA writes (during init) OR compute reads (during inference)
    // Phase 6 Layer 2: Mux between DMA and compute unit
    .en_a(bram_we_weight | bram_en_a_compute),  // Enable for either DMA write or compute read
    .addr_a(bram_we_weight ? (bram_addr_weight >> 2) : bram_addr_a_compute),  // Mux: DMA needs >>2, compute already word addr
    .we_a(bram_we_weight ? bram_we_byte_enable : 4'b0),  // Write only during DMA
    .data_in_a(bram_data_word),                  // Data for DMA writes
    .data_out_a(bram_dout_a_compute),            // Data for compute reads

    // Port B: Compute unit reads (word access)
    // Phase 6 Layer 2: Compute unit now sends word addresses directly (not byte addresses!)
    .en_b(1'b1),  // Always enabled for reads
    .addr_b(bram_addr_compute), // Compute unit sends word address directly (no >>2 needed!)
    .we_b(4'b0),  // Read-only
    .data_in_b(32'b0),
    .data_out_b(bram_dout_compute)
  );

  // ===== Direct Routing: Input DMA → Compute Unit =====
  // Phase 5: Input buffer write mux (DMA or layer results transfer)
  // Muxing logic moved to always block below to support inter-layer data transfer

  // Debug input writes
  reg [31:0] wr_cnt_input;
  initial wr_cnt_input = 0;

  always @(posedge clk) begin
    if (input_write_en_compute) begin
      wr_cnt_input <= wr_cnt_input + 1;
      // Debug: Per-byte write tracing (disabled for cleaner output)
      // if (wr_cnt_input < 10 || (wr_cnt_input >= 774 && wr_cnt_input < 784)) begin
      //   $display("[INPUT_DMA→COMPUTE] Write #%d: addr=%d, data=0x%02x",
      //            wr_cnt_input, input_write_addr_compute, input_write_data_compute);
      // end
    end
    if (input_dma_done) begin
      `ifdef DEBUG
      $display("[INPUT_DMA→COMPUTE] Transfer complete: %d bytes written", wr_cnt_input);
      `endif
      wr_cnt_input <= 0;
    end
  end

  // ===== Systolic Layer Compute Unit (Phase 5) =====
  // Layer-level computation using 8×8 systolic array
  // Computes ALL neurons in layer at once (vs one neuron at a time in Phase 4)
  // CRITICAL: Convert layer number to actual BRAM base offset
  //   Layer 0: 0 (100352 bytes)
  //   Layer 1: 100352 (8192 bytes)
  //   Layer 2: 108544 (640 bytes)
  wire [31:0] layer_bram_base;
  assign layer_bram_base = (neuron_config[31:16] == 16'd2) ? 32'd108544 :  // Layer 3
                           (neuron_config[31:16] == 16'd1) ? 32'd100352 :  // Layer 2
                           32'd0;                                            // Layer 1

  // Phase 5: New signals for layer-level computation
  reg [7:0] layer_num_neurons;        // Number of neurons to compute (1-128)
  reg [31:0] layer_bram_addr;         // Adjusted BRAM address for specific neuron (simplified mode)
  wire [7:0] layer_output_addr;       // Streaming output: neuron index
  wire signed [31:0] layer_output_data; // Streaming output: neuron result
  wire layer_output_valid;            // Streaming output: valid strobe

  // Phase 5: Bias buffer interface (replaces single bias from Phase 4)
  reg bias_write_en_internal;
  reg [7:0] bias_write_addr_internal;
  reg signed [31:0] bias_write_data_internal;

  // Phase 5: Result caching for layer-level computation
  reg signed [31:0] layer_results [0:127];  // Cache all neuron results
  reg [15:0] current_layer;                 // Track current layer being computed
  reg layer_computed;                       // Has current layer been computed?
  reg [7:0] results_captured;               // Number of results captured so far
  integer i;                                // Loop variable for initialization

  systolic_layer_compute_unit #(
    .ARRAY_SIZE(8)
  ) compute_unit (
    .clk(clk),
    .rst(reset),

    // Layer configuration (NEW in Phase 5)
    .num_neurons(layer_num_neurons),              // NEW: Number of neurons (1-128)
    .input_size(input_size),                      // Same as Phase 4
    .layer_base_addr(layer_bram_base),            // Layer base address (0, 100352, or 108544)

    // Control
    .start(compute_start),                        // Same as Phase 4
    .clear_buffers(clear_buffers),                // NEW: Clear buffers for new image
    .done(compute_done),                          // Same as Phase 4

    // Input buffer loading interface (from input DMA)
    .input_write_en(input_write_en_compute),      // Same as Phase 4
    .input_write_addr(input_write_addr_compute),  // Same as Phase 4
    .input_write_data(input_write_data_compute),  // Same as Phase 4

    // Bias buffer interface (NEW in Phase 5)
    .bias_write_en(bias_write_en_internal),       // NEW
    .bias_write_addr(bias_write_addr_internal),   // NEW
    .bias_write_data(bias_write_data_internal),   // NEW

    // Weight BRAM interface - Port B (Phase 6 Layer 1)
    .bram_addr(bram_addr_compute),                // Port B address
    .bram_data_out(bram_dout_compute),            // Port B data
    .bram_en(),                                   // Port B enable (not used)

    // Weight BRAM interface - Port A (Phase 6 Layer 2: dual-port reads)
    .bram_addr_a(bram_addr_a_compute),            // Port A address
    .bram_data_a_out(bram_dout_a_compute),        // Port A data
    .bram_en_a(bram_en_a_compute),                // Port A enable

    // Output streaming interface (NEW in Phase 5)
    .output_addr(layer_output_addr),              // NEW: Which neuron
    .output_data(layer_output_data),              // NEW: Neuron result
    .output_valid(layer_output_valid)             // NEW: Result valid strobe
  );

  // ===== Command Processing FSM =====
  always @(posedge clk) begin
    if (reset) begin
      status <= STATUS_IDLE;
      error_flags <= 8'h00;
      result_ready <= 1'b0;
      result_latch <= 32'h0;
      operation_done <= 1'b0;
      // DUAL DMA: No mode signals to initialize!
      neuron_config <= 32'h0;
      bias <= 32'h0;
      input_size <= 16'h0;
      weight_dma_start <= 1'b0;
      input_dma_start <= 1'b0;
      output_dma_start <= 1'b0;
      compute_start <= 1'b0;
      clear_buffers <= 1'b0;  // Phase 5: Initialize buffer clear signal
      compute_busy <= 1'b0;  // Phase 5: Initialize compute busy flag
      rsp_valid <= 1'b0;
      rsp_payload_outputs_0 <= 32'h0;

      weight_dma_src_addr <= 32'h0;
      weight_dma_dst_addr <= 32'h0;
      weight_dma_byte_count <= 16'h0;
      input_dma_src_addr <= 32'h0;
      input_dma_byte_count <= 16'h0;
      output_dma_dst_addr <= 32'h0;
      output_dma_byte_count <= 16'h4;  // Always 4 bytes (INT32)

      // Phase 5: Initialize layer computation signals
      layer_num_neurons <= 8'd1;                  // Default: 1 neuron at a time (simplified mode)
      layer_bram_addr <= 32'h0;                   // BRAM address for specific neuron
      bias_write_en_internal <= 1'b0;
      bias_write_addr_internal <= 8'h0;
      bias_write_data_internal <= 32'h0;

      // Phase 5: Initialize result caching
      current_layer <= 16'hFFFF;                  // Invalid layer marker
      layer_computed <= 1'b0;
      results_captured <= 8'h0;
      // Initialize result buffer to zero (will be set by actual computation)
      for (i = 0; i < 128; i = i + 1) begin
        layer_results[i] <= 32'sd0;
      end

      // Phase 5: Initialize inter-layer transfer signals
      transfer_prev_layer <= 1'b0;
      transfer_active <= 1'b0;
      transfer_idx <= 16'h0;
      input_write_en_compute <= 1'b0;
      input_write_addr_compute <= 16'h0;
      input_write_data_compute <= 8'h0;
    end else begin
      // Clear bias write enable (one-shot signal)
      bias_write_en_internal <= 1'b0;
      // DUAL DMA: No mode latching logic needed!

      // Clear one-shot signals
      weight_dma_start <= 1'b0;
      input_dma_start <= 1'b0;
      output_dma_start <= 1'b0;
      compute_start <= 1'b0;
      clear_buffers <= 1'b0;  // Phase 5: Clear buffer clear signal

      // Phase 5: Update compute_busy flag
      if (compute_start) begin
        compute_busy <= 1'b1;  // Set when computation starts
      end else if (compute_done) begin
        compute_busy <= 1'b0;  // Clear when computation completes
      end

      // Phase 5: Capture streaming outputs from systolic layer
      if (layer_output_valid) begin
        layer_results[layer_output_addr] <= layer_output_data;
        results_captured <= results_captured + 1;

        // DEBUG: Trace first/last neuron captures unconditionally
        if (layer_output_addr < 2 || layer_output_addr >= 126) begin
          $display("[RESULT_CAPTURE] layer=%d addr=%3d data=%8d (total=%d)",
                   current_layer, layer_output_addr, $signed(layer_output_data), results_captured + 1);
        end

        `ifdef DEBUG
        $display("[CFU_AXI] Captured result: neuron %d = %d (total captured: %d)",
                 layer_output_addr, $signed(layer_output_data), results_captured + 1);
        `endif
      end

      // Phase 5: Mark layer as computed when done
      if (compute_done && compute_busy) begin
        $display("[STATE_DONE] Layer %d complete, setting layer_computed=1 (%d results)",
                 current_layer, results_captured);
        layer_computed <= 1'b1;
      end

      // Phase 5: Inter-layer data transfer logic
      // Mux input buffer writes between DMA and layer results
      // DEBUG: Log MUX state every time DMA tries to write
      if (input_dma_we_out && input_dma_addr_out[15:0] < 10) begin
        $display("[MUX_STATE] DMA write request addr=%d data=%d | transfer_active=%b transfer_prev_layer=%b",
                 input_dma_addr_out[15:0], $signed(input_dma_data_out), transfer_active, transfer_prev_layer);
      end

      if (transfer_active) begin
        // Transfer layer_results to input buffer
        if (transfer_idx < input_size) begin
          input_write_en_compute <= 1'b1;
          input_write_addr_compute <= transfer_idx;
          // Extract INT8 from layer_results INT32 (already requantized)
          input_write_data_compute <= layer_results[transfer_idx][7:0];
          transfer_idx <= transfer_idx + 1;
          // UNCONDITIONAL DEBUG: Log first 10 transfers
          if (transfer_idx < 10) begin
            $display("[CFU_AXI_TRANSFER] Writing result[%d]=%d to input_buffer[%d]",
                     transfer_idx, $signed(layer_results[transfer_idx][7:0]), transfer_idx);
          end
        end else begin
          // Transfer complete
          input_write_en_compute <= 1'b0;
          transfer_active <= 1'b0;
          transfer_prev_layer <= 1'b0;
          // UNCONDITIONAL DEBUG: Log completion
          $display("[CFU_AXI_TRANSFER] Transfer complete: %d elements copied", input_size);
        end
      end else if (transfer_prev_layer) begin
        // Start transfer
        transfer_active <= 1'b1;
        transfer_idx <= 16'h0;
        // UNCONDITIONAL DEBUG: Log transfer start
        $display("[CFU_AXI_TRANSFER] Starting transfer: input_size=%d", input_size);
      end else begin
        // Normal DMA mode
        input_write_en_compute <= input_dma_we_out;
        input_write_addr_compute <= input_dma_addr_out[15:0];
        input_write_data_compute <= input_dma_data_out;
        // DEBUG: Log first 10 DMA writes
        if (input_dma_we_out && input_dma_addr_out[15:0] < 10) begin
          $display("[DMA_WRITE_SUCCESS] addr=%d data=%d", input_dma_addr_out[15:0], $signed(input_dma_data_out));
        end
      end

      // Update error flags
      if (weight_dma_error) error_flags[0] <= 1'b1;
      if (input_dma_error) error_flags[1] <= 1'b1;
      if (output_dma_error) error_flags[2] <= 1'b1;

      // Latch done signals when DMA completes
      if ((weight_dma_done && !weight_dma_busy) || (input_dma_done && !input_dma_busy) || compute_done || output_dma_done) begin
        operation_done <= 1'b1;
        // Phase 5 Debug: Always show DMA completion
        if (weight_dma_done && !weight_dma_busy) begin
          $display("[CFU_AXI] WEIGHT DMA complete (done=1, busy=0)");
        end
        if (input_dma_done && !input_dma_busy) begin
          $display("[CFU_AXI] INPUT DMA complete (done=1, busy=0), loaded %d bytes", input_dma_byte_count);
        end
      end

      // Update status based on operation
      // DUAL DMA: OR both busy signals
      if (weight_dma_busy || input_dma_busy) begin
        status <= STATUS_BUSY_LOADING;
      end else if (output_dma_busy) begin
        status <= STATUS_BUSY_WRITING;
      end else if (compute_busy) begin
        // Phase 5: Check if systolic layer is computing
        status <= STATUS_BUSY_COMPUTING;
      end else if (operation_done || result_ready) begin
        status <= STATUS_DONE;
      end else begin
        status <= STATUS_IDLE;
      end

      // Phase 5: Capture result from streaming output
      // Simplified mode: only one neuron computed, so only one result expected
      if (layer_output_valid) begin
        `ifdef DEBUG
        $display("[CFU_AXI] Phase 5: Capturing result: neuron=%d, result=%d",
                 layer_output_addr, $signed(layer_output_data));
        `endif
        result_latch <= layer_output_data;
        result_ready <= 1'b1;
        `ifdef DEBUG
        $display("[CFU_AXI] result_latch <= %d", $signed(layer_output_data));
        `endif
      end

      // Process CFU commands
      if (cmd_valid && cmd_ready) begin
        rsp_valid <= 1'b1;  // Response ready next cycle
        operation_done <= 1'b0;  // Clear done flag on new command

        case (cmd_payload_function_id)
          CMD_LOAD_WEIGHTS: begin
            // Load weights: src_addr, byte_count
            // DUAL DMA: Use dedicated weight_dma (direct to BRAM)
            // CRITICAL: BRAM destination calculation strategy:
            //   MNIST testbench: Use address range detection (LOAD_WEIGHTS before SET_NEURON_CONFIG)
            //   Other testbenches: Use layer_base + neuron_idx * weight_count (SET_NEURON_CONFIG first)
            // Address range mapping for MNIST (external memory → BRAM):
            //   Layer 1: 0x10000-0x187FF (100352 bytes) → BRAM 0-100351
            //   Layer 2: 0x30000-0x31FFF (8192 bytes)   → BRAM 100352-108543
            //   Layer 3: 0x40000-0x4027F (640 bytes)    → BRAM 108544-109183
            //   Others:  Use layer_bram_base + neuron_index * weight_count
            weight_dma_src_addr <= cmd_payload_inputs_0;
            if (cmd_payload_inputs_0 >= 32'h00040000 && cmd_payload_inputs_0 < 32'h00050000) begin
              // Layer 3 range: offset from 0x40000 + layer 3 base (108544)
              weight_dma_dst_addr <= 32'd108544 + (cmd_payload_inputs_0 - 32'h00040000);
            end else if (cmd_payload_inputs_0 >= 32'h00030000 && cmd_payload_inputs_0 < 32'h00040000) begin
              // Layer 2 range: offset from 0x30000 + layer 2 base (100352)
              weight_dma_dst_addr <= 32'd100352 + (cmd_payload_inputs_0 - 32'h00030000);
            end else if (cmd_payload_inputs_0 >= 32'h00010000 && cmd_payload_inputs_0 < 32'h00030000) begin
              // Layer 1 range: offset from 0x10000
              weight_dma_dst_addr <= cmd_payload_inputs_0 - 32'h00010000;
            end else begin
              // Fallback for other testbenches: use neuron index calculation
              weight_dma_dst_addr <= layer_bram_base + (neuron_config[15:0] * cmd_payload_inputs_1[15:0]);
            end
            weight_dma_byte_count <= cmd_payload_inputs_1[15:0];
            weight_dma_start <= 1'b1;
            `ifdef DEBUG
            $display("[CFU_AXI] CMD_LOAD_WEIGHTS: src=0x%08x, dst=%d, bytes=%d",
                     cmd_payload_inputs_0,
                     (cmd_payload_inputs_0 >= 32'h00040000 && cmd_payload_inputs_0 < 32'h00050000) ? (32'd108544 + (cmd_payload_inputs_0 - 32'h00040000)) :
                     (cmd_payload_inputs_0 >= 32'h00030000 && cmd_payload_inputs_0 < 32'h00040000) ? (32'd100352 + (cmd_payload_inputs_0 - 32'h00030000)) :
                     (cmd_payload_inputs_0 >= 32'h00010000 && cmd_payload_inputs_0 < 32'h00030000) ? (cmd_payload_inputs_0 - 32'h00010000) :
                     (layer_bram_base + (neuron_config[15:0] * cmd_payload_inputs_1[15:0])),
                     cmd_payload_inputs_1[15:0]);
            `endif
            rsp_payload_outputs_0 <= 32'h0;  // ACK
          end

          CMD_LOAD_INPUT: begin
            // Load input to activation buffer: src_addr, byte_count
            // DUAL DMA: Use dedicated input_dma (direct to compute unit)
            // UNCONDITIONAL DEBUG: Critical path for Image 1 bug
            $display("[CMD_LOAD_INPUT_EXEC] byte_count=%d, src=0x%x, setting input_dma_start=1",
                     cmd_payload_inputs_1[15:0], cmd_payload_inputs_0);
            input_dma_src_addr <= cmd_payload_inputs_0;
            input_dma_byte_count <= cmd_payload_inputs_1[15:0];
            input_dma_start <= 1'b1;
            input_size <= cmd_payload_inputs_1[15:0];
            rsp_payload_outputs_0 <= 32'h0;  // ACK
          end

          CMD_SET_NEURON_CONFIG: begin
            // Configure neuron: config word, bias
            neuron_config <= cmd_payload_inputs_0;
            bias <= cmd_payload_inputs_1;

            // Phase 5 Simplified Mode: Calculate BRAM address for specific neuron
            // neuron_config[31:16] = layer number
            // neuron_config[15:0] = neuron index within layer
            // BRAM address = layer_base + (neuron_index * input_size)
            //
            // Example: Layer 1, neuron 5, input_size 784
            //   layer_base = 0
            //   neuron_index = 5
            //   BRAM address = 0 + (5 * 784) = 3920
            layer_bram_addr <= layer_bram_base + (cmd_payload_inputs_0[15:0] * input_size);

            // Phase 5: Write bias to bias buffer at CORRECT neuron index
            bias_write_en_internal <= 1'b1;
            bias_write_addr_internal <= cmd_payload_inputs_0[15:0];  // Use neuron index, not always 0
            bias_write_data_internal <= cmd_payload_inputs_1;

            // Phase 5: Detect layer change and set layer_num_neurons
            if (cmd_payload_inputs_0[31:16] != current_layer) begin
              $display("[STATE_CHANGE] Layer change: %d → %d, resetting layer_computed=0",
                       current_layer, cmd_payload_inputs_0[31:16]);
              current_layer <= cmd_payload_inputs_0[31:16];
              layer_computed <= 1'b0;  // New layer, need to compute
              results_captured <= 8'h0;

              // Set layer_num_neurons based on layer
              case (cmd_payload_inputs_0[31:16])
                16'd0: layer_num_neurons <= 8'd128;  // Layer 1: 784 → 128
                16'd1: layer_num_neurons <= 8'd64;   // Layer 2: 128 → 64
                16'd2: layer_num_neurons <= 8'd10;   // Layer 3: 64 → 10
                default: layer_num_neurons <= 8'd1;  // Fallback
              endcase

              // Phase 5: DISABLED - Inter-layer transfer NOT USED
              // BUG FIX (2025-10-20): Hardware transfer was copying raw INT32 low bytes, not requantized INT8
              // The testbench handles requantization in C++ and loads via DMA, so disable hardware transfer
              // if (cmd_payload_inputs_0[31:16] > 16'd0) begin
              //   transfer_prev_layer <= 1'b1;
              //   // UNCONDITIONAL DEBUG: Trace transfer trigger
              //   $display("[CFU_AXI_TRANSFER] Layer %d: transfer_prev_layer=1 (triggered)",
              //            cmd_payload_inputs_0[31:16]);
              // end

              `ifdef DEBUG
              $display("[CFU_AXI] Layer change detected: %d → %d, layer_num_neurons=%d",
                       current_layer, cmd_payload_inputs_0[31:16],
                       (cmd_payload_inputs_0[31:16] == 16'd0) ? 8'd128 :
                       (cmd_payload_inputs_0[31:16] == 16'd1) ? 8'd64 : 8'd10);
              `endif
            end

            `ifdef DEBUG
            $display("[CFU_AXI] Phase 5 Simplified: layer=%d, neuron=%d, input_size=%d, layer_base=0x%x, bram_addr=0x%x",
                     cmd_payload_inputs_0[31:16], cmd_payload_inputs_0[15:0], input_size,
                     layer_bram_base, layer_bram_base + (cmd_payload_inputs_0[15:0] * input_size));
            `endif

            rsp_payload_outputs_0 <= 32'h0;  // ACK
          end

          CMD_START_COMPUTE: begin
            // Phase 5: Layer-level computation (NOT neuron-by-neuron!)
            // CRITICAL FIX (2025-10-20): Removed neuron-by-neuron trigger logic
            // Old behavior: Wait for neuron_config == layer_num_neurons - 1
            // New behavior: Trigger on FIRST START_COMPUTE call for layer
            $display("[STATE_CHECK] START_COMPUTE: layer=%d layer_computed=%b",
                     current_layer, layer_computed);

            if (!layer_computed) begin
              // First START_COMPUTE call for this layer - compute ALL neurons
              $display("[STATE_COMPUTE] Computing layer %d (layer-level processing)",
                       current_layer);
              compute_start <= 1'b1;
              result_ready <= 1'b0;
              status <= STATUS_BUSY_COMPUTING;
            end else begin
              // Layer already computed, return cached result
              $display("[STATE_SKIP] Layer %d CACHED, result ready",
                       current_layer);
              result_ready <= 1'b1;
              status <= STATUS_DONE;
            end
            rsp_payload_outputs_0 <= 32'h0;  // ACK
          end

          CMD_GET_STATUS: begin
            // Return status register
            rsp_payload_outputs_0 <= {24'h0, status};
          end

          CMD_READ_RESULT: begin
            // Phase 5: Return cached result for this neuron
            `ifdef DEBUG
            $display("[CFU_AXI] CMD_READ_RESULT: neuron %d, returning layer_results[%d]=%d",
                     neuron_config[15:0], neuron_config[15:0], $signed(layer_results[neuron_config[15:0]]));
            `endif
            rsp_payload_outputs_0 <= layer_results[neuron_config[15:0]];
            result_ready <= 1'b0;  // Mark as consumed
          end

          CMD_WRITE_RESULT: begin
            // Write result via DMA: dst_addr
            output_dma_dst_addr <= cmd_payload_inputs_0;
            output_dma_byte_count <= 16'h4;  // INT32
            output_dma_start <= 1'b1;
            rsp_payload_outputs_0 <= 32'h0;  // ACK
          end

          CMD_GET_ERROR: begin
            // Return error flags
            rsp_payload_outputs_0 <= {24'h0, error_flags};
          end

          CMD_CLEAR: begin
            // Clear state
            error_flags <= 8'h00;
            result_ready <= 1'b0;
            status <= STATUS_IDLE;
            // Phase 5: Reset layer_computed flag for new image
            layer_computed <= 1'b0;
            results_captured <= 8'h0;
            // Phase 5: CRITICAL FIX - Reset current_layer to force layer change detection
            current_layer <= 16'hFFFF;
            // Phase 5: Reset inter-layer transfer state
            transfer_prev_layer <= 1'b0;
            transfer_active <= 1'b0;
            transfer_idx <= 16'h0;
            // Phase 5: Reset DMA start signals
            input_dma_start <= 1'b0;
            weight_dma_start <= 1'b0;
            compute_start <= 1'b0;
            // Phase 5: CRITICAL FIX - Clear input/bias buffers for new image
            clear_buffers <= 1'b1;
            $display("[STATE_RESET] CMD_CLEAR: layer_computed=0, current_layer=0xFFFF, clearing buffers");
            rsp_payload_outputs_0 <= 32'h0;  // ACK
          end

          CMD_RESET: begin
            // Full reset (will be handled by reset signal)
            rsp_payload_outputs_0 <= 32'h0;  // ACK
          end

          default: begin
            // Unknown command
            rsp_payload_outputs_0 <= 32'hFFFFFFFF;
          end
        endcase
      end else if (rsp_valid && rsp_ready) begin
        // Response consumed
        rsp_valid <= 1'b0;
      end
    end
  end

endmodule
