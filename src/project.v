/*
 * Copyright (c) 2026 Zachary Johnson
 * SPDX-License-Identifier: Apache-2.0
 *
 * Tiny ML ALU: an 8-bit signed (INT8) ALU built around the operations
 * neural-network inference actually uses: saturating math, ReLU, max/min,
 * a packed 4-bit dot product, and a writeback path so results can be
 * accumulated without leaving the chip. All pins are registered on the
 * way in so every timing path starts and ends at a flip-flop, and a
 * writeback happens once per command even if the command is held.
 */

`default_nettype none

module tt_um_zjohnson2005_ml_alu (
    input  wire [7:0] ui_in,    // data bus: value to load into A or B
    output wire [7:0] uo_out,   // registered result
    input  wire [7:0] uio_in,   // [3:0] opcode, [5:4] command
    output wire [7:0] uio_out,  // [6] overflow/saturated flag, [7] zero flag
    output wire [7:0] uio_oe,
    input  wire       ena,
    input  wire       clk,
    input  wire       rst_n
);

    localparam OP_ADD    = 4'd0;
    localparam OP_SUB    = 4'd1;
    localparam OP_ADDS   = 4'd2;
    localparam OP_SUBS   = 4'd3;
    localparam OP_MAX    = 4'd4;
    localparam OP_MIN    = 4'd5;
    localparam OP_RELU   = 4'd6;
    localparam OP_ABSD   = 4'd7;
    localparam OP_AVG    = 4'd8;
    localparam OP_DOT4   = 4'd9;
    localparam OP_AND    = 4'd10;
    localparam OP_OR     = 4'd11;
    localparam OP_XOR    = 4'd12;
    localparam OP_POPCNT = 4'd13;
    localparam OP_CLZ    = 4'd14;
    localparam OP_ROTL   = 4'd15;

    localparam CMD_HOLD      = 2'b00;
    localparam CMD_LOAD_A    = 2'b01;
    localparam CMD_LOAD_B    = 2'b10;
    localparam CMD_WRITEBACK = 2'b11;

    reg [7:0] data_in;
    reg [3:0] opcode;
    reg [1:0] command;
    reg [1:0] previous_command;

    always @(posedge clk) begin
        if (!rst_n) begin
            data_in          <= 8'd0;
            opcode           <= 4'd0;
            command          <= CMD_HOLD;
            previous_command <= CMD_HOLD;
        end else begin
            data_in          <= ui_in;
            opcode           <= uio_in[3:0];
            command          <= uio_in[5:4];
            previous_command <= command;
        end
    end

    reg signed [7:0] operand_a;
    reg signed [7:0] operand_b;
    reg        [7:0] result;
    reg              overflow_flag;
    reg              zero_flag;

    function [7:0] saturate;
        input signed [9:0] value;
        begin
            if (value > 10'sd127)       saturate = 8'h7F;
            else if (value < -10'sd128) saturate = 8'h80;
            else                        saturate = value[7:0];
        end
    endfunction

    function did_saturate;
        input signed [9:0] value;
        begin
            did_saturate = (value > 10'sd127) || (value < -10'sd128);
        end
    endfunction

    wire signed [9:0] wide_a     = {{2{operand_a[7]}}, operand_a};
    wire signed [9:0] wide_b     = {{2{operand_b[7]}}, operand_b};
    wire signed [9:0] sum        = wide_a + wide_b;
    wire signed [9:0] difference = wide_a - wide_b;
    wire signed [9:0] abs_difference = difference[9] ? -difference : difference;

    wire signed [3:0] a_low  = operand_a[3:0];
    wire signed [3:0] a_high = operand_a[7:4];
    wire signed [3:0] b_low  = operand_b[3:0];
    wire signed [3:0] b_high = operand_b[7:4];
    wire signed [7:0] product_low  = a_low * b_low;
    wire signed [7:0] product_high = a_high * b_high;
    wire signed [9:0] dot_product  = {{2{product_low[7]}}, product_low}
                                   + {{2{product_high[7]}}, product_high};

    wire [3:0] population_count = {3'd0, operand_a[0]} + {3'd0, operand_a[1]}
                                + {3'd0, operand_a[2]} + {3'd0, operand_a[3]}
                                + {3'd0, operand_a[4]} + {3'd0, operand_a[5]}
                                + {3'd0, operand_a[6]} + {3'd0, operand_a[7]};

    reg [3:0] leading_zeros;
    always @(*) begin
        casez (operand_a)
            8'b1???????: leading_zeros = 4'd0;
            8'b01??????: leading_zeros = 4'd1;
            8'b001?????: leading_zeros = 4'd2;
            8'b0001????: leading_zeros = 4'd3;
            8'b00001???: leading_zeros = 4'd4;
            8'b000001??: leading_zeros = 4'd5;
            8'b0000001?: leading_zeros = 4'd6;
            8'b00000001: leading_zeros = 4'd7;
            default:     leading_zeros = 4'd8;
        endcase
    end

    wire signed [9:0] rounded_average = (sum + 10'sd1) >>> 1;

    wire [2:0]  rotate_amount = operand_b[2:0];
    wire [15:0] doubled_a     = {operand_a, operand_a} << rotate_amount;

    wire writeback_now = (command == CMD_WRITEBACK) && (previous_command != CMD_WRITEBACK);

    reg [7:0] next_result;
    reg       next_overflow;

    always @(*) begin
        next_overflow = 1'b0;
        case (opcode)
            OP_ADD: begin
                next_result   = sum[7:0];
                next_overflow = did_saturate(sum);
            end
            OP_SUB: begin
                next_result   = difference[7:0];
                next_overflow = did_saturate(difference);
            end
            OP_ADDS: begin
                next_result   = saturate(sum);
                next_overflow = did_saturate(sum);
            end
            OP_SUBS: begin
                next_result   = saturate(difference);
                next_overflow = did_saturate(difference);
            end
            OP_MAX:    next_result = (operand_a > operand_b) ? operand_a : operand_b;
            OP_MIN:    next_result = (operand_a < operand_b) ? operand_a : operand_b;
            OP_RELU:   next_result = operand_a[7] ? 8'd0 : operand_a;
            OP_ABSD: begin
                next_result   = saturate(abs_difference);
                next_overflow = did_saturate(abs_difference);
            end
            OP_AVG:    next_result = rounded_average[7:0];
            OP_DOT4: begin
                next_result   = saturate(dot_product);
                next_overflow = did_saturate(dot_product);
            end
            OP_AND:    next_result = operand_a & operand_b;
            OP_OR:     next_result = operand_a | operand_b;
            OP_XOR:    next_result = operand_a ^ operand_b;
            OP_POPCNT: next_result = {4'd0, population_count};
            OP_CLZ:    next_result = {4'd0, leading_zeros};
            OP_ROTL:   next_result = doubled_a[15:8];
            default:   next_result = 8'd0;
        endcase
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            operand_a     <= 8'd0;
            operand_b     <= 8'd0;
            result        <= 8'd0;
            overflow_flag <= 1'b0;
            zero_flag     <= 1'b1;
        end else begin
            case (command)
                CMD_LOAD_A:    operand_a <= data_in;
                CMD_LOAD_B:    operand_b <= data_in;
                CMD_WRITEBACK: if (writeback_now) operand_a <= next_result;
                CMD_HOLD:      ;
            endcase

            result        <= next_result;
            overflow_flag <= next_overflow;
            zero_flag     <= (next_result == 8'd0);
        end
    end

    assign uo_out  = result;
    assign uio_out = {zero_flag, overflow_flag, 6'b000000};
    assign uio_oe  = 8'b1100_0000;

    wire _unused = &{ena, uio_in[7:6], 1'b0};

endmodule
