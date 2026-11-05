<!---

This file is used to generate your project datasheet. Please fill in the information below and delete any unused
sections.

You can also include images in this folder and reference them in the markdown. Each image must be less than
512 kb in size, and the combined size of all images must be less than 1 MB.
-->

## How it works

Tiny ML ALU is an 8-bit signed (INT8) ALU built around the operations that neural-network inference uses most. Besides normal add, subtract and logic, it has saturating add and subtract (results clamp to 127 or -128 instead of wrapping), ReLU, max, min, absolute difference, rounded average, and a packed dot product that treats each byte as two signed 4-bit numbers and computes `a_lo*b_lo + a_hi*b_hi`. That last one is the core step of a quantized matrix multiply.

The ALU holds two operand registers, A and B. A 2-bit command on `uio[5:4]` loads either register from the data bus `ui_in`, or writes the current result back into A. Writeback lets you chain operations on chip. For example, you can repeatedly add B into A to accumulate, then apply ReLU, without reloading anything.

All input pins are captured into registers on each clock edge before they are used, so every timing path starts and ends at a flip-flop. Commands take effect one clock after you present them, and the result is registered too, so it appears on `uo_out` two clocks after the operands or opcode change.

Writeback happens once each time the command changes to `11`. Holding `11` for many clocks still adds only once, so an accumulate step is safe even when the command is held by hand. To write back again, return the command to `00` for at least one clock first. Two flags are also registered: `zero` (result is 0) and `overflow` (signed overflow for ADD/SUB, or the result was clamped for the saturating ops).

| Opcode | Name   | Result |
|--------|--------|--------|
| 0  | ADD    | A + B (wraps), overflow flag set on signed overflow |
| 1  | SUB    | A - B (wraps), overflow flag set on signed overflow |
| 2  | ADDS   | A + B, saturated to [-128, 127] |
| 3  | SUBS   | A - B, saturated to [-128, 127] |
| 4  | MAX    | max(A, B), signed |
| 5  | MIN    | min(A, B), signed |
| 6  | RELU   | max(A, 0) |
| 7  | ABSD   | \|A - B\|, saturated to 127 |
| 8  | AVG    | (A + B + 1) >> 1, signed, rounds up |
| 9  | DOT4   | A[3:0]·B[3:0] + A[7:4]·B[7:4] as signed 4-bit lanes, saturated |
| 10 | AND    | A & B |
| 11 | OR     | A \| B |
| 12 | XOR    | A ^ B |
| 13 | POPCNT | number of 1 bits in A |
| 14 | CLZ    | leading zeros in A (8 if A = 0) |
| 15 | ROTL   | A rotated left by B[2:0] |

| Command `uio[5:4]` | Action on the next clock edge |
|---|---|
| 00 | hold |
| 01 | A <= ui_in |
| 10 | B <= ui_in |
| 11 | A <= result of the current opcode (writeback), once per command |

## How to test

1. Hold `rst_n` low for a few clocks, then release it.
2. Put a value on `ui_in` and set `uio[5:4] = 01` for one clock to load A.
3. Put a value on `ui_in` and set `uio[5:4] = 10` for one clock to load B.
4. Set the opcode on `uio[3:0]` with `uio[5:4] = 00`. Wait two clocks and read the result on `uo_out`, overflow on `uio[6]` and zero on `uio[7]`.
5. To accumulate, keep the opcode set (for example 2, ADDS) and set `uio[5:4] = 11`, then back to `00`. Each change to `11` replaces A with A + B exactly once, however long it is held.

Example: load A = 3 (0x03) and B = 0x21 (lanes 1 and 2), then opcode 9 (DOT4) gives 3·1 + 0·2 = 3.

## External hardware

No extra parts are needed beyond the Tiny Tapeout demo board, but the board's DIP switches only reach `ui_in`. The control signals on the `uio` pins have to be driven another way:

- **Recommended:** drive and read every pin from the demo board's RP2040 using the Tiny Tapeout MicroPython SDK. A short script makes timing exact and repeatable.
- **By hand:** wire the `uio` pins to switches through the PMOD headers.

Read results from the RP2040 rather than the 7-segment display, since `uo_out` also drives the display segments and binary values light up as odd segment patterns.

Set up the data and address first, wait one clock, and only then assert the command or write enable. The board's signals are not synchronized to the chip's clock, so changing everything at once can let a clock edge capture a mix of old and new values.

`uio[7:6]` are outputs from this design (the zero and overflow flags). Configure those two pins as inputs on the RP2040 side, and do not drive them from a PMOD, or the two sides will fight over the pin.

If you use hand switches for the command, note that mechanical switches bounce and one flip can look like several quick changes, which would trigger several writebacks. Use the RP2040 to drive the command when accumulating.
