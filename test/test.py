# SPDX-FileCopyrightText: © 2026 Zachary Johnson
# SPDX-License-Identifier: Apache-2.0

import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles

LOAD_A, LOAD_B, WRITEBACK = 0b01, 0b10, 0b11
OP_NAMES = ["ADD", "SUB", "ADDS", "SUBS", "MAX", "MIN", "RELU", "ABSD",
            "AVG", "DOT4", "AND", "OR", "XOR", "POPCNT", "CLZ", "ROTL"]


def signed8(x):
    x &= 0xFF
    return x - 256 if x & 0x80 else x


def signed4(x):
    x &= 0xF
    return x - 16 if x & 0x8 else x


def saturate(v):
    return max(-128, min(127, v))


def model(op, a, b):
    """Return (result as 0..255, overflow flag) for INT8 operands a, b."""
    sa, sb = signed8(a), signed8(b)
    overflow = 0
    if op == 0:
        r = sa + sb; overflow = int(r != saturate(r))
    elif op == 1:
        r = sa - sb; overflow = int(r != saturate(r))
    elif op == 2:
        overflow = int(sa + sb != saturate(sa + sb)); r = saturate(sa + sb)
    elif op == 3:
        overflow = int(sa - sb != saturate(sa - sb)); r = saturate(sa - sb)
    elif op == 4:
        r = max(sa, sb)
    elif op == 5:
        r = min(sa, sb)
    elif op == 6:
        r = max(sa, 0)
    elif op == 7:
        d = abs(sa - sb); overflow = int(d > 127); r = saturate(d)
    elif op == 8:
        r = (sa + sb + 1) >> 1
    elif op == 9:
        d = signed4(a) * signed4(b) + signed4(a >> 4) * signed4(b >> 4)
        overflow = int(d != saturate(d)); r = saturate(d)
    elif op == 10:
        r = a & b
    elif op == 11:
        r = a | b
    elif op == 12:
        r = a ^ b
    elif op == 13:
        r = bin(a).count("1")
    elif op == 14:
        r = 8 - a.bit_length()
    else:
        k = b & 7
        r = ((a << k) | (a >> (8 - k))) & 0xFF
    return r & 0xFF, overflow


async def reset(dut):
    clock = Clock(dut.clk, 10, unit="us")
    cocotb.start_soon(clock.start())
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 1)


async def load(dut, command, value):
    dut.ui_in.value = value
    dut.uio_in.value = command << 4
    await ClockCycles(dut.clk, 1)
    dut.uio_in.value = 0


async def run_op(dut, op):
    dut.uio_in.value = op
    await ClockCycles(dut.clk, 3)
    uio = int(dut.uio_out.value)
    return int(dut.uo_out.value), (uio >> 6) & 1, (uio >> 7) & 1


async def check(dut, op, a, b):
    await load(dut, LOAD_A, a)
    await load(dut, LOAD_B, b)
    result, overflow, zero = await run_op(dut, op)
    want, want_overflow = model(op, a, b)
    assert result == want, f"{OP_NAMES[op]} a={signed8(a)} b={signed8(b)}: got {result}, want {want}"
    assert overflow == want_overflow, f"{OP_NAMES[op]} a={signed8(a)} b={signed8(b)}: overflow {overflow}, want {want_overflow}"
    assert zero == int(want == 0), f"{OP_NAMES[op]} zero flag wrong"


@cocotb.test()
async def test_edge_cases(dut):
    await reset(dut)
    assert int(dut.uio_oe.value) == 0xC0
    edges = [0x00, 0x01, 0x7F, 0x80, 0xFF, 0x0F, 0xF0, 0x88, 0x77]
    for op in range(16):
        for a in edges:
            for b in edges:
                await check(dut, op, a, b)


@cocotb.test()
async def test_random(dut):
    await reset(dut)
    rng = random.Random(2026)
    for _ in range(600):
        await check(dut, rng.randrange(16), rng.randrange(256), rng.randrange(256))


@cocotb.test()
async def test_relu_dot_accumulate(dut):
    """Chain ops with writeback: A <- A + B three times, then ReLU it."""
    await reset(dut)
    await load(dut, LOAD_A, 10)
    await load(dut, LOAD_B, 50)
    dut.uio_in.value = 2  # ADDS
    for _ in range(3):
        await ClockCycles(dut.clk, 1)
        dut.uio_in.value = (WRITEBACK << 4) | 2
        await ClockCycles(dut.clk, 1)
        dut.uio_in.value = 2
    result, overflow, _ = await run_op(dut, 2)
    assert result == 127 and overflow == 1, "10+50+50+50 should saturate to 127 on the next add"

    await load(dut, LOAD_A, 0x80)  # -128
    result, _, zero = await run_op(dut, 6)  # RELU
    assert result == 0 and zero == 1
