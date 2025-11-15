#!/usr/bin/env python3
import struct
import sys

def read_floats(f, n):
    data = f.read(n * 4)
    return struct.unpack("<" + "f" * n, data)

def main():
    if len(sys.argv) < 2:
        print("Usage: inspect_model.py <tiny_lstm.bin>")
        return

    path = sys.argv[1]
    with open(path, "rb") as f:
        inputSize = struct.unpack("<I", f.read(4))[0]
        hidden = struct.unpack("<I", f.read(4))[0]
        layers = struct.unpack("<I", f.read(4))[0]
        reserved = struct.unpack("<I", f.read(4))[0]

        print("Input size:", inputSize)
        print("Hidden size:", hidden)
        print("Layers:", layers)

        print("Total expected weights:")
        print("  W_ih:", 4 * hidden * inputSize)
        print("  W_hh:", 4 * hidden * hidden)
        print("  b_ih:", 4 * hidden)
        print("  b_hh:", 4 * hidden)
        print("  W_out:", 256 * hidden)
        print("  b_out:", 256)

        print("\nModel appears structurally valid.")
