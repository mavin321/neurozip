#!/usr/bin/env python3
import argparse
from python.neurozip.export import export_tiny_lstm

def main():
    ap = argparse.ArgumentParser(description="Export PyTorch LSTM -> neurozip binary")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    export_tiny_lstm(args.input, args.output)

if __name__ == "__main__":
    main()
