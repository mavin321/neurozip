# neurozip — Offline AI-Powered Text Compression

**neurozip** is a machine-learned text compression engine built around a tiny
locally-trained LSTM neural network. It acts like a drop-in replacement for gzip
and other entropy coders, but uses a learned probability model to achieve better
compression on domain-specific text.

## Features

- Offline (no cloud, no remote calls)
- High-speed C/C++ encoder/decoder
- Tiny LSTM model trained in Python with PyTorch
- Deterministic bit-exact range coder
- CLI tools (`neurozip`, `neurounzip`)
- Web demo (optional, local only)
- Simple C, C++, and Python APIs

## Repository Structure
