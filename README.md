````markdown
# NeuroZip — Offline AI-Powered Text Compression

NeuroZip is an **AI-based text compression engine** that uses a **tiny LSTM neural network** instead of traditional fixed algorithms (like gzip). You **train the model on your own text**, export it, and then use a fast C++ engine to compress and decompress files.

It’s built to be:

- ⚡ **Fast** — C++ core with an efficient range coder
- 🧠 **Learned** — a trained LSTM model predicts the next byte
- 🔐 **Offline** — no cloud, no remote calls
- 🧪 **Tested & Benchmarkable** — unit tests, integration tests, and corpus benchmarks
- 🌐 **User-Friendly** — CLI tools, a FastAPI backend, and a React web UI

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [How NeuroZip Works (High-Level)](#how-neurozip-works-high-level)
3. [Prerequisites](#prerequisites)
4. [Building the C++ Core & CLI](#building-the-c-core--cli)
   - [Recommended: Windows (MSYS2 MinGW64)](#recommended-windows-msys2-mingw64)
   - [Linux / macOS (generic)](#linux--macos-generic)
5. [Training the LSTM Model (Python)](#training-the-lstm-model-python)
6. [Exporting the Model to tiny_lstmbin](#exporting-the-model-to-tiny_lstmbin)
7. [Using the Command-Line Tools](#using-the-command-line-tools)
   - [`neurozip` — compress](#neurozip--compress)
   - [`neurounzip` — decompress](#neurounzip--decompress)
   - [`neurozip-inspect` — inspect metadata](#neurozip-inspect--inspect-metadata)
8. [Running the FastAPI Backend](#running-the-fastapi-backend)
9. [Running the React Web UI](#running-the-react-web-ui)
10. [Tests and Benchmarks](#tests-and-benchmarks)
11. [Common Issues & Troubleshooting](#common-issues--troubleshooting)

---

## Project Structure

From the project root:

```text
neurozip/
  CMakeLists.txt           # Top-level CMake config
  include/
    neurozip/
      neurozip.h           # Public umbrella header
  src/
    core/                  # Range coder, file format, model interface
    models/                # Tiny LSTM implementation
    api/                   # C and C++ API wrappers
    cli/                   # Command-line tools (neurozip, neurounzip, etc.)
  python/
    neurozip/
      __init__.py
      tokenize.py
      dataset.py
      model.py             # TinyLSTM PyTorch model
      train.py             # Training script
      export.py            # PyTorch → binary export
      evaluate.py          # Evaluation metrics (bpb, etc.)
  backend/
    server.py              # FastAPI backend
    requirements.txt
  web/
    package.json
    vite.config.js
    index.html
    src/
      main.jsx
      App.jsx
      api.js
      components/
        Header.jsx
        TextArea.jsx
        StatsPanel.jsx
  tests/
    unit/
      test_codec.cpp
      test_file_format.cpp
      test_model_interface.cpp
    integration/
      test_cli.py
      test_roundtrip.cpp
  benchmarks/
    bench_cli.py
    datasets/
      enwiki_sample.txt
      code_snippets.txt
      logs_sample.txt
  tools/
    export_model.py        # CLI wrapper: PyTorch → tiny_lstm.bin
    inspect_model.py
    generate_test_corpus.py
```
````

---

## How NeuroZip Works (High-Level)

1. **Train a small LSTM model in Python**

   - It reads text data and learns to predict the probability of each next byte (0–255).
   - A better predictor → better compression.

2. **Export the trained model to a binary file (`tiny_lstm.bin`)**

   - This file contains all the neural network weights in a compact format C++ can read.

3. **Compress with C++**

   - The C++ core uses your model + a range coder to encode text into compressed `.nzp` format.
   - `.nzp` includes a header with model ID, model hash, original size, checksum, etc.

4. **Decompress with C++**

   - Given the same model file and an `.nzp` file, it reconstructs the original text exactly.

5. **Frontend & Backend (Optional)**

   - A FastAPI backend exposes `/compress` and `/decompress` over HTTP.
   - A React app lets you paste text and visually see compressed output and stats.

---

## Prerequisites

### General

- **Git** (optional but recommended)
- **Python 3.9+**
- **CMake 3.16+**
- **A C++17 compiler**

  - On Windows: MSYS2 MinGW-w64 **or** MSVC
  - On Linux: GCC or Clang
  - On macOS: Xcode / Apple Clang

### Python Packages

From project root:

```bash
pip install torch fastapi uvicorn pydantic pytest numpy
```

_(On CPU-only machines, use CPU PyTorch builds.)_

### Node / npm (for the React UI)

From `web/`:

```bash
npm install
```

---

## Building the C++ Core & CLI

### Recommended: Windows (MSYS2 MinGW64)

**Do NOT use Git Bash for building**. Use **MSYS2 MinGW64**.

1. Install MSYS2
   Download from: [https://www.msys2.org/](https://www.msys2.org/)

2. Open **MSYS2 MinGW 64-bit**
   (Start Menu → MSYS2 MinGW 64-bit)

3. Install toolchain:

```bash
pacman -Syu            # update packages
pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake mingw-w64-x86_64-make
```

4. Build NeuroZip:

```bash
cd /c/Users/admin/python/neurozip   # adjust path to your clone
rm -rf build
mkdir build
cd build

cmake -G "MinGW Makefiles" ..
mingw32-make -j8
```

When it finishes, you should see:

```text
[100%] Built target neurozip_core
[100%] Built target neurozip
[100%] Built target neurounzip
[100%] Built target neurozip-inspect
```

Executables are at:

```text
build/src/cli/neurozip.exe
build/src/cli/neurounzip.exe
build/src/cli/neurozip-inspect.exe
```

### Linux / macOS (generic)

From project root:

```bash
mkdir build
cd build
cmake ..
make -j8
```

Executables will be in:

```text
build/src/cli/neurozip
build/src/cli/neurounzip
build/src/cli/neurozip-inspect
```

---

## Training the LSTM Model (Python)

You need a text corpus to train on. Create a file, for example:

```text
data/corpus.txt
```

Fill it with whatever you want NeuroZip to specialize in (natural language, logs, code, etc.).

From the project root:

```bash
python -m python.neurozip.train \
  --data data/corpus.txt \
  --seq-len 256 \
  --hidden-size 256 \
  --batch-size 64 \
  --epochs 2 \
  --lr 0.001 \
  --output model_checkpoint.pt
```

**Arguments:**

- `--data`: Path to the training text/binary file.
- `--seq-len`: Sequence length (context window).
- `--batch-size`: Training batch size.
- `--hidden-size`: LSTM hidden state size (must match C++ expectations).
- `--epochs`: Number of passes over the data.
- `--lr`: Learning rate.
- `--output`: Path to save the PyTorch checkpoint (`.pt`).

This produces `model_checkpoint.pt`, which contains:

- `hidden_size`
- `model_state` (PyTorch state_dict)

---

## Exporting the Model to `tiny_lstm.bin`

The C++ engine **cannot** read `.pt` files directly. You must export the model to the specific binary format.

From project root:

```bash
python -m tools.export_model   --input model_checkpoint.pt   --output tiny_lstm.bin
```

This writes `tiny_lstm.bin` in the project root (or wherever you run the command).

You can also keep models organized, e.g.:

```bash
mkdir models
python tools/export_model.py \
  --input model_checkpoint.pt \
  --output models/tiny_lstm.bin
```

---

## Using the Command-Line Tools

The CLI tools are in:

```text
build/src/cli/
```

On Windows (MSYS2 MinGW64):

```bash
cd /c/Users/admin/python/neurozip/build

./src/cli/neurozip.exe -m ../tiny_lstm.bin ../test.txt
./src/cli/neurounzip.exe -m ../tiny_lstm.bin test.txt.nzp
```

You can also add them to your PATH (optional):

```bash
export PATH="$PWD/src/cli:$PATH"
```

Then use just `neurozip` and `neurounzip`.

### `neurozip` — compress

**Usage:**

```bash
neurozip [-v] -m <model.bin> [-o <output.nzp>] <input-file>
```

- `-m <model.bin>`: Path to `tiny_lstm.bin` (required).
- `-o <file>`: Output `.nzp` file name (optional; defaults to `<input>.nzp`).
- `-v`: Verbose logging.

**Example:**

```bash
neurozip -m tiny_lstm.bin myfile.txt
```

This creates:

```text
myfile.txt.nzp
```

### `neurounzip` — decompress

**Usage:**

```bash
neurounzip [-v] -m <model.bin> [-o <output.txt>] <input-file.nzp>
```

- `-m <model.bin>`: Same model used to compress.
- `-o <file>`: Output file (optional; defaults to stripping `.nzp`).
- `-v`: Verbose logging.

**Example:**

```bash
neurounzip -m tiny_lstm.bin myfile.txt.nzp
```

This reconstructs `myfile.txt` (or `myfile.txt.out` depending on options).

### `neurozip-inspect` — inspect metadata

**Usage:**

```bash
neurozip-inspect <file.nzp>
```

Outputs info such as:

- Magic & version
- Model ID
- Model hash
- Original size
- CRC32 checksum
- Payload length

Useful for debugging and verifying compatibility.

---

## Running the FastAPI Backend

The backend exposes HTTP endpoints:

- `POST /compress` — compress text
- `POST /decompress` — decompress Base64-encoded data

### 1. Install dependencies

From `backend/`:

```bash
cd backend
pip install -r requirements.txt
```

or directly:

```bash
pip install fastapi uvicorn pydantic
```

### 2. Configure model & binary paths

Open `backend/server.py` and make sure these paths are correct:

```python
NEUROZIP_BIN = r"C:\Users\admin\python\neurozip\build\src\cli\neurozip.exe"
NEUROUNZIP_BIN = r"C:\Users\admin\python\neurozip\build\src\cli\neurounzip.exe"
MODEL_PATH = r"C:\Users\admin\python\neurozip\tiny_lstm.bin"
```

Adjust them **to match your system**.

### 3. Run the server

From `backend/`:

```bash
uvicorn server:app --reload --port 5000
```

The backend now listens on:

```text
http://localhost:5000
```

#### API Contracts

- `POST /compress`

  Request JSON:

  ```json
  { "text": "hello world" }
  ```

  Response JSON:

  ```json
  { "data": "<base64-of-compressed-bytes>" }
  ```

- `POST /decompress`

  Request JSON:

  ```json
  { "data": "<base64-of-compressed-bytes>" }
  ```

  Response JSON:

  ```json
  { "text": "hello world" }
  ```

---

## Running the React Web UI

The React app calls the FastAPI backend to compress/decompress text from a browser.

### 1. Install dependencies

From `web/`:

```bash
cd web
npm install
```

### 2. Ensure backend is running

Backend must be running on `http://localhost:5000` (see previous section).

### 3. Run the dev server

```bash
npm run dev
```

By default, Vite runs on `http://localhost:3000`.

Open the URL in your browser.

### 4. Using the UI

- Top toggle: **Compress / Decompress**
- Input box:

  - In **Compress** mode: paste plain text.
  - In **Decompress** mode: paste Base64 text from a previous compression.

- Click **Run**:

  - Compress:

    - Sends `{ text }` to `/compress`
    - Shows Base64 result in the output text area
    - Shows stats: input bytes, output bytes, ratio, time

  - Decompress:

    - Sends `{ data: base64 }` to `/decompress`
    - Shows decompressed text in the output
    - Shows basic stats

---

## Tests and Benchmarks

### C++ Unit Tests

Unit tests are in `tests/unit/`:

- `test_codec.cpp`
- `test_file_format.cpp`
- `test_model_interface.cpp`

If you’ve integrated them with CTest in CMake:

```bash
cd build
ctest --verbose
```

Or you can compile and run them manually if you add them to CMake.

### Integration Tests

`tests/integration/` contains:

- `test_cli.py` — calls `neurozip` and `neurounzip` and checks roundtrip.
- `test_roundtrip.cpp` — directly uses the C++ API to compress & decompress.

To run Python integration tests (from project root):

```bash
pytest tests/integration/test_cli.py -q
```

Make sure:

- `neurozip` and `neurounzip` are on PATH, or adjust the script.
- `tiny_lstm.bin` exists.

### Benchmarks

`benchmarks/bench_cli.py` runs basic speed and ratio tests on sample datasets.

From project root:

```bash
python benchmarks/bench_cli.py
```

It expects files in:

```text
benchmarks/datasets/enwiki_sample.txt
benchmarks/datasets/code_snippets.txt
benchmarks/datasets/logs_sample.txt
```

You can replace them with your own corpora for more realistic metrics.

---

## Common Issues & Troubleshooting

### 1. `neurozip: command not found`

**Cause:** Executable not on PATH or you’re in the wrong shell.

**Fix:**

- Use full path:

  ```bash
  ./src/cli/neurozip.exe -m tiny_lstm.bin myfile.txt
  ```

- Or export PATH (from `build/`):

  ```bash
  export PATH="$PWD/src/cli:$PATH"
  ```

### 2. `tiny_lstm.bin` not found

**Cause:** You haven’t exported the model yet, or it’s in another directory.

**Fix:**

1. Train:

   ```bash
   python -m neurozip.train --data data/corpus.txt --output model_checkpoint.pt
   ```

2. Export:

   ```bash
   python tools/export_model.py --input model_checkpoint.pt --output tiny_lstm.bin
   ```

3. Run `ls` or `dir` to confirm the file location and update paths accordingly.

### 3. Backend complains: model file not found

**Cause:** `MODEL_PATH` in `backend/server.py` does not match the real location.

**Fix:** Edit:

```python
MODEL_PATH = r"C:\exact\path\to\tiny_lstm.bin"
```

### 4. `cmake: command not found` or `make: command not found` (on Windows)

**Cause:** Using Git Bash without a compiler.

**Fix:** Use **MSYS2 MinGW64** (not Git Bash), and install toolchain:

```bash
pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake mingw-w64-x86_64-make
```

Then build with:

```bash
cmake -G "MinGW Makefiles" ..
mingw32-make -j8
```

### 5. Model mismatch errors

If you see errors about model ID / hash mismatch during decompression:

- You are using a different `tiny_lstm.bin` than the one used for compression.
- Make sure you use **the exact same model file** to compress and decompress.

---

## Summary

NeuroZip is a fully working, end-to-end system:

- 🧠 **Train** a tiny LSTM model on your text with Python
- 📦 **Export** it to `tiny_lstm.bin`
- 🔧 **Build** the C++ core with CMake
- 💻 **Compress & Decompress** via CLI (`neurozip`, `neurounzip`)
- 🌐 **Serve** compression over HTTP via FastAPI
- 🖥️ **Interact** with it using a clean React web UI
- 🧪 **Validate** with tests & benchmarks

You now have a **ship-ready AI compression project** that you can extend, optimize, or open-source.

```

```
