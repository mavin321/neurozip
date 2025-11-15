import base64
import subprocess
import tempfile
import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------

# Path to the neurozip CLI binaries
# 🚨 CRITICAL: Check and update this path 
# It must point to the compiled neurozip.exe file.
NEUROZIP_BIN = r"C:\Users\admin\python\neurozip\build\src\cli\neurozip.exe"

# 🚨 CRITICAL: Check and update this path 
# It must point to the compiled neurounzip.exe file.
NEUROUNZIP_BIN = r"C:\Users\admin\python\neurozip\build\src\cli\neurounzip.exe"

# 🚨 CRITICAL: Check and update this path 
# It must point to the exported tiny_lstm.bin model.
MODEL_PATH = r"C:\Users\admin\python\neurozip\tiny_lstm.bin"

# Ensure model exists
if not os.path.exists(MODEL_PATH):
    print("WARNING: MODEL FILE NOT FOUND:", MODEL_PATH)
    print("Backend will not work until you export a model to tiny_lstm.bin")


app = FastAPI(title="NeuroZip Backend")

# Allow React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # you can restrict to http://localhost:3000
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------------------
# Request Models
# -------------------------------------------------------------------

class CompressRequest(BaseModel):
    text: str


class DecompressRequest(BaseModel):
    data: str  # base64 string


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------

def run_cmd(cmd: list[str]) -> tuple[int, str, str]:
    """Run a system command and return (returncode, stdout, stderr)."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    out, err = proc.communicate()
    return proc.returncode, out, err


# -------------------------------------------------------------------
# API ENDPOINTS
# -------------------------------------------------------------------

@app.post("/compress")
def compress(req: CompressRequest):
    """
    Input:
        { "text": "hello world" }
    Output:
        { "data": "<base64>" }
    """

    # Create temp file for input
    with tempfile.NamedTemporaryFile(delete=False) as tmp_in:
        tmp_in.write(req.text.encode("utf-8"))
        tmp_in.flush()
        input_path = tmp_in.name

    # Output temp file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_out:
        output_path = tmp_out.name

    # Run the neurozip CLI
    cmd = [
        NEUROZIP_BIN,
        "-m", MODEL_PATH,
        "-o", output_path,
        input_path
    ]

    code, out, err = run_cmd(cmd)

    if code != 0:
        return {"error": f"Compression failed: {err}"}

    # Read compressed file
    with open(output_path, "rb") as f:
        raw = f.read()

    # Encode as base64 for JSON transport
    b64 = base64.b64encode(raw).decode("ascii")

    # Cleanup
    os.remove(input_path)
    os.remove(output_path)

    return {
        "data": b64
    }


@app.post("/decompress")
def decompress(req: DecompressRequest):
    """
    Input:
        { "data": "<base64>" }
    Output:
        { "text": "..." }
    """

    raw = base64.b64decode(req.data)

    # Temp compressed file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_in:
        tmp_in.write(raw)
        tmp_in.flush()
        input_path = tmp_in.name

    # Temp output file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_out:
        output_path = tmp_out.name

    # Run the CLI
    cmd = [
        NEUROUNZIP_BIN,
        "-m", MODEL_PATH,
        "-o", output_path,
        input_path
    ]

    code, out, err = run_cmd(cmd)

    if code != 0:
        return {"error": f"Decompression failed: {err}"}

    # Read decompressed file
    with open(output_path, "rb") as f:
        text = f.read().decode("utf-8", errors="replace")

    # Cleanup
    os.remove(input_path)
    os.remove(output_path)

    return {
        "text": text
    }


@app.get("/")
def root():
    return {"status": "ok", "message": "NeuroZip backend running"}
