import os
import subprocess
import tempfile

# 🚨 CRITICAL: Check and update this path 
# It must point to the compiled neurozip.exe file.
NEUROZIP = r"C:\Users\admin\python\neurozip\build\src\cli\neurozip.exe"

# 🚨 CRITICAL: Check and update this path 
# It must point to the compiled neurounzip.exe file.
NEUROUNZIP = r"C:\Users\admin\python\neurozip\build\src\cli\neurounzip.exe"

# 🚨 CRITICAL: Check and update this path 
# It must point to the exported tiny_lstm.bin model.
MODEL = r"C:\Users\admin\python\neurozip\tiny_lstm.bin"

def run(cmd):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    return p.returncode, out, err

def test_cli_roundtrip():
    text = "Hello from CLI test!"

    # Input temp file
    with tempfile.NamedTemporaryFile(delete=False, mode="w") as f:
        f.write(text)
        inpath = f.name

    outpath = inpath + ".nzp"
    restored = outpath + ".txt"

    code, _, err = run([NEUROZIP, "-m", MODEL, "-o", outpath, inpath])
    assert code == 0, err

    code, _, err = run([NEUROUNZIP, "-m", MODEL, "-o", restored, outpath])
    assert code == 0, err

    with open(restored, "r") as f:
        assert f.read() == text

    os.remove(inpath)
    os.remove(outpath)
    os.remove(restored)
