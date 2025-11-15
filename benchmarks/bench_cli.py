import subprocess
import time
import os

NEUROZIP = "neurozip"
NEUROUNZIP = "neurounzip"
MODEL = "tiny_lstm.bin"

DATASETS = [
    "datasets/enwiki_sample.txt",
    "datasets/code_snippets.txt",
    "datasets/logs_sample.txt",
]

def run(cmd):
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    return p.returncode, out, err, (time.time() - t0)

for ds in DATASETS:
    print(f"\n=== Benchmarking {ds} ===")
    outpath = ds + ".nzp"

    code, _, err, t = run([NEUROZIP, "-m", MODEL, "-o", outpath, ds])
    if code != 0:
        print("ERROR:", err)
        continue
    size_in = os.path.getsize(ds)
    size_out = os.path.getsize(outpath)

    print("Input size:     ", size_in)
    print("Output size:    ", size_out)
    print("Ratio:          ", size_out / size_in)
    print("Compress time:  ", t, "sec")

    _, _, _, dt = run([NEUROUNZIP, "-m", MODEL, "-o", outpath + ".txt", outpath])
    print("Decompress time:", dt, "sec")
