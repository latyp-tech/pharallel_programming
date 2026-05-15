import numpy as np
import subprocess
import sys
import os
import re

def genMatrix(fname, N):
    mat = np.random.rand(N, N).astype(np.float64)
    with open(fname, 'w') as f:
        f.write(f"{N}\n")
        np.savetxt(f, mat, fmt="%.6f")
    return mat

def readMatrix(fname):
    with open(fname, 'r') as f:
        N = int(f.readline().strip())
        mat = np.loadtxt(f)
    return mat

def runTest(N, block, exePath, verify=False):
    fA = "A_tmp.txt"
    fB = "B_tmp.txt"
    fC = "C_tmp.txt"

    matA = genMatrix(fA, N)
    matB = genMatrix(fB, N)

    proc = subprocess.run([exePath, fA, fB, fC, str(block)], capture_output=True, text=True)

    match = re.search(r"Execution time: ([0-9.]+)", proc.stdout)
    if not match:
        print("Error: Time not found in output.")
        sys.exit(1)
    duration = float(match.group(1))

    if verify:
        ref = np.dot(matA, matB)
        res = readMatrix(fC)
        if not np.allclose(ref, res, atol=1e-5):
            print(f"Verification failed for N={N}, block={block}x{block}")
            sys.exit(1)
        else:
            print(f"Verification OK for N={N}, block={block}x{block}")

    for f in [fA, fB, fC]:
        if os.path.exists(f):
            os.remove(f)

    return duration

if __name__ == "__main__":
    execName = "gpu_multiply" if os.name != 'nt' else "gpu_multiply.exe"
    execPath = os.path.join(".", execName)

    if not os.path.exists(execPath):
        print(f"Error: Executable '{execPath}' not found.")
        print("Compile: nvcc -O3 gpu_multiply.cu -o gpu_multiply")
        sys.exit(1)

    dims = [200, 400, 800, 1200, 1600, 2000]
    blocks = [8, 16, 32]

    print("Quick verification for N=400, block=16x16...")
    runTest(400, 16, execPath, verify=True)
    print("Verification passed.\n")

    header = "N\\Block" + "".join(f"{b}x{b:>8}" for b in blocks)
    print(header)
    print("-" * len(header))

    outcomes = {}
    for n in dims:
        line = f"{n:<8}"
        for b in blocks:
            elapsed = runTest(n, b, execPath, verify=False)
            outcomes[(n, b)] = elapsed
            line += f"{elapsed:10.5f}"
        print(line)

    with open("cuda_results.txt", "w") as f:
        for n in dims:
            f.write(f"{n}")
            for b in blocks:
                f.write(f" {outcomes[(n,b)]}")
            f.write("\n")
