import numpy as np
import subprocess
import os
import sys
import re

def generate_matrix(filename, N):
    mat = np.random.rand(N, N).astype(np.float64)
    with open(filename, 'w') as f:
        f.write(f"{N}\n")
        np.savetxt(f, mat, fmt="%.6f")
    return mat

def read_matrix(filename):
    with open(filename, 'r') as f:
        N = int(f.readline().strip())
        mat = np.loadtxt(f)
    return mat

def run_experiment(N, threads, exe_path, verify=False):
    A_file = "A_temp.txt"
    B_file = "B_temp.txt"
    C_file = "C_temp.txt"

    matA = generate_matrix(A_file, N)
    matB = generate_matrix(B_file, N)

    cmd = [exe_path, A_file, B_file, C_file, str(threads)]
    result = subprocess.run(cmd, capture_output=True, text=True)

    match = re.search(r"Execution time = ([0-9.]+)", result.stdout)
    if not match:
        print("Error: could not parse execution time")
        print(result.stdout)
        sys.exit(1)
    exec_time = float(match.group(1))

    if verify:
        expected = np.dot(matA, matB)
        actual = read_matrix(C_file)
        if not np.allclose(expected, actual, atol=1e-5):
            print(f"Verification FAILED for N={N}, threads={threads}")
            sys.exit(1)
        else:
            print(f"Verification OK for N={N}, threads={threads}")

    os.remove(A_file)
    os.remove(B_file)
    os.remove(C_file)
    return exec_time

if __name__ == "__main__":
    exe = "./matrix_omp" if os.name != 'nt' else "matrix_omp.exe"
    if not os.path.exists(exe):
        print(f"Executable '{exe}' not found. Compile with: g++ -O3 -fopenmp matrix_omp.cpp -o matrix_omp")
        sys.exit(1)

    sizes = [200, 400, 800, 1200, 1600, 2000]
    threads_list = [1, 2, 4, 6, 8, 12]

    print("Quick verification for N=400, threads=4...")
    run_experiment(400, 4, exe, verify=True)
    print("Verification passed.\n")

    header = "N \\ threads" + "".join(f"{t:>10}" for t in threads_list)
    print(header)
    print("-" * len(header))

    results = {}
    for n in sizes:
        row = f"{n:<12}"
        for t in threads_list:
            t_used = min(t, os.cpu_count() or 1)
            elapsed = run_experiment(n, t_used, exe, verify=False)
            results[(n, t)] = elapsed
            row += f"{elapsed:10.5f}"
        print(row)

    with open("results_omp.txt", "w") as f:
        for n in sizes:
            f.write(f"{n}")
            for t in threads_list:
                f.write(f" {results[(n,t)]}")
            f.write("\n")
