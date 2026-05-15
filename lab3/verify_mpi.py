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

def run_experiment(N, procs, exe_path, verify=False):
    A_file = "A_temp.txt"
    B_file = "B_temp.txt"
    C_file = "C_temp.txt"

    matA = generate_matrix(A_file, N)
    matB = generate_matrix(B_file, N)

    cmd = ["mpiexec", "-n", str(procs), exe_path, A_file, B_file, C_file]
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
            print(f"Verification FAILED for N={N}, procs={procs}")
            sys.exit(1)
        else:
            print(f"Verification OK for N={N}, procs={procs}")

    for f in [A_file, B_file, C_file]:
        if os.path.exists(f):
            os.remove(f)

    return exec_time

if __name__ == "__main__":
    exe = "./matrix_mpi" if os.name != 'nt' else "matrix_mpi.exe"
    if not os.path.exists(exe):
        print(f"Executable '{exe}' not found. Compile with: mpicxx -O3 matrix_mpi.cpp -o matrix_mpi")
        sys.exit(1)

    sizes = [200, 400, 800, 1200, 1600, 2000]
    procs_list = [1, 2, 4, 6, 8, 12]

    print("Quick verification for N=400, procs=4...")
    run_experiment(400, 4, exe, verify=True)
    print("Verification passed.\n")

    header = "N \\ procs" + "".join(f"{p:>10}" for p in procs_list)
    print(header)
    print("-" * len(header))

    results = {}
    for n in sizes:
        row = f"{n:<12}"
        for p in procs_list:
            elapsed = run_experiment(n, p, exe, verify=False)
            results[(n, p)] = elapsed
            row += f"{elapsed:10.5f}"
        print(row)

    with open("results_mpi.txt", "w") as f:
        for n in sizes:
            f.write(f"{n}")
            for p in procs_list:
                f.write(f" {results[(n,p)]}")
            f.write("\n")
