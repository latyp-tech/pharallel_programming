import numpy as np
import subprocess
import os
import sys

def generate_matrix(filename, N):
    """Генерирует случайную матрицу N x N и сохраняет в файл"""
    mat = np.random.rand(N, N)
    with open(filename, 'w') as f:
        f.write(f"{N}\n")
        np.savetxt(f, mat, fmt="%.6f")
    return mat

def read_matrix_from_file(filename):
    """Читает матрицу из файла, созданного C++ программой"""
    with open(filename, 'r') as f:
        N = int(f.readline().strip())
        mat = np.loadtxt(f)
    return mat

def run_test(N, exe_path):
    print(f"\n===== Testing N = {N} =====")
    A_file = "A_temp.txt"
    B_file = "B_temp.txt"
    C_file = "C_temp.txt"


    matA = generate_matrix(A_file, N)
    matB = generate_matrix(B_file, N)

    result = subprocess.run([exe_path, A_file, B_file, C_file],
                            capture_output=True, text=True)
    print(result.stdout)


    expected = np.dot(matA, matB)


    actual = read_matrix_from_file(C_file)


    if np.allclose(expected, actual, atol=1e-5):
        print("✓ Verification PASSED")
    else:
        print("✗ Verification FAILED")


    os.remove(A_file)
    os.remove(B_file)
    os.remove(C_file)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        exe = sys.argv[1]
    else:
        exe = "./matrix" if os.name != 'nt' else "matrix.exe"

    if not os.path.exists(exe):
        print(f"Executable '{exe}' not found. Compile matrix.cpp first.")
        sys.exit(1)

    sizes = [200, 400, 800, 1200, 1600, 2000]
    for n in sizes:
        run_test(n, exe)
