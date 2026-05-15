#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <cstdlib>

using namespace std;

__global__ void matMultKernel(const double* A, const double* B, double* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        double val = 0.0;
        for (int k = 0; k < N; ++k) {
            val += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = val;
    }
}

bool loadMatrix(const string& path, vector<double>& mat, int& dim) {
    ifstream in(path);
    if (!in) return false;
    in >> dim;
    if (dim <= 0) return false;
    mat.resize(dim * dim);
    for (int i = 0; i < dim * dim; ++i)
        in >> mat[i];
    return true;
}

void saveMatrix(const string& path, const vector<double>& mat, int dim) {
    ofstream out(path);
    out << dim << "\n";
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j)
            out << mat[i * dim + j] << " ";
        out << "\n";
    }
}

int main(int argc, char* argv[]) {
    if (argc < 4 || argc > 5) {
        cerr << "Usage: " << argv[0] << " <A.txt> <B.txt> <C.txt> [block_size]\n";
        return 1;
    }

    string fileA = argv[1];
    string fileB = argv[2];
    string fileC = argv[3];

    int blockSize = 16;
    if (argc == 5) {
        blockSize = atoi(argv[4]);
        if (blockSize <= 0) blockSize = 16;
    }

    vector<double> h_A, h_B, h_C;
    int dimA, dimB;

    if (!loadMatrix(fileA, h_A, dimA) || !loadMatrix(fileB, h_B, dimB) || dimA != dimB) {
        cerr << "Error: Cannot read matrices or dimensions mismatch.\n";
        return 1;
    }

    int N = dimA;
    h_C.assign(N * N, 0.0);
    size_t bytes = N * N * sizeof(double);

    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaEvent_t startEv, stopEv;
    cudaEventCreate(&startEv);
    cudaEventCreate(&stopEv);

    cudaEventRecord(startEv);

    cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block(blockSize, blockSize);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    matMultKernel<<<grid, block>>>(d_A, d_B, d_C, N);

    cudaDeviceSynchronize();

    cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost);

    cudaEventRecord(stopEv);
    cudaEventSynchronize(stopEv);

    float elapsedMs = 0;
    cudaEventElapsedTime(&elapsedMs, startEv, stopEv);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(startEv);
    cudaEventDestroy(stopEv);

    saveMatrix(fileC, h_C, N);

    cout << "Matrix dimension: " << N << "\n";
    cout << "Thread block: " << blockSize << "x" << blockSize << "\n";
    cout << "Execution time: " << elapsedMs / 1000.0 << " sec\n";

    return 0;
}
