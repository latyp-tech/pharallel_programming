#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <omp.h>
#include <cstdlib>

using namespace std;

bool readMatrix(const string& filename, vector<double>& mat, int& N) {
    ifstream in(filename);
    if (!in) return false;
    in >> N;
    if (N <= 0) return false;
    mat.resize(N * N);
    for (int i = 0; i < N * N; ++i)
        in >> mat[i];
    return true;
}

void writeMatrix(const string& filename, const vector<double>& mat, int N) {
    ofstream out(filename);
    out << N << "\n";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j)
            out << mat[i * N + j] << " ";
        out << "\n";
    }
}

int main(int argc, char* argv[]) {
    if (argc < 4 || argc > 5) {
        cerr << "Usage: " << argv[0] << " <A.txt> <B.txt> <C.txt> [num_threads]\n";
        return 1;
    }

    string fileA = argv[1];
    string fileB = argv[2];
    string fileC = argv[3];

    int num_threads = omp_get_max_threads();
    if (argc == 5) {
        num_threads = atoi(argv[4]);
        if (num_threads < 1) num_threads = 1;
        omp_set_num_threads(num_threads);
    }

    vector<double> A, B, C;
    int N_A, N_B;

    if (!readMatrix(fileA, A, N_A) || !readMatrix(fileB, B, N_B)) {
        cerr << "Error reading input files\n";
        return 1;
    }
    if (N_A != N_B) {
        cerr << "Matrix sizes do not match\n";
        return 1;
    }

    int N = N_A;
    C.assign(N * N, 0.0);

    auto start = chrono::high_resolution_clock::now();

    #pragma omp parallel for schedule(dynamic, 16) collapse(2)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }

    auto end = chrono::high_resolution_clock::now();
    double elapsed = chrono::duration<double>(end - start).count();

    writeMatrix(fileC, C, N);

    cout << "Matrix size N = " << N << "\n";
    cout << "Threads = " << num_threads << "\n";
    cout << "Execution time = " << elapsed << " seconds\n";

    return 0;
}
