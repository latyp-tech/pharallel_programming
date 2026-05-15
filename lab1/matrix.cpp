#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <string>

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
    if (argc != 4) {
        cerr << "Usage: " << argv[0] << " <matrixA.txt> <matrixB.txt> <result.txt>\n";
        return 1;
    }

    string fileA = argv[1];
    string fileB = argv[2];
    string fileC = argv[3];

    vector<double> A, B, C;
    int N_A, N_B;

    if (!readMatrix(fileA, A, N_A)) {
        cerr << "Error reading " << fileA << endl;
        return 1;
    }
    if (!readMatrix(fileB, B, N_B)) {
        cerr << "Error reading " << fileB << endl;
        return 1;
    }
    if (N_A != N_B) {
        cerr << "Matrix sizes do not match!" << endl;
        return 1;
    }

    int N = N_A;
    C.assign(N * N, 0.0);

    auto start = chrono::high_resolution_clock::now();
 
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            double aik = A[i * N + k];
            for (int j = 0; j < N; ++j) {
                C[i * N + j] += aik * B[k * N + j];
            }
        }
    }

    auto end = chrono::high_resolution_clock::now();
    double elapsed = chrono::duration<double>(end - start).count();

    writeMatrix(fileC, C, N);

    cout << "Matrix size N = " << N << endl;
    cout << "Execution time = " << elapsed << " seconds" << endl;

    return 0;
}
