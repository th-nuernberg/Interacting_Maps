#include <iostream>
#include <ctime>
#include <cblas.h>
#include <lapacke.h>
#include <openblas_config.h>

#define N 1000



void print_matrix(const char* desc, int m, int n, double* a, int lda) {
    int i, j;
    std::cout << desc << std::endl;
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            std::cout << a[i + j * lda] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    openblas_set_num_threads(4); // Set the number of threads to 4

    srand(static_cast<unsigned int>(time(NULL)));

    // Create a large random matrix
    double* A = new double[N * N];
    for (int i = 0; i < N * N; ++i) {
        A[i] = static_cast<double>(rand()) / RAND_MAX;
    }

    // Allocate arrays for the pivot indices
    int* ipiv = new int[N];

    clock_t start_time = clock();

    // Perform LU decomposition using LAPACKE
    int info = LAPACKE_dgetrf(LAPACK_COL_MAJOR, N, N, A, N, ipiv);

    clock_t end_time = clock();

    if (info != 0) {
        std::cerr << "LU decomposition failed." << std::endl;
        delete[] A;
        delete[] ipiv;
        return -1;
    }

    
    double elapsed_time = static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC;

    std::cout << "Matrix size: " << N << " x " << N << std::endl;
    std::cout << "LAPACKE LU decomposition time: " << elapsed_time << " seconds" << std::endl;

    // Extract L and U matrices from A
    double* L = new double[N * N];
    double* U = new double[N * N];
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i > j) {
                L[i + j * N] = A[i + j * N];
                U[i + j * N] = 0.0;
            } else if (i == j) {
                L[i + j * N] = 1.0;
                U[i + j * N] = A[i + j * N];
            } else {
                L[i + j * N] = 0.0;
                U[i + j * N] = A[i + j * N];
            }
        }
    }

    // Construct permutation matrix P from ipiv
    double* P = new double[N * N];
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            P[i + j * N] = (ipiv[i] == j + 1) ? 1.0 : 0.0;
        }
    }

    // Optionally, print L, U, and P matrices
    // print_matrix("Matrix L", N, N, L, N);
    // print_matrix("Matrix U", N, N, U, N);
    // print_matrix("Matrix P", N, N, P, N);

    delete[] A;
    delete[] L;
    delete[] U;
    delete[] P;
    delete[] ipiv;

    return 0;
}
