#include <iostream>
#include <cstdlib>   // For rand()
#include <ctime>     // For timing
#include <complex>

#define EIGEN_SUPERLU_SUPPORT
#define EIGEN_USE_BLAS
#define EIGEN_USE_LAPACKE

#define LAPACK_COMPLEX_CUSTOM
#define lapack_complex_float std::complex<float>
#define lapack_complex_double std::complex<double>

#include "../eigen/Eigen/Dense"   // Eigen library

// Link with OpenBLAS (if not already done in your build configuration)
extern "C" {
    #include <cblas.h>
}

// Define matrix size
#define N 1000    // Size of the matrices (N x N)

int main() {
    openblas_set_num_threads(4); // Set the number of threads to 4
    srand(static_cast<unsigned int>(time(NULL)));

    // Use RowMajor storage order for Eigen matrices
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A = Eigen::MatrixXd::Random(N, N);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> B = Eigen::MatrixXd::Random(N, N);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> C(N, N);

    clock_t start_time = clock();

    // Use OpenBLAS for matrix multiplication
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N, 1.0, A.data(), N, B.data(), N, 0.0, C.data(), N);

    clock_t end_time = clock();
    double elapsed_time = static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC;

    std::cout << "Matrix size: " << N << " x " << N << std::endl;
    std::cout << "Eigen + OpenBLAS time: " << elapsed_time << " seconds" << std::endl;

    Eigen::MatrixXd L, U, P;
    Eigen::VectorXi ipiv(N);

    // Convert Eigen matrix to column-major order if necessary
    Eigen::MatrixXd Ac = A;

    start_time = clock();

    // Use LAPACK for LU decomposition
    int info = LAPACKE_dgetrf(LAPACK_COL_MAJOR, N, N, Ac.data(), N, ipiv.data());

    if (info != 0) {
        std::cerr << "LU decomposition failed." << std::endl;
        return -1;
    }

    end_time = clock();
    elapsed_time = static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC;

    std::cout << "Eigen + OpenBLAS LU decomposition time: " << elapsed_time << " seconds" << std::endl;

    return 0;
}
