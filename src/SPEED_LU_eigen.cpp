#include <iostream>
#include <cstdlib>   // For rand()
#include <ctime>     // For timing
#include <complex>

#define EIGEN_SUPERLU_SUPPORT
#define EIGEN_USE_BLAS
#define EIGEN_USE_LAPACKE
#define EIGEN_USE_LAPACKE_STRICT

#define LAPACK_COMPLEX_CUSTOM
#define lapack_complex_float std::complex<float>
#define lapack_complex_double std::complex<double>

#include "../eigen/Eigen/Dense"   // Eigen library#

// Define matrix size
#define N 1000    // Size of the matrices (N x N)

int main() {
    srand(static_cast<unsigned int>(time(NULL)));

    // Define matrices A, B, and C using Eigen's dynamic size matrices
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(N, N);
    Eigen::MatrixXd B = Eigen::MatrixXd::Random(N, N);
    Eigen::MatrixXd C(N, N);

    // Benchmarking matrix multiplication using Eigen
    clock_t start_time = clock();   // Start timing

    C = A * B;   // Matrix multiplication using Eigen

    clock_t end_time = clock();     // End timing
    double elapsed_time = static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC;

    // Print results and timing
    std::cout << "Matrix size: " << N << " x " << N << std::endl;
    std::cout << "Eigen time: " << elapsed_time << " seconds" << std::endl;

    start_time = clock();

    // Perform LU decomposition using Eigen
    Eigen::FullPivLU<Eigen::MatrixXd> lu(A);

    end_time = clock();
    elapsed_time = static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC;

    std::cout << "Eigen LU decomposition time: " << elapsed_time << " seconds" << std::endl;

    return 0;
}