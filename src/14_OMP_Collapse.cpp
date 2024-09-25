//
// Created by arbeit on 9/25/24.
//
#include <iostream>
#include <omp.h>
#include <vector>
#include <chrono>  // For timing

const int N = 2000;  // Matrix size


// Function to perform matrix multiplication without collapse
void matrix_multiply_serial(const std::vector<std::vector<int>>& A,
                                 const std::vector<std::vector<int>>& B,
                                 std::vector<std::vector<int>>& C) {
//#pragma omp parallel for  // Only outer loop is parallelized
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Function to perform matrix multiplication without collapse
void matrix_multiply_no_collapse(const std::vector<std::vector<int>>& A,
                                 const std::vector<std::vector<int>>& B,
                                 std::vector<std::vector<int>>& C) {
    #pragma omp parallel for  // Only outer loop is parallelized
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Function to perform matrix multiplication with collapse
void matrix_multiply_with_collapse(const std::vector<std::vector<int>>& A,
                                   const std::vector<std::vector<int>>& B,
                                   std::vector<std::vector<int>>& C) {
    #pragma omp parallel for collapse(2)  // Collapses both the i and j loops
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main() {
    // Initialize matrices A, B, and C
    int max_threads = 16;
    std::vector<std::vector<int>> A(N, std::vector<int>(N, 1));
    std::vector<std::vector<int>> B(N, std::vector<int>(N, 1));
    std::vector<std::vector<int>> C1(N, std::vector<int>(N, 0));
    std::vector<std::vector<int>> C2(N, std::vector<int>(N, 0));
    std::vector<std::vector<int>> C3(N, std::vector<int>(N, 0));

    // Time the serial version
    auto start_serial = std::chrono::high_resolution_clock::now();
    matrix_multiply_serial(A, B, C3);
    auto end_serial = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff_serial = end_serial - start_serial;
    std::cout << "Time serial: " << diff_serial.count() << " seconds\n";
    for (int n_threads = 2; n_threads <= max_threads; ++n_threads) {
        // Set number of threads
        omp_set_num_threads(n_threads);

        std::cout << "Number of threads: " << n_threads << "\n";
        // Time the no collapse version
        auto start_no_collapse = std::chrono::high_resolution_clock::now();
        matrix_multiply_no_collapse(A, B, C1);
        auto end_no_collapse = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff_no_collapse = end_no_collapse - start_no_collapse;
        std::cout << "Time without collapse: " << diff_no_collapse.count() << " seconds\n";

        // Time the collapse version
        auto start_with_collapse = std::chrono::high_resolution_clock::now();
        matrix_multiply_with_collapse(A, B, C2);
        auto end_with_collapse = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff_with_collapse = end_with_collapse - start_with_collapse;
        std::cout << "Time with collapse: " << diff_with_collapse.count() << " seconds\n";
    }
    return 0;

}
