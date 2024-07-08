#include <iostream>
#include "../eigen/Eigen/Sparse"
#include "../eigen/Eigen/Dense"
#include "../eigen/Eigen/Core"
#include <chrono>


int main() {
    int n_threads = 4;
    Eigen::setNbThreads(n_threads);
    constexpr int N = 180*240;
    constexpr int M = 10000;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i<M; i++){
        // Define the sparse matrix and the right-hand side vector
        Eigen::SparseMatrix<double> A(N*3, N+3);
        Eigen::VectorXd b(N*3);

        // Initialize the matrix with random sparse entries
        std::vector<Eigen::Triplet<double>> tripletList;
        tripletList.reserve(6 * N); // Assuming on average 2 non-zero entries per row
        int j = 3;
        for (int i = 0; i < N*3; i++) {
            tripletList.emplace_back(i, i%3, static_cast<double>(1.0)); // Diagonal element
            tripletList.emplace_back(i, j, static_cast<double>(rand()) / RAND_MAX);
            if (i%3 == 2){
                j++;
            }
        }
        
        A.setFromTriplets(tripletList.begin(), tripletList.end());
        
        // Initialize the right-hand side vector with random values
        for (int i = 0; i < N*3; ++i) {
            b[i] = static_cast<double>(rand()) / RAND_MAX;
        }

        // Perform the computation and measure the time
        Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<double>> solver;
        solver.compute(A);
        if(solver.info() != Eigen::Success) {
            std::cerr << "Decomposition failed!" << std::endl;
            return -1;
        }

        Eigen::VectorXd x = solver.solve(b);
        if(solver.info() != Eigen::Success) {
            std::cerr << "Solving failed!" << std::endl;
            return -1;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = (end - start);
    double time = elapsed.count()/M;

    std::cout << "Average time taken (over " << M << " iterations) to solve the system of size " << N << ": " << time << " seconds" << std::endl;

    return 0;
}
