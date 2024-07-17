#include <iostream>
#include "../eigen/Eigen/Sparse"
#include "../eigen/Eigen/Dense"
#include "../eigen/Eigen/Core"
#include "../eigen/Eigen/SparseQR"
#include <chrono>
#include <random>

std::vector<double> GenerateRandomVector(int NumberCount,int minimum, int maximum) {
    std::random_device rd; 
    std::mt19937 gen(rd()); // these can be global and/or static, depending on how you use random elsewhere

    std::vector<double> values(NumberCount); 
    std::uniform_real_distribution<double> dis(minimum, maximum);
    std::generate(values.begin(), values.end(), [&](){ return dis(gen); });
    return values;
}

// void update_R(Eigen::MatrixXd &V, Eigen::VectorXd &R_extended, Eigen::VectorXd &points, int N){
//     // Initialize the matrix with random sparse entries
//     // Flatten the matrix to a vector
//     Eigen::SparseMatrix<double> A(N*3, N+3);
//     Eigen::VectorXd V_flat = Eigen::Map<Eigen::VectorXd>(V.data(), V.size());
//     std::vector<Eigen::Triplet<double>> tripletList;
//     tripletList.reserve(6 * N); // Assuming on average 2 non-zero entries per row
//     int j = 3;
//     for (int i = 0; i < N*3; i++) {
//         tripletList.emplace_back(i, i%3, static_cast<double>(1.0)); // Diagonal element
//         tripletList.emplace_back(i, j, static_cast<double>(V_flat[i]));
//         if (i%3 == 2){
//             j++;
//         }
//     }
    
//     A.setFromTriplets(tripletList.begin(), tripletList.end());

//     // Perform the computation and measure the time
//     // Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<double>> solver;
//     Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<double>, Eigen::LeastSquareDiagonalPreconditioner<double>> solver;
//     // solver.setTolerance(1e-6);
//     solver.setTolerance(1e-4);
//     solver.compute(A);
//     R_extended = solver.solveWithGuess(points, R_extended);
// }

int main() {
    std::cout << "test" << std::endl;
    int n_threads = 1;
    Eigen::setNbThreads(n_threads);
    constexpr int N = 180*240;
    constexpr int M = 1000;
    Eigen::SparseMatrix<double> A(N*3, N+3);
    Eigen::VectorXd b(N*3);
    Eigen::VectorXd x = Eigen::VectorXd::Zero(N+3);

    std::random_device rd; 
    std::mt19937 gen(rd()); // these can be global and/or static, depending on how you use random elsewhere

    std::vector<double> V_data = GenerateRandomVector(N*3, -10, 10);
    std::vector<double> b_data = GenerateRandomVector(N*3, -10, 10);
    std::vector<double> iteration_data = GenerateRandomVector(N*3*2*M, -10, 10);

    std::cout << "test" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    
    for (int k = 0; k<M; k++){
        // Initialize the matrix with random sparse entries
        std::vector<Eigen::Triplet<double>> tripletList;
        tripletList.reserve(6 * N); // Assuming on average 2 non-zero entries per row
        int j = 3;
        for (int i = 0; i < N*3; i++) {
            tripletList.emplace_back(i, i%3, static_cast<double>(1.0)); // Diagonal element
            tripletList.emplace_back(i, j, static_cast<double>(V_data[i] + iteration_data[i+(k*N*6)]));
            if (i%3 == 2){
                j++;
            }
        }
        
        A.setFromTriplets(tripletList.begin(), tripletList.end());
        
        // Initialize the right-hand side vector with random values
        for (int i = 0; i < N*3; ++i) {
            b[i] = b_data[i] + static_cast<double>(iteration_data[i+N*3+(k*N*6)]);
        }

        // Perform the computation and measure the time
        // Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<double>> solver;
        Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<double>, Eigen::LeastSquareDiagonalPreconditioner<double>> solver;
        // solver.setTolerance(1e-6);
        solver.setTolerance(1e-4);
        // std::cout << "A rows: " << A.rows() << " cols: " << A.cols() << std::endl;
        solver.compute(A);
        if(solver.info() != Eigen::Success) {
            std::cerr << "Decomposition failed during iteration " << k << std::endl;
            // return -1;
            continue;
        }

        // x = solver.solve(b);
        // std::cout << "b rows: " << b.rows() << " cols: " << b.cols() << std::endl;
        // std::cout << "x rows: " << x.rows() << " cols: " << x.cols() << std::endl;
        x = solver.solveWithGuess(b, x);
        if(solver.info() != Eigen::Success) {
            std::cerr << "Solving failed during iteration " << k << std::endl;
            // return -1;
            continue;
        }
        // if (i%100==0){
        //     std::cout << i << std::endl;
        // }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = (end - start);
    double time = elapsed.count()/M;

    std::cout << "Average time taken (over " << M << " iterations) to solve the system of size " << N << ": " << time << " seconds" << std::endl;

    return 0;
}
