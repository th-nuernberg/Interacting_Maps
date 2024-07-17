#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "Eigen/Core"
#include "../.venv/lib/python3.10/site-packages/pybind11/include/pybind11/pybind11.h"
#include "../.venv/lib/python3.10/site-packages/pybind11/include/pybind11/numpy.h"
#include "../.venv/lib/python3.10/site-packages/pybind11/include/pybind11/eigen.h"
#include <iostream>

// #include "../eigen/Eigen/IterativeSolvers"

namespace py = pybind11;

void update_R(Eigen::Ref<Eigen::MatrixXd> &V, Eigen::Ref<Eigen::VectorXd> &R_extended, const Eigen::Ref<Eigen::VectorXd> &points, int N) {
    // Initialize the sparse matrix A
    // std::cout << "V rows: " << V.rows() << " cols: " << V.cols() << std::endl;

    Eigen::SparseMatrix<double> A(N * 3, N + 3);
    // std::cout << "A rows: " << A.rows() << " cols: " << A.cols() << std::endl;
    // std::cout << "x rows: " << R_extended.rows() << " cols: " << R_extended.cols() << std::endl;
    // std::cout << "b rows: " << points.rows() << " cols: " << points.cols() << std::endl;
    Eigen::VectorXd V_flat = Eigen::Map<Eigen::VectorXd>(V.data(), V.size());
    // std::cout << "Test1" << std::endl;
    std::vector<Eigen::Triplet<double>> tripletList;
    tripletList.reserve(6 * N); // Assuming on average 2 non-zero entries per row
    int j = 3;
    for (int i = 0; i < N * 3; i++) {
        tripletList.emplace_back(i, i % 3, static_cast<double>(1.0)); // Diagonal element
        tripletList.emplace_back(i, j, static_cast<double>(V_flat[i]));
        if (i % 3 == 2) {
            j++;
        }
    }
    // std::cout << "Test2" << std::endl;
    
    A.setFromTriplets(tripletList.begin(), tripletList.end());

    // std::cout << "Test3" << std::endl;
    // Create and configure the solver
    Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<double>, Eigen::LeastSquareDiagonalPreconditioner<double>> solver;
    solver.setTolerance(1e-4); // Set tolerance
    solver.compute(A);
    // std::cout << "Test4" << std::endl;
    // Solve for R_extended
    R_extended = solver.solveWithGuess(points, R_extended);
}

// Define the Python module
PYBIND11_MODULE(pybind11_sparse_matrix, m) {
    m.doc() = "pybind11 example plugin"; // Optional module docstring

    m.def("update_R", &update_R, "Update R_extended using Eigen");
}
