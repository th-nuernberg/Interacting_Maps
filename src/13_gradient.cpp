//
// Created by Arbeit on 9/17/24.
//
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

Eigen::Tensor<double, 3> computeGradient(const Eigen::MatrixXd& data) {
    int rows = data.rows();
    int cols = data.cols();

    // Initialize the tensor: 3 dimensions (rows, cols, gradient direction)
    Eigen::Tensor<double, 3> gradients(rows, cols, 2);

    // Compute gradient along rows (x-direction)
    for (int i = 0; i < rows; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            gradients(i, j, 0) = (data(i, j + 1) - data(i, j - 1)) / 2.0;
        }
        gradients(i, 0, 0) = data(i, 1) - data(i, 0); // Forward difference for the first column
        gradients(i, cols - 1, 0) = data(i, cols - 1) - data(i, cols - 2); // Backward difference for the last column
    }

    // Compute gradient along columns (y-direction)
    for (int j = 0; j < cols; ++j) {
        for (int i = 1; i < rows - 1; ++i) {
            gradients(i, j, 1) = (data(i + 1, j) - data(i - 1, j)) / 2.0;
        }
        gradients(0, j, 1) = data(1, j) - data(0, j); // Forward difference for the first row
        gradients(rows - 1, j, 1) = data(rows - 1, j) - data(rows - 2, j); // Backward difference for the last row
    }

    return gradients;
}

int main() {
    Eigen::MatrixXd data(3, 3);
    data << 1, 2, 3,
            4, 5, 6,
            7, 8, 9;

    Eigen::Tensor<double, 3> gradients = computeGradient(data);

    std::cout << "Gradient (x-direction):\n";
    for (int i = 0; i < gradients.dimension(0); ++i) {
        for (int j = 0; j < gradients.dimension(1); ++j) {
            std::cout << gradients(i, j, 0) << " ";
        }
        std::cout << "\n";
    }

    std::cout << "\nGradient (y-direction):\n";
    for (int i = 0; i < gradients.dimension(0); ++i) {
        for (int j = 0; j < gradients.dimension(1); ++j) {
            std::cout << gradients(i, j, 1) << " ";
        }
        std::cout << "\n";
    }

    return 0;
}

