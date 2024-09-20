//
// Created by Arbeit on 9/17/24.
//
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

namespace Eigen {
    typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXfRowMajor;
}

Eigen::Tensor<float,3,Eigen::RowMajor> computeGradient(const Eigen::MatrixXfRowMajor & data, const std::vector<int> direction={0,1}) {
    int rows = data.rows();
    int cols = data.cols();

//    DEBUG_LOG("data: " << std::endl << data);
    // Initialize the tensor: 3 dimensions (rows, cols, gradient direction)
    if (direction.size() == 2){
        Eigen::Tensor<float,3,Eigen::RowMajor> gradients(rows, cols, 2);
        // Compute gradient along columns (up-down, x-direction)
        for (int j = 0; j < cols; ++j) {
            for (int i = 1; i < rows - 1; ++i) {
                gradients(i, j, 0) = (data(i + 1, j) - data(i - 1, j)) / 2.0;
            }
//            gradients(0, j, 1) = data(1, j) - data(0, j); // Forward difference for the first row
//            gradients(rows - 1, j, 1) = data(rows - 1, j) - data(rows - 2, j); // Backward difference for the last row
            gradients(0, j, 0) = (data(1, j) - data(0,j)) / 2.0; // Central difference with replicate border
            gradients(rows - 1, j, 0) = (data(rows-1, j) - data(rows - 2, j)) / 2.0; // Central difference with replicate border
        }
        // Compute gradient along rows (left-right, y-direction)
        for (int i = 0; i < rows; ++i) {
            for (int j = 1; j < cols - 1; ++j) {
                gradients(i, j, 1) = (data(i, j + 1) - data(i, j - 1)) / 2.0;
            }
//            gradients(i, 0, 0) = data(i, 1) - data(i, 0); // Forward difference for the first column
//            gradients(i, cols - 1, 0) = data(i, cols - 1) - data(i, cols - 2); // Backward difference for the last column
            gradients(i, 0, 1) = (data(i, 1) - data(i,0)) / 2.0; // Central difference with replicate border
            gradients(i, cols - 1, 1) = (data(i, cols - 1) - data(i, cols - 2)) / 2.0; // Central difference with replicate border
        }
        return gradients;
    }
    else if (direction[0] == 1){
        Eigen::Tensor<float,3,Eigen::RowMajor> gradients(rows, cols, 1);
        // Compute gradient along rows (y-direction)
        for (int i = 0; i < rows; ++i) {
            for (int j = 1; j < cols - 1; ++j) {
                gradients(i, j, 0) = (data(i, j + 1) - data(i, j - 1)) / 2.0;
            }
//            gradients(i, 0, 0) = data(i, 1) - data(i, 0); // Forward difference for the first column
//            gradients(i, cols - 1, 0) = data(i, cols - 1) - data(i, cols - 2); // Backward difference for the last colums
            gradients(i, 0, 0) = (data(i, 1) - data(i,0)) / 2.0; // Central difference with replicate border
            gradients(i, cols - 1, 0) = (data(i, cols - 1) - data(i, cols - 2)) / 2.0; // Central difference with replicate border
        }
        return gradients;
    }
    else if (direction[0] == 0) {
        Eigen::Tensor<float,3,Eigen::RowMajor> gradients(rows, cols, 1);
        // Compute gradient along columns (x-direction)
        for (int j = 0; j < cols; ++j) {
            for (int i = 1; i < rows - 1; ++i) {
                gradients(i, j, 0) = (data(i + 1, j) - data(i - 1, j)) / 2.0;
            }
//            gradients(0, j, 0) = data(1, j) - data(0, j); // Forward difference for the first row
//            gradients(rows - 1, j, 0) = data(rows - 1, j) - data(rows - 2, j); // Backward difference for the last row
            gradients(0, j, 0) = (data(1, j) - data(0,j)) / 2.0; // Central difference with replicate border
            gradients(rows - 1, j, 0) = (data(rows-1, j) - data(rows - 2, j)) / 2.0; // Central difference with replicate border
        }
//        DEBUG_LOG(gradients(0, 0, 0));
        return gradients;
    }
    else{
        throw std::invalid_argument("invalid directions");
    }
}

int main() {
    Eigen::MatrixXfRowMajor data(3, 3);
    data << 1, 2, 3,
            4, 5, 6,
            7, 8, 9;

    Eigen::Tensor<float,3,Eigen::RowMajor> gradients = computeGradient(data);

    std::cout << "Gradient (up-down-x-direction, double):\n";
    for (int i = 0; i < gradients.dimension(0); ++i) {
        for (int j = 0; j < gradients.dimension(1); ++j) {
            std::cout << gradients(i, j, 0) << " ";
        }
        std::cout << "\n";
    }

    std::cout << "Gradient (left-right-y-direction, double):\n";
    for (int i = 0; i < gradients.dimension(0); ++i) {
        for (int j = 0; j < gradients.dimension(1); ++j) {
            std::cout << gradients(i, j, 1) << " ";
        }
        std::cout << "\n";
    }

    std::vector<int> x_direction = {0};
    gradients = computeGradient(data, x_direction);
    std::cout << "Gradient (up-down-x-direction, single):\n";
    for (int i = 0; i < gradients.dimension(0); ++i) {
        for (int j = 0; j < gradients.dimension(1); ++j) {
            std::cout << gradients(i, j, 0) << " ";
        }
        std::cout << "\n";
    }

    std::vector<int> y_direction = {1};
    gradients = computeGradient(data, y_direction);
    std::cout << "Gradient (left-right-y-direction, single):\n";
    for (int i = 0; i < gradients.dimension(0); ++i) {
        for (int j = 0; j < gradients.dimension(1); ++j) {
            std::cout << gradients(i, j, 0) << " ";
        }
        std::cout << "\n";
    }


    return 0;
}

