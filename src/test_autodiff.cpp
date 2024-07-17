#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>

// Define a type for convenience
using namespace autodiff;
// using ADfloat = autodiff::var;

// Function to compute C_star
Vector3real C_star(real x, real y, int N_x, int N_y, float height, float width, float rs) {
    Vector3real result;
    result << height * (-1 + (2 * x) / (N_x - 1)),
              width * (1 - (2 * y) / (N_y - 1)),
              rs;
    return result;
}

// Function to compute C
Vector3real C(real x, real y, int N_x, int N_y, float height, float width, float rs) {
    Vector3real c_star = C_star(x, y, N_x, N_y, height, width, rs);
    real norm = sqrt(c_star.squaredNorm());
    return c_star / norm;
}

void find_c(int N_x, int N_y, float view_angle_x = 3.1415 / 4, float view_angle_y = 3.1415 / 4, float rs = 1) {
    float height = tan(view_angle_x / 2);
    float width = tan(view_angle_y / 2);

    // Create grid of points
    Eigen::MatrixXd XX(N_x, N_y);
    Eigen::MatrixXd YY(N_x, N_y);
    for (int i = 0; i < N_x; ++i) {
        for (int j = 0; j < N_y; ++j) {
            XX(i, j) = i;
            YY(i, j) = j;
        }
    }

    // Compute the camera calibration map and the Jacobians
    std::vector<std::vector<Vector3real>> camera_calibration_map;
    std::vector<std::vector<Eigen::VectorXd>> C_x;
    std::vector<std::vector<Eigen::VectorXd>> C_y;
    for (int i = 0; i < N_x; ++i) {
        std::vector<Vector3real> row_C;
        std::vector<Eigen::VectorXd> row_Cx;
        std::vector<Eigen::VectorXd> row_Cy;
        for (int j = 0; j < N_y; ++j) {
            real x = XX(i, j);
            real y = YY(i, j);

            // Compute the function value
            Vector3real c_val = C(x, y, N_x, N_y, height, width, rs);
            row_C.push_back(c_val);

            // Compute the Jacobians
            // Vector3real dCdx;
            // Vector3real dCdy;
            VectorXreal F;

            Eigen::VectorXd dCdx = jacobian(C, wrt(x), at(x,y,N_x, N_y, height, width, rs), F);
            Eigen::VectorXd dCdy = jacobian(C, wrt(y), at(x,y,N_x, N_y, height, width, rs), F);

            row_Cx.push_back(dCdx);
            row_Cy.push_back(dCdy);
        }
        camera_calibration_map.push_back(row_C);
        C_x.push_back(row_Cx);
        C_y.push_back(row_Cy);
    }

    // Output the results
    std::cout << "Camera Calibration Map: \n";
    for (const auto& vec : camera_calibration_map) {
        for (const auto& vec1 : vec) {
            for (const auto& val : vec1) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
    }

    std::cout << "\nC_x: \n";
    for (const auto& vec : C_x) {
        for (const auto& vec1 : vec) {
            for (const auto& val : vec1) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
    }

    std::cout << "\nC_y: \n";
    for (const auto& vec : C_y) {
        for (const auto& vec1 : vec) {
            for (const auto& val : vec1) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
    }
}

int main() {
    find_c(5, 5);
    return 0;
}
