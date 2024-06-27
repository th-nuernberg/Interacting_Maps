#pragma once
#include <iostream>
#include <vector>
#include <string>
#include "../Eigen/Dense"
#include "../Eigen/Sparse"
#include "../unsupported/Eigen/CXX11/Tensor"
using namespace Eigen;

void create_folder_and_update_gitignore(const std::string foldername);

void find_c(int N_x, int N_y, float view_angle_x, float view_angle_y, float rs, Tensor& calibration_matrix);