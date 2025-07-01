//
// Created by daniel on 11/25/24.
//
#ifndef INTERACTINGMAPS_CONVERSIONS_H
#define INTERACTINGMAPS_CONVERSIONS_H

#include "datatypes.h"
#include <opencv4/opencv2/opencv.hpp>

using namespace Eigen;

VectorXf Tensor2Vector(const Tensor1f& input);


Tensor1f Vector2Tensor(const VectorXf& input);


MatrixXfRowMajor Tensor2Matrix(const Tensor2f& input);


Tensor2f Matrix2Tensor(const MatrixXfRowMajor& input);


cv::Mat eigenToCvMat(const MatrixXfRowMajor& eigen_matrix);


cv::Mat eigenToCvMatCopy(const MatrixXfRowMajor& eigen_matrix);


MatrixXfRowMajor cvMatToEigen(const cv::Mat& mat);


MatrixXfRowMajor cvMatToEigenCopy(const cv::Mat& mat);


cv::Mat convertTofloat(cv::Mat& mat);
#endif //INTERACTINGMAPS_CONVERSIONS_H
