//
// Created by daniel on 11/25/24.
//
#ifndef INTERACTINGMAPS_CONVERSIONS_H
#define INTERACTINGMAPS_CONVERSIONS_H

#include "datatypes.h"
#include <opencv2/core.hpp>
#include <xtensor/containers/xadapt.hpp>


using namespace Eigen;

VectorXf Tensor2Vector(const Tensor1f& input);


Tensor1f Vector2Tensor(const VectorXf& input);


MatrixXfRowMajor Tensor2Matrix(const Tensor2f& input);


Tensor2f Matrix2Tensor(const MatrixXfRowMajor& input);

cv::Mat eigenToCvMat(const MatrixXfRowMajor& eigen_matrix);

cv::Mat eigenToCvMatCopy(const MatrixXfRowMajor& eigen_matrix);

cv::Mat xtensorToCvMat(const xt::xtensor<float, 2>& tensor);

cv::Mat xtensorToCvMat(const xt::xtensor<float, 3>& tensor);

MatrixXfRowMajor cvMatToEigen(const cv::Mat& mat);

MatrixXfRowMajor cvMatToEigenCopy(const cv::Mat& mat);

xt::xtensor<float, 2> cvMatToV(const cv::Mat& cv_image, int neutral = 128, float contribution = 10.0);

xt::xtensor<float, 2> cvMatToI(const cv::Mat& cv_image);

cv::Mat convertTofloat(cv::Mat& mat);
#endif //INTERACTINGMAPS_CONVERSIONS_H
