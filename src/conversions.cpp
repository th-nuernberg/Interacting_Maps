//
// Created by daniel on 11/25/24.
//
#include "conversions.h"

/**
 * Converts an Eigen 1Tensor to an Eigen Vector
 * @param input The tensor, any size
 * @return corresponding vector
 */
VectorXf Tensor2Vector(const Tensor1f& input) {
    // RowMajor Version
    array<Index, 1> dims = input.dimensions();
    const float* data_ptr = input.data();
    Map<const VectorXf> result(data_ptr, dims[0]);
    return result;
}
/**
 * Converts an Eigen vector to an Eigen 1Tensor
 * @param input Eigen vector
 * @return Eigen 1Tensor
 */
Tensor1f Vector2Tensor(const VectorXf& input) {
    const int cols = input.cols();
    const float* data_ptr = input.data();
    TensorMap<const Tensor1f> result(data_ptr, cols);
    return result;
}
/**
 * Converts an Eigen 2Tensor to an Eigen Matrix
 * @param input Eigen 2Tensor, any size
 * @return corresponding matrix
 */
MatrixXfRowMajor Tensor2Matrix(const Tensor2f& input){
    array<Index, 2> dims = input.dimensions();
    const float* data_ptr = &input(0); // Points to beginning of array;
    Map<const MatrixXfRowMajor> result(data_ptr, dims[0], dims[1]);
    return result;
}
/**
 * Converts an Eigen matrix to an Eigen 2Tensor
 * @param input Eigen matrix
 * @return Eigen 2Tensor
 */
Tensor2f Matrix2Tensor(const MatrixXfRowMajor& input) {
    // Get Pointer to data
    float const *data_ptr = &input(0);
    // Map data to Tensor
    TensorMap<const Tensor2f> result(data_ptr, input.rows(), input.cols());
    // Swap the layout and preserve the order of the dimensions
    return result;
}
/**
 * Converts a Eigen matrix to a opencv matrix, without copying the data
 * @param eigen_matrix
 * @return opencv matrix
 */
cv::Mat eigenToCvMat(const MatrixXfRowMajor& eigen_matrix) {
    return {static_cast<int>(eigen_matrix.rows()), static_cast<int>(eigen_matrix.cols()), CV_32F, (void*)eigen_matrix.data()};
}
/**
 * Creates a copy of an Eigen matrix and saves it in a opencv matrix
 * @param eigen_matrix
 * @return opencv matrix
 */
cv::Mat eigenToCvMatCopy(const MatrixXfRowMajor& eigen_matrix) {
    // Create a cv::Mat and copy Eigen matrix data into it
    cv::Mat mat(static_cast<int>(eigen_matrix.rows()), static_cast<int>(eigen_matrix.cols()), CV_32F);
    for (int i = 0; i < eigen_matrix.rows(); ++i) {
        for (int j = 0; j < eigen_matrix.cols(); ++j) {
            mat.at<float>(i, j) = eigen_matrix(i, j);
        }
    }
    return mat;
}
/**
 * Converts a opencv matrix to an Eigen matrix without creating a copy
 * @param mat opencv matrix
 * @return eigen matrix
 */
MatrixXfRowMajor cvMatToEigen(const cv::Mat& mat) {
    // Ensure the cv::Mat has the correct type
    CV_Assert(mat.type() == CV_32F);
    array<Index, 2> dims;
    dims[0] = mat.rows;
    dims[1] = mat.cols;
    const float* data_ptr = mat.ptr<float>();
    Map<const MatrixXfRowMajor> result(data_ptr, dims[0], dims[1]);
    return result;
}
/**
 * Converts a opencv matrix to an Eigen matrix while creating a copy
 * @param mat opencv matrix
 * @return eigen matrix
 */
MatrixXfRowMajor cvMatToEigenCopy(const cv::Mat& mat) {
    // Ensure the cv::Mat has the correct type
    CV_Assert(mat.type() == CV_32F);
    MatrixXfRowMajor eigen_matrix(mat.rows, mat.cols);
    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j) {
            eigen_matrix(i, j) = mat.at<float>(i, j);
        }
    }
    return eigen_matrix;
}
/**
 * Converts a opencv matrix from integer [0,255] to a float [0,1] matrix
 * @param mat
 * @return
 */
cv::Mat convertTofloat(cv::Mat& mat) {
    // Ensure the source matrix is of type CV_8U
    CV_Assert(mat.type() == CV_8U);

    // Convert the source matrix to CV_32F
    mat.convertTo(mat, CV_32F, 1.0 / 255.0); // Scaling from [0, 255] to [0, 1]

    return mat;
}
