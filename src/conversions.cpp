//
// Created by daniel on 11/25/24.
//
#include "conversions.h"
#include <imaging.h>
#include <boost/stacktrace/detail/frame_decl.hpp>

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
    return {static_cast<int>(eigen_matrix.rows()), static_cast<int>(eigen_matrix.cols()), CV_32F, (float*)eigen_matrix.data()};
}

/**
 * Converts a xtensor matrix to a opencv matrix, without copying the data
 * @param xtensor 2
 * @return opencv matrix
 */
cv::Mat xtensorToCvMat(const xt::xtensor<float, 2>& tensor) {
    // Create a cv::Mat that shares the data (no copy)
    cv::Mat mat(
        static_cast<int>(tensor.shape()[0]), // rows
        static_cast<int>(tensor.shape()[1]), // cols
        CV_32F,                               // type
        const_cast<float*>(tensor.data())    // data pointer (const_cast because cv::Mat constructor is not const-correct)
    );
    return mat;
}

xt::xtensor<float, 2> cvMatToXtensor(const cv::Mat& mat) {
    // Check if the input is a 2D float matrix
    if (mat.empty() || mat.dims != 2 || mat.type() != CV_32F) {
        throw std::runtime_error("Input must be a non-empty 2D CV_32F cv::Mat.");
    }

    // Adapt the cv::Mat data to an xtensor
    // Use xt::no_ownership() to avoid double-free (xtensor won't manage the memory)
    std::vector<int> shape = {mat.rows, mat.cols};
    return xt::adapt(
        mat.ptr<float>(),                           // Data pointer (const for safety)
        mat.rows*mat.cols,                          // size
        xt::no_ownership(),                         // Let OpenCV manage the memory
        shape                                       // Shape
    );
}

/**
 * Converts a xtensor tensor to a opencv matrix, without copying the data
 * @param xtensor 3
 * @return opencv matrix
 */
cv::Mat xtensorToCvMat(const xt::xtensor<float, 3>& tensor) {
    int sz[] = {static_cast<int>(tensor.shape()[0]),static_cast<int>(tensor.shape()[1]),static_cast<int>(tensor.shape()[2])};
    return {3, sz, CV_32F, (float*)tensor.data()};
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

xt::xtensor<float, 2> cvMatToV(const cv::Mat& cv_image, const int neutral, const float contribution) {
    cv::Mat float_image;
    //Convert to float if not already
    if (cv_image.type() != CV_32FC3) {
        cv_image.convertTo(float_image, CV_32FC3);
    } else {
        float_image = cv_image.clone();
    }
    size_t size = float_image.total();
    size_t channels = float_image.channels();
    std::vector<int> boxOutputArrShape = { float_image.rows, float_image.cols , float_image.channels()};
    xt::xtensor<float, 3> adapt = xt::adapt((float*)float_image.data, size * channels, xt::no_ownership(), boxOutputArrShape, xt::layout_type::row_major);

    // USED FOR DEBUGGING
    //std::cout << adapt << std::endl;
    // std::cout<< xt::view(adapt, xt::all(), xt::all(), 0) << std::endl;
    // std::cout<< xt::view(adapt, xt::all(), xt::all(), 1) << std::endl;
    // std::cout<< xt::view(adapt, xt::all(), xt::all(), 2) << std::endl;
    // xt::xtensor<float, 2> V = xt::view(adapt, xt::all(), xt::all(), 1)/128*contribution - xt::view(adapt, xt::all(), xt::all(), 2)/255*contribution;
    // cv::Mat imageV = V2image(V, 1.0);
    return xt::view(adapt, xt::all(), xt::all(), 1)/128*contribution - xt::view(adapt, xt::all(), xt::all(), 2)/255*contribution; // Bracket Initializer from xarray to xtensor
}

xt::xtensor<float, 2> cvMatToI(const cv::Mat& cv_image) {

    //Convert to float if not already
    // if (cv_image.type() != CV_32FC1) {
    //     cv_image.convertTo(float_image, CV_32FC1);
    // } else {
    //     float_image = cv_image.clone();
    // }
    size_t size = cv_image.total();
    size_t channels = cv_image.channels();
    std::vector<int> boxOutputArrShape = { cv_image.rows, cv_image.cols};
    xt::xtensor<uint8_t, 2> adapt = xt::adapt(cv_image.data, size * channels, xt::no_ownership(), boxOutputArrShape);
    xt::xtensor<float, 2> res = xt::cast<float>(adapt);
    cv::Mat test = frame2grayscale(res);
    // xt::xtensor<float, 3> res = xt::empty<float>(boxOutputArrShape);
    // res = xt::cast<float>(adapt);
    return res; // Bracket Initializer from xarray to xtensor
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