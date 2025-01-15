//
// Created by daniel on 11/25/24.
//

#ifndef INTERACTINGMAPS_DATATYPES_H
#define INTERACTINGMAPS_DATATYPES_H

#include <iostream>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace Eigen;

/*
 * Define some common datatypes. Sometimes the non-standard RowMajor versions of eigen matrices/tensors are prefered
 * If other libraries have their data stored in a row-major format.
 */
namespace{
    typedef Matrix<float, Dynamic, Dynamic, RowMajor> MatrixXfRowMajor;
    typedef Matrix<double, Dynamic, Dynamic, RowMajor> MatrixXdRowMajor;
    typedef Matrix<int, Dynamic, Dynamic, RowMajor> MatrixXiRowMajor;
    typedef Tensor<float,1,RowMajor> Tensor1f;
    typedef Tensor<float,2,RowMajor> Tensor2f;
    typedef Tensor<float,3,RowMajor> Tensor3f;
}

/*
 * Holds a single event defined by time t the event occurred at, location (x,y) where the event happend on the receptor
 * and a polarity p (-1,1) of the event.
 */
struct Event{
    float time;
    std::vector<int> coordinates;
    int polarity;
    std::string toString() const;
};

/*
 * Images of a camera can be distorted by said camera. If certain parameters about the camera are known, the images can
 * be undistorted.
 */
struct Calibration_Data{
    std::vector<float> focal_point;
    MatrixXf camera_matrix;
    std::vector<float> distortion_coefficients;
    std::vector<float> view_angles;
};

#endif //INTERACTINGMAPS_DATATYPES_H
