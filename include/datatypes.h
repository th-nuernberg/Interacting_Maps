//
// Created by daniel on 11/25/24.
//
#ifndef INTERACTINGMAPS_DATATYPES_H
#define INTERACTINGMAPS_DATATYPES_H

#include <vector>
#include <unsupported/Eigen/CXX11/Tensor>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <xtensor.hpp>

using namespace Eigen;

using array_type = xt::xarray<float>;
using shape_type = array_type::shape_type;

using tensor_type3 = xt::xtensor<float, 3>;
using tensor_type2 = xt::xtensor<float, 2>;
using tensor_type1 = xt::xtensor<float, 1>;
using shape_type3 = tensor_type3::shape_type;
using shape_type2 = tensor_type2::shape_type;
using shape_type1 = tensor_type1::shape_type;

/*
 * Define some common datatypes. Sometimes the non-standard RowMajor versions of eigen matrices/tensors are prefered
 * If other libraries have their data stored in a row-major format.
 */
namespace Eigen{
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
struct Event {
    float time{0.0f}; // initialize to 0

    // Default constructor
    Event() = default;

    // Constructor
    explicit Event(float t) : time(t) {}

    // Destructor
    virtual ~Event() = default; // Ensure the base class has a virtual destructor
};

struct CameraEvent : Event {
    std::vector<int> coordinates;
    int polarity{0};

    // Default constructor
    CameraEvent() = default;

    // Constructor
    CameraEvent(float t, std::vector<int>& c, int p)
            : Event(t), coordinates(c), polarity(p) {}
};

struct IMUEvent : Event {
    std::vector<float> accelerations;
    std::vector<float> ang_velocities;

    // Default constructor
    IMUEvent() = default;

    // Constructor
    IMUEvent(float t, const std::vector<float>& a, const std::vector<float>& v)
            : Event(t), accelerations(a), ang_velocities(v) {}
};

struct ImageEvent : Event {
    cv::Mat image;

    // Default constructor
    ImageEvent() = default;

    // Constructor
    ImageEvent(float t, cv::Mat& img)
            : Event(t), image(img) {}
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

struct CalibrationData{
    std::vector<float> focal_point;
    xt::xarray<float> camera_matrix;
    std::vector<float> distortion_coefficients;
    std::vector<float> view_angles;
};

#endif //INTERACTINGMAPS_DATATYPES_H
