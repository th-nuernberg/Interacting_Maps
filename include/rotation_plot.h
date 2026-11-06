#pragma once
// Live plot of the rotation estimate against a reference (the gyro). Display only: the reference never enters the
// network (that is --fuseR).

#include <Eigen/Dense>
#include <filesystem>
#include <opencv2/core.hpp>
#include <vector>

// Gyro readings as camera angular velocity (x, y, z in rad/s), for comparison.
struct GyroReference {
    std::vector<double> time;
    std::vector<Eigen::Vector3f> omega;

    // imu.txt: t ax ay az gx gy gz (IMU frame); R_cam_imu rotates into the camera frame
    static GyroReference load(const std::filesystem::path &path, const Eigen::Matrix3f &R_cam_imu);
    bool empty() const { return time.empty(); }
    // linear interpolation, clamped at the ends
    Eigen::Vector3f at(double t) const;
};

class RotationPlot {
public:
    RotationPlot(double startTime, double endTime, double window = 10.0, double statsWindow = 1.0,
                 float minReferenceSpeed = 1.0f);

    // One sample per output frame. Rolling statistics (median over the last statsWindow seconds, only samples where
    // the reference turns faster than minReferenceSpeed) are updated here, so render() stays cheap.
    void add(double t, const Eigen::Vector3f &estimate, const Eigen::Vector3f &reference);
    cv::Mat render(int width, int height) const;

    struct Sample {
        double t;
        Eigen::Vector3f estimate, reference;
        float cos, ratio;               // of this sample, NaN when the reference is too slow
        float rollingCos, rollingRatio; // NaN until enough samples
    };
    const std::vector<Sample> &samples() const { return samples_; }

private:
    double startTime_, endTime_, window_, statsWindow_;
    float minReferenceSpeed_;
    std::vector<Sample> samples_;
};
