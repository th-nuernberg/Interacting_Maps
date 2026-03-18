//
// Created by Daniel Pommer on 01.12.25.
//
#include <dv-processing/camera/calibration_set.hpp>


int main() {
    /// Writing calibration JSON
    // Initialize a calibration set
    dv::camera::CalibrationSet calibration;

    // Add a camera calibration with hardcoded calibration parameters, the exact values are just for illustration.
    calibration.addCameraCalibration(dv::camera::calibrations::CameraCalibration("DAVIS346_00001088", "left", true,
        cv::Size(346, 260), cv::Point2f(173, 130), cv::Point2f(346, 346), {}, dv::camera::DistortionModel::NONE,
        dv::kinematics::Transformationf(), dv::camera::calibrations::CameraCalibration::Metadata()));
    // Add an IMU calibration as well, the exact values are just for illustration of use here.
    calibration.addImuCalibration(dv::camera::calibrations::IMUCalibration("DAVIS346_00001088", 100.f, 98.1f,
        cv::Point3f(0.f, 0.f, 0.f), cv::Point3f(0.f, 0.f, 0.f), 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 3500,
        dv::kinematics::Transformationf(), dv::camera::calibrations::IMUCalibration::Metadata()));
    // Just write the generated calibration set into a file.
    calibration.writeToFile("calibration.json");

    /// Reading Calibration JSON
    // Initialize a calibration set
    const auto calibrationSet = dv::camera::CalibrationSet::LoadFromFile("calibration.json");

    // Iterate through available camera calibrations. The designation here is an internal camera abbreviation
    // used to refer to a specific sensor in the camera rig.
    for (const auto &[designation, calibrationRead] : calibrationSet.getCameraCalibrations()) {
        // Print the designation and the camera name of current calibration
        std::cout << "[" << designation << "] Found calibration for camera with name [" << calibrationRead.name << "]"
                  << std::endl;

        // Print the intrinsic calibration parameters for this camera: focal length, principal point, distortion model
        // and parameters of the distortion model
        std::cout << "\t Focal length: " << calibrationRead.focalLength << std::endl;
        std::cout << "\t Principal point: " << calibrationRead.principalPoint << std::endl;
        std::cout << "\t Distortion model: " << calibrationRead.getDistortionModelString() << std::endl;
        std::cout << "\t Distortion parameters: "
                  << fmt::format("[{}]", fmt::join(calibrationRead.distortion.begin(), calibrationRead.distortion.end(), ", "))
                  << std::endl;

    }


    // Iterate through available IMU calibrations in the file

    for (const auto &[designation, calibrationRead] : calibrationSet.getImuCalibrations()) {
        // Print the designation and the camera name of current calibration
        std::cout << "[" << designation << "] Found IMU calibration for camera with name [" << calibrationRead.name << "]"
                  << std::endl;

        // Print some available information: accelerometer and gyroscope measurement limits, calibrated time offset and
        // biases
        std::cout << "\t Maximum acceleration: " << calibrationRead.accMax << " [m/s^2]" << std::endl;
        std::cout << "\t Maximum angular velocity: " << calibrationRead.omegaMax << " [rad/s]" << std::endl;
        std::cout << "\t Time offset: " << calibrationRead.timeOffsetMicros << " [μs]" << std::endl;
        std::cout << "\t Accelerometer bias: " << calibrationRead.accOffsetAvg << " [m/s^2]" << std::endl;
        std::cout << "\t Gyroscope bias: " << calibrationRead.omegaOffsetAvg << " [rad/s]" << std::endl;

        // Print noise density values for the IMU sensor
        std::cout << "\t Accelerometer noise density: " << calibrationRead.accNoiseDensity << " [m/s^2/sqrt(Hz)]"
                  << std::endl;
        std::cout << "\t Gyroscope noise density: " << calibrationRead.omegaNoiseDensity << " [rad/s/sqrt(Hz)]"
                  << std::endl;
    }

    return 0;
}