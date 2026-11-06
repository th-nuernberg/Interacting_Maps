#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>

#include <rotation_plot.h>

using Catch::Matchers::WithinAbs;

TEST_CASE("GyroReference interpolates the gyro and rotates it into the camera frame", "[plot][imu]") {
    namespace fs = std::filesystem;
    const fs::path file = fs::temp_directory_path() / "im_tests_gyro_reference.txt";
    {
        std::ofstream f(file);
        f << "0.0 0 0 0 1.0 2.0 3.0\n1.0 0 0 0 3.0 2.0 1.0\n";
    }
    Eigen::Matrix3f R_cam_imu;
    R_cam_imu << 0, -1, 0, 1, 0, 0, 0, 0, 1;   // 90 degrees about z
    const GyroReference ref = GyroReference::load(file, R_cam_imu);
    fs::remove(file);

    REQUIRE(ref.time.size() == 2);
    const Eigen::Vector3f mid = ref.at(0.5);   // IMU (2, 2, 2) -> camera (-2, 2, 2)
    REQUIRE_THAT(mid(0), WithinAbs(-2.0, 1e-6));
    REQUIRE_THAT(mid(1), WithinAbs(2.0, 1e-6));
    REQUIRE_THAT(mid(2), WithinAbs(2.0, 1e-6));
    REQUIRE_THAT(ref.at(-1.0)(0), WithinAbs(-2.0, 1e-6));   // clamped: first sample (1, 2, 3) -> (-2, 1, 3)
}

TEST_CASE("RotationPlot's rolling statistics are 1 for a perfect estimate and skip slow reference motion", "[plot]") {
    RotationPlot plot(0.0, 10.0);
    for (int i = 0; i < 120; ++i) {
        const double t = i / 60.0;
        const Eigen::Vector3f omega(std::sin(t) + 2.0f, 0.5f, -1.0f);
        plot.add(t, omega, omega);
    }
    const auto &last = plot.samples().back();
    REQUIRE_THAT(last.rollingCos, WithinAbs(1.0, 1e-6));
    REQUIRE_THAT(last.rollingRatio, WithinAbs(1.0, 1e-6));

    RotationPlot slow(0.0, 10.0);
    slow.add(0.0, Eigen::Vector3f(0.1f, 0, 0), Eigen::Vector3f(0.2f, 0, 0));   // reference below 1 rad/s
    REQUIRE(std::isnan(slow.samples().back().cos));
}

TEST_CASE("RotationPlot renders an image of the requested size", "[plot]") {
    RotationPlot plot(0.0, 10.0);
    for (int i = 0; i < 30; ++i) plot.add(i / 60.0, Eigen::Vector3f(2, 0, 0), Eigen::Vector3f(2.1f, 0, 0));
    const cv::Mat img = plot.render(630, 400);
    REQUIRE(img.cols == 630);
    REQUIRE(img.rows == 400);
    REQUIRE(img.type() == CV_8UC3);
}
