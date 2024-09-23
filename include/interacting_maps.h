#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/CXX11/Tensor>
#include <opencv2/opencv.hpp>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <filesystem>

namespace fs = std::filesystem;

using namespace Eigen;

typedef SparseMatrix<float> SpMat;

struct Event{
    float time;
    std::vector<int> coordinates;
    float polarity;

    std::string toString() const;
};

struct Calibration_Data{
    std::vector<float> focal_point;
    Eigen::MatrixXf camera_matrix;
    std::vector<float> distortion_coefficients;
    std::vector<float> view_angles;
};

struct Parameters{
    std::unordered_map<std::string,float> weights;
    int iteratons;
    std::string results_directory;
    float time_step;
    float start_time;
    float end_time;
    float view_angle_x;
    float view_angle_y;
};

fs::path create_folder_and_update_gitignore(const std::string& foldername);

/**
 * @brief Reads the camera calibration file from a .txt-file.
 * 
 * @param file_path Path to file
 * @param calibration_data Read out calibration data as float std::vector
 */
void read_calib(const std::string& file_path, std::vector<float>& calibration_data);

Calibration_Data get_calibration_data(const std::vector<float>& calibration_data, int height, int width);

/**
 * @brief Read the events from a .txt-file.
 * 
 * @param file_path path to file
 * @param events Read out events as a std::vector of Events
 * @param start_time Starting point from which events are considered
 * @param end_time Ending point from which events are considered
 * @param max_events maximum number of events to consider, overwrites time 
 */
void read_events(const std::string& file_path, std::vector<Event>& events, float start_time, float end_time, int max_events);

/**
 * @brief Put events into bins which form the frames for the interacting maps algorithm
 * 
 * @param events std::vector of Events to put into bins
 * @param bin_size Size of the bins to put the events into, in seconds
 * @return std::vector<std::vector<Event>> binned Events
 */
std::vector<std::vector<Event>> bin_events(std::vector<Event>& events, float bin_size);

/**
 * @brief Creates frames out of the event information
 * 
 * @param binned_events Binned Events
 * @param frames output
 * @param camera_height number of pixels in x direction
 * @param camera_width number of pixels in y direction
 */
void create_frames(const std::vector<std::vector<Event>>& binned_events, std::vector<Tensor<float,2,Eigen::RowMajor>>& frames, const int camera_height, const int camera_width);

/**
 * @brief Create a sparse matrix from temporal derivative data (Events) for the update function update_RF
 * 
 * @param N Number of pixels (rows*cols)
 * @param V temporal derivative matrix
 * @param result sparse matrix with N*6 entries and dimensions N*3xN+3
 */
void create_sparse_matrix(const int N, const Tensor<float,2,Eigen::RowMajor>& V, SpMat& result);

/**
 * @brief Undo lens distortion 
 * 
 * @param image distorted image 
 * @param camera_matrix camera matrix obtained from read_calib (intrinsic to camera, focal length and optical centers)
 * @param distortion_parameters distortion paramaters obtained from read_calib (radial distortion and tangential distortion)
 * @return cv::Mat undistorted openCV image
 */
cv::Mat undistort_image(const cv::Mat& image, const cv::Mat& camera_matrix, const cv::Mat& distortion_parameters);

// void undistort_images(std::vector<Tensor<float,2,Eigen::RowMajor>>& images, Tensor<float,2,Eigen::RowMajor> camera_matrix, std::vector<float> distortion_parameters, int height, int width);

// void undistort_frames(std::vector<Tensor<float,2,Eigen::RowMajor>>& frames, Tensor<float,2,Eigen::RowMajor> camera_matrix, std::vector<float> distortion_parameters, int height, int width);

cv::Mat frame2grayscale(const Eigen::MatrixXf& frame);

cv::Mat V2image(const Eigen::MatrixXf& V);

cv::Mat vector_field2image(const Eigen::Tensor<float, 3>& vector_field);

void create_VIFG_image(Tensor<float,2,Eigen::RowMajor>& V, Tensor<float,2,Eigen::RowMajor>& I, Tensor<float,3,Eigen::RowMajor>& F, Tensor<float,3,Eigen::RowMajor>& G, Tensor<float,2,Eigen::RowMajor>& VIFG);

// void find_C(int N_x, int N_y, float view_angle_x, float view_angle_y, float rs, Tensor<float,3,Eigen::RowMajor>& C, Tensor<float,3,Eigen::RowMajor>& dXfC, Tensor<float,3,Eigen::RowMajor>& dydC);
void find_C(int N_x, int N_y, float view_angle_x, float view_angle_y, float rs, Tensor<float,3,Eigen::RowMajor>& CCM, Tensor<float,3,Eigen::RowMajor>& C_x, Tensor<float,3,Eigen::RowMajor>& C_y);

void update_FG(Tensor<float,3,Eigen::RowMajor>& F, Tensor<float,2,Eigen::RowMajor>& V, Tensor<float,3,Eigen::RowMajor>& G, float lr, float weight_FG, float eps, float gamma);

void update_GF(Tensor<float,3,Eigen::RowMajor>& G, Tensor<float,2,Eigen::RowMajor>& V, Tensor<float,3,Eigen::RowMajor>& F, float lr, float weight_GF, float eps, float gamma);

void update_GI(Tensor<float,3,Eigen::RowMajor>& G, Tensor<float,3,Eigen::RowMajor>& I_gradient, float weight_GI, float eps, float gamma);

void update_IV(Tensor<float,2,Eigen::RowMajor>& I, Tensor<float,2,Eigen::RowMajor>& cum_V, float weight_IV, float time_step);

void update_IG(Tensor<float,2,Eigen::RowMajor>& I, Tensor<float,3,Eigen::RowMajor>& I_gradient, Tensor<float,3,Eigen::RowMajor>& G, float weight_IG);

void update_FR(Tensor<float,3,Eigen::RowMajor>& F, Tensor<float,3,Eigen::RowMajor>& CCM, Tensor<float,3,Eigen::RowMajor>& Cx, Tensor<float,3,Eigen::RowMajor>& Cy, Tensor<float,1> R, float weight_FR, float eps, float gamma);

void update_RF(Tensor<float,1> R, Tensor<float,3,Eigen::RowMajor>& F, Tensor<float,3,Eigen::RowMajor>& C, Tensor<float,3,Eigen::RowMajor>& Cx, Tensor<float,3,Eigen::RowMajor>& Cy, float weight_RF, int N);

void vector_distance(const Tensor<float,3,Eigen::RowMajor> &vec1, const Tensor<float,3,Eigen::RowMajor> &vec2, Tensor<float,2,Eigen::RowMajor> &distance);

void m32(const Tensor<float,3,Eigen::RowMajor>& In, const Tensor<float,3,Eigen::RowMajor>& Cx, const Tensor<float,3,Eigen::RowMajor>& Cy, Tensor<float,3,Eigen::RowMajor>& Out);

void interacting_maps_step(Tensor<float,2,Eigen::RowMajor>& V, Tensor<float,2,Eigen::RowMajor>& cum_V, Tensor<float,2,Eigen::RowMajor>& I, Tensor<float,3,Eigen::RowMajor>& delta_I, Tensor<float,3,Eigen::RowMajor>& F, Tensor<float,3,Eigen::RowMajor>& G, Tensor<float,1>& R, const Tensor<float,3,Eigen::RowMajor>& CCM, const Tensor<float,3,Eigen::RowMajor>& dCdx, const Tensor<float,3,Eigen::RowMajor>& dCdy, SpMat& sparse_m, std::unordered_map<std::string,float>& weights, std::vector<int>& permutation, const int N);

void interacting_maps(std::vector<Tensor<float,2,Eigen::RowMajor>>& Vs, std::vector<Tensor<float,2,Eigen::RowMajor>>& cum_Vs, std::unordered_map<std::string,float> weights, int iterations, std::string results_directory);

bool isApprox(Tensor<float,3,Eigen::RowMajor>& t1, Tensor<float,2,Eigen::RowMajor>& t2, float precision);

bool isApprox(Tensor<float,2,Eigen::RowMajor>& t1, Tensor<float,2,Eigen::RowMajor>& t2, float precision);

