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

typedef SparseMatrix<double> SpMat;

struct Event{
    double time;
    std::vector<int> coordinates;
    int polarity;

    std::string toString() const;
};

struct Calibration_Data{
    std::vector<double> focal_point;
    Eigen::MatrixXd camera_matrix;
    std::vector<double> distortion_coefficients;
    std::vector<double> view_angles;
};

struct Parameters{
    std::unordered_map<std::string,double> weights;
    int iteratons;
    std::string results_directory;
    double time_step;
    double start_time;
    double end_time;
    double view_angle_x;
    double view_angle_y;
};

fs::path create_folder_and_update_gitignore(const std::string& foldername);

/**
 * @brief Reads the camera calibration file from a .txt-file.
 * 
 * @param file_path Path to file
 * @param calibration_data Read out calibration data as double std::vector
 */
void read_calib(const std::string& file_path, std::vector<double>& calibration_data);

Calibration_Data get_calibration_data(const std::vector<double>& calibration_data, int height, int width);

/**
 * @brief Read the events from a .txt-file.
 * 
 * @param file_path path to file
 * @param events Read out events as a std::vector of Events
 * @param start_time Starting point from which events are considered
 * @param end_time Ending point from which events are considered
 * @param max_events maximum number of events to consider, overwrites time 
 */
void read_events(const std::string& file_path, std::vector<Event>& events, double start_time, double end_time, int max_events);

/**
 * @brief Put events into bins which form the frames for the interacting maps algorithm
 * 
 * @param events std::vector of Events to put into bins
 * @param bin_size Size of the bins to put the events into, in seconds
 * @return std::vector<std::vector<Event>> binned Events
 */
std::vector<std::vector<Event>> bin_events(std::vector<Event>& events, double bin_size);

/**
 * @brief Creates frames out of the event information
 * 
 * @param binned_events Binned Events
 * @param frames output
 * @param camera_height number of pixels in x direction
 * @param camera_width number of pixels in y direction
 */
void create_frames(const std::vector<std::vector<Event>>& binned_events, std::vector<Tensor<double,2,Eigen::RowMajor>>& frames, const int camera_height, const int camera_width);

/**
 * @brief Create a sparse matrix from temporal derivative data (Events) for the update function update_R_from_F
 * 
 * @param N Number of pixels (rows*cols)
 * @param V temporal derivative matrix
 * @param result sparse matrix with N*6 entries and dimensions N*3xN+3
 */
void create_sparse_matrix(const int N, const Tensor<double,2,Eigen::RowMajor>& V, SpMat& result);

/**
 * @brief Undo lens distortion 
 * 
 * @param image distorted image 
 * @param camera_matrix camera matrix obtained from read_calib (intrinsic to camera, focal length and optical centers)
 * @param distortion_parameters distortion paramaters obtained from read_calib (radial distortion and tangential distortion)
 * @return cv::Mat undistorted openCV image
 */
cv::Mat undistort_image(const cv::Mat& image, const cv::Mat& camera_matrix, const cv::Mat& distortion_parameters);

// void undistort_images(std::vector<Tensor<double,2,Eigen::RowMajor>>& images, Tensor<double,2,Eigen::RowMajor> camera_matrix, std::vector<double> distortion_parameters, int height, int width);

// void undistort_frames(std::vector<Tensor<double,2,Eigen::RowMajor>>& frames, Tensor<double,2,Eigen::RowMajor> camera_matrix, std::vector<double> distortion_parameters, int height, int width);

cv::Mat frame2grayscale(const Eigen::MatrixXd& frame);

cv::Mat V2image(const Eigen::MatrixXd& V);

cv::Mat vector_field2image(const Eigen::Tensor<double, 3>& vector_field);

void create_VIFG_image(Tensor<double,2,Eigen::RowMajor>& V, Tensor<double,2,Eigen::RowMajor>& I, Tensor<double,3,Eigen::RowMajor>& F, Tensor<double,3,Eigen::RowMajor>& G, Tensor<double,2,Eigen::RowMajor>& VIFG);

// void find_C(int N_x, int N_y, double view_angle_x, double view_angle_y, double rs, Tensor<double,3,Eigen::RowMajor>& C, Tensor<double,3,Eigen::RowMajor>& dCdx, Tensor<double,3,Eigen::RowMajor>& dCdy);
void find_C(int N_x, int N_y, double view_angle_x, double view_angle_y, double rs, Tensor<double,3,Eigen::RowMajor>& CCM, Tensor<double,3,Eigen::RowMajor>& C_x, Tensor<double,3,Eigen::RowMajor>& C_y);

void update_F_from_G(Tensor<double,3,Eigen::RowMajor>& F, Tensor<double,2,Eigen::RowMajor>& V, Tensor<double,3,Eigen::RowMajor>& G, double lr, double weight_FG);

void update_G_from_F(Tensor<double,3,Eigen::RowMajor>& G, Tensor<double,2,Eigen::RowMajor>& V, Tensor<double,3,Eigen::RowMajor>& F, double lr, double weight_GF);

void update_G_from_I(Tensor<double,3,Eigen::RowMajor>& G, Tensor<double,3,Eigen::RowMajor>& I_gradient, double weight_GI);

void update_I_from_V(Tensor<double,2,Eigen::RowMajor>& I, Tensor<double,2,Eigen::RowMajor>& cum_V, double weight_IV, double time_step);

void update_I_from_G(Tensor<double,2,Eigen::RowMajor>& I, Tensor<double,3,Eigen::RowMajor>& I_gradient, Tensor<double,3,Eigen::RowMajor>& G, double weight_IG);

void update_F_from_R(Tensor<double,3,Eigen::RowMajor>& F, Tensor<double,3,Eigen::RowMajor>& CCM, Tensor<double,3,Eigen::RowMajor>& Cx, Tensor<double,3,Eigen::RowMajor>& Cy, Tensor<double,1> R, double weight_FR);

void update_R_from_F(Tensor<double,1> R, Tensor<double,3,Eigen::RowMajor>& F, Tensor<double,3,Eigen::RowMajor>& C, Tensor<double,3,Eigen::RowMajor>& Cx, Tensor<double,3,Eigen::RowMajor>& Cy, double weight_RF, int N);

void vector_distance(const Tensor<double,3,Eigen::RowMajor> &vec1, const Tensor<double,3,Eigen::RowMajor> &vec2, Tensor<double,2,Eigen::RowMajor> &distance);

void m32(const Tensor<double,3,Eigen::RowMajor>& In, const Tensor<double,3,Eigen::RowMajor>& Cx, const Tensor<double,3,Eigen::RowMajor>& Cy, Tensor<double,3,Eigen::RowMajor>& Out);

void interacting_maps_step(Tensor<double,2,Eigen::RowMajor>& V, Tensor<double,2,Eigen::RowMajor>& cum_V, Tensor<double,2,Eigen::RowMajor>& I, Tensor<double,3,Eigen::RowMajor>& F, Tensor<double,3,Eigen::RowMajor>& G, Tensor<double,1>& R, const Tensor<double,3,Eigen::RowMajor>& CCM, const Tensor<double,3,Eigen::RowMajor>& dCdx, const Tensor<double,3,Eigen::RowMajor>& dCdy,  Matrix3f& A, std::vector<Matrix3f>& Identity_minus_outerProducts, std::unordered_map<std::string,double>& weights, std::vector<int>& permutation, const int N);

void interacting_maps(std::vector<Tensor<double,2,Eigen::RowMajor>>& Vs, std::vector<Tensor<double,2,Eigen::RowMajor>>& cum_Vs, std::unordered_map<std::string,double> weights, int iterations, std::string results_directory);

bool isApprox(Tensor<double,3,Eigen::RowMajor>& t1, Tensor<double,2,Eigen::RowMajor>& t2, double precision);

bool isApprox(Tensor<double,2,Eigen::RowMajor>& t1, Tensor<double,2,Eigen::RowMajor>& t2, double precision);

