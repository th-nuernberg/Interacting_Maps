#pragma once
#include <iostream>
#include <vector>
#include <string>
#include "../eigen/Eigen/Dense"
#include "../eigen/Eigen/Sparse"
#include "../eigen/unsupported/Eigen/CXX11/Tensor"
#include <opencv2/opencv.hpp>
using namespace Eigen;

typedef SparseMatrix<float> SpMat;

struct Event{
    float time;
    std::vector<int> coordinates;
    int polarity;

    std::string toString() const;
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

std::string  create_folder_and_update_gitignore(const std::string& foldername);

void read_calib(const std::string file, std::vector<float>& calibration_data);

void read_events(const std::string file_path, std::vector<Event>& events, float start_time, float end_time, int max_events);

std::vector<std::vector<Event>> bin_events(std::vector<Event>& events, float bucketsize);

void create_frames(std::vector<std::vector<Event>>& bucketed_events, std::vector<Tensor<float,2>>& frames, int camera_height, int camera_width);

void create_sparse_matrix(int N, Tensor<float,2>& V, SpMat& result);

cv::Mat undistort_image(const cv::Mat& image, const cv::Mat& camera_matrix, const cv::Mat& distortion_parameters);

// void undistort_images(std::vector<Tensor<float,2>>& images, Tensor<float,2> camera_matrix, std::vector<float> distortion_parameters, int height, int width);

// void undistort_frames(std::vector<Tensor<float,2>>& frames, Tensor<float,2> camera_matrix, std::vector<float> distortion_parameters, int height, int width);

cv::Mat frame2grayscale(const Eigen::MatrixXf& frame);

cv::Mat V2image(const Eigen::MatrixXf& V);

void vectorfield2image(Tensor<float,3> vectorfield);

void create_VIFG_image(Tensor<float,2>& V, Tensor<float,2>& I, Tensor<float,3>& F, Tensor<float,3>& G, Tensor<float,2>& VIFG);

// void find_C(int N_x, int N_y, float view_angle_x, float view_angle_y, float rs, Tensor<float,3>& C, Tensor<float,3>& dxdC, Tensor<float,3>& dydC);
void find_C(int N_x, int N_y, float view_angle_x, float view_angle_y, float rs, Tensor<float,3>& CCM, Tensor<float,3>& C_x, Tensor<float,3>& C_y);

void update_F_from_G(Tensor<float,3>& F, Tensor<float,2>& V, Tensor<float,3>& G, float lr, float weight_FG);

void update_G_from_F(Tensor<float,3>& G, Tensor<float,2>& V, Tensor<float,3>& F, float lr, float weight_GF);

void update_G_from_I(Tensor<float,3>& G, Tensor<float,3>& I_gradient, float weight_GI);

void update_I_from_V(Tensor<float,2>& I, Tensor<float,2>& cum_V, float weight_IV, float time_step);

void update_I_from_G(Tensor<float,2>& I, Tensor<float,3>& I_gradient, Tensor<float,3>& G, float weight_IG);

void update_F_from_R(Tensor<float,3>& F, Tensor<float,3>& C, Tensor<float,3>& Cx, Tensor<float,3>& Cy, Tensor<float,1> R, float weight_FR);

void update_R_from_F(Tensor<float,1> R, Tensor<float,3>& F, Tensor<float,3>& C, Tensor<float,3>& Cx, Tensor<float,3>& Cy, float weight_RF, int N);

void vector_distance(Tensor<float,3> &vec1, Tensor<float,3> &vec2, Tensor<float,2> &distance);

void m23(Tensor<float,3>& In, Tensor<float,3>& Cx, Tensor<float,3>& Cy, Tensor<float,3>& Out);

void m32(Tensor<float,3>& In, Tensor<float,3>& Cx, Tensor<float,3>& Cy, Tensor<float,3>& Out);

void interacting_maps_step(Tensor<float,2>& V, Tensor<float,2>& cum_V, Tensor<float,2>& I, Tensor<float,3>& F, Tensor<float,3>& G, Tensor<float,1>& R, Tensor<float,3>& CCM, Tensor<float,3>& dCdx, Tensor<float,3>& dCdy, std::unordered_map<std::string,float> weights, std::vector<int> permutation, int N);

void interacting_maps(std::vector<Tensor<float,2>>& Vs, std::vector<Tensor<float,2>>& cum_Vs, std::unordered_map<std::string,float> weights, int iterations, std::string results_directory);



