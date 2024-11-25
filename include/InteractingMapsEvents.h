//
// Created by arbeit on 10/15/24.
//

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
struct Event{
    float time;
    std::vector<int> coordinates;
    float polarity;
    std::string toString() const;
};

struct Pixel{
    std::vector<int> coordinates;
    float V;
    float I;
    float G;
    float F;
    float Igrad;
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

std::vector<float> update_FG(std::vector<float> F, std::vector<float> G, std::vector<float> V, float weight);




