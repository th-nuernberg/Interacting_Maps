//
// Created by daniel on 11/25/24.
//

#ifndef INTERACTINGMAPS_IMAGING_H
#define INTERACTINGMAPS_IMAGING_H
#include "datatypes.h"
#include <opencv2/opencv.hpp>


MatrixXfRowMajor undistort_frame(const MatrixXfRowMajor &frame, const cv::Mat &camera_matrix, const cv::Mat &distortion_parameters);

cv::Mat undistort_image(const cv::Mat &image, const cv::Mat &camera_matrix, const cv::Mat &distortion_parameters);

std::vector<cv::Mat> undistort_images(const std::vector<cv::Mat> &images, const MatrixXfRowMajor &camera_matrix, const MatrixXfRowMajor &distortion_coefficients);

cv::Mat frame2grayscale(const MatrixXfRowMajor &frame);

cv::Mat V2image(const MatrixXfRowMajor &V, float cutoff);

cv::Mat vector_field2image(const Tensor3f &vector_field);


Tensor3f create_outward_vector_field(int grid_size);


cv::Mat create_circular_band_mask(const cv::Size &image_size, float inner_radius, float outer_radius);


cv::Mat create_colorwheel(int grid_size);


cv::Mat create_VIGF(const MatrixXfRowMajor &V, const MatrixXfRowMajor &I, const Tensor3f &G, const Tensor3f &F, const std::string &path, bool save, float cutoff);


cv::Mat createColorbar(double globalMin, double globalMax, int height, int width, int colormapType);


cv::Mat plot_VvsFG(const MatrixXfRowMajor &V, const Tensor3f &F, const Tensor3f &G, const std::string &path, bool save);
#endif //INTERACTINGMAPS_IMAGING_H
