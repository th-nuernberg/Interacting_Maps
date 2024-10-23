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

namespace Eigen{
    typedef Matrix<float, Dynamic, Dynamic, RowMajor> MatrixXfRowMajor;
    typedef Matrix<double, Dynamic, Dynamic, RowMajor> MatrixXdRowMajor;
    typedef Matrix<int, Dynamic, Dynamic, RowMajor> MatrixXiRowMajor;
    typedef Tensor<float,1,RowMajor> Tensor1f;
    typedef Tensor<float,2,RowMajor> Tensor2f;
    typedef Tensor<float,3,RowMajor> Tensor3f;
}

struct Event{
    float time;
    std::vector<int> coordinates;
    float polarity;
    std::string toString() const;
};

struct Calibration_Data{
    std::vector<float> focal_point;
    MatrixXf camera_matrix;
    std::vector<float> distortion_coefficients;
    std::vector<float> view_angles;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  STRING OPERATIONS  /////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Enables use of Event in outstream
 * @param os outstream to add Event to
 * @param e Event to add
 * @return new outstream
 */
std::ostream& operator << (std::ostream &os, const Event &e);

/**
 * Splits a stringstream at a provided delimiter. Delimiter is removed
 * @param sstream stringstream to be split
 * @param delimiter The delimiter, can be any char
 * @return Vector of split string
 */
std::vector<std::string> split_string(std::stringstream sstream, char delimiter);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  GRADIENT CALCULATIONS  /////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Computes the two gradients of a single entry of a matrix. Direction is from left to right
 * and down to up.
 * @param data Matrix from which gradient is calculated
 * @param y height/row coordinate of the entry
 * @param x width/column coordinate of the entry
 * @return 2 dimensional Vector of the gradients. First entry is vertical, second entry is horizontal direction
 */
Vector2f computeGradient(const MatrixXfRowMajor &data, int y, int x);

/**
 * Computes the two gradients of a single entry of a 2Tensor. Direction is from left to right
 * and down to up.
 * @param data 2Tensor from which gradient is calculated
 * @param gradients 3Tensor which should contain resulting gradients, same size as data in first 2 dimensions
 * @param y height/row coordinate of the entry
 * @param x width/column coordinate of the entry
 */
void computeGradient(const Tensor2f& data, Tensor3f& gradients, int y, int x);

/**
 * Computes the gradients of a single entry of 3Tensor, where the tensor is understood as a 2D array
 * of vectors. y and x index the position of the vector in the array. Calculates the gradient in vertical (down-up) direction
 * for the first entry and the horizontal (left-right) direction for the second entry of the vector
 * @param data 2Tensor from which gradient is calculated
 * @param gradients 3Tensor which should contain resulting gradients, same size as data in dimensions
 * @param y height/row coordinate of the entry
 * @param x width/column coordinate of the entry
 */
void computeGradient(const Tensor3f& data, Tensor3f& gradients, int y, int x);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  CONVERSION FUNCTIONS  //////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Converts an Eigen 1Tensor to an Eigen Vector
 * @param input The tensor, any size
 * @return corresponding vector
 */
VectorXf Tensor2Vector(const Tensor1f& input);

/**
 * Converts an Eigen vector to an Eigen 1Tensor
 * @param input Eigen vector
 * @return Eigen 1Tensor
 */
Tensor1f Vector2Tensor(const VectorXf& input);

/**
 * Converts an Eigen 2Tensor to an Eigen Matrix
 * @param input Eigen 2Tensor, any size
 * @return corresponding matrix
 */
MatrixXfRowMajor Tensor2Matrix(const Tensor2f& input);

/**
 * Converts an Eigen matrix to an Eigen 2Tensor
 * @param input Eigen matrix
 * @return Eigen 2Tensor
 */
Tensor2f Matrix2Tensor(const MatrixXfRowMajor& input);

/**
 * Converts a Eigen matrix to a opencv matrix, without copying the data
 * @param eigen_matrix
 * @return opencv matrix
 */
cv::Mat eigenToCvMat(const MatrixXfRowMajor& eigen_matrix);

/**
 * Creates a copy of an Eigen matrix and saves it in a opencv matrix
 * @param eigen_matrix
 * @return opencv matrix
 */
cv::Mat eigenToCvMatCopy(const MatrixXfRowMajor& eigen_matrix);

/**
 * Converts a opencv matrix to an Eigen matrix without creating a copy
 * @param mat opencv matrix
 * @return eigen matrix
 */
MatrixXfRowMajor cvMatToEigen(const cv::Mat& mat);

/**
 * Converts a opencv matrix to an Eigen matrix while creating a copy
 * @param mat opencv matrix
 * @return eigen matrix
 */
MatrixXfRowMajor cvMatToEigenCopy(const cv::Mat& mat);

/**
 * Converts a opencv matrix from integer [0,255] to a float [0,1] matrix
 * @param mat
 * @return
 */
cv::Mat convertTofloat(cv::Mat& mat);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  OpenCV FUNCTIONS  //////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Takes a frame and applies the camera_matrix and distortiona parameters to undistort the image
 * @param frame Image in form of an Eigen matrix
 * @param camera_matrix contains camera parameters like focal length; in form of an opencv matrix
 * @param distortion_parameters contains distortion parameters for radial distortion, etc. opencv matrix
 * @return Image in form of an Eigen matrix
 */
MatrixXfRowMajor undistort_frame(const MatrixXfRowMajor& frame, const cv::Mat& camera_matrix, const cv::Mat& distortion_parameters);

/**
 * Takes an image and undistorts it.
 * @param image Distorted image in form of an opencv matrix
 * @param camera_matrix contains camera parameters like focal length; in form of an opencv matrix
 * @param distortion_parameters contains distortion parameters for radial distortion, etc. opencv matrix
 * @return Undistorted image in form of an opencv matrix
 */
cv::Mat undistort_image(const cv::Mat& image, const cv::Mat& camera_matrix, const cv::Mat& distortion_parameters);

/**
 * Takes a vector of images to undisort
 * @param images std::vector of distorted images in form of opencv matrices
 * @param camera_matrix contains camera parameters like focal length; in form of an opencv matrix
 * @param distortion_parameters contains distortion parameters for radial distortion, etc. opencv matrix
 * @return std::vector of undistorted images
 */
std::vector<cv::Mat> undistort_images(const std::vector<cv::Mat>& images, const MatrixXfRowMajor& camera_matrix, const MatrixXfRowMajor& distortion_coefficients);

/**
 * receives a frame and converts it to an greyscale image
 * @param frame in form of an Eigen matrix
 * @return opencv matrix of the greyscale image
 */
cv::Mat frame2grayscale(const MatrixXfRowMajor& frame);

/**
 * Converts the an Event frame to an image. positive polarity results in green coloring, negative in red.
 * @param V agglomerated Events in a Eigen matrix
 * @param cutoff Events with intensity less than cutoff are not visualised
 * @return opencv matrix of the colorcoded event frame
 */
cv::Mat V2image(const MatrixXfRowMajor& V, const float cutoff);

/**
 * Converts a vector field to an opencv image. Vectors are color coded according to their direction
 * @param vector_field 3Tensor containing a 2 dimensional vector for each pixel of the image
 * @return returns a bgr_image as an opencv matrix
 */
cv::Mat vector_field2image(const Tensor3f& vector_field);

/**
 * Creates a 3 Tensor of a square 2D grid with 2D vectors which point from the center radially outward
 * @param grid_size square side length of the grid
 * @return 3 Tensor of 2D vectors
 */
Tensor3f create_outward_vector_field(int grid_size);

/**
 * Creates a image mask for a square image which only shows a circular ring around the center of the image
 * @param image_size
 * @param inner_radius
 * @param outer_radius
 * @return
 */
cv::Mat create_circular_band_mask(const cv::Size& image_size, float inner_radius, float outer_radius);

/**
 * Creates a square image colored ring to visualise the direction vectors are pointing in a vectorfield
 * @param grid_size Size if the image of the colored ring
 * @return bgr image of the ring in form of a opencv matrix
 */
cv::Mat create_colorwheel(int grid_size);

/**
 * Visualise the Event information V, image I, spatial gradient field G, and optical flow F in a single image
 * @param V Eigen matrix of Event frame
 * @param I Eigen matrix of light intensities
 * @param G 3 Tensor containing the spatial gradients of the image
 * @param F 3 Tensor containing the optical flow of the image
 * @param path where to save the image on the disk if desired
 * @param save if a save to disk is desired
 * @param cutoff Events with intensity less than cutoff are not visualised
 * @return bgr image of the joined visualisation as opencv matrix
 */
cv::Mat create_VIGF(const MatrixXfRowMajor& V, const MatrixXfRowMajor& I, const Tensor3f& G, const Tensor3f& F, const std::string& path, const bool save, const float cutoff);

/**
 * Creates a colorbar of given height and width representing given max and min values with a specific colormap
 * @param globalMin minimum value of the colorbar
 * @param globalMax maximum value of the colorbar
 * @param height height of the colorbar image
 * @param width width of the colorbar image
 * @param colormapType which colormap to use, default is VIRIDIS
 * @return returns an image of the colormap to be included in plots
 */
cv::Mat createColorbar(double globalMin, double globalMax, int height, int width, int colormapType);

/**
 * Plots the Event information (as stand in for the temporal gradient) against the dot product of
 * the spatial gradient G and the optical flow F
 * @param V Event information as Eigen matrix
 * @param F Optical flow as 3Tensor
 * @param G Spatial gradient as 3Tensor
 * @param path where to save the image on the disk if desired
 * @param save if a save to disk is desired
 * @return bgr image of the plot as opencv matrix
 */
cv::Mat plot_VvsFG(const MatrixXfRowMajor& V, const Tensor3f& F, const Tensor3f& G, const std::string& path, bool save);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  FILE FUNCTIONS  ////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * saves a 3Tensor as string to a file on disk
 * @param t 3Tensor to be saved
 * @param fileName path on disk
 */
void writeToFile(const Tensor3f& t, const std::string fileName);

/**
 * saves a 2Tensor as string to a file on disk
 * @param t 2Tensor to be saved
 * @param fileName path on disk
 */
void writeToFile(const Tensor2f& t, const std::string fileName);

/**
 * saves a Eigen matrix as string to a file on disk
 * @param t matrix to be saved
 * @param fileName path on disk
 */
void writeToFile(const MatrixXfRowMajor& t, const std::string fileName);

/**
 * Creates a results at current directory/output/name and ads the path to .gitignore to contain git bloat
 * @param folder_name path to folder which is to be created
 * @return path to created folder
 */
fs::path create_folder_and_update_gitignore(const std::string& folder_name);

/**
 * Reads out a single line text file consisting of a string of float
 * @param file_path path to the file
 * @param calibration_data std::vector of contained floats
 */
void read_single_line_txt(const std::string& file_path, std::vector<float>& calibration_data);

/**
 * Converts a vector of calibration data floats to an Calibration_Data struct for further use
 * @param raw_data std::vector of floats of calibration data
 * @param frame_height height of the image for which the calibration data is for
 * @param frame_width width of the image for which the calibration data is for
 * @return Combined Calibration_Data
 */
Calibration_Data get_calibration_data(const std::vector<float>& raw_data, int frame_height, int frame_width);

/**
 * Reads out events from disk and converts it to a std::vector of Events. Expects a file with on event
 * per line in chronological order.
 * @param file_path Path to the event file
 * @param events Empty vector in which the events get written
 * @param start_time time stamp from which on out to consider frames
 * @param end_time time stamp after which events get ignored
 * @param event_factor allows scaling the event intensity with an factor
 * @param max_events upper limit on the amount of events to save if end_time is not reached before
 */
void read_events(const std::string& file_path, std::vector<Event>& events, float start_time, float end_time, float event_factor, int max_events);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// EVENT HANDLING  //////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 *
 * @param events
 * @param bin_size
 * @return
 */
std::vector<std::vector<Event>> bin_events(std::vector<Event>& events, float bin_size);

void create_frames(const std::vector<std::vector<Event>>& bucketed_events, std::vector<Tensor2f>& frames, const int camera_height, const int camera_width);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  INTERACTING MAPS HELPER FUNCTIONS  /////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//TODO ensure Tensors are actually const
bool isApprox(Tensor3f& t1, Tensor3f& t2, const float precision);

bool isApprox(Tensor2f& t1, Tensor2f& t2, const float precision);

void norm_tensor_along_dim3(const Tensor3f& T, Tensor2f& norm);

// Function to compute C_star
autodiff::Vector3real C_star(autodiff::real x, autodiff::real y, int N_x, int N_y, float height, float width, float rs);

// Function to compute C
autodiff::Vector3real C(autodiff::real x, autodiff::real y, int N_x, int N_y, float height, float width, float rs);

// Jacobian for x value tested by hand
void find_C(int N_x, int N_y, float view_angle_x, float view_angle_y, float rs, Tensor3f& CCM, Tensor3f& C_x, Tensor3f& C_y);

// Function to compute the cross product of two 3-tensors
void crossProduct3x3(const Tensor3f& A, const Tensor3f& B, Tensor3f& C);

// Function to compute the cross product of two 3-tensors
void crossProduct3x3(const Tensor3f& A, const Vector3f& B, Vector3f& C, int y, int x);

void crossProduct1x3(const Tensor<float,1>& A, const Tensor3f& B, Tensor3f& C);

void vector_distance(const Tensor3f &vec1, const Tensor3f &vec2, Tensor2f &distance);

float sign_func(float x);

// Function to time the performance of a given dot product computation function
template<typename Func>
void timeDotProductComputation(Func func, const Tensor3f& A, const Tensor3f& B, Tensor2f& D, int iterations);

// Function using nested loops to compute the dot product
void computeDotProductWithLoops(const Tensor3f& A, const Tensor3f& B, Tensor2f& D);
// Function using .chip() to compute the dot product
void computeDotProductWithChip(const Tensor3f& A, const Tensor3f& B, Tensor2f& D);

void m32(const Tensor3f &In, const Tensor3f &C_x, const Tensor3f &C_y, Tensor3f &Out);

void m23(const Tensor3f& In, const Tensor3f& Cx, const Tensor3f& Cy, Vector3f& Out, int y, int x);

void setup_R_update(const Tensor3f& CCM, Matrix3f& A, Vector3f& B, std::unique_ptr<Matrix3f[]>& Identity_minus_outerProducts, std::unique_ptr<Vector3f[]>& points);

float VFG_check(Tensor2f& V, Tensor3f& F, Tensor3f& G, float precision);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  INTERACTING MAPS UPDATE FUNCTIONS  /////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void update_FG(Tensor3f& F, Tensor2f& V, Tensor3f& G, int y, int x, const float lr, const float weight_FG, float eps, float gamma);

void update_GF(Tensor3f& G, Tensor2f V, Tensor3f& F, int y, int x, const float lr, const float weight_GF, float eps, float gamma);

void update_GI(Tensor3f& G, const Tensor3f& I_gradient, const int y, const int x, const float weight_GI, const float eps, const float gamma);

void update_IV(Tensor2f& I, Tensor2f& MI, int y, int x, const float weight_IV, const float time_step);

void updateGIDiffGradient(Tensor3f& G, Tensor3f& I_gradient, Tensor3f& GIDiff, Tensor3f& GIDiffGradient, int y, int x);

void update_IG(Tensor2f& I, Tensor3f& GIDiffGradient, int y, int x, const float weight_IG);

void update_FR(Tensor3f& F, const Tensor3f& CCM, const Tensor3f& Cx, const Tensor3f& Cy, const Tensor<float,1>& R, const float weight_FR, float eps, float gamma);

void update_RF(Tensor<float,1>& R, const Tensor3f& F, const Tensor3f& C, const Tensor3f& Cx, const Tensor3f& Cy, const Matrix3f& A, Vector3f& B, const std::unique_ptr<Matrix3f[]>& Identity_minus_outerProducts, Vector3f& old_point, const int y, const int x, const float weight_RF);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  INTERACTING MAPS MAIN FUNCTION  ////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void event_step(Tensor2f& V, Tensor2f& MI, Tensor2f& I, Tensor3f& delta_I, Tensor3f& GIDiff, Tensor3f& GIDiffGradient, Tensor3f& F, Tensor3f& G, Tensor<float,1>& R, const Tensor3f& CCM, const Tensor3f& dCdx, const Tensor3f& dCdy, const Matrix3f& A, Vector3f& B, const std::unique_ptr<Matrix3f[]>& Identity_minus_outerProducts, Vector3f& old_point, std::unordered_map<std::string,float>& weights, std::vector<int>& permutation, int y, int x);