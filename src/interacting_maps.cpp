#include <interacting_maps.h>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <sstream>
#include <cassert>
#include <numeric>
#include "Instrumentor.h"
#include <cmath>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define PI 3.14159265
#define EXECUTE_TEST 0
#define SMALL_MATRIX_METHOD 1

// Define DEBUG_LOG macro that logs with function name in debug mode
#ifdef DEBUG
#define DEBUG_LOG(message) \
        std::cout << "DEBUG (" << __func__ << "): " << message << std::endl << \
        "###########################################" << std::endl;
#else
#define DEBUG_LOG(message) // No-op in release mode
#endif

#define PROFILING 1
#if PROFILING
#define PROFILE_SCOPE(name) InstrumentationTimer timer##__LINE__(name)
#define PROFILE_FUNCTION() PROFILE_SCOPE(__FUNCTION__)
// #define PROFILE_FUNCTION() PROFILE_SCOPE(__FUNCSIG__) (Includes call attributes, whole signature of function)
#else
#define PROFILE_SCOPE(name)
#define PROFILE_FUNCTION()
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  STRING OPERATIONS  /////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::ostream& operator << (std::ostream &os, const Event &e) {
    return (os << "Time: " << e.time << " Coords: " << e.coordinates[0] << " " << e.coordinates[1] << " Polarity: " << e.polarity);
}

std::string Event::toString() const {
    std::stringstream ss;
    ss << (*this);
    return ss.str();
}

std::vector<std::string> split_string(std::stringstream sstream, char delimiter){
    std::string segment;
    std::vector<std::string> seglist;
    while(std::getline(sstream, segment, delimiter))
    {
        seglist.push_back(segment);
    }
    return seglist;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  GRADIENT CALCULATIONS  /////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Vector2f computeGradient(const MatrixXfRowMajor &data, int y, int x) {
    PROFILE_FUNCTION();
    int rows = data.rows();
    int cols = data.cols();
    assert(y < rows);
    assert(x < cols);
    Vector2f gradients;
    // Compute gradient along columns (down-up, y-direction)
    if (y == 0) {
        gradients(0) = (data(y, x) - data(y + 1, x)) / 2.0; // Central difference with replicate border
    } else if (y == rows - 1) {
        gradients(0) = (data(y - 2, x) - data(y - 1, x)) / 2.0; // Central difference with replicate border
    } else {
        gradients(0) = (data(y - 1, x) - data(y + 1, x)) / 2.0;
    }
    // Compute gradient along rows (left-right, x-direction)
    if (x == 0) {
        gradients(1) = (data(y, x + 1) - data(y, x)) / 2.0; // Central difference with replicate border
    } else if (x == cols - 1) {
        gradients(1) = (data(y, x - 1) - data(y, x - 2)) / 2.0; // Central difference with replicate border
    } else {
        gradients(1) = (data(y, x + 1) - data(y, x - 1)) / 2.0;
    }
    return gradients;
}

void computeGradient(const Tensor2f& data, Tensor3f& gradients, int y, int x) {
    PROFILE_FUNCTION();
    // Compute gradient for update_IG
    const auto& ddimensions = data.dimensions();
    int rows = ddimensions[0];
    int cols = ddimensions[1];
    const auto& gdimensions = gradients.dimensions();
    assert(ddimensions[0] == gdimensions[0]);
    assert(ddimensions[1] == gdimensions[1]);
    assert(y < rows);
//    if (x >= cols){
//        std::cout << x << "too big" << std::endl;
//    }
    assert(x < cols);

    // Compute gradient along columns (down-up, y-direction)
    if (y == 0) {
        gradients(y, x, 0) = (data(y, x) - data(y + 1, x)) / 2.0; // Central difference with replicate border
    } else if (y == rows - 1) {
        gradients(y, x, 0) = (data(y - 2, x) - data(y - 1, x)) / 2.0; // Central difference with replicate border
    } else {
        gradients(y, x, 0) = (data(y - 1, x) - data(y + 1, x)) / 2.0;
    }
    // Compute gradient along rows (left-right, x-direction)
    if (x == 0) {
        gradients(y, x, 1) = (data(y, x + 1) - data(y, x)) / 2.0; // Central difference with replicate border
    } else if (x == cols - 1) {
        gradients(y, x, 1) = (data(y, x - 1) - data(y, x - 2)) / 2.0; // Central difference with replicate border
    } else {
        gradients(y, x, 1) = (data(y, x + 1) - data(y, x - 1)) / 2.0;
    }
}

void computeGradient(const Tensor3f& data, Tensor3f& gradients, int y, int x) {
    // Compute gradient for update_IG
    const auto& dimensions = data.dimensions();
    int rows = dimensions[0];
    int cols = dimensions[1];
    assert(y < rows);
    assert(x < cols);
    // Compute gradient along columns (down-up, y-direction)
    if (y == 0) {
        gradients(y, x, 0) = (data(y, x, 0) - data(y + 1, x, 0)) / 2.0; // Central difference with replicate border
    } else if (y == rows - 1) {
        gradients(y, x, 0) = (data(y - 2, x, 0) - data(y - 1, x, 0)) / 2.0; // Central difference with replicate border
    } else {
        gradients(y, x, 0) = (data(y - 1, x, 0) - data(y + 1, x, 0)) / 2.0;
    }
    // Compute gradient along rows (left-right, x-direction)
    if (x == 0) {
        gradients(y, x, 1) = (data(y, x + 1, 1) - data(y, x, 1)) / 2.0; // Central difference with replicate border
    } else if (x == cols - 1) {
        gradients(y, x, 1) = (data(y, x - 1, 1) - data(y, x - 2, 1)) / 2.0; // Central difference with replicate border
    } else {
        gradients(y, x, 1) = (data(y, x + 1, 1) - data(y, x - 1, 1)) / 2.0;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  CONVERSION FUNCTIONS  //////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

VectorXf Tensor2Vector(const Tensor1f& input) {
    // RowMajor Version
    array<Index, 1> dims = input.dimensions();
    const float* data_ptr = input.data();
    Map<const VectorXf> result(data_ptr, dims[0]);
    return result;
}

Tensor1f Vector2Tensor(const VectorXf& input) {
    const int cols = input.cols();
    const float* data_ptr = input.data();
    TensorMap<const Tensor1f> result(data_ptr, cols);
    return result;
}

MatrixXfRowMajor Tensor2Matrix(const Tensor2f& input){
    array<Index, 2> dims = input.dimensions();
    const float* data_ptr = &input(0); // Points to beginning of array;
    Map<const MatrixXfRowMajor> result(data_ptr, dims[0], dims[1]);
    return result;
}

Tensor2f Matrix2Tensor(const MatrixXfRowMajor& input) {
    // Get Pointer to data
    float const *data_ptr = &input(0);
    // Map data to Tensor
    TensorMap<const Tensor2f> result(data_ptr, input.rows(), input.cols());
    // Swap the layout and preserve the order of the dimensions
    return result;
}

cv::Mat eigenToCvMat(const MatrixXfRowMajor& eigen_matrix) {
    // Map Eigen matrix data to cv::Mat without copying
    return cv::Mat(eigen_matrix.rows(), eigen_matrix.cols(), CV_32F, (void*)eigen_matrix.data());
}

cv::Mat eigenToCvMatCopy(const MatrixXfRowMajor& eigen_matrix) {
    // Create a cv::Mat and copy Eigen matrix data into it
    cv::Mat mat(eigen_matrix.rows(), eigen_matrix.cols(), CV_32F);
    for (int i = 0; i < eigen_matrix.rows(); ++i) {
        for (int j = 0; j < eigen_matrix.cols(); ++j) {
            mat.at<float>(i, j) = eigen_matrix(i, j);
        }
    }
    return mat;
}

MatrixXfRowMajor cvMatToEigen(const cv::Mat& mat) {
    // Ensure the cv::Mat has the correct type
    CV_Assert(mat.type() == CV_32F);
    array<Index, 2> dims;
    dims[0] = mat.rows;
    dims[1] = mat.cols;
    const float* data_ptr = mat.ptr<float>();
    Map<const MatrixXfRowMajor> result(data_ptr, dims[0], dims[1]);
    return result;
}

MatrixXfRowMajor cvMatToEigenCopy(const cv::Mat& mat) {
    // Ensure the cv::Mat has the correct type
    CV_Assert(mat.type() == CV_32F);
    MatrixXfRowMajor eigen_matrix(mat.rows, mat.cols);
    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j) {
            eigen_matrix(i, j) = mat.at<float>(i, j);
        }
    }
    return eigen_matrix;
}

cv::Mat convertTofloat(cv::Mat& mat) {
    // Ensure the source matrix is of type CV_8U
    CV_Assert(mat.type() == CV_8U);

    // Convert the source matrix to CV_32F
    mat.convertTo(mat, CV_32F, 1.0 / 255.0); // Scaling from [0, 255] to [0, 1]

    return mat;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  OpenCV FUNCTIONS  //////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

MatrixXfRowMajor undistort_frame(const MatrixXfRowMajor& frame, const cv::Mat& camera_matrix, const cv::Mat& distortion_parameters) {
    cv::Mat image = eigenToCvMat(frame);
    return cvMatToEigen(undistort_image(image, camera_matrix, distortion_parameters));
}

cv::Mat undistort_image(const cv::Mat& image, const cv::Mat& camera_matrix, const cv::Mat& distortion_parameters) {
    // cv::Mat new_camera_matrix;
    // cv::Rect roi;
    // new_camera_matrix = cv::getOptimalNewCameraMatrix(camera_matrix, distortion_parameters, cv::Size(width, height), 1, cv::Size(width, height), &roi);
    cv::Mat undistorted_image;
    cv::undistort(image, undistorted_image, camera_matrix, distortion_parameters, camera_matrix);
    return undistorted_image;
}

std::vector<cv::Mat> undistort_images(const std::vector<cv::Mat>& images, const MatrixXfRowMajor& camera_matrix, const MatrixXfRowMajor& distortion_coefficients) {
    std::vector<cv::Mat> undistorted_images;

    // Convert Eigen matrices to cv::Mat
    cv::Mat camera_matrix_cv = eigenToCvMat(camera_matrix);
    cv::Mat distortion_coefficients_cv = eigenToCvMat(distortion_coefficients);

    for (const auto& image : images) {
        undistorted_images.push_back(undistort_image(image, camera_matrix_cv, distortion_coefficients_cv));
    }
    return undistorted_images;
}

// Function to convert MatrixXfRowMajor to grayscale image
cv::Mat frame2grayscale(const MatrixXfRowMajor& frame) {
    // Convert MatrixXfRowMajor to cv::Mat
    cv::Mat frame_cv = eigenToCvMat(frame);
    cv::Mat output;

    // Find min and max polarity
    double min_polarity, max_polarity;
    cv::minMaxLoc(frame_cv, &min_polarity, &max_polarity);

    // Normalize the frame
    frame_cv.convertTo(frame_cv, CV_32FC3, 1.0 / (max_polarity - min_polarity), -min_polarity / (max_polarity - min_polarity));

    // Scale to 0-255 and convert to CV_8U
    frame_cv.convertTo(output, CV_8UC1, 255.0);

    return output;
}

cv::Mat V2image(const MatrixXfRowMajor& V, const float cutoff=0.1) {
    // Determine the shape of the image
    long rows = V.rows();
    long cols = V.cols();
    
    // Create an empty image with 3 channels (BGR)
    cv::Mat image = cv::Mat::zeros(rows, cols, CV_8UC3);
    
    // Process on_events (V > 0)
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (V(i, j) > cutoff) {
                image.at<cv::Vec3b>(i, j)[1] = 255; // Set green channel
            }
            if (V(i, j) < -cutoff) {
                image.at<cv::Vec3b>(i, j)[2] = 255; // Set red channel
            }
        }
    }
    
    return image;
}

cv::Mat vector_field2image(const Tensor3f& vector_field) {
    const int rows = vector_field.dimension(0);
    const int cols = vector_field.dimension(1);

    // Calculate angles and saturations
    MatrixXfRowMajor angles(rows, cols);
    MatrixXfRowMajor saturations(rows, cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float y = vector_field(i, j, 0);
            float x = vector_field(i, j, 1);
            float angle = std::atan2(y, x);
            if (angle < 0){
                angle += 2*PI;
            }
            angles(i,j) = angle;
            saturations(i, j) = std::sqrt(x * x + y * y);
        }
    }

    // Normalize angles to [0, 179]
    cv::Mat hue(rows, cols, CV_8UC1);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            hue.at<uint8_t>(i, j) = static_cast<uint8_t>((angles(i, j))/(2 * M_PI) * 179);
        }
    }

    // Normalize saturations to [0, 255]
    float max_saturation = saturations.maxCoeff();
    cv::Mat saturation(rows, cols, CV_8UC1);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            saturation.at<uint8_t>(i, j) = static_cast<uint8_t>(std::min(saturations(i, j) / max_saturation * 255.0, 255.0));
        }
    }

    // Value channel (full brightness)
    cv::Mat value(rows, cols, CV_8UC1, cv::Scalar(255));

    // Merge HSV channels
    std::vector<cv::Mat> hsv_channels = {hue, saturation, value};
    cv::Mat hsv_image;
    cv::merge(hsv_channels, hsv_image);

    // Convert HSV image to BGR format
    cv::Mat bgr_image;
    cv::cvtColor(hsv_image, bgr_image, cv::COLOR_HSV2BGR);

    return bgr_image;
}

// Function to create outward vector field using Eigen
Tensor3f create_outward_vector_field(int grid_size) {
    // Create a 2D grid of points using linspace equivalent
    VectorXf x = VectorXf::LinSpaced(grid_size, -1.0, 1.0);
    VectorXf y = VectorXf::LinSpaced(grid_size, 1.0, -1.0);

    // Initialize matrices for meshgrid-like behavior (xv, yv)
    MatrixXfRowMajor xv(grid_size, grid_size);
    MatrixXfRowMajor yv(grid_size, grid_size);

    // Create meshgrid by repeating x and y values across rows and columns
    for (int i = 0; i < grid_size; ++i) {
        xv.row(i) = x.transpose();
        yv.col(i) = y;
    }

    // Create a 3D Tensor to store the 2D vector field (x, y components at each point)
    Tensor3f vectors(grid_size, grid_size, 2);

    // Calculate distances from origin and store the normalized vectors
    for (int i = 0; i < grid_size; ++i) {
        for (int j = 0; j < grid_size; ++j) {
            float x_val = xv(i, j);
            float y_val = yv(i, j);
            float distance = std::sqrt(x_val * x_val + y_val * y_val);

            // Prevent division by zero at the origin
            if (distance == 0) distance = 1.0;

            // Normalize the vector and assign it
            vectors(i, j, 1) = x_val / distance;
            vectors(i, j, 0) = y_val / distance;
        }
    }

    return vectors;
}

// Function to create the circular band mask
cv::Mat create_circular_band_mask(const cv::Size& image_size, float inner_radius, float outer_radius) {
    cv::Mat mask(image_size, CV_8UC1, cv::Scalar(0));

    int center_x = image_size.width / 2;
    int center_y = image_size.height / 2;

    for (int i = 0; i < image_size.height; ++i) {
        for (int j = 0; j < image_size.width; ++j) {
            float distance = std::sqrt((i - center_y) * (i - center_y) + (j - center_x) * (j - center_x));
            if (distance >= inner_radius && distance <= outer_radius) {
                mask.at<uchar>(i, j) = 255;  // Inside the band
            }
        }
    }

    return mask;
}

// Function to create the color wheel with a circular band
cv::Mat create_colorwheel(int grid_size) {
    Tensor3f vector_field = create_outward_vector_field(grid_size);

    cv::Mat image;
    image = vector_field2image(vector_field);

    // Define inner and outer radius
    float inner_radius = grid_size / 4.0f;
    float outer_radius = grid_size / 2.0f;

    // Create the circular band mask
    cv::Mat mask = create_circular_band_mask(image.size(), inner_radius, outer_radius);

    // Apply mask to the image
    cv::Mat colourwheel = image.clone();
    colourwheel.setTo(cv::Scalar(255, 255, 255), ~mask);  // Outside the mask: set to white
    return colourwheel;
}

// Function to create the VIGF image
cv::Mat create_VIGF(const MatrixXfRowMajor& V, const MatrixXfRowMajor& I, const Tensor3f& G, const Tensor3f& F, const std::string& path = "VIGF", const bool save = false, const float cutoff=0.1) {
    cv::Mat V_img = V2image(V, cutoff);
    cv::Mat I_img = frame2grayscale(I);
    cv::Mat G_img = vector_field2image(G);
    cv::Mat F_img = vector_field2image(F);

    long rows = V.rows();
    long cols = V.cols();
    long I_rows = I.rows();
    long I_cols = I.cols();

    // Calculate the size of the color wheel
    int colourwheel_size = cols / 2;

    // Calculate the size of the final output image (double size + padding and color wheel)
    int y_size = rows * 2 + 20;
    int x_size = cols * 2 + 30 + colourwheel_size;

    cv::Mat image(y_size, x_size, CV_8UC3, cv::Scalar(0, 0, 0));
    std::cout << image.size() << std::endl;

    // Place V image
    V_img.copyTo(image(cv::Rect(5, 5, cols, rows)));

    // Place I image (convert to BGR first)
    cvtColor(I_img, I_img, cv::COLOR_GRAY2BGR);
    I_img.copyTo(image(cv::Rect(cols + 15 + colourwheel_size, 5, I_cols, I_rows)));

    // Place G image
    G_img.copyTo(image(cv::Rect(5, rows + 10, cols, rows)));

    // Place F image
    F_img.copyTo(image(cv::Rect(cols + 15 + colourwheel_size, rows + 10, cols, rows)));

    // Create and place the color wheel
    cv::Mat colourwheel = create_colorwheel(colourwheel_size);
    colourwheel.copyTo(image(cv::Rect(cols + 10, 2 * rows + 10 - colourwheel_size, colourwheel_size, colourwheel_size)));

    // Save the image if required
    if (save && !path.empty()) {
        cv::imwrite(path, image);
    }

    return image;
}

// Function to create a colorbar
cv::Mat createColorbar(double globalMin, double globalMax, int height, int width, int colormapType = cv::COLORMAP_VIRIDIS) {
    // Create a vertical gradient (values from globalMin to globalMax)
    cv::Mat colorbar(height, width, CV_32F);

    // Fill the colorbar with values from globalMin to globalMax
    for (int i = 0; i < height; ++i) {
        float value = globalMin + i * (globalMax - globalMin) / height;
        colorbar.row(i).setTo(value);
    }

    // Normalize the colorbar to the range [0, 255]
    cv::Mat colorbarNormalized;
    colorbar.convertTo(colorbarNormalized, CV_8U, 255.0 / (globalMax - globalMin), -255.0 * globalMin / (globalMax - globalMin));

    // Apply the same colormap to the colorbar
    cv::Mat colorbarColored;
    cv::applyColorMap(colorbarNormalized, colorbarColored, colormapType);

    // Add labels to the colorbar (min, max, and optionally intermediate ticks)
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 0.4;
    int thickness = 1;
    int baseline = 0;

    // Put the min value at the bottom
    std::string minLabel = std::to_string(globalMin);
    cv::Size minTextSize = cv::getTextSize(minLabel, fontFace, fontScale, thickness, &baseline);
    cv::putText(colorbarColored, minLabel, cv::Point(5, height - 5), fontFace, fontScale, cv::Scalar(255, 255, 255), thickness);

    // Put the max value at the top
    std::string maxLabel = std::to_string(globalMax);
    cv::Size maxTextSize = cv::getTextSize(maxLabel, fontFace, fontScale, thickness, &baseline);
    cv::putText(colorbarColored, maxLabel, cv::Point(5, maxTextSize.height), fontFace, fontScale, cv::Scalar(255, 255, 255), thickness);

    // Optionally, you can add intermediate ticks and labels here as needed
    return colorbarColored;
}

cv::Mat plot_VvsFG(const MatrixXfRowMajor& V, const Tensor3f& F, const Tensor3f& G, const std::string& path = "VvsFG", bool save = false){
    long rows = V.rows();
    long cols = V.cols();
    cv::Mat image(rows + 50, cols * 3 + 50, CV_8UC3, cv::Scalar(0, 0, 0));

    // Step 1: Prepare Matrices
    // Convert V to cvMat
    cv::Mat V_img;
    V_img = eigenToCvMat(V);

    // Calculate the dot product of F and G, convert to cvMat
    MatrixXfRowMajor FdotG(rows, cols);
    cv::Mat FdotG_img;
    FdotG = Tensor2Matrix(-(F*G).sum(array<int,1>({2})));
    FdotG_img = eigenToCvMat(FdotG); // Multiply FG elementwise then sum over 3rd dim > dot product

    // Calculate the difference between V and the dot product
    MatrixXfRowMajor diff(rows, cols);
    cv::Mat diff_img;
    diff = V - FdotG;
    diff_img = eigenToCvMat(diff);

    // Step 2: Find the global min and max values across all three matrices
    double minVal1, maxVal1, minVal2, maxVal2, minVal3, maxVal3;
    cv::minMaxLoc(V_img, &minVal1, &maxVal1);
    cv::minMaxLoc(FdotG_img, &minVal2, &maxVal2);
    cv::minMaxLoc(diff_img, &minVal3, &maxVal3);

    double globalMin = std::min({minVal1, minVal2, minVal3});
    double globalMax = std::max({maxVal1, maxVal2, maxVal3});

    // Step 3: Normalize all matrices to the global range [0, 255]
    cv::Mat normalizedMatrix1, normalizedMatrix2, normalizedMatrix3;
    // Normalize each matrix to the range [0, 255] using the global min and max
    V_img.convertTo(normalizedMatrix1, CV_8U, 255.0 / (globalMax - globalMin), -255.0 * globalMin / (globalMax - globalMin));
    FdotG_img.convertTo(normalizedMatrix2, CV_8U, 255.0 / (globalMax - globalMin), -255.0 * globalMin / (globalMax - globalMin));
    diff_img.convertTo(normalizedMatrix3, CV_8U, 255.0 / (globalMax - globalMin), -255.0 * globalMin / (globalMax - globalMin));

    // Step 4: Apply the same colormap to each normalized matrix
    cv::Mat coloredMatrix1, coloredMatrix2, coloredMatrix3;
    cv::applyColorMap(normalizedMatrix1, coloredMatrix1, cv::COLORMAP_VIRIDIS);
    cv::applyColorMap(normalizedMatrix2, coloredMatrix2, cv::COLORMAP_VIRIDIS);
    cv::applyColorMap(normalizedMatrix3, coloredMatrix3, cv::COLORMAP_VIRIDIS);

    // Step 5: Concatenate the three matrices horizontally or vertically for display
    cv::Mat combinedMatrix;
    cv::hconcat(std::vector<cv::Mat>{coloredMatrix1, coloredMatrix2, coloredMatrix3}, combinedMatrix);

    // Step 6: Create a colorbar (we'll use height of the matrix and width of 50 pixels for the colorbar)
    int colorbarWidth = 50;
    cv::Mat colorbar = createColorbar(globalMin, globalMax, combinedMatrix.rows, colorbarWidth);

    // Step 7: Concatenate the colorbar with the combined image
    cv::Mat coloredImage;
    cv::hconcat(combinedMatrix, colorbar, coloredImage);

    // Step 8: Add a black region at the top of the image for the title
    int titleHeight = 50;  // Height of the title area
    cv::Mat titleImage = cv::Mat::zeros(titleHeight, coloredImage.cols, CV_8UC3);

    // Step 9: Add the title text to the top region
    std::string title = "V | - F dot G | Difference";
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 0.5;
    int thickness = 2;
    cv::Scalar color(255, 255, 255); // White text
    int baseline = 0;

    // Get the text size to center the title
    cv::Size textSize = cv::getTextSize(title, fontFace, fontScale, thickness, &baseline);
    cv::Point textOrg((titleImage.cols - textSize.width) / 2, (titleHeight + textSize.height) / 2);

    // Put the title on the titleImage
    cv::putText(titleImage, title, textOrg, fontFace, fontScale, color, thickness);

    // Step 10: Concatenate the title image with the matrix image
    cv::Mat finalImage;
    cv::vconcat(titleImage, coloredImage, finalImage);

    // Step 11: Display the final image with the title
    if (save && !path.empty()) {
        imwrite(path, finalImage);
    }
    else{
        cv::imshow("V vs F dot G", finalImage);
        cv::waitKey(0); // Wait for a key press before closing the window
    }
    return finalImage;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  FILE FUNCTIONS  ////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void writeToFile(const Tensor3f& t, const std::string fileName){
    std::ofstream file(fileName);
    if (file.is_open())
    {
        file << t;
    }
}

void writeToFile(const Tensor2f& t, const std::string fileName){
    std::ofstream file(fileName);
    if (file.is_open())
    {
        file << t;
    }
}

void writeToFile(const MatrixXfRowMajor& t, const std::string fileName){
    std::ofstream file(fileName);
    if (file.is_open())
    {
        file << t;
    }
}

fs::path create_folder_and_update_gitignore(const std::string& folder_name) {
    // Get the absolute path of the current working directory
    fs::path current_directory = fs::current_path();
    
    // Create the output folder if it does not exist
    fs::path output_folder_path = current_directory / "output";
    // Create the directory if it does not exist
    if (!fs::exists(output_folder_path)) {
        fs::create_directory(output_folder_path);
    }

    // Same for the actual folder
    fs::path folder_path = output_folder_path / folder_name;
    if (!fs::exists(folder_path)) {
        fs::create_directory(folder_path);
    }
    
    // Path to the .gitignore file
    fs::path gitignore_path = current_directory/ ".gitignore";
    
    // Check if the folder is already in .gitignore
    bool folder_in_gitignore = false;
    if (fs::exists(gitignore_path)) {
        std::ifstream gitignore_file(gitignore_path);
        std::string line;
        while (std::getline(gitignore_file, line)) {
            if (line == folder_name || line == "/" + folder_name) {
                folder_in_gitignore = true;
                break;
            }
        }
    }
    
    // Add the folder to .gitignore if it's not already there
    if (!folder_in_gitignore) {
        std::ofstream gitignore_file(gitignore_path, std::ios_base::app);
        gitignore_file << "\n" << folder_name << "\n";
    }
    
    // Return the absolute path of the new folder
    return folder_path;
}

void read_single_line_txt(const std::string& file_path, std::vector<float>& calibration_data){
    fs::path current_directory = fs::current_path();
    std::string path = current_directory / file_path;
    if (fs::exists(path)) {
        std::ifstream calibration_file(path);
        std::string::size_type size;
        for (std::string line; std::getline(calibration_file, line, ' ');) {
            calibration_data.push_back(std::stof(line, &size));
        }
    }
}

Calibration_Data get_calibration_data(const std::vector<float>& raw_data, int frame_height, int frame_width){
    Calibration_Data data;
    data.focal_point = std::vector<float>(raw_data.begin(), raw_data.begin()+2);
    data.camera_matrix = MatrixXf({{data.focal_point[0], 0, raw_data[2]},
                                              {0, data.focal_point[1], raw_data[3]},
                                              {0, 0, 1}});
    data.distortion_coefficients = std::vector<float>(raw_data.begin()+4, raw_data.end());
    data.view_angles = std::vector<float>({2*std::atan(frame_height/(2*data.focal_point[0])),
                                            2*std::atan(frame_width/(2*data.focal_point[1]))});
    return data;
}

void read_events(const std::string& file_path, std::vector<Event>& events, float start_time, float end_time, int max_events = INT32_MAX){
    fs::path current_directory = fs::current_path();
    std::string path = current_directory / file_path;
    if (fs::exists(path)) {
        std::ifstream event_file(path);
        int counter = 0;
        float time;
        int width, height, polarity;
        while (event_file >> time >> width >> height >> polarity){
            if (time < start_time) continue;
            if (time > end_time) break;
            if (counter > max_events) break;
            Event event;
            event.time = time;
            std::vector<int> coords = {height, width};
            event.coordinates = coords;
            event.polarity = (polarity*2 - 1);
            events.push_back(event);
            counter++;
        }
        DEBUG_LOG("Final time stamp: " << time)
        DEBUG_LOG("Number of events: " << events.size())
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// EVENT HANDLING  //////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<std::vector<Event>> bin_events(std::vector<Event>& events, float bin_size = 0.05){
    std::vector<std::vector<Event>> bins;
    if (events.empty()) {
        return bins;  // Return empty if the input vector is empty
    }

    Event minVal = events.front();  // The lowest number in the sorted vector
    float currentBinStart = minVal.time;

    std::vector<Event> currentBin;

    for (Event event : events) {
        if (event.time >= currentBinStart && event.time < currentBinStart + bin_size) {
            currentBin.push_back(event);
        } else {
            // Push the current bin into bins and start a new bin
            bins.push_back(currentBin);
            currentBin.clear();
            currentBinStart += bin_size;
            // Keep adjusting the currentBinStart if the number falls outside the current bin
            while (event.time >= currentBinStart + bin_size) {
                currentBinStart += bin_size;
                bins.emplace_back(); // Add an empty bin for skipped bins
            }
            currentBin.push_back(event);
        }
    }
    // Push the last bin
    bins.push_back(currentBin);
    return bins;
}

void create_frames(const std::vector<std::vector<Event>>& bucketed_events, std::vector<Tensor2f>& frames, const int camera_height, const int camera_width, float eventContribution){
    int i = 0;
    Tensor2f frame(camera_height, camera_width);
    Tensor2f cum_frame(camera_height, camera_width);
    for (std::vector<Event> event_vector : bucketed_events){

        frame.setZero();
        cum_frame.setZero();
        for (Event event : event_vector){
//            std::cout << event << std::endl;
            frame(event.coordinates.at(0), event.coordinates.at(1)) = event.polarity * eventContribution;
//            cum_frame(event.coordinates.at(0), event.coordinates.at(1)) += (float)event.polarity;
        }
        frames[i] = frame;
        i++;

        DEBUG_LOG("Eventvector size: " << event_vector.size());
        DEBUG_LOG("Last Event: " << event_vector.back());
    }

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  INTERACTING MAPS HELPER FUNCTIONS  /////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//TODO ensure Tensors are actually const
bool isApprox(Tensor3f& t1, Tensor3f& t2, const float precision = 1e-8){
    Map<VectorXf> mt1(t1.data(), t1.size());
    Map<VectorXf> mt2(t2.data(), t2.size());
    return mt1.isApprox(mt2, precision);
}

bool isApprox(Tensor2f& t1, Tensor2f& t2, const float precision = 1e-8){
    Map<VectorXf> mt1(t1.data(), t1.size());
    Map<VectorXf> mt2(t2.data(), t2.size());
    return mt1.isApprox(mt2, precision);
}


void norm_tensor_along_dim3(const Tensor3f& T, Tensor2f& norm){
    array<int,1> dims({2});
    norm = T.square().sum(dims).sqrt();
}

// Function to compute C_star
autodiff::Vector3real C_star(autodiff::real x, autodiff::real y, int N_x, int N_y, float height, float width, float rs) {
    autodiff::Vector3real result;
    result << height * (1 - (2 * y) / (N_y - 1)),
              width * (-1 + (2 * x) / (N_x - 1)),
              rs;
    return result;
}

// Function to compute C
autodiff::Vector3real C(autodiff::real x, autodiff::real y, int N_x, int N_y, float height, float width, float rs) {
    autodiff::Vector3real c_star = C_star(x, y, N_x, N_y, height, width, rs);
    autodiff::real norm = sqrt(c_star.squaredNorm());
    return c_star / norm;
}

// Jacobian for x value tested by hand
void find_C(int N_x, int N_y, float view_angle_x, float view_angle_y, float rs, Tensor3f& CCM, Tensor3f& C_x, Tensor3f& C_y) {
    float height = tan(view_angle_y / 2);
    float width = tan(view_angle_x / 2);

    DEBUG_LOG("view_angle_x: " << view_angle_x);
    DEBUG_LOG("view_angle_y: " << view_angle_y);
    DEBUG_LOG("Height: " << height);
    DEBUG_LOG("Width: " << width);
    // Create grid of points
    MatrixXf XX(N_y, N_x);
    MatrixXf YY(N_y, N_x);
    for (int i = 0; i < N_y; ++i) {
        for (int j = 0; j < N_x; ++j) {
            XX(i, j) = float(j);
            YY(i, j) = float(i);
        }
    }
//    std::cout << "X Grid: " << XX << std::endl;
//    std::cout << "Y Grid: " << YY << std::endl;


    // Compute the camera calibration map (CCM) and the Jacobians
    // std::vector<std::vector<autodiff::Vector3real>> CCM;
    // Tensor3f CCM_T;
    // Tensor3f C_x;
    // Tensor3f C_y;
    // std::vector<std::vector<VectorXf>> C_y;
    for (int i = 0; i < N_y; ++i) {
        for (int j = 0; j < N_x; ++j) {
            autodiff::real x = XX(i, j);
            autodiff::real y = YY(i, j);

            // Compute the function value
            autodiff::Vector3real c_val = C(x, y, N_x, N_y, height, width, rs);
            CCM(i,j,0) = static_cast<float>(c_val(0)); // y
            CCM(i,j,1) = static_cast<float>(c_val(1)); // x
            CCM(i,j,2) = static_cast<float>(c_val(2)); // z
            // Compute the Jacobians
            // Vector3real dCdx;
            // Vector3real dCdy;
            autodiff::VectorXreal F;

            // NEEDS TO STAY D O U B L E
            VectorXd dCdx = autodiff::jacobian(C, wrt(x), at(x,y,N_x, N_y, height, width, rs), F);
            VectorXd dCdy = autodiff::jacobian(C, wrt(y), at(x,y,N_x, N_y, height, width, rs), F);

            // C_x = dCdx
            C_x(i,j,0) = dCdx(0); // y
            C_x(i,j,1) = dCdx(1); // x
            C_x(i,j,2) = dCdx(2); // z

            // C_y = -dCdy
            C_y(i,j,0) = -dCdy(0); // y
            C_y(i,j,1) = -dCdy(1); // x
            C_y(i,j,2) = -dCdy(2); // z

//            std::cout << "CCM (i: " << i << "), (j: " << j << "):" << CCM << std::endl;
//            std::cout << "C_x (i: " << i << "), (j: " << j << "):" << C_x << std::endl;
//            std::cout << "C_y (i: " << i << "), (j: " << j << "):" << C_y << std::endl;

            // C_x(i,j,0) = 1.0f;
            // C_x(i,j,1) = 1.0f;
            // C_x(i,j,2) = 1.0f;
            // C_y(i,j,0) = 1.0f;
            // C_y(i,j,1) = 1.0f;
            // C_y(i,j,2) = 1.0f;
        }
    }
}

// Function to compute the cross product of two 3-tensors
void crossProduct3x3(const Tensor3f& A, const Tensor3f& B, Tensor3f& C) {
    const auto& dims = A.dimensions();
    int height, rows = dims[0];
    int width, cols = dims[1];
    for (size_t i = 0; i < rows; ++i){
        for (size_t j = 0; j < cols; ++j){
            C(i, j, 0) = A(i, j, 2) * B(i, j, 1) - A(i, j, 1) * B(i, j, 2);  // y
            C(i, j, 1) = A(i, j, 0) * B(i, j, 2) - A(i, j, 2) * B(i, j, 0);  // x
            C(i, j, 2) = A(i, j, 1) * B(i, j, 0) - A(i, j, 0) * B(i, j, 1);  // z
        }
    }
}

// Function to compute the cross product of two 3-tensors
void crossProduct3x3(const Tensor3f& A, const Vector3f& B, Vector3f& C, int y, int x) {
    C(0) = A(y, x, 2) * B(1) - A(y, x, 1) * B(2);  // y
    C(1) = A(y, x, 0) * B(2) - A(y, x, 2) * B(0);  // x
    C(2) = A(y, x, 1) * B(0) - A(y, x, 0) * B(1);  // z
}


void crossProduct1x3(const Tensor<float,1>& A, const Tensor3f& B, Tensor3f& C){
    const auto& dimensions = B.dimensions();
    for (size_t i = 0; i < dimensions[0]; ++i){
        for (size_t j = 0; j < dimensions[1]; ++j){
            C(i, j, 0) = A(2) * B(i, j, 1) - A(1) * B(i, j, 2);  // y
            C(i, j, 1) = A(0) * B(i, j, 2) - A(2) * B(i, j, 0);  // x
            C(i, j, 2) = A(1) * B(i, j, 0) - A(0) * B(i, j, 1);  // z
        }
    }

}

void vector_distance(const Tensor3f &vec1, const Tensor3f &vec2, Tensor2f &distance){
    PROFILE_FUNCTION();
    const auto& dimensions = vec1.dimensions();
    Tensor3f cross_product(dimensions);
    Tensor2f norm(dimensions[0], dimensions[1]);
    Tensor2f norm2(dimensions[0], dimensions[1]);
    crossProduct3x3(vec1, vec2, cross_product);
    norm_tensor_along_dim3(cross_product, norm);
    norm_tensor_along_dim3(vec2, norm2);
//    std::cout << vec1 << " " << vec2 << std::endl;
//    std::cout << "cross product " << cross_product << std::endl;
//    std::cout << "norm " << norm << std::endl;
//    std::cout << "norm2 " << norm2 << std::endl;
    distance = norm/norm2;
}

float sign_func(float x){
    // Apply via a.unaryExpr(std::ptr_fun(sign_func))
    if (x > 0)
        return +1.0;
    else if (x == 0)
        return 0.0;
    else
        return -1.0;
}

// Function to time the performance of a given dot product computation function
template<typename Func>
void timeDotProductComputation(Func func, const Tensor3f& A, const Tensor3f& B, Tensor2f& D, int iterations) {
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        func(A, B, D);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> elapsed = end - start;
    std::cout << "Time elapsed: " << elapsed.count() << " seconds" << std::endl;
}

// Function using nested loops to compute the dot product
void computeDotProductWithLoops(const Tensor3f& A, const Tensor3f& B, Tensor2f& D) {
    PROFILE_FUNCTION();

    const int height = A.dimension(0);
    const int width = A.dimension(1);
    const int depth  = A.dimension(2);

    // #pragma omp parallel for collapse(2)
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            float dotProduct = 0.0f; // Initialize the dot product for position (i, j)
            for (int k = 0; k < depth; ++k) {
                dotProduct += A(i, j, k) * B(i, j, k);
            }
            D(i, j) = dotProduct; // Store the result in tensor D
        }
    }
}

// // Function using .chip() to compute the dot product
// void computeDotProductWithChip(const Tensor3f& A, const Tensor3f& B, Tensor2f& D) {
//     const int m = A.dimension(0);
//     const int n = A.dimension(1);
//     Tensor<float, 1> a_slice;
//     Tensor<float, 1> b_slice;
//     Tensor<float, 0> res;
//     for (int i = 0; i < m; ++i) {
//         for (int j = 0; j < n; ++j) {
//             // Extract the slices and compute the dot product
//             a_slice = A.chip(i, 0).chip(j, 0);
//             b_slice = B.chip(i, 0).chip(j, 0);
//             res = (a_slice * b_slice).sum().sqrt();
//             D(i, j) = res(); // Compute the dot product
//         }
//     }
// }

void m32(const Tensor3f &In, const Tensor3f &C_x, const Tensor3f &C_y, Tensor3f &Out){
    const auto& dimensions = In.dimensions();
    Tensor3f C1(dimensions);
    Tensor3f C2(dimensions);
    Tensor2f dot(dimensions[0], dimensions[1]);
    Tensor2f sign(dimensions[0], dimensions[1]);
    Tensor2f distance1(dimensions[0], dimensions[1]);
    Tensor2f distance2(dimensions[0], dimensions[1]);

//    std::cout << "In " << In << std::endl;
//    std::cout << "C_x " << C_x << std::endl;
//    std::cout << "C_y " << C_y << std::endl;

    crossProduct3x3(C_x,C_y,C1);
    crossProduct3x3(C_y,C1,C2);
    computeDotProductWithLoops(In,C2,dot);
    sign = dot.unaryExpr(std::ptr_fun(sign_func));
    vector_distance(In, C_y, distance1);
    vector_distance(C_x, C_y, distance2);
    Out.chip(1,2) = sign * distance1/distance2;

//    std::cout << "C1 " << C1 << std::endl;
//    std::cout << "C2 " << C2 << std::endl;
//    std::cout << "dot " << dot << std::endl;
//    std::cout << "sign" << sign << std::endl;
//    std::cout << "distance1 " << distance1 << std::endl;
//    std::cout << "distance2 " << distance2 << std::endl;
//    std::cout << "Out " << Out << std::endl;

    crossProduct3x3(C_y,C_x,C1);
    crossProduct3x3(C_x,C1,C2);
    computeDotProductWithLoops(In,C2,dot);
    sign = dot.unaryExpr(std::ptr_fun(sign_func));
    vector_distance(In, C_x, distance1);
    vector_distance(C_y, C_x, distance2);
    Out.chip(0,2) = sign * distance1/distance2;

//    std::cout << "C1 " << C1 << std::endl;
//    std::cout << "C2 " << C2 << std::endl;
//    std::cout << "dot " << dot << std::endl;
//    std::cout << "sign" << sign << std::endl;
//    std::cout << "distance1 " << distance1 << std::endl;
//    std::cout << "distance2 " << distance2 << std::endl;
//    std::cout << "Out " << Out << std::endl;
}

void m23(const Tensor3f& In, const Tensor3f& Cx, const Tensor3f& Cy, Vector3f& Out, int y, int x) {
    Out(0) = In(y, x, 1) * Cx(y, x, 0) + In(y, x, 0) * Cy(y, x, 0);
    Out(1) = In(y, x, 1) * Cx(y, x, 1) + In(y, x, 0) * Cy(y, x, 1);
    Out(2) = In(y, x, 1) * Cx(y, x, 2) + In(y, x, 0) * Cy(y, x, 2);
}

void setup_R_update(const Tensor3f& CCM, Matrix3f& A, Vector3f& B, std::vector<std::vector<Matrix3f>>& Identity_minus_outerProducts, std::vector<std::vector<Vector3f>>& points){
    PROFILE_FUNCTION();
    const auto &dimensions = CCM.dimensions();
    int rows = dimensions[0];
    int cols = dimensions[1];
    Matrix3f Identity = Matrix3f::Identity();
    Vector3f d;
    B.setZero();

    for (size_t i = 0; i < rows; ++i){
        for (size_t j = 0; j < cols; ++j){
            d(0) = CCM(i, j, 0);
            d(1) = CCM(i, j, 1);
            d(2) = CCM(i, j, 2);
            Identity_minus_outerProducts[i][j] = Identity - d * d.transpose();
            A += Identity_minus_outerProducts[i][j];
            points[i][j].setZero();
        }
    }
}

float VFG_check(Tensor2f& V, Tensor3f& F, Tensor3f& G, float precision){
//    InstrumentationTimer timer("VFG_check");
    const auto& dimensions = F.dimensions();
    MatrixXfRowMajor dot(dimensions[0], dimensions[1]);
    MatrixXfRowMajor diff(dimensions[0], dimensions[1]);

    for (int i = 0; i<dimensions[0]; i++){
        for (int j = 0; j<dimensions[1]; j++){
            dot(i,j) = -(F(i,j,0)*G(i,j,0) + F(i,j,1)*G(i,j,1));
            diff(i,j) = (V(i,j) - dot(i,j));
        }
    }
    return diff.lpNorm<Infinity>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  INTERACTING MAPS UPDATE FUNCTIONS  /////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void update_FG(Tensor3f& F, float V, Tensor3f& G, int y, int x, const float lr, const float weight_FG, float eps=1e-8, float gamma=255.0){
//    InstrumentationTimer timer("update_FG");
    PROFILE_FUNCTION();
    Vector2f update_F;
    update_F.setZero();
    float norm = std::abs((G(y, x, 0) * G(y, x, 0) + G(y, x, 1) * G(y, x, 1)));
    if (norm != 0.0) {
        update_F(0) = F(y, x, 0) - ((G(y, x, 0) / norm) * (V + (F(y, x, 0) * G(y, x, 0) + F(y, x, 1) * G(y, x, 1))));
        update_F(1) = F(y, x, 1) - ((G(y, x, 1) / norm) * (V + (F(y, x, 0) * G(y, x, 0) + F(y, x, 1) * G(y, x, 1))));
        F(y, x, 0) = (1 - weight_FG) * F(y, x, 0) + lr * weight_FG * update_F(0);
        F(y, x, 1) = (1 - weight_FG) * F(y, x, 1) + lr * weight_FG * update_F(1);
        if (F(y, x, 0) > gamma){
            F(y, x, 0) = gamma;
        }
        if (F(y, x, 1) > gamma){
            F(y, x, 1) = gamma;
        }
        if (F(y, x, 0) < -gamma){
            F(y, x, 0) = -gamma;
        }
        if (F(y, x, 1) < -gamma){
            F(y, x, 1) = -gamma;
        }
        if (std::abs(F(y, x, 0)) < eps){
            F(y, x, 0) = 0;
        }
        if (std::abs(F(y, x, 1)) < eps){
            F(y, x, 1) = 0;
        }
    }
}

void update_GF(Tensor3f& G, float V, Tensor3f& F, int y, int x, const float lr, const float weight_GF, float eps=1e-8, float gamma=255.0){
//    InstrumentationTimer timer("update_GF");
    PROFILE_FUNCTION();
    Vector2f update_G;
    update_G.setZero();
    float norm = std::abs((F(y, x, 0) * F(y, x, 0) + F(y, x, 1) * F(y, x, 1)));
    if (norm != 0.0) {
        update_G(0) = G(y, x, 0) - ((F(y, x, 0) / norm) * (V + (G(y, x, 0) * F(y, x, 0) + G(y, x, 1) * F(y, x, 1))));
        update_G(1) = G(y, x, 1) - ((F(y, x, 1) / norm) * (V + (G(y, x, 0) * F(y, x, 0) + G(y, x, 1) * F(y, x, 1))));
        G(y, x, 0) = (1 - weight_GF) * G(y, x, 0) + lr * weight_GF * update_G(0);
        G(y, x, 1) = (1 - weight_GF) * G(y, x, 0) + lr * weight_GF * update_G(1);
        if (G(y, x, 0) > gamma){
            G(y, x, 0) = gamma;
        }
        if (G(y, x, 1) > gamma){
            G(y, x, 1) = gamma;
        }
        if (G(y, x, 0) < -gamma){
            G(y, x, 0) = -gamma;
        }
        if (G(y, x, 1) < -gamma){
            G(y, x, 1) = -gamma;
        }
        if (std::abs(G(y, x, 0)) < eps){
            G(y, x, 0) = 0;
        }
        if (std::abs(G(y, x, 1)) < eps){
            G(y, x, 1) = 0;
        }
    }
}

void update_GI(Tensor3f& G, const Tensor3f& I_gradient, const int y, const int x, const float weight_GI, const float eps=1e-8, const float gamma=255.0){
    PROFILE_FUNCTION();
//    DEBUG_LOG(G(y, x, 0));
//    DEBUG_LOG(I_gradient(y, x, 0));
//    DEBUG_LOG(G(y, x, 1));
//    DEBUG_LOG(I_gradient(y, x, 1));
    G(y, x, 0) = (1 - weight_GI) * G(y, x, 0) + weight_GI*I_gradient(y, x, 0);
    G(y, x, 1) = (1 - weight_GI) * G(y, x, 1) + weight_GI*I_gradient(y, x, 1);
    if (G(y, x, 0) > gamma){
        G(y, x, 0) = gamma;
    }
    if (G(y, x, 1) > gamma){
        G(y, x, 1) = gamma;
    }
    if (G(y, x, 0) < -gamma){
        G(y, x, 0) = -gamma;
    }
    if (G(y, x, 1) < -gamma){
        G(y, x, 1) = -gamma;
    }
    if (std::abs(G(y, x, 0)) < eps){
        G(y, x, 0) = 0;
    }
    if (std::abs(G(y, x, 1)) < eps){
        G(y, x, 1) = 0;
    }
//    DEBUG_LOG(G(y, x, 0));
//    DEBUG_LOG(G(y, x, 1));
}

void contribute(Tensor2f& I, float V, int y, int x, float minPotential, float maxPotential){
    I(y, x) = std::min(std::max(I(y, x) + V, minPotential), maxPotential);

}

void decay(Tensor2f& I, Tensor2f& decayTimeSurface, const int y, const int x, const float time, const float neutralPotential, const float decayParam){
//    DEBUG_LOG(I(y, x))
    I(y, x) = (I(y, x) - neutralPotential) * expf(-(time - decayTimeSurface(y, x)) / decayParam) + neutralPotential;
    decayTimeSurface(y, x) = time;
//    DEBUG_LOG(I(y, x))
}

void update_IV(Tensor2f& I, const float V, Tensor2f& decayTimeSurface, const int y, const int x, const float time, const float minPotential, const float maxPotential, const float neutralPotential, const float decayParam){
    PROFILE_FUNCTION();
//    decay(I, decayTimeSurface, y, x, time, neutralPotential, decayParam);
    contribute(I, V, y, x, minPotential, maxPotential);
}

void updateGIDiffGradient(Tensor3f& G, Tensor3f& I_gradient, Tensor3f& GIDiff, Tensor3f& GIDiffGradient, int y, int x){
    PROFILE_FUNCTION();
    GIDiff(y, x, 0) = G(y, x, 0) - I_gradient(y, x, 0);
    GIDiff(y, x, 1) = G(y, x, 1) - I_gradient(y, x, 1);
    computeGradient(GIDiff, GIDiffGradient, y, x);
}

void update_IG(Tensor2f& I, Tensor3f& GIDiffGradient, int y, int x, const float weight_IG=0.5){
    PROFILE_FUNCTION();
    I(y, x) = I(y, x) + weight_IG * (- GIDiffGradient(y, x, 0) - GIDiffGradient(y, x, 1));
}

void update_FR(Tensor3f& F, const Tensor3f& CCM, const Tensor3f& Cx, const Tensor3f& Cy, const Tensor<float,1>& R, const float weight_FR, float eps=1e-8, float gamma=255.0){
    PROFILE_FUNCTION();
    Tensor3f cross(CCM.dimensions());
    const auto& dimensions = F.dimensions();
    Tensor3f update(F.dimensions());
    {
        PROFILE_SCOPE("FR CROSSPRODUCT");
        crossProduct1x3(R, CCM, cross);
    }
    {
        PROFILE_SCOPE("FR M32");
        m32(cross, Cx, Cy, update);
    }
    F = (1 - weight_FR)*F + weight_FR*update;
    for (int i = 0; i<dimensions[0]; i++) {
        for (int j = 0; j < dimensions[1]; j++) {
            if (F(i, j, 0) > gamma){
                F(i, j, 0) = gamma;
            }
            if (F(i, j, 1) > gamma){
                F(i, j, 1) = gamma;
            }
            if (F(i, j, 0) < -gamma){
                F(i, j, 0) = -gamma;
            }
            if (F(i, j, 1) < -gamma){
                F(i, j, 1) = -gamma;
            }
            if (std::abs(F(i,j,0)) < eps){
                F(i,j,0) = 0;
            }
            if (std::abs(F(i,j,1)) < eps){
                F(i,j,1) = 0;
            }
        }
    }
}

void update_RF(Tensor<float,1>& R, const Tensor3f& F, const Tensor3f& C, const Tensor3f& Cx, const Tensor3f& Cy, const Matrix3f& A, Vector3f& B, const std::vector<std::vector<Matrix3f>>& Identity_minus_outerProducts, std::vector<std::vector<Vector3f>>& old_points, const float weight_RF, std::vector<Event>& frameEvents) {
    //InstrumentationTimer timer1("update_RF");
    PROFILE_FUNCTION();
    const auto &dimensions = F.dimensions();
    int rows = dimensions[0];
    int cols = dimensions[1];
    Vector3f transformed_F(3);
    Vector3f point(3);
    Vector3f solution(3);
    {
        PROFILE_SCOPE("RF Pre");
        for (auto event : frameEvents){
            // Transform F from 2D image space to 3D world space with C
            m23(F, Cx, Cy, transformed_F, event.coordinates[0], event.coordinates[1]);
            // calculate crossproduct between world space F and calibration matrix.
            // this gives us the point on which the line stands
            crossProduct3x3(C, transformed_F, point, event.coordinates[0], event.coordinates[1]);
            // right hand side B consists of a sum of a points
            // subtract the contribution of the old_point at y,x and add the contribution of the new point
            B = B - Identity_minus_outerProducts[event.coordinates[0]][event.coordinates[1]]*old_points[event.coordinates[0]][event.coordinates[1]] + Identity_minus_outerProducts[event.coordinates[0]][event.coordinates[1]]*point;
            // new point is now old
            old_points[event.coordinates[0]][event.coordinates[1]] = point;
        }
    }
    // solve for the new rotation vector
    solution = A.partialPivLu().solve(B);
    R(0) = (1 - weight_RF) * R(0) + weight_RF * solution(0);
    R(1) = (1 - weight_RF) * R(1) + weight_RF * solution(1);
    R(2) = (1 - weight_RF) * R(2) + weight_RF * solution(2);
}

void update_RF(Tensor<float,1>& R, const Tensor3f& F, const Tensor3f& C, const Tensor3f& Cx, const Tensor3f& Cy, const Matrix3f& A, Vector3f& B, const std::vector<std::vector<Matrix3f>>& Identity_minus_outerProducts, std::vector<std::vector<Vector3f>>& old_points, const float weight_RF, int y, int x) {
    //InstrumentationTimer timer1("update_RF");
    PROFILE_FUNCTION();
    const auto &dimensions = F.dimensions();
    int rows = dimensions[0];
    int cols = dimensions[1];
    Vector3f transformed_F(3);
    Vector3f point(3);
    Vector3f solution(3);
    {
        PROFILE_SCOPE("RF Pre");
        // Transform F from 2D image space to 3D world space with C
        m23(F, Cx, Cy, transformed_F, y, x);
        // calculate crossproduct between world space F and calibration matrix.
        // this gives us the point on which the line stands
        crossProduct3x3(C, transformed_F, point, y, x);
        // right hand side B consists of a sum of a points
        // subtract the contribution of the old_point at y,x and add the contribution of the new point
        B = B - Identity_minus_outerProducts[y][x]*old_points[y][x] + Identity_minus_outerProducts[y][x]*point;
        // new point is now old
        old_points[y][x] = point;
    }
    // solve for the new rotation vector
    solution = A.partialPivLu().solve(B);
    R(0) = (1 - weight_RF) * R(0) + weight_RF * solution(0);
    R(1) = (1 - weight_RF) * R(1) + weight_RF * solution(1);
    R(2) = (1 - weight_RF) * R(2) + weight_RF * solution(2);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  INTERACTING MAPS MAIN FUNCTION  ////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void event_step(const float V, Tensor2f& MI, Tensor2f& decayTimeSurface, Tensor3f& delta_I, Tensor3f& GIDiff, Tensor3f& GIDiffGradient, Tensor3f& F, Tensor3f& G, Tensor<float,1>& R, const Tensor3f& CCM, const Tensor3f& dCdx, const Tensor3f& dCdy, const Matrix3f& A, Vector3f& B, const std::vector<std::vector<Matrix3f>>& Identity_minus_outerProducts, std::vector<std::vector<Vector3f>>& old_points, std::unordered_map<std::string,float>& parameters, std::vector<int>& permutation, int y, int x, float time){
    PROFILE_FUNCTION();
    array<Index, 2> dimensions = MI.dimensions();
    update_IV(MI, V, decayTimeSurface, y, x, time, parameters["minPotential"], parameters["maxPotential"], parameters["neutralPotential"], parameters["decayParam"]);
    // Image (MI) got changed through update by V. we need to update all surrounding gradient values. Because of the change at this pixel
    {
        PROFILE_SCOPE("GRADIENTS");
        if (y>0){
            computeGradient(MI, delta_I, y-1, x);
            update_GI(G, delta_I, y-1, x, parameters["weight_GI"], parameters["eps"], parameters["gamma"]);
        }
        if (x>0){
            computeGradient(MI, delta_I, y, x-1);
            update_GI(G, delta_I, y, x-1, parameters["weight_GI"], parameters["eps"], parameters["gamma"]);
        }
        if (y<dimensions[0]){
            computeGradient(MI, delta_I, y+1, x);
            update_GI(G, delta_I, y+1, x, parameters["weight_GI"], parameters["eps"], parameters["gamma"]);
        }
        if (x<dimensions[1]){
            computeGradient(MI, delta_I, y, x+1);
            update_GI(G, delta_I, y, x+1, parameters["weight_GI"], parameters["eps"], parameters["gamma"]);
        }
    }


    computeGradient(MI, delta_I, y, x);
    update_GI(G, delta_I, y, x, parameters["weight_GI"], parameters["eps"], parameters["gamma"]);
    updateGIDiffGradient(G, delta_I, GIDiff, GIDiffGradient, y, x);

//    update_IG(I, GIDiffGradient, y, x, parameters["weight_IG"]);
//    computeGradient(I, delta_I, y, x);


    for (const auto& element : permutation){
        PROFILE_SCOPE("PERMUTATION");
        switch( element ){
            default:
                std::cout << "Unknown number in permutation" << std::endl;
            case 0:
                update_FG(F, V, G, y, x, parameters["lr"], parameters["weight_FG"], parameters["eps"], parameters["gamma"]);
                break;
            case 1:
                update_FR(F, CCM, dCdx, dCdy, R, parameters["weight_FR"], parameters["eps"], parameters["gamma"]);
                break;
            case 2:
                update_GF(G, V, F, y, x, parameters["lr"], parameters["weight_GF"], parameters["eps"], parameters["gamma"]);
                break;
            case 3:
                update_RF(R, F, CCM, dCdx, dCdy, A, B, Identity_minus_outerProducts, old_points, parameters["weight_RF"], y, x);
                break;
         }
     }
 }


//
//int test(){
//    //##################################################################################################################
//    // TEST PARAMETERS
//    int n = 2;
//    int m = 2;
//    int nm = n*m;
//    int N = 50;
//    int M = 70;
//    int NM = N*M;
//    bool test_conversion = true;
//    bool test_update = true;
//    bool test_helper = true;
//    bool test_file = true;
//    bool test_step = true;
//    bool test_cv = true;
//
//    if (test_step){
//        test_helper = true;
//    }
//
//    //##################################################################################################################
//    // Camera calibration matrix (C/CCM) and dCdx/dCdy
//    Tensor3f CCM(n,m,3);
//    CCM.setZero();
//    Tensor3f dCdx(n,m,3);
//    dCdx.setZero();
//    Tensor3f dCdy(n,m,3);
//    dCdy.setZero();
//
//    //##################################################################################################################
//    // Optic flow F, temporal derivative V, spatial derivative G
//    Tensor3f F(n,m,2);
//    F.setZero();
//    F.chip(1,2).setConstant(1.0);
//    Tensor2f V(n,m);
//    V.setZero();
//    Tensor3f G(n,m,2);
//    G.setZero();
//    G.chip(1,2).setConstant(-1.0);
//
//    //##################################################################################################################
//    // Intesity I
//    Tensor2f I(n+1,m+1);
//    I.setZero();
//    Tensor3f delta_I(n,m,2);
//    delta_I.setRandom();
//
//    //##################################################################################################################
//    // Rotation Vector R
//    Tensor<float,1> R(3);
//    R.setRandom();
//
//    if (test_conversion){
//        std::cout << "TESTING CONVERSION" << std::endl;
//        //##############################################################################################################
//        // Test Tensor casts
//        Tensor2f T2M (N,M);
//        Tensor<float,1> T2V (N);
//        Tensor<float,2> M2T_res (N,M);
//        Tensor<float,1> V2T_res (N);
//        T2M.setConstant(1.0);
//        T2V.setConstant(1.0);
//        MatrixXfRowMajor M2T (N,M);
//        VectorXf V2T (N);
//        MatrixXfRowMajor T2M_res (N,M);
//        VectorXf T2V_res (N);
//        M2T.setConstant(2.0);
//        V2T.setConstant(2.0);
//        T2M_res = Tensor2Matrix(T2M);
//        T2V_res = Tensor2Vector(T2V);
//        M2T_res = Matrix2Tensor(M2T);
//        V2T_res = Vector2Tensor(V2T);
//        std::cout << "Implemented Tensor/Matrix/Vector casts" << std::endl;
//
//        //##################################################################################################################
//        // TESTING OPENCV CONVERSION
//        MatrixXfRowMajor eigen_matrix = MatrixXfRowMajor::Constant(3, 3, 2.0);
//        cv::Mat mat = eigenToCvMat(eigen_matrix);
//        std::cout << "Implemented Eigen to CV" << std::endl << mat << std::endl;
//
//        cv::Mat mat2(3, 3, CV_32F, cv::Scalar::all(1));
//        MatrixXfRowMajor eigen_matrix2 = cvMatToEigen(mat2);
//        std::cout << "Implemented CV to eigen" << std::endl << eigen_matrix2 << std::endl;
//
//        std::cout << "CONVERSION TEST PASSED" << std::endl;
//    }
//
//    if (test_update) {
//        std::cout << "TESTING UPDATE FUNCTIONS" << std::endl;
//        //##################################################################################################################
//        // Update F/G from G/F
//        Tensor3f F_comparison(n,m,2);
//        F.setConstant(0.0);
//        F.chip(1,2).setConstant(1.0);
//        V(0,0) = 1;
//        V(0,1) = 2;
//        V(1,0) = 3;
//        V(1,1) = 4;
//        G.setConstant(0.0);
//        G.chip(1,2).setConstant(-1.0);
//        F_comparison.setConstant(0.0);
//        F_comparison(0,0,1) = 1;
//        F_comparison(0,1,1) = 1.5;
//        F_comparison(1,0,1) = 2.0;
//        F_comparison(1,1,1) = 2.5;
//        update_FG(F, V, G, 1.0, 0.5);
//        if(isApprox(F, F_comparison)){
//            std::cout << "UPDATE FUNCTION FG CORRECT" << std::endl;
//        }else{
//            std::cout << "UPDATE FUNCTION FG FALSE" << std::endl;
//            std::cout << "F after update" << std::endl;
//            std::cout << F << std::endl;
//            std::cout << "F should be" << std::endl;
//            std::cout << F_comparison << std::endl;
//        }
//        //##################################################################################################################
//        // gradient I
//        Tensor2f delta_I_x_comparison(3,3);
//        Tensor2f delta_I_y_comparison(3,3);
//
//        I.setValues({{0,1,3}, {1,3,6}, {3,6,10}});
//        delta_I_x_comparison.setValues({{0.5,1.5,1},{1,2.5,1.5},{1.5,3.5,2}});
//        delta_I_y_comparison.setValues({{0.5,1,1.5},{1.5,2.5,3.5},{1,1.5,2}});
//        G.chip(1,2).setConstant(1.0);
//        array<Index, 2> dimensions = I.dimensions();
//        Tensor2f delta_I_x(dimensions[0], dimensions[1]);
//        Tensor2f delta_I_y(dimensions[0], dimensions[1]);
//        array<int, 2> shuffle({1, 0});
////        delta_I.chip(0,2) = Matrix2Tensor(gradient_x(Tensor2Matrix(I))).swap_layout().shuffle(shuffle); // Swap Layout of delta_I_x back 2 RowMajor as Matrix2Tensor returns ColMajor.
////        delta_I.chip(1,2) = Matrix2Tensor(gradient_y(Tensor2Matrix(I))).swap_layout().shuffle(shuffle);
//        delta_I = computeGradient(Tensor2Matrix(I));
//        delta_I_y = delta_I.chip(0,2);
//        delta_I_x = delta_I.chip(1,2);
//        if (isApprox(delta_I_x_comparison, delta_I_x) and isApprox(delta_I_y_comparison, delta_I_y)){
//            std::cout << "I GRADIENT FUNCTION CORRECT" << std::endl;
//        }else{
//            std::cout << "I GRADIENT FUNCTION FALSE" << std::endl;
//            std::cout << "I gradient x after update" << std::endl;
//            std::cout << delta_I_x << std::endl;
//            std::cout << "I gradient y after update" << std::endl;
//            std::cout << delta_I_y << std::endl;
//            std::cout << "I gradient should be" << std::endl;
//            std::cout << delta_I_x_comparison << std::endl;
//            std::cout << delta_I_y_comparison << std::endl;
//        }
//
//        //##################################################################################################################
//        // Update I from G
//        Tensor2f I_comparison(n+1,m+1);
//        delta_I.chip(0,2) = delta_I_y_comparison;
//        delta_I.chip(1,2) = delta_I_x_comparison;
//        I_comparison.setValues({{1,4,3},{3,5,6},{3,6,10}});
//        update_IG(I, delta_I, G, 1.0);
//        if (isApprox(I_comparison, I)){
//            std::cout << "UPDATE FUNCTION IG CORRECT" << std::endl;
//        }else{
//            std::cout << "UPDATE FUNCTION IG FALSE" << std::endl;
//            std::cout << "I update" << std::endl;
//            std::cout << I << std::endl;
//            std::cout << "I should be" << std::endl;
//            std::cout << I_comparison << std::endl;
//        }
//
//        //##################################################################################################################
//        // Update I from V
//        I.setValues({{0,1,3}, {1,3,6}, {3,6,10}});
//        V.setValues({{1,1},{0,0}});
//        I_comparison.setValues({{0.4,0.9,2.9},{0.4,1.4,5.9},{2.9,5.9,9.9}});
//        update_IV(I, V, 0.5, 0.1);
//        if (isApprox(I_comparison, I)){
//            std::cout << "UPDATE FUNCTION IV CORRECT" << std::endl;
//        }else{
//            std::cout << "UPDATE FUNCTION IV FALSE" << std::endl;
//            std::cout << "I update" << std::endl;
//            std::cout << I << std::endl;
//            std::cout << "I should be" << std::endl;
//            std::cout << I_comparison << std::endl;
//        }
//
//        //##################################################################################################################
//        // Update G from I
//        Tensor3f G_comparison(n,m,2);
//        G.setValues({{{1,4},{2,3}},{{3,2},{4,1}}});
//        delta_I.setValues({{{1,0},{0,1}},{{0,1},{1,0}}});
//        G_comparison.setValues({{{1,2},{1,2}},{{1.5,1.5},{2.5,0.5}}});
//        std::cout << G << std::endl;
//        std::cout << delta_I << std::endl;
//        update_GI(G, delta_I, 0.5);
//        if (isApprox(G_comparison, G)){
//            std::cout << "UPDATE FUNCTION GI CORRECT" << std::endl;
//        }else{
//            std::cout << "UPDATE FUNCTION GI FALSE" << std::endl;
//            std::cout << "G update" << std::endl;
//            std::cout << G << std::endl;
//            std::cout << "G should be" << std::endl;
//            std::cout << G_comparison << std::endl;
//        }
//        std::cout << "UPDATE FUNCTIONS TEST PASSED" << std::endl;
//    }
//
//    if (test_file){
//        std::cout << "TESTING FILE READOUT AND EVENT HANDLING" << std::endl;
//        //##################################################################################################################
//        // Create results_folder
//        std::string folder_name = "results";
//        fs::path folder_path = create_folder_and_update_gitignore(folder_name);
//        std::cout << "Implemented Folder creation" << std::endl;
//
//        //##################################################################################################################
//        // Read calibration file
//        std::string calib_path = "../res/shapes_rotation/calib.txt";
//        std::vector<float> calibration_data;
//        read_single_line_txt(calib_path, calibration_data);
//        std::cout << "Implemented calibration data readout" << std::endl;
//
//        //##################################################################################################################
//        // Read events file
//        std::string event_path = "../res/shapes_rotation/events.txt";
//        std::vector<Event> event_data;
//        read_events(event_path, event_data, 0.0, 1.0);
//        std::cout << "Implemented events readout" << std::endl;
//
//        //##################################################################################################################
//        // Bin events
//        std::vector<std::vector<Event>> binned_events;
//        binned_events = bin_events(event_data, 0.05);
//
//        //##################################################################################################################
//        // Create frames
//        size_t frame_count = binned_events.size();
//        std::vector<Tensor2f> frames(frame_count);
//        float eventContribution
//        create_frames(binned_events, frames, 180, 240, eventContribution);
//        std::cout << "Implemented event binning and event frame creation" << std::endl;
//        std::cout << "FILE READOUT AND EVENT HANDLING TEST PASSED" << std::endl;
//    }
//
//    if (test_helper){
//        std::cout << "TESTING HELPER FUNCTIONS" << std::endl;
//        //##################################################################################################################
//        // Test sparse matrix creation
//        Tensor3f In2(n,m,2);
//        Tensor3f Cx(n,m,3);
//        Tensor3f Cy(n,m,3);
//        Tensor3f Out3(n,m,3);
//        Tensor3f Out3_comparison(n,m,3);
//        In2.setValues({{{1,4},{2,3}},{{3,2},{4,1}}});
//        Cx.setValues({{{1,2,3},{1,2,3}},{{1,2,3},{1,2,3}}});
//        Cy.setValues({{{.1,.2,.3},{.1,.2,.3}},{{.1,.2,.3},{.1,.2,.3}}});
//        Out3_comparison.setValues({{{1.4,2.8,4.2},{2.3,4.6,6.9}},{{3.2,6.4,9.6},{4.1,8.2,12.3}}});
//        m23(In2, Cx, Cy, Out3);
//        if (isApprox(Out3_comparison, Out3)){
//            std::cout << "HELPER FUNCTION M23 CORRECT" << std::endl;
//        }else{
//            std::cout << "HELPER FUNCTION M23 FALSE" << std::endl;
//            std::cout << "Out" << std::endl;
//            std::cout << Out3 << std::endl;
//            std::cout << "Out should be" << std::endl;
//            std::cout << Out3_comparison << std::endl;
//        }
//
//        //##################################################################################################################
//        // Test Vector distance function needed for M32
//        Tensor3f dCdx_small(1,1,3);
//        Tensor3f dCdy_small(1,1,3);
//        Tensor2f distance_comparison(1,1);
//        Tensor2f distance(1,1);
//        dCdx_small.setValues({{{1,2,3}}});
//        dCdy_small.setValues({{{1,0,1}}});
//        distance_comparison.setValues({{(float)std::sqrt(6.0)}});
//        vector_distance(dCdx_small, dCdy_small, distance);
//        if (isApprox(distance_comparison, distance)){
//            std::cout << "HELPER FUNCTION VECTOR_DISTANCE CORRECT" << std::endl;
//        }else{
//            std::cout << "HELPER FUNCTION VECTOR_DISTANCE FALSE" << std::endl;
//            std::cout << "distance" << std::endl;
//            std::cout << distance << std::endl;
//            std::cout << "distance should be" << std::endl;
//            std::cout << distance_comparison << std::endl;
//        }
//
//        //##################################################################################################################
//        // M32 Test
//        Tensor3f In3(1,1,3);
//        Tensor3f Out2(1,1,2);
//        Tensor3f Out2_comparison(1,1,2);
//        // TAKE dCdx and dCdy from vector_distance test above
//        In3.setValues({{{1,1,1}}});
//        Out2_comparison.setValues({{{(float)(1/std::sqrt(6)), (float)(std::sqrt(6)/std::sqrt(12))}}});
//        m32(In3, dCdx_small, dCdy_small, Out2);
//        if (isApprox(Out2_comparison, Out2, 1e-6)){
//            std::cout << "HELPER FUNCTION M32 CORRECT" << std::endl;
//        }else{
//            std::cout << "HELPER FUNCTION M32 FALSE" << std::endl;
//            std::cout << "Out2" << std::endl;
//            std::cout << Out2 << std::endl;
//            std::cout << "Out2 should be" << std::endl;
//            std::cout << Out2_comparison << std::endl;
//        }
//
//        //##################################################################################################################
//        // Camera calibration map
//        find_C(n, m, PI/2, PI/2, 1.0f, CCM, dCdx, dCdy);
////        std::cout << CCM << std::endl;
////        std::cout << dCdx << std::endl;
////        std::cout << dCdy << std::endl;
////        std::cout << "Implemented find_C function with autodiff" << std::endl;
//        float epsilon = 1e-6;
//        if (std::abs(CCM(1,1,0) - 1/std::sqrt(3)) < epsilon and std::abs(dCdx(1,1,0) - 4*std::sqrt(3)/9) < epsilon and std::abs(dCdx(1,1,1) - 2/(std::sqrt(3)*3)) < epsilon){
//            std::cout << "HELPER FUNCTION FIND_C CORRECT" << std::endl;
//        }else{
//            std::cout << "HELPER FUNCTION FIND_C FALSE" << std::endl;
//        }
//
//
//        //##################################################################################################################
//        // Tensor cross product
//        // Define the dimensions of the tensors
//        array<Index, 3> dimensions = {n, m, 3};
//        // Initialize tensors A and B with random values
//        Tensor3f A(dimensions);
//        Tensor3f B(dimensions);
//        Tensor3f D(dimensions);
//        A.setRandom();
//        B.setRandom();
//
//        int iterations = 1000;
//
//        // Timing the chip-based implementation
//        std::cout << "Timing cross product for tensors implementations " << std::endl;
//        auto start_chip = std::chrono::high_resolution_clock::now();
//        for (int i = 0; i < iterations; ++i) {
//            crossProduct3x3(A, B, D);
//        }
//        auto end_chip = std::chrono::high_resolution_clock::now();
//        std::chrono::duration<float> duration_chip = end_chip - start_chip;
//        std::cout << "Time for chip-based implementation: " << duration_chip.count()/iterations << " seconds\n";
//
//        // Timing the loop-based implementation
//        auto start_loop = std::chrono::high_resolution_clock::now();
//        for (int i = 0; i < iterations; ++i) {
//            crossProduct3x3_loop(A, B, D);
//        }
//        auto end_loop = std::chrono::high_resolution_clock::now();
//        std::chrono::duration<float> duration_loop = end_loop - start_loop;
//        std::cout << "Time for loop-based implementation: " << duration_loop.count()/iterations << " seconds\n";
//        std::cout << "Implemented cross product for Tensors" << std::endl;
//        // Define the resulting tensor D with shape (m, n)
//        Tensor2f E(n, m);
//        // Time the dot product computation using nested loops
//        std::cout << "Timing dot product computation with loops:" << std::endl;
//        timeDotProductComputation(computeDotProductWithLoops, A, B, E, 1000);
//        std::cout << "Implemented dot product for tensors" << std::endl;
//
//        // // Time the dot product computation using .chip() (CHATGPT USES CHIP JUST FOR FANCY INDEXING)
//        // std::cout << "Timing dot product computation with .chip():" << std::endl;
//        // timeDotProductComputation(computeDotProductWithChip, A, B, E, 1000);
//
//
//        //##################################################################################################################
//        // Tensor cross product 1x3
//        {
//            Tensor<float,1> A(3);
//            Tensor3f B(2,2,3);
//            Tensor3f C(2,2,3);
//            Tensor3f C_comp(2,2,3);
//            A.setValues({1,2,3});
//            B.setValues({{{0,1,2},{0,1,2}},{{0,1,2},{0,1,2}}});
//            C_comp.setValues({{{-1,2,-1},{-1,2,-1}},{{-1,2,-1},{-1,2,-1}}});
//            crossProduct1x3(A, B, C);
//            if (isApprox(C_comp, C)){
//                std::cout << "HELPER FUNCTION CROSSPRODUCT1x3 CORRECT" << std::endl;
//            }else{
//                std::cout << "HELPER FUNCTION CROSSPRODUCT1x3 FALSE" << std::endl;
//                std::cout << "C" << std::endl;
//                std::cout << C << std::endl;
//                std::cout << "C should be" << std::endl;
//                std::cout << C_comp << std::endl;
//            }
//        }
//
//        std::cout << "HELPER FUNCTIONS TEST PASSED" << std::endl;
//    }
//
//    if (test_step){
//        std::cout << "TESTING UPDATE STEP FUNCTION" << std::endl;
//
//        //##################################################################################################################
//        // Camera calibration matrix (C/CCM) and dCdx/dCdy
//        Tensor3f CCM(n,m,3);
//        CCM.setZero();
//        Tensor3f dCdx(n,m,3);
//        dCdx.setZero();
//        Tensor3f dCdy(n,m,3);
//        dCdy.setZero();
//        find_C(m, n, PI/2, PI/2, 1.0f, CCM, dCdx, dCdy);
//
//        //##################################################################################################################
//        // Optic flow F, temporal derivative V, spatial derivative G
//        Tensor3f F(n,m,2);
//        F.setConstant(0.1);
//        F.chip(1,2).setConstant(1.0);
//        Tensor2f V(n,m);
//        V.setConstant(1.0);
//        Tensor3f G(n,m,2);
//        G.setConstant(0.0);
//        G.chip(1,2).setConstant(-1.0);
//
//        //##################################################################################################################
//        // Intesity I
//        Tensor2f I(n+1,m+1);
//        I.setValues({{1,4,0},{9,16,0},{0,0,0}});
//        Tensor3f delta_I(n,m,2);
//        delta_I.setZero();
//
//        //##################################################################################################################
//        // Rotation Vector R
//        Tensor<float,1> R(3);
//        R.setValues({1,2,3});
//        Matrix3f A = Matrix3f::Zero();
//        std::vector<std::vector<Matrix3f>> Identity_minus_outerProducts(n*m);
//        setup_R_update(CCM, A, Identity_minus_outerProducts);
//
//
//        //##################################################################################################################
//        // Testing Interacting Maps step
//        std::unordered_map<std::string,float> weights;
//        parameters["weight_FG"] = 0.2;
//        parameters["weight_FR"] = 0.8;
//        parameters["weight_GF"] = 0.2;
//        parameters["weight_GI"] = 0.5;
//        parameters["weight_IG"] = 0.2;
//        parameters["weight_IV"] = 0.2;
//        parameters["weight_RF"] = 0.8;
//        parameters["lr"] = 1.0; // optimisation of FG,GF alone is inaccurate with this; why is was this in the Paper?
//        parameters["time_step"] = 0.05f;
//        std::vector<int> permutation {0,1,2,3,4,5,6};
//        for (int i = 0; i < 100; ++i){
//            interacting_maps_step(V, V, I, delta_I, F, G, R, CCM, dCdx, dCdy, A, Identity_minus_outerProducts, weights, permutation, nm);
//        }
//        std::cout << "R: " << R << std::endl;
//        std::cout << "Implemented interacting maps update step" << std::endl;
//        std::cout << "UPDATE STEP FUNCTION TEST PASSED" << std::endl;
//    }
//
//    if (test_cv){
//        std::cout << "TESTING OPENCV FUNCTIONS" << std::endl;
//        //##################################################################################################################
//        // Convert to grayscale image
//        MatrixXfRowMajor grayscale_test = MatrixXfRowMajor::Random(100,100);
//        cv::Mat grayscale_image = frame2grayscale(grayscale_test);
//        cv::imshow("Grayscale Image", grayscale_image);
////        cv::waitKey(0);
//
//        //##################################################################################################################
//        // TESTING UNDISTORT IMAGE
//        // Camera matrix (3x3)
//        MatrixXfRowMajor camera_matrix_eigen(3, 3);
//        camera_matrix_eigen << 800, 0, 320,
//                0, 800, 240,
//                0, 0, 1;
//        cv::Mat camera_matrix_cv = eigenToCvMat(camera_matrix_eigen);
//
//        // Distortion coefficients (1x5)
//        MatrixXfRowMajor distortion_coefficients_eigen(1, 5);
//        distortion_coefficients_eigen << 0.1, -0.05, 0, 0, 0; // Simple distortion
//        cv::Mat distortion_coefficients_cv = eigenToCvMat(distortion_coefficients_eigen);
//
//        // Convert Eigen matrices to cv::Mat and back to Eigen
//        // Test undistort_image function
//        cv::Mat undistorted_image = undistort_image(grayscale_image, camera_matrix_cv, distortion_coefficients_cv);
//
//        // Print result
//        cv::imshow("Undistorted Image", undistorted_image);
////        cv::waitKey(0);
//
//        // V to Image
//        MatrixXfRowMajor V2image_eigen = MatrixXfRowMajor::Random(100,100);
//        cv::Mat V2image_cv = V2image(V2image_eigen);
//        cv::imshow("V Image", V2image_cv);
////        cv::waitKey(0);
//
//        // Vectorfield to Image
//        Tensor3f vector_field(100,100,2);
//        // Fill example tensor data (normally you would have real data)
//        for (int i = 0; i < 100; ++i) {
//            for (int j = 0; j < 100; ++j) {
//                vector_field(i, j, 0) = std::sin(i * 0.05);
//                vector_field(i, j, 1) = std::cos(j * 0.05);
//            }
//        }
//        cv::Mat image = vector_field2image(vector_field);
//        cv::imshow("Vector Field Image", image);
////        cv::waitKey(0);
//
//        // Matrix V I F G to OpenCV Image
//        MatrixXfRowMajor Vimage = MatrixXfRowMajor::Random(100, 100);  // Example data
//        MatrixXfRowMajor Iimage = MatrixXfRowMajor::Random(100+1, 100+1);  // Example data
//        Tensor3f Fimage(100, 100, 2);          // Example data
//        Tensor3f Gimage(100, 100, 2);          // Example data
//
//        // Fill example tensor data (normally you would have real data)
//        for (int i = 0; i < 100; ++i) {
//            for (int j = 0; j < 100; ++j) {
//                Fimage(i, j, 0) = std::sin(i * 0.33);
//                Fimage(i, j, 1) = std::cos(j * 0.33);
//                Gimage(i, j, 0) = std::cos(i * 0.1);
//                Gimage(i, j, 1) = std::sin(j * 0.1);
//            }
//        }
//        cv::Mat result = create_VIGF(Vimage, Iimage, Gimage, Fimage, "Test VIFG.png", true);
//        cv::imshow("Test VIFG", result);
////        cv::waitKey(0);
//
//        plot_VvsFG(Vimage, Fimage, Gimage, "Test VvsFG.png", true);
//        std::cout << "OPENCV TEST PASSED" << std::endl;
//    }
//    return 0;
//}

int main() {
    //##################################################################################################################
    // Parameters

    std::setprecision(8);

    if (EXECUTE_TEST){
//        test();
    }
    else{
        auto clock_time = std::chrono::system_clock::now();
        std::time_t time = std::chrono::system_clock::to_time_t(clock_time);
        std::string results_name = "Event 023 - iterate FR 10 times";
        std::string folder_name = results_name + " " + std::ctime(&time);
        std::string resource_name = "shapes_rotation";
        std::string calib_path = "../res/" + resource_name + "/calib.txt";
        std::string event_path = "../res/" + resource_name + "/events.txt";
        std::string settings_path = "../res/" + resource_name + "/settings.txt";

        float start_time_events = 10.0; // in s
        float end_time_events = 20.0; // in s
        float time_bin_size_in_s = 0.05; // in s
        int iterations = 1500;
        
        float memoryWeight = 0.99;

        // Plotting
        float cutoff = 0.1;

        std::vector<float> settings;
        read_single_line_txt(settings_path, settings);

//        int height = 180; // in pixels
//        int width = 240; // in pixels
        int height = settings[0]+1; // in pixels
        int rows = settings[0]+1; // in pixels
        int width = settings[1]+1; // in pixels
        int cols = settings[1]+1; // in pixels
        int N = height*width;

        std::unordered_map<std::string,float> parameters;
        parameters["weight_FG"] = 1.0; // 0
        parameters["weight_FR"] = 0.8; // 1
        parameters["weight_GF"] = 1.0; // 2
        parameters["weight_GI"] = 1.0; // 3
        parameters["weight_IG"] = 1.0; // 4
        parameters["weight_IV"] = 1.0; // 5
        parameters["weight_RF"] = 0.8; // 6
        parameters["lr"] = 1.0;
        parameters["time_step"] = time_bin_size_in_s;
        parameters["eventContribution"] = 25.5;
        parameters["eps"] = 0.00001;
        parameters["gamma"] = 255;
        parameters["decayParam"] = 1e6; // exp: 1e6; linear: 0.000001
        parameters["minPotential"] = 0.0;
        parameters["maxPotential"] = 255;
        parameters["neutralPotential"] = 128;
        parameters["fps"] = 30;
        parameters["updateIterationsFR"] = 10; // more iterations -> F caputures general movement of scene/camera better but significantly more computation time
        // iterations are done after event calculations for a frame are done
        std::vector<int> permutation {0,2,3}; // Which update steps to take
        auto rng = std::default_random_engine {};

        //##################################################################################################################
        // Optic flow F, temporal derivative V, spatial derivative G, intensity I, rotation vector R
        Tensor2f V_Vis(height, width);
        V_Vis.setZero();
        float V;
        Tensor3f F(height, width, 2);
        Tensor3f F1(height, width, 2);
        Tensor3f F2(height, width, 2);
        Tensor3f F3(height, width, 2);

        F1.setConstant(1.0);
        F2.setConstant(2.0);
        F3.setConstant(0.01);

        F.setRandom();
        F = F*F2 - F1;
        F = F * F3;

//        F.setZero();
//        F.chip(0,2) = F1.chip(0,2);
        Tensor3f G(height, width, 2);
        G.setZero();
        Tensor2f I(height, width);
        I.setConstant(128.0);
        Tensor2f decayTimeSurface(height, width);
        decayTimeSurface.setZero();
        Tensor3f delta_I(height, width,2);
        delta_I.setZero();
        Tensor3f GIDiff(height, width,2);
        GIDiff.setRandom();
        Tensor3f GIDiffGradient(height, width,2);
        GIDiffGradient.setRandom();
        Tensor<float,1> R(3);
        Tensor<float,1> R2(3);
        Tensor<float,1> R3(3);
        R.setRandom(); // between 0 and 1
        R2.setConstant(2);
        R3.setConstant(1);
        R = R*R2 - R3; // between -1 and 1
//        R.setValues({0.0,-1.0,0.0});

        //##################################################################################################################
        // Create results_folder
        fs::path folder_path = create_folder_and_update_gitignore(folder_name);
        std::cout << "Created Folder " << folder_name << std::endl;

        //##################################################################################################################
        // Read calibration file
        std::vector<float> raw_calibration_data;
        read_single_line_txt(calib_path, raw_calibration_data);
        Calibration_Data calibration_data = get_calibration_data(raw_calibration_data, height, width);
        std::cout << "Readout calibration file at " << calib_path << std::endl;

        //##################################################################################################################
        // Read events file
        std::vector<Event> event_data;
        read_events(event_path, event_data, start_time_events, end_time_events);
        std::cout << "Readout events at " << event_path << std::endl;

        //##################################################################################################################
        // Bin events
        std::vector<std::vector<Event>> binned_events;
        binned_events = bin_events(event_data, time_bin_size_in_s);
        std::cout << "Binned events" << std::endl;

        //##################################################################################################################
        // Create frames
        size_t frame_count = binned_events.size();
        std::vector<Tensor2f> frames(frame_count);
        create_frames(binned_events, frames, height, width, parameters["eventContribution"]);
        std::cout << "Created frames " << frame_count << " out of " << event_data.size() << " events" << std::endl;

        //##################################################################################################################
        // Camera calibration matrix (C/CCM) and dCdx/dCdy
        Tensor3f CCM(height, width,3);
        CCM.setZero();
        Tensor3f dCdx(height, width,3);
        dCdx.setZero();
        Tensor3f dCdy(height, width,3);
        dCdy.setZero();
        find_C(width, height, calibration_data.view_angles[1], calibration_data.view_angles[0], 1.0f, CCM, dCdx, dCdy);
        std::cout << "Calculated Camera Matrix" << std::endl;

        //##################################################################################################################
        // A matrix and outerProducts for update_R
        Matrix3f A = Matrix3f::Zero();
        Vector3f B = Vector3f::Zero();
        // Create a 2D vector with uninitialized elements
        std::vector<std::vector<Matrix3f>> Identity_minus_outerProducts;
        std::vector<std::vector<Vector3f>> old_points;
        Identity_minus_outerProducts.resize(rows);  // Resize to have the number of rows
        old_points.resize(rows);  // Resize to have the number of rows

        for (int i = 0; i < rows; ++i) {
            Identity_minus_outerProducts[i].resize(cols);  // Resize each row but do not initialize
            old_points[i].resize(cols);  // Resize each row but do not initialize
        }


        setup_R_update(CCM, A, B, Identity_minus_outerProducts, old_points);

        //##################################################################################################################
        // Memory Image for I to remember previous image

        Tensor2f MI(height, width);
        MI.setConstant(parameters["neutralPotential"]);

        Tensor2f decayBase(height, width);
        decayBase.setConstant(parameters["neutralPotential"]);

        Tensor2f expDecay(height, width);
        expDecay.setConstant(1.0);

        std::string profiler_name = "Profiler.json";
        fs::path profiler_path = folder_path / profiler_name;
        Instrumentor::Get().BeginSession("Interacting Maps", profiler_path);
        std::cout << "Setup Profiler" << std::endl;
        int counter = 0;
        std::vector<Event> frameEvents;
        for (Event event : event_data){
            std::shuffle(std::begin(permutation), std::end(permutation), rng);

            int y = event.coordinates[0];
            int x = event.coordinates[1];
            V = (float) event.polarity * parameters["eventContribution"];
            V_Vis(y, x) = (float) event.polarity * parameters["eventContribution"];

            frameEvents.push_back(event);
            for (int i = 0; i < 2; ++i){
//                DEBUG_LOG(MI(y, x));
                event_step(V, MI, decayTimeSurface, delta_I, GIDiff, GIDiffGradient, F, G, R, CCM, dCdx, dCdy, A, B, Identity_minus_outerProducts, old_points, parameters, permutation, y, x, event.time);
//                DEBUG_LOG(MI(y, x));
            }
            if (start_time_events + counter * (float) 1/parameters["fps"] < event.time){

                for (int i = 0; i < parameters["updateIterationsFR"]; ++i){
                    update_FR(F, CCM, dCdx, dCdy, R, parameters["weight_FR"], parameters["eps"], parameters["gamma"]);
                }

                {
                    PROFILE_SCOPE("BETWEEN FRAMES");
                    counter++;
//                    writeToFile(V_Vis, folder_path / "V.txt");
//                    writeToFile(MI, folder_path / "MI.txt");
//                    writeToFile(I, folder_path / "I.txt");
//                    writeToFile(delta_I, folder_path / "I_gradient.txt");
//                    writeToFile(F, folder_path / "F.txt");
//                    writeToFile(G, folder_path / "G.txt");
//                    std::cout << "Time: " << event.time << std::endl;
//                    std::cout << "R: " << R << std::endl;
                    std::string image_name = "VIGF_" + std::to_string(counter) + ".png";
                    fs::path image_path = folder_path / image_name;
                    create_VIGF(Tensor2Matrix(V_Vis), Tensor2Matrix(MI), G, F, image_path, true, cutoff);
//                    image_name = "VvsFG" + std::to_string(counter) + ".png";
//                    image_path = folder_path / image_name;
//                    plot_VvsFG(Tensor2Matrix(V_Vis), F, G, image_path, true);
                    V_Vis.setZero();
                    F.setRandom();
                    F = F*F2 - F1;
                    F = F * F3;
                    G.setZero();
                }

            }
        }
        Instrumentor::Get().EndSession();
    }
}
