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

namespace fs = std::filesystem;

namespace Eigen{
    typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXfRowMajor;
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXdRowMajor;
    typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXiRowMajor;
    typedef Eigen::Tensor<float,2,Eigen::RowMajor> Tensor2f;
    typedef Eigen::Tensor<float,3,Eigen::RowMajor> Tensor3f;
}

std::ostream& operator << (std::ostream &os, const Event &e) {
    return (os << "Time: " << e.time << " Coords: " << e.coordinates[0] << " " << e.coordinates[1] << " Polarity: " << e.polarity);
}

std::string Event::toString() const {
    std::stringstream ss;
    ss << (*this);
    return ss.str();
}

// Function to write an unordered_map to a file
template <typename KeyType, typename ValueType>
void writeMapToFile(const std::string& filePath, const std::unordered_map<KeyType, ValueType>& myMap) {
    // Open the file in write mode
    std::ofstream outFile(filePath);

    // Check if the file is opened successfully
    if (!outFile) {
        std::cerr << "Error: Could not open the file " << filePath << " for writing!" << std::endl;
        return;
    }

    // Iterate through the unordered_map and write key-value pairs to the file
    for (const auto& pair : myMap) {
        outFile << pair.first << " " << pair.second << "\n";  // Key and value separated by a space
    }

    // Close the file after writing
    outFile.close();

    std::cout << "Map successfully written to " << filePath << std::endl;
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

void writeToFile(const Tensor<float,3,Eigen::RowMajor>& t, const std::string fileName){
    std::ofstream file(fileName);
    if (file.is_open())
    {
        file << t;
    }
}

void writeToFile(const Tensor<float,2,Eigen::RowMajor>& t, const std::string fileName){
    std::ofstream file(fileName);
    if (file.is_open())
    {
        file << t;
    }
}

void writeToFile(const Eigen::MatrixXfRowMajor& t, const std::string fileName){
    std::ofstream file(fileName);
    if (file.is_open())
    {
        file << t;
    }
}

// TENSOR CASTING

Eigen::VectorXf Tensor2Vector(const Eigen::Tensor<float,1>& input) {
    Eigen::array<Eigen::Index, 1> dims = input.dimensions();
    const float* data_ptr = input.data();
    Eigen::Map<const Eigen::VectorXf> result(data_ptr, dims[0]);
    return result;
}

Eigen::VectorXf Tensor2VectorRM(const Eigen::Tensor<float,1,Eigen::RowMajor>& input) {
    // RowMajor Version
    Eigen::array<Eigen::Index, 1> dims = input.dimensions();
    const float* data_ptr = input.data();
    Eigen::Map<const Eigen::VectorXf> result(data_ptr, dims[0]);
    return result;
}

Eigen::Tensor<float,1> Vector2Tensor(const Eigen::VectorXf& input) {
    const int cols = input.cols();
    const float* data_ptr = input.data();
    Eigen::TensorMap<const Eigen::Tensor<float,1>> result(data_ptr, cols);
    return result;
}

Eigen::MatrixXfRowMajor Tensor2Matrix(const Eigen::Tensor<float,2,Eigen::RowMajor>& input){
    Eigen::array<Eigen::Index, 2> dims = input.dimensions();
//
//    // As OpenCV uses a RowMajor layout, all Eigen::Matrix are also RowMajor. Tensors are ColMajor by default and it is
//    // recommended to only uses ColMajor Tensors. So at conversion the layout needs to be considered.
//    Eigen::Tensor<float,2,RowMajor> row_major(dims[0], dims[1]);
//    // Swap the layout and preserve the order of the dimensions; https://eigen.tuXfamily.org/dox/unsupported/eigen_tensors.html
//    Eigen::array<int,2> shuffle({1, 0});
//    row_major = input.swap_layout().shuffle(shuffle);
    const float* data_ptr = &input(0); // Points to beginning of array;
    Eigen::Map<const Eigen::MatrixXfRowMajor> result(data_ptr, dims[0], dims[1]);
    return result;
}

Eigen::Tensor<float,2> Matrix2Tensor(const Eigen::MatrixXfRowMajor& input) {
    // Returns ColMajorData

    // Get Pointer to data
    float const *data_ptr = &input(0);
    // Map data to Tensor
    Eigen::TensorMap<const Eigen::Tensor<float,2,Eigen::RowMajor>> result(data_ptr, input.rows(), input.cols());
    // Swap the layout and preserve the order of the dimensions
    array<int, 2> shuffle({1, 0});
    Tensor<float, 2> col_major_result =  result.swap_layout().shuffle(shuffle);
    return col_major_result;
}

// OPENCV PART

cv::Mat eigenToCvMat(const Eigen::MatrixXfRowMajor& eigen_matrix) {
    // Map Eigen matrix data to cv::Mat without copying
    return cv::Mat(eigen_matrix.rows(), eigen_matrix.cols(), CV_32F, (void*)eigen_matrix.data());
}

cv::Mat eigenToCvMatCopy(const Eigen::MatrixXfRowMajor& eigen_matrix) {
    // Create a cv::Mat and copy Eigen matrix data into it
    cv::Mat mat(eigen_matrix.rows(), eigen_matrix.cols(), CV_32F);
    for (int i = 0; i < eigen_matrix.rows(); ++i) {
        for (int j = 0; j < eigen_matrix.cols(); ++j) {
            mat.at<float>(i, j) = eigen_matrix(i, j);
        }
    }
    return mat;
}

Eigen::MatrixXfRowMajor cvMatToEigen(const cv::Mat& mat) {
    // Ensure the cv::Mat has the correct type
    CV_Assert(mat.type() == CV_32F);
    Eigen::array<Eigen::Index, 2> dims;
    dims[0] = mat.rows;
    dims[1] = mat.cols;
    const float* data_ptr = mat.ptr<float>();
    Eigen::Map<const Eigen::MatrixXfRowMajor> result(data_ptr, dims[0], dims[1]);
    return result;
}

Eigen::MatrixXfRowMajor cvMatToEigenCopy(const cv::Mat& mat) {
    // Ensure the cv::Mat has the correct type
    CV_Assert(mat.type() == CV_32F);
    Eigen::MatrixXfRowMajor eigen_matrix(mat.rows, mat.cols);
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

//TODO ensure Tensors are actually const
bool isApprox(Tensor<float,3,Eigen::RowMajor>& t1, Tensor<float,3,Eigen::RowMajor>& t2, const float precision = 1e-8){
    Map<VectorXf> mt1(t1.data(), t1.size());
    Map<VectorXf> mt2(t2.data(), t2.size());
    return mt1.isApprox(mt2, precision);
}

bool isApprox(Tensor<float,2,Eigen::RowMajor>& t1, Tensor<float,2,Eigen::RowMajor>& t2, const float precision = 1e-8){
    Map<VectorXf> mt1(t1.data(), t1.size());
    Map<VectorXf> mt2(t2.data(), t2.size());
    return mt1.isApprox(mt2, precision);
}

// OPENCV

// Function to create a video from a sequence of images
bool createVideoFromImages(const std::vector<std::string>& imagePaths, const std::string& outputVideoFile,
                           const std::string& codecStr, double fps) {
    // Check if there are any images
    if (imagePaths.empty()) {
        std::cerr << "No images provided." << std::endl;
        return false;
    }

    // Read the first image to determine the frame size
    cv::Mat firstFrame = cv::imread(imagePaths[0]);
    if (firstFrame.empty()) {
        std::cerr << "Error: Could not load the first image." << std::endl;
        return false;
    }

    // Define the codec using the fourcc code (e.g., "MJPG", "XVID", "MP4V")
    int codec = cv::VideoWriter::fourcc(codecStr[0], codecStr[1], codecStr[2], codecStr[3]);
    cv::Size frameSize(firstFrame.cols, firstFrame.rows);  // Frame size of the video

    // Create VideoWriter object
    cv::VideoWriter videoWriter(outputVideoFile, codec, fps, frameSize);
    if (!videoWriter.isOpened()) {
        std::cerr << "Error: Could not open video file for writing." << std::endl;
        return false;
    }

    // Loop over all image paths and write them to the video
    for (const auto& imagePath : imagePaths) {
        cv::Mat frame = cv::imread(imagePath);
        if (frame.empty()) {
            std::cerr << "Warning: Could not load image " << imagePath << ", skipping." << std::endl;
            continue; // Skip this frame if it can't be loaded
        }

        // Ensure the frame size is consistent
        if (frame.size() != frameSize) {
            std::cerr << "Error: Image " << imagePath << " has a different size than the first image." << std::endl;
            return false;
        }

        // Write the frame to the video file
        videoWriter.write(frame);
    }

    // Release the video writer
    videoWriter.release();
    std::cout << "Video saved successfully to " << outputVideoFile << std::endl;
    return true;
}

Eigen::MatrixXfRowMajor undistort_frame(const Eigen::MatrixXfRowMajor& frame, const cv::Mat& camera_matrix, const cv::Mat& distortion_parameters) {
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

std::vector<cv::Mat> undistort_images(const std::vector<cv::Mat>& images, const Eigen::MatrixXfRowMajor& camera_matrix, const Eigen::MatrixXfRowMajor& distortion_coefficients) {
    std::vector<cv::Mat> undistorted_images;

    // Convert Eigen matrices to cv::Mat
    cv::Mat camera_matrix_cv = eigenToCvMat(camera_matrix);
    cv::Mat distortion_coefficients_cv = eigenToCvMat(distortion_coefficients);

    for (const auto& image : images) {
        undistorted_images.push_back(undistort_image(image, camera_matrix_cv, distortion_coefficients_cv));
    }
    return undistorted_images;
}

// Function to convert Eigen::MatrixXfRowMajor to grayscale image
cv::Mat frame2grayscale(const Eigen::MatrixXfRowMajor& frame) {
    // Convert Eigen::MatrixXfRowMajor to cv::Mat
    cv::Mat frame_cv = eigenToCvMat(frame);

    // Find min and max polarity
    double min_polarity, max_polarity;
    cv::minMaxLoc(frame_cv, &min_polarity, &max_polarity);
//    DEBUG_LOG("Min Polarity: " << min_polarity)
//    DEBUG_LOG("Max Polarity: " << max_polarity)

    // Normalize the frame
    cv::Mat normalized_frame;
    frame_cv.convertTo(normalized_frame, CV_32FC3, 1.0 / (max_polarity - min_polarity), -min_polarity / (max_polarity - min_polarity));

    // Scale to 0-255 and convert to CV_8U
    cv::Mat grayscale_frame;
    normalized_frame.convertTo(grayscale_frame, CV_8UC1, 255.0);

    return grayscale_frame;
}

cv::Mat V2image(const Eigen::MatrixXfRowMajor& V) {
    // Determine the shape of the image
    long rows = V.rows();
    long cols = V.cols();
    
    // Create an empty image with 3 channels (BGR)
    cv::Mat image = cv::Mat::zeros(rows, cols, CV_8UC3);
    
    // Process on_events (V > 0)
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (V(i, j) > 0) {
                image.at<cv::Vec3b>(i, j)[1] = 255; // Set green channel
            }
            if (V(i, j) < 0) {
                image.at<cv::Vec3b>(i, j)[2] = 255; // Set red channel
            }
        }
    }
    
    return image;
}

cv::Mat vector_field2image(const Eigen::Tensor<float,3,Eigen::RowMajor>& vector_field) {

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

//cv::Mat create_VIGF(const MatrixXfRowMajor& V, const MatrixXfRowMajor& I, const Tensor<float,3,Eigen::RowMajor>& G, const Tensor<float,3,Eigen::RowMajor>& F, const std::string& path = "VIGF", bool save = false) {
//    cv::Mat V_img = V2image(V);
//    cv::Mat I_img = frame2grayscale(I);
//    cv::Mat G_img = vector_field2image(G);
//    cv::Mat F_img = vector_field2image(G);
//
//    long rows = V.rows();
//    long cols = V.cols();
//    cv::Mat image(rows * 2 + 20, cols * 2 + 20, CV_8UC3, cv::Scalar(0, 0, 0));
//
//    V_img.copyTo(image(cv::Rect(5, 5, cols, rows)));
//    cvtColor(I_img, I_img, cv::COLOR_GRAY2BGR);
//    I_img.copyTo(image(cv::Rect(cols + 10, 5, cols+1, rows+1)));
//    G_img.copyTo(image(cv::Rect(5, rows + 10, cols, rows)));
//    F_img.copyTo(image(cv::Rect(cols + 10, rows + 10, cols, rows)));
//
//    if (save && !path.empty()) {
//        imwrite(path, image);
//    }
//
//    return image;
//}

// Function to create outward vector field using Eigen
Tensor<float,3,Eigen::RowMajor> create_outward_vector_field(int grid_size) {
    // Create a 2D grid of points using linspace equivalent
    Eigen::VectorXf x = Eigen::VectorXf::LinSpaced(grid_size, -1.0, 1.0);
    Eigen::VectorXf y = Eigen::VectorXf::LinSpaced(grid_size, 1.0, -1.0);

    // Initialize matrices for meshgrid-like behavior (xv, yv)
    Eigen::MatrixXfRowMajor xv(grid_size, grid_size);
    Eigen::MatrixXfRowMajor yv(grid_size, grid_size);

    // Create meshgrid by repeating x and y values across rows and columns
    for (int i = 0; i < grid_size; ++i) {
        xv.row(i) = x.transpose();
        yv.col(i) = y;
    }

    // Create a 3D Tensor to store the 2D vector field (x, y components at each point)
    Tensor<float,3,Eigen::RowMajor> vectors(grid_size, grid_size, 2);

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
    Tensor<float,3,Eigen::RowMajor> vector_field = create_outward_vector_field(grid_size);

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
cv::Mat create_VIGF(const MatrixXfRowMajor& V, const MatrixXfRowMajor& I, const Tensor<float,3,Eigen::RowMajor>& G, const Tensor<float,3,Eigen::RowMajor>& F, const std::string& path = "VIGF", bool save = false) {
    cv::Mat V_img = V2image(V);
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

    //    V_img.copyTo(image(cv::Rect(5, 5, cols, rows)));
    //    cvtColor(I_img, I_img, cv::COLOR_GRAY2BGR);
    //    I_img.copyTo(image(cv::Rect(cols + 10, 5, cols+1, rows+1)));
    //    G_img.copyTo(image(cv::Rect(5, rows + 10, cols, rows)));
    //    F_img.copyTo(image(cv::Rect(cols + 10, rows + 10, cols, rows)));

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

cv::Mat plot_VvsFG(const MatrixXfRowMajor& V, const Tensor<float,3,Eigen::RowMajor>& F, const Tensor<float,3,Eigen::RowMajor>& G, const std::string& path = "VvsFG", bool save = false){
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
    FdotG = Tensor2Matrix(-(F*G).sum(Eigen::array<int,1>({2})));
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


// INTERACTING MAPS

// GRADIENT CALCUATIONS

Eigen::VectorXf gradient(const Eigen::VectorXf& x) {
    int n = x.size();
    Eigen::VectorXf grad(n);

    // Central differences in the interior
    for (int i = 1; i < n - 1; ++i) {
        grad(i) = (x(i + 1) - x(i - 1)) / 2.0;
    }

    // Forward difference at the start
    grad(0) = x(1) - x(0);

    // Backward difference at the end
    grad(n - 1) = x(n - 1) - x(n - 2);

    return grad;
}

Eigen::Tensor<float,3,Eigen::RowMajor> computeGradient(const Eigen::MatrixXfRowMajor & data, const std::vector<int> direction={0,1}) {
    int rows = data.rows();
    int cols = data.cols();

//    DEBUG_LOG("data: " << std::endl << data);
    // Initialize the tensor: 3 dimensions (rows, cols, gradient direction)
    if (direction.size() == 2){
        Eigen::Tensor<float,3,Eigen::RowMajor> gradients(rows, cols, 2);
        // Compute gradient along columns (down-up, y-direction)
        for (int j = 0; j < cols; ++j) {
            for (int i = 1; i < rows - 1; ++i) {
                gradients(i, j, 0) = (data(i - 1, j) - data(i + 1, j)) / 2.0;
            }
//            gradients(0, j, 1) = data(1, j) - data(0, j); // Forward difference for the first row
//            gradients(rows - 1, j, 1) = data(rows - 1, j) - data(rows - 2, j); // Backward difference for the last row
            gradients(0, j, 0) = (data(0, j) - data(1,j)) / 2.0; // Central difference with replicate border
            gradients(rows - 1, j, 0) = (data(rows - 2, j) - data(rows - 1, j)) / 2.0; // Central difference with replicate border
        }
        // Compute gradient along rows (left-right, x-direction)
        for (int i = 0; i < rows; ++i) {
            for (int j = 1; j < cols - 1; ++j) {
                gradients(i, j, 1) = (data(i, j + 1) - data(i, j - 1)) / 2.0;
            }
//            gradients(i, 0, 0) = data(i, 1) - data(i, 0); // Forward difference for the first column
//            gradients(i, cols - 1, 0) = data(i, cols - 1) - data(i, cols - 2); // Backward difference for the last column
            gradients(i, 0, 1) = (data(i, 1) - data(i,0)) / 2.0; // Central difference with replicate border
            gradients(i, cols - 1, 1) = (data(i, cols - 1) - data(i, cols - 2)) / 2.0; // Central difference with replicate border
        }
        return gradients;
    }
    else if (direction[0] == 1){
        Eigen::Tensor<float,3,Eigen::RowMajor> gradients(rows, cols, 1);
        // Compute gradient along rows (x-direction)
        for (int i = 0; i < rows; ++i) {
            for (int j = 1; j < cols - 1; ++j) {
                gradients(i, j, 0) = (data(i, j + 1) - data(i, j - 1)) / 2.0;
            }
//            gradients(i, 0, 0) = data(i, 1) - data(i, 0); // Forward difference for the first column
//            gradients(i, cols - 1, 0) = data(i, cols - 1) - data(i, cols - 2); // Backward difference for the last colums
            gradients(i, 0, 0) = (data(i, 1) - data(i,0)) / 2.0; // Central difference with replicate border
            gradients(i, cols - 1, 0) = (data(i, cols - 1) - data(i, cols - 2)) / 2.0; // Central difference with replicate border
        }
        return gradients;
    }
    else if (direction[0] == 0) {
        Eigen::Tensor<float,3,Eigen::RowMajor> gradients(rows, cols, 1);
        // Compute gradient along columns (y-direction)
        for (int j = 0; j < cols; ++j) {
            for (int i = 1; i < rows - 1; ++i) {
                gradients(i, j, 0) = (data(i + 1, j) - data(i - 1, j)) / 2.0;
            }
//            gradients(0, j, 0) = data(1, j) - data(0, j); // Forward difference for the first row
//            gradients(rows - 1, j, 0) = data(rows - 1, j) - data(rows - 2, j); // Backward difference for the last row
            gradients(0, j, 0) = (data(1, j) - data(0,j)) / 2.0; // Central difference with replicate border
            gradients(rows - 1, j, 0) = (data(rows-1, j) - data(rows - 2, j)) / 2.0; // Central difference with replicate border
        }
//        DEBUG_LOG(gradients(0, 0, 0));
        return gradients;
    }
    else{
        throw std::invalid_argument("invalid directions");
    }
}

Eigen::MatrixXfRowMajor gradient_y(const Eigen::MatrixXfRowMajor& mat) {
    int rows = mat.rows();
    int cols = mat.cols();
    Eigen::MatrixXfRowMajor grad_x(rows-1, cols-1);
//    // Compute central differences for interior points
//    grad_x.block(1, 0, rows - 2, cols) = (mat.block(2, 0, rows - 2, cols) - mat.block(0, 0, rows - 2, cols)) / 2.0;
//    // Compute forward difference for the first row
//    grad_x.row(0) = mat.row(1) - mat.row(0);
//    // Compute backward difference for the last row
//    grad_x.row(rows - 1) = mat.row(rows - 1) - mat.row(rows - 2);

//    grad_x.block(0,0,1,cols-1) = mat.block(1,0,1,cols-1) - mat.block(0,0,1,cols-1);
    grad_x = mat.block(1,0,rows-1,cols-1) - mat.block(0,0,rows-1,cols-1);
    return grad_x;
}

Eigen::MatrixXfRowMajor gradient_x(const Eigen::MatrixXfRowMajor& mat) {
    int rows = mat.rows();
    int cols = mat.cols();
    Eigen::MatrixXfRowMajor grad_y(rows-1, cols-1);
//    // Compute central differences for interior points
//    grad_y.block(0, 1, rows, cols - 2) = (mat.block(0, 2, rows, cols - 2) - mat.block(0, 0, rows, cols - 2)) / 2.0;
//    // Compute forward difference for the first column
//    grad_y.col(0) = mat.col(1) - mat.col(0);
//    // Compute backward difference for the last column
//    grad_y.col(cols - 1) = mat.col(cols - 1) - mat.col(cols - 2);
    grad_y = mat.block(0,1,rows-1,cols-1) - mat.block(0,0,rows-1,cols-1);
    return grad_y;
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
    data.camera_matrix = Eigen::MatrixXf({{data.focal_point[0], 0, raw_data[2]},
                                              {0, data.focal_point[1], raw_data[3]},
                                              {0, 0, 1}});
    data.distortion_coefficients = std::vector<float>(raw_data.begin()+4, raw_data.end());
    data.view_angles = std::vector<float>({2*std::atan(frame_height/(2*data.focal_point[0])),
                                            2*std::atan(frame_width/(2*data.focal_point[1]))});
    return data;
}

void read_events(const std::string& file_path, std::vector<Event>& events, float start_time, float end_time, float event_factor = 1.0, int max_events = INT32_MAX){
    fs::path current_directory = fs::current_path();
    std::string path = current_directory / file_path;
    if (fs::exists(path)) {
        std::ifstream event_file(path);
        int counter = 0;
        float time, polarity;
        int width, height;
        while (event_file >> time >> width >> height >> polarity){
            if (time < start_time) continue;
            if (time > end_time) break;
            if (counter > max_events) break;
            Event event;
            event.time = time;
            std::vector<int> coords = {height, width};
            event.coordinates = coords;
            event.polarity = (polarity*2 - 1) * event_factor;
            events.push_back(event);
            counter++;
        }
        DEBUG_LOG("Final time stamp: " << time)
        DEBUG_LOG("Number of events: " << events.size())
    }
}

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

void create_frames(const std::vector<std::vector<Event>>& bucketed_events, std::vector<Tensor<float,2,Eigen::RowMajor>>& frames, const int camera_height, const int camera_width){
    int i = 0;
    Tensor<float,2,Eigen::RowMajor> frame(camera_height, camera_width);
    Tensor<float,2,Eigen::RowMajor> cum_frame(camera_height, camera_width);
    for (std::vector<Event> event_vector : bucketed_events){

        frame.setZero();
        cum_frame.setZero();
        for (Event event : event_vector){
//            std::cout << event << std::endl;
            frame(event.coordinates.at(0), event.coordinates.at(1)) = event.polarity;
//            cum_frame(event.coordinates.at(0), event.coordinates.at(1)) += (float)event.polarity;
        }
        frames[i] = frame;
        i++;

        DEBUG_LOG("Eventvector size: " << event_vector.size());
        DEBUG_LOG("Last Event: " << event_vector.back());
    }

}

void create_sparse_matrix(const int N, const Tensor<float,2,Eigen::RowMajor>& V, SpMat& result){
    // Step 1: Create the repeated identity matrix part
    std::vector<Triplet<float>> tripletList;
    tripletList.reserve(N*3*2);
    for (int i = 0; i<N*3; i++){
        tripletList.push_back(Triplet<float>(i,i%3,1.0));
    }

    // Step 2: Create the diagonal and off-diagonal values from V
    int j = 3;
    array<long,1> one_dim{{V.size()}};
    Eigen::Tensor<float,1,Eigen::RowMajor> k = V.reshape(one_dim);
    for (int i = 0; i<N*3; i++){
        tripletList.push_back(Triplet<float>(i,j,k(i)));
        if (i%3 == 2){
            j++;
        }
    }
    result.setFromTriplets(tripletList.begin(), tripletList.end());
}

void norm_tensor_along_dim3(const Tensor<float,3,Eigen::RowMajor>& T, Tensor<float,2,Eigen::RowMajor>& norm){
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
void find_C(int N_x, int N_y, float view_angle_x, float view_angle_y, float rs, Tensor<float,3,Eigen::RowMajor>& CCM, Tensor<float,3,Eigen::RowMajor>& C_x, Tensor<float,3,Eigen::RowMajor>& C_y) {
    float height = tan(view_angle_y / 2);
    float width = tan(view_angle_x / 2);

    DEBUG_LOG("view_angle_x: " << view_angle_x);
    DEBUG_LOG("view_angle_y: " << view_angle_y);
    DEBUG_LOG("Height: " << height);
    DEBUG_LOG("Width: " << width);
    // Create grid of points
    Eigen::MatrixXf XX(N_y, N_x);
    Eigen::MatrixXf YY(N_y, N_x);
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
    // Tensor<float,3,Eigen::RowMajor> CCM_T;
    // Tensor<float,3,Eigen::RowMajor> C_x;
    // Tensor<float,3,Eigen::RowMajor> C_y;
    // std::vector<std::vector<Eigen::VectorXf>> C_y;
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
            Eigen::VectorXd dCdx = autodiff::jacobian(C, wrt(x), at(x,y,N_x, N_y, height, width, rs), F);
            Eigen::VectorXd dCdy = autodiff::jacobian(C, wrt(y), at(x,y,N_x, N_y, height, width, rs), F);

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

void crossProduct3x3(const Eigen::Tensor<float,3,Eigen::RowMajor>& A, const Eigen::Tensor<float,3,Eigen::RowMajor>& B, Eigen::Tensor<float,3,Eigen::RowMajor>& C) {
    PROFILE_FUNCTION();
    assert(A.dimensions() == B.dimensions() && "Tensors A and B must have the same shape");

    // Use Eigen's tensor operations to compute the cross product
    // x is the second value in a vector as it corresponds to columns (left right coordinate)
    // y is the first value in a vector as it corresponds to rows (up down coordinate)
    C.chip(0, 2) = A.chip(2, 2) * B.chip(1, 2) - A.chip(1, 2) * B.chip(2, 2);  // y
    C.chip(1, 2) = A.chip(0, 2) * B.chip(2, 2) - A.chip(2, 2) * B.chip(0, 2);  // x
    C.chip(2, 2) = A.chip(1, 2) * B.chip(0, 2) - A.chip(0, 2) * B.chip(1, 2);  // z
}

void crossProduct1x3(const Eigen::Tensor<float,1>& A, const Eigen::Tensor<float,3,Eigen::RowMajor>& B, Eigen::Tensor<float,3,Eigen::RowMajor>& C){
    // Use Eigen's tensor operations to compute the cross product
    for (int i = 0; i < B.dimension(0); ++i) {
        for (int j = 0; j < B.dimension(1); ++j) {
            C(i, j, 0) = A(2) * B(i, j, 1) - A(1) * B(i, j, 2);  // y
            C(i, j, 1) = A(0) * B(i, j, 2) - A(2) * B(i, j, 0);  // x
            C(i, j, 2) = A(1) * B(i, j, 0) - A(0) * B(i, j, 1);  // z
        }
    }
}

// Function to compute the cross product of two 3-tensors
void crossProduct3x3_loop(const Eigen::Tensor<float,3,Eigen::RowMajor>& A, const Eigen::Tensor<float,3,Eigen::RowMajor>& B, Eigen::Tensor<float,3,Eigen::RowMajor>& C) {
    assert(A.dimensions() == B.dimensions() && "Tensors A and B must have the same shape");
    for (int i = 0; i < A.dimension(0); ++i) {
        for (int j = 0; j < A.dimension(1); ++j) {
            C(i, j, 0) = A(i, j, 2) * B(i, j, 1) - A(i, j, 1) * B(i, j, 2);  // y
            C(i, j, 1) = A(i, j, 0) * B(i, j, 2) - A(i, j, 2) * B(i, j, 0);  // x
            C(i, j, 2) = A(i, j, 1) * B(i, j, 0) - A(i, j, 0) * B(i, j, 1);  // z
        }
    }
}

void vector_distance(const Tensor<float,3,Eigen::RowMajor> &vec1, const Tensor<float,3,Eigen::RowMajor> &vec2, Tensor<float,2,Eigen::RowMajor> &distance){
    PROFILE_FUNCTION();
    const auto& dimensions = vec1.dimensions();
    Tensor<float,3,Eigen::RowMajor> cross_product(dimensions);
    Tensor<float,2,Eigen::RowMajor> norm(dimensions[0], dimensions[1]);
    Tensor<float,2,Eigen::RowMajor> norm2(dimensions[0], dimensions[1]);
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

// Function using nested loops to compute the dot product
void computeDotProductWithLoops(const Eigen::Tensor<float,3,Eigen::RowMajor>& A, const Eigen::Tensor<float,3,Eigen::RowMajor>& B, Eigen::Tensor<float,2,Eigen::RowMajor>& D) {
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

void m32(const Tensor<float,3,Eigen::RowMajor> &In, const Tensor<float,3,Eigen::RowMajor> &C_x, const Tensor<float,3,Eigen::RowMajor> &C_y, Tensor<float,3,Eigen::RowMajor> &Out){
    const auto& dimensions = In.dimensions();
    Tensor<float,3,Eigen::RowMajor> C1(dimensions);
    Tensor<float,3,Eigen::RowMajor> C2(dimensions);
    Tensor<float,2,Eigen::RowMajor> dot(dimensions[0], dimensions[1]);
    Tensor<float,2,Eigen::RowMajor> sign(dimensions[0], dimensions[1]);
    Tensor<float,2,Eigen::RowMajor> distance1(dimensions[0], dimensions[1]);
    Tensor<float,2,Eigen::RowMajor> distance2(dimensions[0], dimensions[1]);

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

void m23(const Tensor<float,3,Eigen::RowMajor>& In, const Tensor<float,3,Eigen::RowMajor>& Cx, const Tensor<float,3,Eigen::RowMajor>& Cy, Tensor<float,3,Eigen::RowMajor>& Out) {
    const auto& dimensions = Cx.dimensions();
    for (int i = 0; i < dimensions[0]; i++){
        for (int j = 0; j < dimensions[1]; j++){
            Out(i,j,0) = In(i,j,1) * Cx(i,j,0) + In(i,j,0) * Cy(i,j,0);
            Out(i,j,1) = In(i,j,1) * Cx(i,j,1) + In(i,j,0) * Cy(i,j,1);
            Out(i,j,2) = In(i,j,1) * Cx(i,j,2) + In(i,j,0) * Cy(i,j,2);
        }
    }
}
// // Function using .chip() to compute the dot product
// void computeDotProductWithChip(const Eigen::Tensor<float,3,Eigen::RowMajor>& A, const Eigen::Tensor<float,3,Eigen::RowMajor>& B, Eigen::Tensor<float,2,Eigen::RowMajor>& D) {
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

float VFG_check(Eigen::Tensor<float,2,Eigen::RowMajor>& V, Eigen::Tensor<float,3,Eigen::RowMajor>& F, Eigen::Tensor<float,3,Eigen::RowMajor>& G, float precision){
//    InstrumentationTimer timer("VFG_check");
    const auto& dimensions = F.dimensions();
    Eigen::MatrixXfRowMajor dot(dimensions[0], dimensions[1]);
    Eigen::MatrixXfRowMajor diff(dimensions[0], dimensions[1]);

    for (int i = 0; i<dimensions[0]; i++){
        for (int j = 0; j<dimensions[1]; j++){
            dot(i,j) = -(F(i,j,0)*G(i,j,0) + F(i,j,1)*G(i,j,1));
            diff(i,j) = (V(i,j) - dot(i,j));
        }
    }
    return diff.lpNorm<Infinity>();
}

float F_check(Eigen::Tensor<float,3,Eigen::RowMajor>& F1, Eigen::Tensor<float,3,Eigen::RowMajor>& F2, float precision){
    const auto& dimensions = F1.dimensions();
    Eigen::MatrixXfRowMajor dot(dimensions[0], dimensions[1]);
    Eigen::MatrixXfRowMajor diff(dimensions[0], dimensions[1]);
    for (int i = 0; i<dimensions[0]; i++){
        for (int j = 0; j<dimensions[1]; j++){
            diff(i,j) = (std::abs(F1(i,j,0)-F2(i,j,0)) + std::abs(F1(i,j,1)-F2(i,j,1)));
        }
    }
    return diff.lpNorm<2>();
}

// Function to time the performance of a given dot product computation function
template<typename Func>
void timeDotProductComputation(Func func, const Eigen::Tensor<float,3,Eigen::RowMajor>& A, const Eigen::Tensor<float,3,Eigen::RowMajor>& B, Eigen::Tensor<float,2,Eigen::RowMajor>& D, int iterations) {
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        func(A, B, D);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> elapsed = end - start;
    std::cout << "Time elapsed: " << elapsed.count() << " seconds" << std::endl;
}

void update_FG(Tensor<float,3,Eigen::RowMajor>& F, Tensor<float,2,Eigen::RowMajor>& V, Tensor<float,3,Eigen::RowMajor>& G, const float lr, const float weight_FG, float eps=1e-8, float gamma=255.0){
//    InstrumentationTimer timer("update_FG");
    PROFILE_FUNCTION();
    const auto& dimensions = F.dimensions();
    Tensor<float,3,Eigen::RowMajor> update_F(dimensions);
    update_F.setZero();
    for (int i = 0; i<dimensions[0]; i++) {
        for (int j = 0; j < dimensions[1]; j++) {
            float norm = std::abs((G(i, j, 0) * G(i, j, 0) + G(i, j, 1) * G(i, j, 1)));
            if (norm != 0.0) {
                update_F(i, j, 0) = F(i, j, 0) - ((G(i, j, 0) / norm) *
                                                  (V(i, j) + (F(i, j, 0) * G(i, j, 0) + F(i, j, 1) * G(i, j, 1))));
                update_F(i, j, 1) = F(i, j, 1) - ((G(i, j, 1) / norm) *
                                                  (V(i, j) + (F(i, j, 0) * G(i, j, 0) + F(i, j, 1) * G(i, j, 1))));
                F(i, j, 0) = (1 - weight_FG) * F(i, j, 0) + lr * weight_FG * update_F(i, j, 0);
                F(i, j, 1) = (1 - weight_FG) * F(i, j, 1) + lr * weight_FG * update_F(i, j, 1);
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
}

void update_GF(Tensor<float,3,Eigen::RowMajor>& G, Tensor<float,2,Eigen::RowMajor>& V, Tensor<float,3,Eigen::RowMajor>& F, const float lr, const float weight_GF, float eps=1e-8, float gamma=255.0){
//    InstrumentationTimer timer("update_GF");
    PROFILE_FUNCTION();
    const auto& dimensions = G.dimensions();
    Tensor<float,3,Eigen::RowMajor> update_G(dimensions);
    for (int i = 0; i<dimensions[0]; i++){
        for (int j = 0; j<dimensions[1]; j++){
            float norm = std::abs((F(i,j,0) * F(i,j,0) + F(i,j,1) * F(i,j,1)));
            if (norm != 0.0) {
                update_G(i, j, 0) = G(i, j, 0) - ((F(i, j, 0) / norm) *
                                                  (V(i, j) + (G(i, j, 0) * F(i, j, 0) + G(i, j, 1) * F(i, j, 1))));
                update_G(i, j, 1) = G(i, j, 1) - ((F(i, j, 1) / norm) *
                                                  (V(i, j) + (G(i, j, 0) * F(i, j, 0) + G(i, j, 1) * F(i, j, 1))));
                G(i, j, 0) = (1 - weight_GF) * G(i, j, 0) + lr * weight_GF * update_G(i, j, 0);
                G(i, j, 1) = (1 - weight_GF) * G(i, j, 1) + lr * weight_GF * update_G(i, j, 1);
                if (G(i, j, 0) > gamma){
                    G(i, j, 0) = gamma;
                }
                if (G(i, j, 1) > gamma){
                    G(i, j, 1) = gamma;
                }
                if (G(i, j, 0) < -gamma){
                    G(i, j, 0) = -gamma;
                }
                if (G(i, j, 1) < -gamma){
                    G(i, j, 1) = -gamma;
                }
                if (std::abs(G(i,j,0)) < eps){
                    G(i,j,0) = 0;
                }
                if (std::abs(G(i,j,1)) < eps){
                    G(i,j,1) = 0;
                }
            }
        }
    }
}

void update_GF_gradient(Tensor<float,3,Eigen::RowMajor>& G, Tensor<float,2,Eigen::RowMajor>& V, Tensor<float,3,Eigen::RowMajor>& F, const float lr, const float weight_GF){
//    InstrumentationTimer timer("update_GF_gradient");
    PROFILE_FUNCTION();
    const auto& dimensions = G.dimensions();
    Tensor<float,3,Eigen::RowMajor> update_G(dimensions);
    for (int i = 0; i<dimensions[0]; i++){
        for (int j = 0; j<dimensions[1]; j++){
            update_G(i,j,0) = 2 * F(i,j,0) * (V(i,j) + F(i,j,0) * G(i,j,0) + F(i,j,1) * G(i,j,1));
            update_G(i,j,1) = 2 * F(i,j,1) * (V(i,j) + F(i,j,0) * G(i,j,0) + F(i,j,1) * G(i,j,1));
        }
    }
    G += lr * weight_GF * update_G;
}

void update_GI(Tensor<float,3,Eigen::RowMajor> &G, Tensor<float,3,Eigen::RowMajor> &I_gradient, const float weight_GI, float eps=1e-8, float gamma=255.0){
//    InstrumentationTimer timer("update_GI");
    PROFILE_FUNCTION();
    const auto& dimensions = G.dimensions();
//    DEBUG_LOG("G: " << std::endl << G);
    G = (1 - weight_GI) * G + weight_GI*I_gradient;
    for (int i = 0; i<dimensions[0]; i++){
        for (int j = 0; j<dimensions[1]; j++){
            if (G(i, j, 0) > gamma){
                G(i, j, 0) = gamma;
            }
            if (G(i, j, 1) > gamma){
                G(i, j, 1) = gamma;
            }
            if (G(i, j, 0) < -gamma){
                G(i, j, 0) = -gamma;
            }
            if (G(i, j, 1) < -gamma){
                G(i, j, 1) = -gamma;
            }
            if (std::abs(G(i,j,0)) < eps){
                G(i,j,0) = 0;
            }
            if (std::abs(G(i,j,1)) < eps){
                G(i,j,1) = 0;
            }
        }
    }
//    DEBUG_LOG("G: " << std::endl << G);
}

void update_IV(Tensor<float,2,Eigen::RowMajor> &I, Tensor<float,2,Eigen::RowMajor> &MI, const float weight_IV=0.5, const float time_step=0.05){
    //    InstrumentationTimer timer("update_IV");
    PROFILE_FUNCTION();
    const auto& dimensions_V = MI.dimensions();
    const auto& dimensions = I.dimensions();
//    Eigen::array<Eigen::Index, 2> offsets = {0, 0};
//    Eigen::array<Eigen::Index, 2> extents = {dimensions_V.at(0), dimensions_V.at(1)};
//    DEBUG_LOG("I: " << I);
//    DEBUG_LOG("Cum_V: " << cum_V);
//    I.slice(offsets,extents) = (1-weight_IV) * I.slice(offsets,extents) + weight_IV*MI;
    I = (1-weight_IV)*I + weight_IV*MI;
//    DEBUG_LOG("I: " << I);
//    for (int i = 0; i<dimensions[0]; i++){
//        for (int j = 0; j<dimensions[1]; j++){
//            if (I(i,j)>0){
//                if (I(i,j)>time_step){
//                    I(i,j) -= time_step;
//                }
//                else{
//                    I(i,j) = 0;
//                }
//            }
//            if (I(i,j)<0){
//                if (I(i,j)<time_step){
//                    I(i,j) += time_step;
//                }
//                else{
//                    I(i,j) = 0;
//                }
//            }
//        }
//    }
}

void update_IG(Tensor<float,2,Eigen::RowMajor> &I, Tensor<float,3,Eigen::RowMajor> &I_gradient, Tensor<float,3,Eigen::RowMajor> &G, const float weight_IG=0.5){
//    InstrumentationTimer timer("update_IG");
    PROFILE_FUNCTION();
    const auto& dimensions = I.dimensions();

//    DEBUG_LOG("I: " << std::endl << I);
    Tensor<float,3,Eigen::RowMajor> temp_map = G - I_gradient;
    Tensor<float,3,Eigen::RowMajor> x_update(dimensions[0], dimensions[1], 1);
    Tensor<float,3,Eigen::RowMajor> y_update(dimensions[0], dimensions[1], 1);
    DEBUG_LOG("temp_map: " << std::endl << temp_map)
    std::vector<int> y_direction = {0};
    std::vector<int> x_direction = {1};
    y_update = computeGradient(Tensor2Matrix(temp_map.chip(0,2)), y_direction);
    x_update = computeGradient(Tensor2Matrix(temp_map.chip(1,2)), x_direction);
    DEBUG_LOG("x_update: " << std::endl << x_update)
    DEBUG_LOG("y_update: " << std::endl << y_update)
    array<int, 2> two_dims{{int(dimensions[0]), int(dimensions[1])}};
    Tensor<float,2,Eigen::RowMajor> x_update2 = x_update.reshape(two_dims);
    Tensor<float,2,Eigen::RowMajor> y_update2 = y_update.reshape(two_dims);

    I = I + weight_IG * (- x_update2 - y_update2);
//    for (int i = 0; i<dimensions[0]; i++){
//        for (int j = 0; j<dimensions[1]; j++){
//            if (I(i,j)<0){
//                I(i,j) = 0;
//            }
//            if (I(i,j)>255.0){
//                I(i,j) = 255.0;
//            }
//
//        }
//    }
//    DEBUG_LOG("I: " << std::endl << I);

//    std::cout << G << std::endl;
//    std::cout << I_gradient << std::endl;
//    std::cout << temp_map << std::endl;
//    x_update.setZero();
//    y_update.setZero();
//    for (int i = 0; i<dimensions[0]-1; i++){
//        for (int j = 0; j<dimensions[1]-1; j++){
//            if (i==0){
////                std::cout << temp_map(i,j,0) << std::endl;
//                x_update(i,j) = temp_map(i,j,0);
//            }
//            else{
//                x_update(i,j) = temp_map(i,j,0) - temp_map(i-1,j,0);
//            }
//            if (j==0){
//                y_update(i,j) = temp_map(i,j,1);
//            }
//            else{
//                y_update(i,j) = temp_map(i,j,1) - temp_map(i,j-1,1);
//            }
//        }
    }
//    std::cout << x_update << std::endl;
//    std::cout << y_update << std::endl;

    // # Gradient Implementation, the paper mentions an effect in x-direction which gets computed as a difference between matrix components in x-direction. 
    // # This is very similar to how gradients are implemented which is why they are used here. The temp map consists of an x and a y component since, the
    // # gradient of I consists of two components. One for the x and one for the y direction.
    // x_update = jnp.gradient(temp_map[:,:,0])
    // y_update = jnp.gradient(temp_map[:,:,1])
    // return (1 - weight_IG)*I + weight_IG*(I - x_update[0] - y_update[1])

    // Like Paper Implementation
    // x_update = np.zeros((temp_map.shape[:-1]))
    // y_update = np.zeros((temp_map.shape[:-1]))
    // x_update[0,:] = temp_map[0,:,0] # - 0 // Out of bound entries are set to 0
    // x_update[1:,:] = temp_map[1:,:,0] - temp_map[:-1,:,0]
    // y_update[:,0] = temp_map[:,0,1] # - 0 // Out of bound entries are set to 0
    // y_update[:,1:] = temp_map[:,1:,1] - temp_map[:,:-1,1]
    // return (1 - weight_IG)*I + weight_IG*(I - x_update - y_update)
void update_FR(Tensor<float,3,Eigen::RowMajor>& F, const Tensor<float,3,Eigen::RowMajor>& CCM, const Tensor<float,3,Eigen::RowMajor>& Cx, const Tensor<float,3,Eigen::RowMajor>& Cy, const Tensor<float,1>& R, const float weight_FR, float eps=1e-8, float gamma=255.0){
    PROFILE_FUNCTION();
    Tensor<float,3,Eigen::RowMajor> cross(CCM.dimensions());
    const auto& dimensions = F.dimensions();
    Tensor<float,3,Eigen::RowMajor> update(F.dimensions());
    {
        PROFILE_SCOPE("FR CROSSPRODUCT");
        crossProduct1x3(R, CCM, cross);
    }
//    std::cout << cross << std::endl;
    {
        PROFILE_SCOPE("FR M32");
        m32(cross, Cx, Cy, update);
    }
//    std::cout << update << std::endl;
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

void update_RF(Tensor<float,1>& R, const Tensor<float,3,Eigen::RowMajor>& F, const Tensor<float,3,Eigen::RowMajor>& C, const Tensor<float,3,Eigen::RowMajor>& Cx, const Tensor<float,3,Eigen::RowMajor>& Cy, const Matrix3f& A, const std::vector<Matrix3f>& Identity_minus_outerProducts, const float weight_RF, const int N) {
    //InstrumentationTimer timer1("update_RF");
    PROFILE_FUNCTION();
    const auto &dimensions = F.dimensions();
    Tensor<float, 3, Eigen::RowMajor> transformed_F(dimensions[0], dimensions[1], 3);
    Tensor<float, 3, Eigen::RowMajor> points_tensor_3(dimensions[0], dimensions[1], 3);
    Tensor<float, 2, Eigen::RowMajor> points_tensor_2;
    Eigen::Vector3f solution(3);
    Eigen::array<int , 2> reshaper_2({N, 3});
    Eigen::MatrixXf points_matrix;
    Vector3f B = Vector3f::Zero();
    Eigen::Vector<float, 3> p;
    float tmp;
    {
        PROFILE_SCOPE("RF Pre");
        m23(F, Cx, Cy, transformed_F);
        crossProduct3x3(C, transformed_F, points_tensor_3);
        points_tensor_2 = points_tensor_3.reshape(reshaper_2); // reshaped_points need to be Eigen::Vector for solver
        points_matrix = Tensor2Matrix(points_tensor_2);
        for (size_t i = 0; i < N; ++i) {
            p = points_matrix.block<1, 3>(i, 0);
            B += (Identity_minus_outerProducts[i]) * p;
        }
    }
    solution = A.partialPivLu().solve(B);
//        std::cout << "R update: " << solution_short << std::endl;
    R(0) = (1 - weight_RF) * R(0) + weight_RF * solution(0);
    R(1) = (1 - weight_RF) * R(1) + weight_RF * solution(1);
    R(2) = (1 - weight_RF) * R(2) + weight_RF * solution(2);
//    if (std::abs(R(0)) < 1e-14 or std::isnan(R(0))) {
//        R(0) = 0.0;
//    }
//    if (std::abs(R(1)) < 1e-14 or std::isnan(R(1))) {
//        R(1) = 0.0;
//    }
//    if (std::abs(R(2)) < 1e-14 or std::isnan(R(2))) {
//        R(2) = 0.0;
//    }
}

void setup_R_update(const Tensor<float,3,Eigen::RowMajor>& CCM, Matrix3f& A, std::vector<Matrix3f>& Identity_minus_outerProducts){
    PROFILE_FUNCTION();
    const auto &dimensions = CCM.dimensions();
    int height = dimensions[0];
    int width = dimensions[1];
    Matrix3f Identity = Matrix3f::Identity();
    Matrix3f outerProduct;
    Eigen::Vector<float,3> d;
    float tmp;
    Tensor<float, 3, Eigen::RowMajor> directions_tensor_3(height, width, 3);
    Tensor<float, 2, Eigen::RowMajor> directions_tensor_2(height*width, 3);
    Eigen::array<int , 2> reshaper_2({height*width, 3});
    Eigen::MatrixXf directions_matrix(height*width, 3);
    directions_tensor_2 = CCM.reshape(reshaper_2);
    directions_matrix = Tensor2Matrix(directions_tensor_2);

    for (size_t i = 0; i < height*width; ++i){
//        d = directions_matrix.block<1,3>(i,0).normalized(); // Normalize direction vector
        d = directions_matrix.block<1,3>(i,0); // d[0] is th y component (height first), d[1] the x component (width second)
        Identity_minus_outerProducts[i] = Identity - d * d.transpose();
        A += Identity_minus_outerProducts[i];
    }
}

// TODO: add return number
void interacting_maps_step(Tensor<float,2,Eigen::RowMajor>& V, Tensor<float,2,Eigen::RowMajor>& MI, Tensor<float,2,Eigen::RowMajor>& I, Tensor<float,3,Eigen::RowMajor>& delta_I, Tensor<float,3,Eigen::RowMajor>& F, Tensor<float,3,Eigen::RowMajor>& G, Tensor<float,1>& R, const Tensor<float,3,Eigen::RowMajor>& CCM, const Tensor<float,3,Eigen::RowMajor>& dCdx, const Tensor<float,3,Eigen::RowMajor>& dCdy, const Matrix3f& A, const std::vector<Matrix3f>& Identity_minus_outerProducts, std::unordered_map<std::string,float>& weights, std::vector<int>& permutation, const int N){
    PROFILE_FUNCTION();
    Eigen::array<Eigen::Index, 2> dimensions = V.dimensions();

    array<int, 2> shuffle({1, 0});
//    delta_I.chip(0,2) = Matrix2Tensor(gradient_x(Tensor2Matrix(I))).swap_layout().shuffle(shuffle); // Swap Layout of delta_I_x back 2 RowMajor as Matrix2Tensor returns ColMajor.
//    delta_I.chip(1,2) = Matrix2Tensor(gradient_y(Tensor2Matrix(I))).swap_layout().shuffle(shuffle);
//    delta_I = computeGradient(Tensor2Matrix(I));

    for (const auto& element : permutation){
        switch( element ){
            default:
                std::cout << "Unknown number in permutation" << std::endl;
            case 0:
//                DEBUG_LOG("F: "<< std::endl << F);
                update_FG(F, V, G, weights["lr"], weights["weight_FG"], weights["eps"], weights["gamma"]);
//                DEBUG_LOG("F: "<<  F);
//                DEBUG_LOG("V: "<<  V);
//                DEBUG_LOG("G: "<<  G);
//                    std::cout << "Case " << element << std::endl;
                break;
            case 1:
//                std::cout << F << std::endl;
                update_FR(F, CCM, dCdx, dCdy, R, weights["weight_FR"], weights["eps"], weights["gamma"]);
//                std::cout << F << std::endl;
//                std::cout << CCM << std::endl;
//                std::cout << dCdx << std::endl;
//                std::cout << dCdy << std::endl;
//                std::cout << R << std::endl;
//                    std::cout << "Case " << element << std::endl;
                break;
            case 2:
//                std::cout << G << std::endl;
                update_GF(G, V, F, weights["lr"], weights["weight_GF"], weights["eps"], weights["gamma"]);
//                std::cout << G << std::endl;
//                std::cout << V << std::endl;
//                std::cout << F << std::endl;
//                    std::cout << "Case " << element << std::endl;
                break;
            case 3:
//                std::cout << G << std::endl;
                update_GI(G, delta_I, weights["weight_GI"], weights["eps"], weights["gamma"]);
//                std::cout << G << std::endl;
//                std::cout << delta_I << std::endl;
//                    std::cout << "Case " << element << std::endl;
                break;
            case 4:
//                std::cout << I << std::endl;
                update_IG(I, delta_I, G, weights["weight_IG"]);
                delta_I = computeGradient(Tensor2Matrix(I));
//                DEBUG_LOG("delta_I: "<< std::endl << delta_I);
//                std::cout << I << std::endl;
//                std::cout << G << std::endl;
//                std::cout << delta_I << std::endl;
//                    std::cout << "Case " << element << std::endl;
                break;
            case 5:
//                std::cout << I << std::endl;
                update_IV(I, MI, weights["weight_IV"], weights["time_step"]);
                delta_I = computeGradient(Tensor2Matrix(I));
//                std::cout << I << std::endl;
//                std::cout << V << std::endl;
//                std::cout << delta_I << std::endl;
//                    std::cout << "Case " << element << std::endl;
                break;
            case 6:
//                std::cout << R << std::endl;
                update_RF(R, F, CCM, dCdx, dCdy, A, Identity_minus_outerProducts, weights["weight_RF"], N);
//                std::cout << R << std::endl;
//                std::cout << F << std::endl;
//                std::cout << CCM << std::endl;
//                std::cout << dCdx << std::endl;
//                std::cout << dCdy << std::endl;
//                    std::cout << "Case " << element << std::endl;
                break;
         }
     }
 }

int test(){
    //##################################################################################################################
    // TEST PARAMETERS
    int n = 2;
    int m = 2;
    int nm = n*m;
    int N = 50;
    int M = 70;
    int NM = N*M;
    bool test_conversion = true;
    bool test_update = true;
    bool test_helper = true;
    bool test_file = true;
    bool test_step = true;
    bool test_cv = true;

    if (test_step){
        test_helper = true;
    }

    //##################################################################################################################
    // Camera calibration matrix (C/CCM) and dCdx/dCdy
    Tensor<float,3,Eigen::RowMajor> CCM(n,m,3);
    CCM.setZero();
    Tensor<float,3,Eigen::RowMajor> dCdx(n,m,3);
    dCdx.setZero();
    Tensor<float,3,Eigen::RowMajor> dCdy(n,m,3);
    dCdy.setZero();

    //##################################################################################################################
    // Optic flow F, temporal derivative V, spatial derivative G
    Tensor<float,3,Eigen::RowMajor> F(n,m,2);
    F.setZero();
    F.chip(1,2).setConstant(1.0);
    Tensor<float,2,Eigen::RowMajor> V(n,m);
    V.setZero();
    Tensor<float,3,Eigen::RowMajor> G(n,m,2);
    G.setZero();
    G.chip(1,2).setConstant(-1.0);

    //##################################################################################################################
    // Intesity I
    Tensor<float,2,Eigen::RowMajor> I(n+1,m+1);
    I.setZero();
    Tensor<float,3,Eigen::RowMajor> delta_I(n,m,2);
    delta_I.setRandom();

    //##################################################################################################################
    // Rotation Vector R
    Tensor<float,1> R(3);
    R.setRandom();

    if (test_conversion){
        std::cout << "TESTING CONVERSION" << std::endl;
        //##############################################################################################################
        // Test Tensor casts
        Tensor<float,2,Eigen::RowMajor> T2M (N,M);
        Tensor<float,1> T2V (N);
        Tensor<float,2> M2T_res (N,M);
        Tensor<float,1> V2T_res (N);
        T2M.setConstant(1.0);
        T2V.setConstant(1.0);
        Eigen::MatrixXfRowMajor M2T (N,M);
        Eigen::VectorXf V2T (N);
        Eigen::MatrixXfRowMajor T2M_res (N,M);
        Eigen::VectorXf T2V_res (N);
        M2T.setConstant(2.0);
        V2T.setConstant(2.0);
        T2M_res = Tensor2Matrix(T2M);
        T2V_res = Tensor2Vector(T2V);
        M2T_res = Matrix2Tensor(M2T);
        V2T_res = Vector2Tensor(V2T);
        std::cout << "Implemented Tensor/Matrix/Vector casts" << std::endl;

        //##################################################################################################################
        // TESTING OPENCV CONVERSION
        Eigen::MatrixXfRowMajor eigen_matrix = Eigen::MatrixXfRowMajor::Constant(3, 3, 2.0);
        cv::Mat mat = eigenToCvMat(eigen_matrix);
        std::cout << "Implemented Eigen to CV" << std::endl << mat << std::endl;

        cv::Mat mat2(3, 3, CV_32F, cv::Scalar::all(1));
        Eigen::MatrixXfRowMajor eigen_matrix2 = cvMatToEigen(mat2);
        std::cout << "Implemented CV to eigen" << std::endl << eigen_matrix2 << std::endl;

        std::cout << "CONVERSION TEST PASSED" << std::endl;
    }

    if (test_update) {
        std::cout << "TESTING UPDATE FUNCTIONS" << std::endl;
        //##################################################################################################################
        // Update F/G from G/F
        Tensor<float,3,Eigen::RowMajor> F_comparison(n,m,2);
        F.setConstant(0.0);
        F.chip(1,2).setConstant(1.0);
        V(0,0) = 1;
        V(0,1) = 2;
        V(1,0) = 3;
        V(1,1) = 4;
        G.setConstant(0.0);
        G.chip(1,2).setConstant(-1.0);
        F_comparison.setConstant(0.0);
        F_comparison(0,0,1) = 1;
        F_comparison(0,1,1) = 1.5;
        F_comparison(1,0,1) = 2.0;
        F_comparison(1,1,1) = 2.5;
        update_FG(F, V, G, 1.0, 0.5);
        if(isApprox(F, F_comparison)){
            std::cout << "UPDATE FUNCTION FG CORRECT" << std::endl;
        }else{
            std::cout << "UPDATE FUNCTION FG FALSE" << std::endl;
            std::cout << "F after update" << std::endl;
            std::cout << F << std::endl;
            std::cout << "F should be" << std::endl;
            std::cout << F_comparison << std::endl;
        }
        //##################################################################################################################
        // gradient I
        Tensor<float,2,Eigen::RowMajor> delta_I_x_comparison(3,3);
        Tensor<float,2,Eigen::RowMajor> delta_I_y_comparison(3,3);

        I.setValues({{0,1,3}, {1,3,6}, {3,6,10}});
        delta_I_x_comparison.setValues({{0.5,1.5,1},{1,2.5,1.5},{1.5,3.5,2}});
        delta_I_y_comparison.setValues({{0.5,1,1.5},{1.5,2.5,3.5},{1,1.5,2}});
        G.chip(1,2).setConstant(1.0);
        Eigen::array<Eigen::Index, 2> dimensions = I.dimensions();
        Eigen::Tensor<float,2,Eigen::RowMajor> delta_I_x(dimensions[0], dimensions[1]);
        Eigen::Tensor<float,2,Eigen::RowMajor> delta_I_y(dimensions[0], dimensions[1]);
        array<int, 2> shuffle({1, 0});
//        delta_I.chip(0,2) = Matrix2Tensor(gradient_x(Tensor2Matrix(I))).swap_layout().shuffle(shuffle); // Swap Layout of delta_I_x back 2 RowMajor as Matrix2Tensor returns ColMajor.
//        delta_I.chip(1,2) = Matrix2Tensor(gradient_y(Tensor2Matrix(I))).swap_layout().shuffle(shuffle);
        delta_I = computeGradient(Tensor2Matrix(I));
        delta_I_y = delta_I.chip(0,2);
        delta_I_x = delta_I.chip(1,2);
        if (isApprox(delta_I_x_comparison, delta_I_x) and isApprox(delta_I_y_comparison, delta_I_y)){
            std::cout << "I GRADIENT FUNCTION CORRECT" << std::endl;
        }else{
            std::cout << "I GRADIENT FUNCTION FALSE" << std::endl;
            std::cout << "I gradient x after update" << std::endl;
            std::cout << delta_I_x << std::endl;
            std::cout << "I gradient y after update" << std::endl;
            std::cout << delta_I_y << std::endl;
            std::cout << "I gradient should be" << std::endl;
            std::cout << delta_I_x_comparison << std::endl;
            std::cout << delta_I_y_comparison << std::endl;
        }

        //##################################################################################################################
        // Update I from G
        Tensor<float,2,Eigen::RowMajor> I_comparison(n+1,m+1);
        delta_I.chip(0,2) = delta_I_y_comparison;
        delta_I.chip(1,2) = delta_I_x_comparison;
        I_comparison.setValues({{1,4,3},{3,5,6},{3,6,10}});
        update_IG(I, delta_I, G, 1.0);
        if (isApprox(I_comparison, I)){
            std::cout << "UPDATE FUNCTION IG CORRECT" << std::endl;
        }else{
            std::cout << "UPDATE FUNCTION IG FALSE" << std::endl;
            std::cout << "I update" << std::endl;
            std::cout << I << std::endl;
            std::cout << "I should be" << std::endl;
            std::cout << I_comparison << std::endl;
        }

        //##################################################################################################################
        // Update I from V
        I.setValues({{0,1,3}, {1,3,6}, {3,6,10}});
        V.setValues({{1,1},{0,0}});
        I_comparison.setValues({{0.4,0.9,2.9},{0.4,1.4,5.9},{2.9,5.9,9.9}});
        update_IV(I, V, 0.5, 0.1);
        if (isApprox(I_comparison, I)){
            std::cout << "UPDATE FUNCTION IV CORRECT" << std::endl;
        }else{
            std::cout << "UPDATE FUNCTION IV FALSE" << std::endl;
            std::cout << "I update" << std::endl;
            std::cout << I << std::endl;
            std::cout << "I should be" << std::endl;
            std::cout << I_comparison << std::endl;
        }

        //##################################################################################################################
        // Update G from I
        Tensor<float,3,Eigen::RowMajor> G_comparison(n,m,2);
        G.setValues({{{1,4},{2,3}},{{3,2},{4,1}}});
        delta_I.setValues({{{1,0},{0,1}},{{0,1},{1,0}}});
        G_comparison.setValues({{{1,2},{1,2}},{{1.5,1.5},{2.5,0.5}}});
        std::cout << G << std::endl;
        std::cout << delta_I << std::endl;
        update_GI(G, delta_I, 0.5);
        if (isApprox(G_comparison, G)){
            std::cout << "UPDATE FUNCTION GI CORRECT" << std::endl;
        }else{
            std::cout << "UPDATE FUNCTION GI FALSE" << std::endl;
            std::cout << "G update" << std::endl;
            std::cout << G << std::endl;
            std::cout << "G should be" << std::endl;
            std::cout << G_comparison << std::endl;
        }
        std::cout << "UPDATE FUNCTIONS TEST PASSED" << std::endl;
    }

    if (test_file){
        std::cout << "TESTING FILE READOUT AND EVENT HANDLING" << std::endl;
        //##################################################################################################################
        // Create results_folder
        std::string folder_name = "results";
        fs::path folder_path = create_folder_and_update_gitignore(folder_name);
        std::cout << "Implemented Folder creation" << std::endl;

        //##################################################################################################################
        // Read calibration file
        std::string calib_path = "../res/shapes_rotation/calib.txt";
        std::vector<float> calibration_data;
        read_single_line_txt(calib_path, calibration_data);
        std::cout << "Implemented calibration data readout" << std::endl;

        //##################################################################################################################
        // Read events file
        std::string event_path = "../res/shapes_rotation/events.txt";
        std::vector<Event> event_data;
        read_events(event_path, event_data, 0.0, 1.0);
        std::cout << "Implemented events readout" << std::endl;

        //##################################################################################################################
        // Bin events
        std::vector<std::vector<Event>> binned_events;
        binned_events = bin_events(event_data, 0.05);

        //##################################################################################################################
        // Create frames
        size_t frame_count = binned_events.size();
        std::vector<Tensor<float,2,Eigen::RowMajor>> frames(frame_count);
        create_frames(binned_events, frames, 180, 240);
        std::cout << "Implemented event binning and event frame creation" << std::endl;
        std::cout << "FILE READOUT AND EVENT HANDLING TEST PASSED" << std::endl;
    }

    if (test_helper){
        std::cout << "TESTING HELPER FUNCTIONS" << std::endl;
        //##################################################################################################################
        // Test sparse matrix creation
        Tensor3f In2(n,m,2);
        Tensor3f Cx(n,m,3);
        Tensor3f Cy(n,m,3);
        Tensor3f Out3(n,m,3);
        Tensor3f Out3_comparison(n,m,3);
        In2.setValues({{{1,4},{2,3}},{{3,2},{4,1}}});
        Cx.setValues({{{1,2,3},{1,2,3}},{{1,2,3},{1,2,3}}});
        Cy.setValues({{{.1,.2,.3},{.1,.2,.3}},{{.1,.2,.3},{.1,.2,.3}}});
        Out3_comparison.setValues({{{1.4,2.8,4.2},{2.3,4.6,6.9}},{{3.2,6.4,9.6},{4.1,8.2,12.3}}});
        m23(In2, Cx, Cy, Out3);
        if (isApprox(Out3_comparison, Out3)){
            std::cout << "HELPER FUNCTION M23 CORRECT" << std::endl;
        }else{
            std::cout << "HELPER FUNCTION M23 FALSE" << std::endl;
            std::cout << "Out" << std::endl;
            std::cout << Out3 << std::endl;
            std::cout << "Out should be" << std::endl;
            std::cout << Out3_comparison << std::endl;
        }

        //##################################################################################################################
        // Test Vector distance function needed for M32
        Tensor3f dCdx_small(1,1,3);
        Tensor3f dCdy_small(1,1,3);
        Tensor2f distance_comparison(1,1);
        Tensor2f distance(1,1);
        dCdx_small.setValues({{{1,2,3}}});
        dCdy_small.setValues({{{1,0,1}}});
        distance_comparison.setValues({{(float)std::sqrt(6.0)}});
        vector_distance(dCdx_small, dCdy_small, distance);
        if (isApprox(distance_comparison, distance)){
            std::cout << "HELPER FUNCTION VECTOR_DISTANCE CORRECT" << std::endl;
        }else{
            std::cout << "HELPER FUNCTION VECTOR_DISTANCE FALSE" << std::endl;
            std::cout << "distance" << std::endl;
            std::cout << distance << std::endl;
            std::cout << "distance should be" << std::endl;
            std::cout << distance_comparison << std::endl;
        }

        //##################################################################################################################
        // M32 Test
        Tensor3f In3(1,1,3);
        Tensor3f Out2(1,1,2);
        Tensor3f Out2_comparison(1,1,2);
        // TAKE dCdx and dCdy from vector_distance test above
        In3.setValues({{{1,1,1}}});
        Out2_comparison.setValues({{{(float)(std::sqrt(6)/std::sqrt(12)), (float)(1/std::sqrt(6))}}});
        m32(In3, dCdx_small, dCdy_small, Out2);
        if (isApprox(Out2_comparison, Out2, 1e-6)){
            std::cout << "HELPER FUNCTION M32 CORRECT" << std::endl;
        }else{
            std::cout << "HELPER FUNCTION M32 FALSE" << std::endl;
            std::cout << "Out2" << std::endl;
            std::cout << Out2 << std::endl;
            std::cout << "Out2 should be" << std::endl;
            std::cout << Out2_comparison << std::endl;
        }

        //##################################################################################################################
        // Camera calibration map
        find_C(n, m, PI/2, PI/2, 1.0f, CCM, dCdx, dCdy);
//        std::cout << CCM << std::endl;
//        std::cout << dCdx << std::endl;
//        std::cout << dCdy << std::endl;
//        std::cout << "Implemented find_C function with autodiff" << std::endl;
        float epsilon = 1e-6;
        if (std::abs(CCM(1,1,1) - 1/std::sqrt(3)) < epsilon){
            std::cout << "CCM CORRECT" << std::endl;
        }else{
            std::cout << "CCM FALSE" << std::endl;
        }
        if(std::abs(dCdx(1,1,1) - 4*std::sqrt(3)/9) < epsilon and std::abs(dCdx(1,1,0) - 2/(std::sqrt(3)*3)) < epsilon){
            std::cout << "dCdx CORRECT" << std::endl;
        }
        else{
            std::cout << "dCdx FALSE" << std::endl;
        }


        //##################################################################################################################
        // Tensor cross product
        // Define the dimensions of the tensors
        Eigen::array<Eigen::Index, 3> dimensions = {n, m, 3};
        // Initialize tensors A and B with random values
        Eigen::Tensor<float,3,Eigen::RowMajor> A(dimensions);
        Eigen::Tensor<float,3,Eigen::RowMajor> B(dimensions);
        Eigen::Tensor<float,3,Eigen::RowMajor> D(dimensions);
        A.setRandom();
        B.setRandom();

        int iterations = 1000;

        // Timing the chip-based implementation
        std::cout << "Timing cross product for tensors implementations " << std::endl;
        auto start_chip = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            crossProduct3x3(A, B, D);
        }
        auto end_chip = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> duration_chip = end_chip - start_chip;
        std::cout << "Time for chip-based implementation: " << duration_chip.count()/iterations << " seconds\n";

        // Timing the loop-based implementation
        auto start_loop = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            crossProduct3x3_loop(A, B, D);
        }
        auto end_loop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> duration_loop = end_loop - start_loop;
        std::cout << "Time for loop-based implementation: " << duration_loop.count()/iterations << " seconds\n";
        std::cout << "Implemented cross product for Tensors" << std::endl;
        // Define the resulting tensor D with shape (m, n)
        Eigen::Tensor<float,2,Eigen::RowMajor> E(n, m);
        // Time the dot product computation using nested loops
        std::cout << "Timing dot product computation with loops:" << std::endl;
        timeDotProductComputation(computeDotProductWithLoops, A, B, E, 1000);
        std::cout << "Implemented dot product for tensors" << std::endl;

        // // Time the dot product computation using .chip() (CHATGPT USES CHIP JUST FOR FANCY INDEXING)
        // std::cout << "Timing dot product computation with .chip():" << std::endl;
        // timeDotProductComputation(computeDotProductWithChip, A, B, E, 1000);


        //##################################################################################################################
        // Tensor cross product 1x3
        {
            Eigen::Tensor<float,1> A(3);
            Eigen::Tensor<float,3,Eigen::RowMajor> B(2,2,3);
            Eigen::Tensor<float,3,Eigen::RowMajor> C(2,2,3);
            Eigen::Tensor<float,3,Eigen::RowMajor> C_comp(2,2,3);
            A.setValues({1,2,3});
            B.setValues({{{0,1,2},{0,1,2}},{{0,1,2},{0,1,2}}});
            C_comp.setValues({{{-1,2,-1},{-1,2,-1}},{{-1,2,-1},{-1,2,-1}}});
            crossProduct1x3(A, B, C);
            if (isApprox(C_comp, C)){
                std::cout << "HELPER FUNCTION CROSSPRODUCT1x3 CORRECT" << std::endl;
            }else{
                std::cout << "HELPER FUNCTION CROSSPRODUCT1x3 FALSE" << std::endl;
                std::cout << "C" << std::endl;
                std::cout << C << std::endl;
                std::cout << "C should be" << std::endl;
                std::cout << C_comp << std::endl;
            }
        }

        std::cout << "HELPER FUNCTIONS TEST PASSED" << std::endl;
    }

    if (test_step){
        std::cout << "TESTING UPDATE STEP FUNCTION" << std::endl;

        //##################################################################################################################
        // Camera calibration matrix (C/CCM) and dCdx/dCdy
        Tensor<float,3,Eigen::RowMajor> CCM(n,m,3);
        CCM.setZero();
        Tensor<float,3,Eigen::RowMajor> dCdx(n,m,3);
        dCdx.setZero();
        Tensor<float,3,Eigen::RowMajor> dCdy(n,m,3);
        dCdy.setZero();
        find_C(m, n, PI/2, PI/2, 1.0f, CCM, dCdx, dCdy);

        //##################################################################################################################
        // Optic flow F, temporal derivative V, spatial derivative G
        Tensor<float,3,Eigen::RowMajor> F(n,m,2);
        F.setConstant(0.1);
        F.chip(1,2).setConstant(1.0);
        Tensor<float,2,Eigen::RowMajor> V(n,m);
        V.setConstant(1.0);
        Tensor<float,3,Eigen::RowMajor> G(n,m,2);
        G.setConstant(0.0);
        G.chip(1,2).setConstant(-1.0);

        //##################################################################################################################
        // Intesity I
        Tensor<float,2,Eigen::RowMajor> I(n+1,m+1);
        I.setValues({{1,4,0},{9,16,0},{0,0,0}});
        Tensor<float,3,Eigen::RowMajor> delta_I(n,m,2);
        delta_I.setZero();

        //##################################################################################################################
        // Rotation Vector R
        Tensor<float,1> R(3);
        R.setValues({1,2,3});
        Matrix3f A = Matrix3f::Zero();
        std::vector<Matrix3f> Identity_minus_outerProducts(n*m);
        setup_R_update(CCM, A, Identity_minus_outerProducts);


        //##################################################################################################################
        // Testing Interacting Maps step
        std::unordered_map<std::string,float> weights;
        weights["weight_FG"] = 0.2;
        weights["weight_FR"] = 0.8;
        weights["weight_GF"] = 0.2;
        weights["weight_GI"] = 0.5;
        weights["weight_IG"] = 0.2;
        weights["weight_IV"] = 0.2;
        weights["weight_RF"] = 0.8;
        weights["lr"] = 1.0; // optimisation of FG,GF alone is inaccurate with this; why is was this in the Paper?
        weights["time_step"] = 0.05f;
        std::vector<int> permutation {0,1,2,3,4,5,6};
        for (int i = 0; i < 100; ++i){
            interacting_maps_step(V, V, I, delta_I, F, G, R, CCM, dCdx, dCdy, A, Identity_minus_outerProducts, weights, permutation, nm);
        }
        std::cout << "R: " << R << std::endl;
        std::cout << "Implemented interacting maps update step" << std::endl;
        std::cout << "UPDATE STEP FUNCTION TEST PASSED" << std::endl;
    }

    if (test_cv){
        std::cout << "TESTING OPENCV FUNCTIONS" << std::endl;
        //##################################################################################################################
        // Convert to grayscale image
        Eigen::MatrixXfRowMajor grayscale_test = Eigen::MatrixXfRowMajor::Random(100,100);
        cv::Mat grayscale_image = frame2grayscale(grayscale_test);
        cv::imshow("Grayscale Image", grayscale_image);
//        cv::waitKey(0);

        //##################################################################################################################
        // TESTING UNDISTORT IMAGE
        // Camera matrix (3x3)
        Eigen::MatrixXfRowMajor camera_matrix_eigen(3, 3);
        camera_matrix_eigen << 800, 0, 320,
                0, 800, 240,
                0, 0, 1;
        cv::Mat camera_matrix_cv = eigenToCvMat(camera_matrix_eigen);

        // Distortion coefficients (1x5)
        Eigen::MatrixXfRowMajor distortion_coefficients_eigen(1, 5);
        distortion_coefficients_eigen << 0.1, -0.05, 0, 0, 0; // Simple distortion
        cv::Mat distortion_coefficients_cv = eigenToCvMat(distortion_coefficients_eigen);

        // Convert Eigen matrices to cv::Mat and back to Eigen
        // Test undistort_image function
        cv::Mat undistorted_image = undistort_image(grayscale_image, camera_matrix_cv, distortion_coefficients_cv);

        // Print result
        cv::imshow("Undistorted Image", undistorted_image);
//        cv::waitKey(0);

        // V to Image
        Eigen::MatrixXfRowMajor V2image_eigen = Eigen::MatrixXfRowMajor::Random(100,100);
        cv::Mat V2image_cv = V2image(V2image_eigen);
        cv::imshow("V Image", V2image_cv);
//        cv::waitKey(0);

        // Vectorfield to Image
        Eigen::Tensor<float,3,Eigen::RowMajor> vector_field(100,100,2);
        // Fill example tensor data (normally you would have real data)
        for (int i = 0; i < 100; ++i) {
            for (int j = 0; j < 100; ++j) {
                vector_field(i, j, 0) = std::sin(i * 0.05);
                vector_field(i, j, 1) = std::cos(j * 0.05);
            }
        }
        cv::Mat image = vector_field2image(vector_field);
        cv::imshow("Vector Field Image", image);
//        cv::waitKey(0);

        // Matrix V I F G to OpenCV Image
        Eigen::MatrixXfRowMajor Vimage = Eigen::MatrixXfRowMajor::Random(100, 100);  // Example data
        Eigen::MatrixXfRowMajor Iimage = Eigen::MatrixXfRowMajor::Random(100+1, 100+1);  // Example data
        Eigen::Tensor<float,3,Eigen::RowMajor> Fimage(100, 100, 2);          // Example data
        Eigen::Tensor<float,3,Eigen::RowMajor> Gimage(100, 100, 2);          // Example data

        // Fill example tensor data (normally you would have real data)
        for (int i = 0; i < 100; ++i) {
            for (int j = 0; j < 100; ++j) {
                Fimage(i, j, 0) = std::sin(i * 0.33);
                Fimage(i, j, 1) = std::cos(j * 0.33);
                Gimage(i, j, 0) = std::cos(i * 0.1);
                Gimage(i, j, 1) = std::sin(j * 0.1);
            }
        }
        cv::Mat result = create_VIGF(Vimage, Iimage, Gimage, Fimage, "Test VIFG.png", true);
        cv::imshow("Test VIFG", result);
//        cv::waitKey(0);

        plot_VvsFG(Vimage, Fimage, Gimage, "Test VvsFG.png", true);
        std::cout << "OPENCV TEST PASSED" << std::endl;
    }
    return 0;
}

int main() {
    //##################################################################################################################
    // Parameters

    std::setprecision(8);

    if (EXECUTE_TEST){
        test();
    }
    else{
        auto clock_time = std::chrono::system_clock::now();
        std::time_t time = std::chrono::system_clock::to_time_t(clock_time);
        std::string results_name = "0123456, weightFR=0,2 academic rotation, ts=0,05";
        std::string folder_name = results_name + " " + std::ctime(&time);
        std::string resource_name = "academic_rotation";
        std::string calib_path = "../res/" + resource_name + "/calib.txt";
        std::string event_path = "../res/" + resource_name + "/events.txt";
        std::string settings_path = "../res/" + resource_name + "/settings.txt";

        float start_time_events = 0.0; // in s
        float end_time_events = 2.0; // in s
        float time_bin_size_in_s = 0.05; // in s
        int iterations = 1500;
        float imageBase = 0.0;
        float memoryWeight = 0.99;

        std::vector<float> settings;
        read_single_line_txt(settings_path, settings);

//        int height = 180; // in pixels
//        int width = 240; // in pixels
        int height = settings[0]; // in pixels
        int width = settings[1]; // in pixels
        int N = height*width;

        std::unordered_map<std::string,float> weights;
        weights["weight_FG"] = 0.2; // 0
        weights["weight_FR"] = 0.2; // 1
        weights["weight_GF"] = 0.2; // 2
        weights["weight_GI"] = 0.2; // 3
        weights["weight_IG"] = 0.2; // 4
        weights["weight_IV"] = 0.5; // 5
        weights["weight_RF"] = 0.8; // 6
        weights["lr"] = 1.0;
        weights["time_step"] = time_bin_size_in_s;
        weights["event_factor"] = 1.0;
        weights["eps"] = 0.00001;
        weights["gamma"] = 255;
        std::vector<int> permutation {0,1,2,3,4,5,6}; // Which update steps to take
        auto rng = std::default_random_engine {};

        //##################################################################################################################
        // Optic flow F, temporal derivative V, spatial derivative G, intensity I, rotation vector R
        Tensor<float,3,Eigen::RowMajor> F(height, width, 2);
//        F.setRandom();
        F.setZero();
//        F.chip(0,2).setConstant(1);
//        F.chip(1,2).setConstant(1);
        Tensor<float,3,Eigen::RowMajor> G(height, width, 2);
        G.setRandom();
        Tensor<float,2,Eigen::RowMajor> I(height, width);
        Tensor<float,2,Eigen::RowMajor> TmpTensor2(height, width);
        I.setRandom();
//        I = I * TmpTensor2.setConstant(255.0);
//        Tensor<float,1,Eigen::RowMajor> row(width+1);
//        Tensor<float,1,Eigen::RowMajor> col(height+1);
//        row.setZero();
//        col.setZero();
//        I.chip(height,0) = row;
//        I.chip(width,1) = col;
        Tensor<float,3,Eigen::RowMajor> delta_I(height, width,2);
        delta_I = computeGradient(Tensor2Matrix(I));
        DEBUG_LOG(delta_I);
        Tensor<float,1> R(3);
        Tensor<float,1> R2(3);
        Tensor<float,1> R3(3);
//        R.setRandom(); // between 0 and 1
//        R2.setConstant(2);
//        R3.setConstant(1);
//        R = R*R2 - R3; // between -1 and 1
        R.setValues({0.0,0.0,0.0});

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
        read_events(event_path, event_data, start_time_events, end_time_events, weights["event_factor"]);
        std::cout << "Readout events at " << event_path << std::endl;

        //##################################################################################################################
        // Bin events
        std::vector<std::vector<Event>> binned_events;
        binned_events = bin_events(event_data, time_bin_size_in_s);
        std::cout << "Binned events" << std::endl;

        //##################################################################################################################
        // Create frames
        size_t frame_count = binned_events.size();
        std::vector<Tensor<float,2,Eigen::RowMajor>> frames(frame_count);
        create_frames(binned_events, frames, height, width);
        std::cout << "Created frames " << frame_count << " out of " << event_data.size() << " events" << std::endl;

        //##################################################################################################################
        // Camera calibration matrix (C/CCM) and dCdx/dCdy
        Tensor<float,3,Eigen::RowMajor> CCM(height, width,3);
        CCM.setZero();
        Tensor<float,3,Eigen::RowMajor> dCdx(height, width,3);
        dCdx.setZero();
        Tensor<float,3,Eigen::RowMajor> dCdy(height, width,3);
        dCdy.setZero();
        find_C(width, height, calibration_data.view_angles[1], calibration_data.view_angles[0], 1.0f, CCM, dCdx, dCdy);
        std::cout << "Calculated Camera Matrix" << std::endl;

        //##################################################################################################################
        // A matrix and outerProducts for update_R
        Matrix3f A = Matrix3f::Zero();
        std::vector<Matrix3f> Identity_minus_outerProducts(height*width);
        setup_R_update(CCM, A, Identity_minus_outerProducts);

        //##################################################################################################################
        // Memory Image for I to remember previous image

        Tensor<float,2,Eigen::RowMajor> MI(height, width);
        MI.setConstant(imageBase);

        Tensor<float,2,Eigen::RowMajor> decayBase(height, width);
        decayBase.setConstant(imageBase);




//        Tensor<float, 3, Eigen::RowMajor> directions_tensor_3(height, width, 3);
//        Tensor<float, 2, Eigen::RowMajor> directions_tensor_2(height*width, 3);
//        Eigen::array<int , 2> reshaper_2({height*width, 3});
//        Eigen::MatrixXf directions_matrix(height*width, 3);
//
//        Matrix3f Identity = Matrix3f::Identity();
//        Matrix3f outerProduct;
//        Eigen::Vector<float,3> d;
//
//        directions_tensor_2 = CCM.reshape(reshaper_2);
//        directions_matrix = Tensor2Matrix(directions_tensor_2);
//
//        for (size_t i = 0; i < height*width; ++i){
////                d = directions_matrix.block<1,3>(i,0).normalized(); // Normalize direction vector
//            d = directions_matrix.block<1,3>(i,0);
//            Identity_minus_outerProducts[i] = Identity - d * d.transpose();
//            A += Identity_minus_outerProducts[i];
//        }

        std::string profiler_name = "Profiler.json";
        fs::path profiler_path = folder_path / profiler_name;
        Instrumentor::Get().BeginSession("Interacting Maps", profiler_path);
        std::cout << "Setup Profiler" << std::endl;
        int counter = 0;
        for (Tensor<float,2,Eigen::RowMajor> V : frames){
//            std::cout << V << std::endl;

            F.setZero();


            writeToFile(V, folder_path / "V.txt");
            writeToFile(I, folder_path / "I.txt");
            writeToFile(delta_I, folder_path / "I_gradient.txt");
            writeToFile(F, folder_path / "F.txt");
            writeToFile(G, folder_path / "G.txt");

            // For each new frame slowly decay the image back to the base gray image
            MI = (1-memoryWeight)*decayBase + memoryWeight*MI;
            MI += V;
            // Image can't have negative intensity Values
            for (int i = 0; i<height; i++){
                for (int j = 0; j<width; j++){
                    if (MI(i,j) < 0.0){
                        MI(i,j) = 0.0;
                    }
                    if (MI(i,j) > 255.0){
                        MI(i,j) = 255.0;
                    }
                }
            }

            for (int iter = 0; iter < iterations; ++iter){
//                std::shuffle(std::begin(permutation), std::end(permutation), rng);
                interacting_maps_step(V, MI, I, delta_I,F, G, R, CCM, dCdx, dCdy, A, Identity_minus_outerProducts, weights, permutation, N);
                if (iter%100==0){
                    std::cout << iter << "/" << iterations << "\t";
                    std::cout << "-V=FG?: " << VFG_check(V, F, G, 0.1) << std::endl;
                    writeToFile(V, folder_path / "V2.txt");
                    writeToFile(I, folder_path / "I2.txt");
                    writeToFile(delta_I, folder_path / "I_gradient2.txt");
                    writeToFile(F, folder_path / "F2.txt");
                    writeToFile(G, folder_path / "G2.txt");
                }
//                if (iter%2270==0){
//                    // std::cout << "Here!" << std::endl;
//                    std::string image_name = "VIGF_testIter_" + std::to_string(iter) + ".png";
//                    fs::path image_path = folder_path / image_name;
//                    create_VIGF(Tensor2Matrix(V), Tensor2Matrix(I), G, F, image_path, true);
//                }
//                if (iter%2271==0){
//                    // std::cout << "Here!" << std::endl;
//                    std::string image_name = "VIGF_testIter_" + std::to_string(iter) + ".png";
//                    fs::path image_path = folder_path / image_name;
//                    create_VIGF(Tensor2Matrix(V), Tensor2Matrix(I), G, F, image_path, true);
//                }
            }
            writeToFile(V, folder_path / "V2.txt");
            writeToFile(I, folder_path / "I2.txt");
            writeToFile(delta_I, folder_path / "I_gradient2.txt");
            writeToFile(F, folder_path / "F2.txt");
            writeToFile(G, folder_path / "G2.txt");
            std::cout << "R: " << R << std::endl;
            counter++;
            std::cout << "Frame: " << counter << std::endl;
            std::string image_name = "VIGF_" + std::to_string(counter) + ".png";
            fs::path image_path = folder_path / image_name;
            create_VIGF(Tensor2Matrix(V), Tensor2Matrix(I), G, F, image_path, true);
            image_name = "VvsFG" + std::to_string(counter) + ".png";
            image_path = folder_path / image_name;
            plot_VvsFG(Tensor2Matrix(V), F, G, image_path, true);
        }
        Instrumentor::Get().EndSession();
    }

}



