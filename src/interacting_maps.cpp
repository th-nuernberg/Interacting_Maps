#include "interacting_maps.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <sstream>
#include <cassert>
#include <numeric>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>


// Define a type for convenience
// using namespace autodiff;

namespace fs = std::filesystem;

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

// GRADIENT CALCUATIONS

Eigen::VectorXd gradient(const Eigen::VectorXd& x) {
    int n = x.size();
    Eigen::VectorXd grad(n);

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

Eigen::MatrixXf gradient_x(const Eigen::MatrixXf& mat) {
    int rows = mat.rows();
    int cols = mat.cols();
    Eigen::MatrixXf grad_x(rows, cols);

    // Compute central differences for interior points
    grad_x.block(1, 0, rows - 2, cols) = (mat.block(2, 0, rows - 2, cols) - mat.block(0, 0, rows - 2, cols)) / 2.0;

    // Compute forward difference for the first row
    grad_x.row(0) = mat.row(1) - mat.row(0);

    // Compute backward difference for the last row
    grad_x.row(rows - 1) = mat.row(rows - 1) - mat.row(rows - 2);

    return grad_x;
}

Eigen::MatrixXf gradient_y(const Eigen::MatrixXf& mat) {
    int rows = mat.rows();
    int cols = mat.cols();
    Eigen::MatrixXf grad_y(rows, cols);

    // Compute central differences for interior points
    grad_y.block(0, 1, rows, cols - 2) = (mat.block(0, 2, rows, cols - 2) - mat.block(0, 0, rows, cols - 2)) / 2.0;

    // Compute forward difference for the first column
    grad_y.col(0) = mat.col(1) - mat.col(0);

    // Compute backward difference for the last column
    grad_y.col(cols - 1) = mat.col(cols - 1) - mat.col(cols - 2);

    return grad_y;
}

// TENSOR CASTING

Eigen::VectorXf Tensor2Vector(const Eigen::Tensor<float,1>& input) {
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

Eigen::MatrixXf Tensor2Matrix(const Eigen::Tensor<float,2>& input){
    Eigen::array<Eigen::Index, 2> dims = input.dimensions();
    const float* data_ptr = input.data();
    Eigen::Map<const Eigen::MatrixXf> result(data_ptr, dims[0], dims[1]);
    return result;
}

Eigen::Tensor<float,2> Matrix2Tensor(const Eigen::MatrixXf& input) {
    const float* data_ptr = input.data();
    Eigen::TensorMap<const Eigen::Tensor<float,2>> result(data_ptr, input.rows(), input.cols());
    return result;
}

// OPENCV PART

cv::Mat eigenToCvMat(const Eigen::MatrixXf& eigen_matrix) {
    // Map Eigen matrix data to cv::Mat without copying
    return cv::Mat(eigen_matrix.rows(), eigen_matrix.cols(), CV_32F, (void*)eigen_matrix.data());
}

cv::Mat eigenToCvMatCopy(const Eigen::MatrixXf& eigen_matrix) {
    // Create a cv::Mat and copy Eigen matrix data into it
    cv::Mat mat(eigen_matrix.rows(), eigen_matrix.cols(), CV_32F);
    for (int i = 0; i < eigen_matrix.rows(); ++i) {
        for (int j = 0; j < eigen_matrix.cols(); ++j) {
            mat.at<float>(i, j) = eigen_matrix(i, j);
        }
    }
    return mat;
}

Eigen::MatrixXf cvMatToEigenMap(const cv::Mat& mat) {
    // Ensure the cv::Mat has the correct type
    CV_Assert(mat.type() == CV_32F);
    Eigen::array<Eigen::Index, 2> dims;
    dims[0] = mat.rows;
    dims[1] = mat.cols;
    const float* data_ptr = mat.ptr<float>();
    Eigen::Map<const Eigen::MatrixXf> result(data_ptr, dims[0], dims[1]);
    return result;
}

Eigen::MatrixXf cvMatToEigenCopy(const cv::Mat& mat) {
    // Ensure the cv::Mat has the correct type
    CV_Assert(mat.type() == CV_32F);
    Eigen::MatrixXf eigen_matrix(mat.rows, mat.cols);
    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j) {
            eigen_matrix(i, j) = mat.at<float>(i, j);
        }
    }
    return eigen_matrix;
}

cv::Mat convertToFloat(cv::Mat& mat) {
    // Ensure the source matrix is of type CV_8U
    CV_Assert(mat.type() == CV_8U);

    // Convert the source matrix to CV_32F
    mat.convertTo(mat, CV_32F, 1.0 / 255.0); // Scaling from [0, 255] to [0, 1]

    return mat;
}

cv::Mat undistort_image(const cv::Mat& image, const cv::Mat& camera_matrix, const cv::Mat& distortion_parameters) {
    // cv::Mat new_camera_matrix;
    // cv::Rect roi;
    // new_camera_matrix = cv::getOptimalNewCameraMatrix(camera_matrix, distortion_parameters, cv::Size(width, height), 1, cv::Size(width, height), &roi);
    cv::Mat undistorted_image;
    cv::undistort(image, undistorted_image, camera_matrix, distortion_parameters, camera_matrix);
    return undistorted_image;
}

std::vector<cv::Mat> undistort_images(const std::vector<cv::Mat>& images, const Eigen::MatrixXf& camera_matrix, const Eigen::MatrixXf& distortion_coefficients) {
    std::vector<cv::Mat> undistorted_images;

    // Convert Eigen matrices to cv::Mat
    cv::Mat camera_matrix_cv = eigenToCvMat(camera_matrix);
    cv::Mat distortion_coefficients_cv = eigenToCvMat(distortion_coefficients);

    for (const auto& image : images) {
        undistorted_images.push_back(undistort_image(image, camera_matrix_cv, distortion_coefficients_cv));
    }
    return undistorted_images;
}

// Function to convert Eigen::MatrixXf to grayscale image
cv::Mat frame2grayscale(const Eigen::MatrixXf& frame) {
    // Convert Eigen::MatrixXf to cv::Mat
    cv::Mat frame_cv = eigenToCvMat(frame);

    // Find min and max polarity
    double min_polarity, max_polarity;
    cv::minMaxLoc(frame_cv, &min_polarity, &max_polarity);

    // Normalize the frame
    cv::Mat normalized_frame;
    frame_cv.convertTo(normalized_frame, CV_32F, 1.0 / (max_polarity - min_polarity), -min_polarity / (max_polarity - min_polarity));

    // Scale to 0-255 and convert to CV_8U
    cv::Mat grayscale_frame;
    normalized_frame.convertTo(grayscale_frame, CV_8U, 255.0);

    return grayscale_frame;
}

cv::Mat V2image(const Eigen::MatrixXf& V) {
    // Determine the shape of the image
    int rows = V.rows();
    int cols = V.cols();
    
    // Create an empty image with 3 channels (BGR)
    cv::Mat image = cv::Mat::zeros(rows, cols, CV_8UC3);
    
    // Process on_events (V > 0)
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (V(i, j) > 0) {
                image.at<cv::Vec3b>(i, j)[0] = 255; // Set blue channel
            }
        }
    }
    
    // Process off_events (V < 0)
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (V(i, j) < 0) {
                image.at<cv::Vec3b>(i, j)[2] = 255; // Set red channel
            }
        }
    }
    
    return image;
}


cv::Mat vector_field2image(const Eigen::Tensor<float, 3>& vector_field) {
    // oben rechts: blau
    // oben links: pink
    // unten rechts: grün
    // unten links: gelb/orange


    int rows = vector_field.dimension(0);
    int cols = vector_field.dimension(1);

    // Calculate angles and saturations
    Eigen::MatrixXf angles(rows, cols);
    Eigen::MatrixXf saturations(rows, cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float x = vector_field(i, j, 0);
            float y = vector_field(i, j, 1);
            angles(i, j) = std::atan2(y, x);
            saturations(i, j) = std::sqrt(x * x + y * y);
        }
    }

    // Normalize angles to [0, 179]
    cv::Mat hue(rows, cols, CV_8U);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            hue.at<uint8_t>(i, j) = static_cast<uint8_t>((angles(i, j) + M_PI) / (2 * M_PI) * 179);
        }
    }

    // Normalize saturations to [0, 255]
    float max_saturation = saturations.maxCoeff();
    cv::Mat saturation(rows, cols, CV_8U);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            saturation.at<uint8_t>(i, j) = static_cast<uint8_t>(std::min(saturations(i, j) / max_saturation * 255, 255.0f));
        }
    }

    // Value channel (full brightness)
    cv::Mat value(rows, cols, CV_8U, cv::Scalar(255));

    // Merge HSV channels
    std::vector<cv::Mat> hsv_channels = { hue, saturation, value };
    cv::Mat hsv_image;
    cv::merge(hsv_channels, hsv_image);

    // Convert HSV image to BGR format
    cv::Mat bgr_image;
    cv::cvtColor(hsv_image, bgr_image, cv::COLOR_HSV2BGR);

    return bgr_image;
}

cv::Mat create_VIFG(const MatrixXf& V, const MatrixXf& I, const Tensor<float, 3>& F, const Tensor<float, 3>& G, const std::string& name = "VIFG", bool save = false) {
    cv::Mat V_img = V2image(V);
    cv::Mat I_img = frame2grayscale(I);
    cv::Mat F_img = vector_field2image(F);
    cv::Mat G_img = vector_field2image(G);

    int rows = V.rows();
    int cols = V.cols();
    cv::Mat image(rows * 2 + 20, cols * 2 + 20, CV_8UC3, cv::Scalar(0, 0, 0));

    V_img.copyTo(image(cv::Rect(5, 5, cols, rows)));
    cvtColor(I_img, I_img, cv::COLOR_GRAY2BGR);
    I_img.copyTo(image(cv::Rect(cols + 10, 5, cols, rows)));
    G_img.copyTo(image(cv::Rect(5, rows + 10, cols, rows)));
    F_img.copyTo(image(cv::Rect(cols + 10, rows + 10, cols, rows)));

    if (save && !name.empty()) {
        imwrite(name, image);
    }

    return image;
}


// INTERACTING MAPS

std::vector<int> digitize(const std::vector<double>& input, const std::vector<double>& bins) {
    std::vector<int> indices;
    for (const double& value : input) {
        // Find the index where 'value' should be inserted to keep 'bins' sorted
        auto it = std::upper_bound(bins.begin(), bins.end(), value);
        // Calculate the index
        int index = it - bins.begin();
        indices.push_back(index);
    }
    return indices;
}

std::string create_folder_and_update_gitignore(const std::string& folder_name) {
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
    return folder_path.string();
}

void read_calib(const std::string file_path, std::vector<float>& calibration_data){
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

void read_events(const std::string file_path, std::vector<Event>& events, float start_time, float end_time, int max_events = INT32_MAX){
    fs::path current_directory = fs::current_path();
    std::string path = current_directory / file_path;
    if (fs::exists(path)) {
        std::ifstream event_file(path);
        int counter;
        float time;
        int width, height, polarity;
        while (event_file >> time >> width >> height >> polarity){
            if (time < start_time) continue;
            if (time > end_time) break;
            if (counter > max_events) break;
            Event event;
            event.time = time;
            std::vector<int> coords = {width, height};
            event.coordinates = coords;
            event.polarity = polarity;
            events.push_back(event);
            counter++;
        }
    }
}

std::vector<std::vector<Event>> bin_events(std::vector<Event>& events, float bucketsize = 0.05){
    float min_time = events.front().time;
    float max_time = events.back().time;
    int n_buckets = int((max_time-min_time)/bucketsize);

    Eigen::VectorXf bins = Eigen::VectorXf::LinSpaced(n_buckets, min_time, max_time);
    std::vector<std::vector<Event>> buckets(n_buckets, std::vector<Event>());


    // Find the indices where each event belongs according to its time stamp
    // std::vector<int> indices;
    int index;
    for (const Event& event : events) {
        // Find the index where 'value' should be inserted to keep 'bins' sorted
        auto it = std::lower_bound(bins.begin(), bins.end(), event.time);

        // If the iterator points to the end, timeValue is greater than all bins
        if (it == bins.end()) {
            index = bins.size(); // timeValue is beyond the last bin
        }

        index = std::distance(bins.begin(), it);

        // std::cout << index << " ";
        // indices.push_back(index);
        buckets[*it].push_back(event);
    }
    return buckets;
}

void create_frames(std::vector<std::vector<Event>>& bucketed_events, std::vector<Tensor<float,2>>& frames, int camera_height, int camera_width){
    for (std::vector<Event> event_vector : bucketed_events){
        Tensor<float,2> frame(camera_height, camera_width);
        Tensor<float,2> cum_frame(camera_height, camera_width);
        for (Event event : event_vector){
            frame(event.coordinates.at(0), event.coordinates.at(1)) = event.polarity;
            cum_frame(event.coordinates.at(0), event.coordinates.at(1)) += event.polarity;
        }
        frames.push_back(frame);
    }
}

void create_sparse_matrix(int N, Tensor<float,2>& V, SpMat& result){

    // Step 1: Create the repeated identity matrix part
    std::vector<Triplet<float>> tripletList;
    tripletList.reserve(N*3*2);
    for (int i = 0; i<N*3; i++){
        tripletList.push_back(Triplet<float>(i,i%3,1.0));
    }

    // Step 2: Create the diagonal and off-diagonal values from V
    int j = 3;
    array<int,1> one_dim{{V.size()}};
    Tensor<float,1> k = V.reshape(one_dim);
    for (int i = 0; i<N*3; i++){
        // std::cout << " i: " << i << " j: " << j << ", " << std::endl;
        tripletList.push_back(Triplet<float>(i,j,k(i)));
        if (i%3 == 2){
            j++;
        }
    }

    result.setFromTriplets(tripletList.begin(), tripletList.end());
}

void norm_tensor_along_dim3(Tensor<float,3>& T, Tensor<float,2>& norm){
    array<int,1> dims({2});
    norm = T.square().sum(dims).sqrt();
}

// Function to compute C_star
autodiff::Vector3real C_star(autodiff::real x, autodiff::real y, int N_x, int N_y, float height, float width, float rs) {
    autodiff::Vector3real result;
    result << height * (-1 + (2 * x) / (N_x - 1)),
              width * (1 - (2 * y) / (N_y - 1)),
              rs;
    return result;
}

// Function to compute C
autodiff::Vector3real C(autodiff::real x, autodiff::real y, int N_x, int N_y, float height, float width, float rs) {
    autodiff::Vector3real c_star = C_star(x, y, N_x, N_y, height, width, rs);
    autodiff::real norm = sqrt(c_star.squaredNorm());
    return c_star / norm;
}

void find_C(int N_x, int N_y, float view_angle_x, float view_angle_y, float rs, Tensor<float,3>& CCM, Tensor<float,3>& C_x, Tensor<float,3>& C_y) {
    float height = tan(view_angle_x / 2);
    float width = tan(view_angle_y / 2);

    // Create grid of points
    Eigen::MatrixXd XX(N_x, N_y);
    Eigen::MatrixXd YY(N_x, N_y);
    for (int i = 0; i < N_x; ++i) {
        for (int j = 0; j < N_y; ++j) {
            XX(i, j) = i;
            YY(i, j) = j;
        }
    }

    // Compute the camera calibration map (CCM) and the Jacobians
    // std::vector<std::vector<autodiff::Vector3real>> CCM;
    // Tensor<float, 3> CCM_T;
    // Tensor<float, 3> C_x;
    // Tensor<float, 3> C_y;
    // std::vector<std::vector<Eigen::VectorXd>> C_y;
    for (int i = 0; i < N_x; ++i) {
        for (int j = 0; j < N_y; ++j) {
            autodiff::real x = XX(i, j);
            autodiff::real y = YY(i, j);

            // Compute the function value
            autodiff::Vector3real c_val = C(x, y, N_x, N_y, height, width, rs);
            CCM(i,j,0) = static_cast<float>(c_val(0));
            CCM(i,j,1) = static_cast<float>(c_val(1));
            CCM(i,j,2) = static_cast<float>(c_val(2));


            // Compute the Jacobians
            // Vector3real dCdx;
            // Vector3real dCdy;
            autodiff::VectorXreal F;

            Eigen::VectorXd dCdx = autodiff::jacobian(C, wrt(x), at(x,y,N_x, N_y, height, width, rs), F);
            Eigen::VectorXd dCdy = autodiff::jacobian(C, wrt(y), at(x,y,N_x, N_y, height, width, rs), F);



            C_x(i,j,0) = dCdx(0);
            C_x(i,j,1) = dCdx(1);
            C_x(i,j,2) = dCdx(2);
            C_y(i,j,0) = dCdy(0);
            C_y(i,j,1) = dCdy(1);
            C_y(i,j,2) = dCdy(2);

            // C_x(i,j,0) = 1.0f;
            // C_x(i,j,1) = 1.0f;
            // C_x(i,j,2) = 1.0f;
            // C_y(i,j,0) = 1.0f;
            // C_y(i,j,1) = 1.0f;
            // C_y(i,j,2) = 1.0f;
        }
    }
}

void crossProductTensors(const Eigen::Tensor<float, 3>& A, const Eigen::Tensor<float, 3>& B, Eigen::Tensor<float, 3>& C) {
    assert(A.dimensions() == B.dimensions() && "Tensors A and B must have the same shape");

    // Use Eigen's tensor operations to compute the cross product
    C.chip(0, 2) = A.chip(1, 2) * B.chip(2, 2) - A.chip(2, 2) * B.chip(1, 2);
    C.chip(1, 2) = A.chip(2, 2) * B.chip(0, 2) - A.chip(0, 2) * B.chip(2, 2);
    C.chip(2, 2) = A.chip(0, 2) * B.chip(1, 2) - A.chip(1, 2) * B.chip(0, 2);
}

// Function to compute the cross product of two 3-tensors
void crossProductTensors_loop(const Eigen::Tensor<float, 3>& A, const Eigen::Tensor<float, 3>& B, Eigen::Tensor<float, 3>& C) {
    assert(A.dimensions() == B.dimensions() && "Tensors A and B must have the same shape");
    for (int i = 0; i < A.dimension(0); ++i) {
        for (int j = 0; j < A.dimension(1); ++j) {
            C(i, j, 0) = A(i, j, 1) * B(i, j, 2) - A(i, j, 2) * B(i, j, 1);
            C(i, j, 1) = A(i, j, 2) * B(i, j, 0) - A(i, j, 0) * B(i, j, 2);
            C(i, j, 2) = A(i, j, 0) * B(i, j, 1) - A(i, j, 1) * B(i, j, 0);
        }
    }
}

// def m32(V, C_x, C_y):
//     x_comp = jnp.sign((jnp.einsum("ijk,ijk->ij", V, (jnp.cross(C_y, jnp.cross(C_x, C_y)))))) * vector_distance(V, C_y)/vector_distance(C_x, C_y)
//     y_comp = jnp.sign((jnp.einsum("ijk,ijk->ij", V, (jnp.cross(C_x, jnp.cross(C_y, C_x)))))) * vector_distance(V, C_x)/vector_distance(C_y, C_x)
//     return jnp.array([x_comp, y_comp]).transpose((1,2,0))

// def vector_distance(vec1, vec2):
//     return jnp.linalg.norm(jnp.cross(vec1, vec2), ord=2, axis=2)/jnp.linalg.norm(vec2, ord=2,axis=2)

void vector_distance(Tensor<float,3> &vec1, Tensor<float,3> &vec2, Tensor<float,2> &distance){
    const auto& dimensions = vec1.dimensions();
    Tensor<float,3> cross_product(dimensions);
    Tensor<float,2> norm(dimensions[0], dimensions[1]);
    Tensor<float,2> norm2(dimensions[0], dimensions[1]);
    // std::cout << vec1.dimensions() << vec2.dimensions() << std::endl;
    crossProductTensors(vec1, vec2, cross_product);
    norm_tensor_along_dim3(cross_product, norm);
    norm_tensor_along_dim3(vec2, norm2);
    distance = norm/norm2;

}

float sign_func(float x)
{
    // Apply via a.unaryExpr(std::ptr_fun(sign_func))
    if (x > 0)
        return +1.0;
    else if (x == 0)
        return 0.0;
    else
        return -1.0;
}

// Function using nested loops to compute the dot product
void computeDotProductWithLoops(const Eigen::Tensor<float, 3>& A, const Eigen::Tensor<float, 3>& B, Eigen::Tensor<float, 2>& D) {
    const int m = A.dimension(0);
    const int n = A.dimension(1);
    const int d = A.dimension(2);

    // #pragma omp parallel for collapse(2)
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float dotProduct = 0.0f; // Initialize the dot product for position (i, j)
            for (int k = 0; k < d; ++k) {
                dotProduct += A(i, j, k) * B(i, j, k);
            }
            D(i, j) = dotProduct; // Store the result in tensor D
        }
    }
}

void m32(Tensor<float,3> &In, Tensor<float,3> &C_x, Tensor<float,3> &C_y, Tensor<float,3> &Out){
    const auto& dimensions = In.dimensions();
    Tensor<float,3> C1(dimensions);
    Tensor<float,3> C2(dimensions);
    Tensor<float,2> dot(dimensions[0], dimensions[1]);
    Tensor<float,2> sign(dimensions[0], dimensions[1]);
    Tensor<float,2> distance1(dimensions[0], dimensions[1]);
    Tensor<float,2> distance2(dimensions[0], dimensions[1]);
    
    crossProductTensors(C_x,C_y,C1);

    crossProductTensors(C_y,C1,C2);
    computeDotProductWithLoops(In,C2,dot);
    sign = dot.unaryExpr(std::ptr_fun(sign_func));
    vector_distance(In, C_y, distance1);
    vector_distance(C_x, C_y, distance2);
    Out.chip(0,2) = sign * distance1/distance2;
    // std::cout << "C1" << C1 << std::endl;
    // std::cout << "C2" << C2 << std::endl;
    // std::cout << "dot" << dot << std::endl;
    // std::cout << "sign" << sign << std::endl;
    // std::cout << "distance1" << distance1 << std::endl;
    // std::cout << "distance2" << distance2 << std::endl;
    // std::cout << "Out" << Out << std::endl;
    crossProductTensors(C_x,-C1,C2);
    computeDotProductWithLoops(In,C2,dot);
    sign = dot.unaryExpr(std::ptr_fun(sign_func));
    vector_distance(In, C_x, distance1);
    vector_distance(C_y, C_x, distance2);
    // Tensor<float,2>res1 = sign * distance1/distance2;
    Out.chip(1,2) = sign * distance1/distance2;
}

void m23(Tensor<float,3>& In, Tensor<float,3>& Cx, Tensor<float,3>& Cy, Tensor<float,3>& Out) {
    const auto& dimensions = Cx.dimensions();
    for (int i = 0; i < dimensions[0]; i++){
        for (int j = 0; j < dimensions[1]; j++){
            Out(i,j,0) = In(i,j,0) * Cx(i,j,0) + In(i,j,1) * Cy(i,j,0);
            Out(i,j,1) = In(i,j,0) * Cx(i,j,1) + In(i,j,1) * Cy(i,j,1);
            Out(i,j,2) = In(i,j,0) * Cx(i,j,2) + In(i,j,1) * Cy(i,j,2);
        }
    }
}
// // Function using .chip() to compute the dot product
// void computeDotProductWithChip(const Eigen::Tensor<float, 3>& A, const Eigen::Tensor<float, 3>& B, Eigen::Tensor<float, 2>& D) {
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



// Function to time the performance of a given dot product computation function
template<typename Func>
void timeDotProductComputation(Func func, const Eigen::Tensor<float, 3>& A, const Eigen::Tensor<float, 3>& B, Eigen::Tensor<float, 2>& D, int iterations) {
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        func(A, B, D);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time elapsed: " << elapsed.count() << " seconds" << std::endl;
}


void update_F_from_G(Tensor<float,3>& F, Tensor<float,2>& V, Tensor<float,3>& G, float lr, float weight_FG){

    const auto& dimensions = F.dimensions();
    Tensor<float,3> update_F(dimensions);
    for (int i = 0; i<dimensions[0]; i++){
        for (int j = 0; j<dimensions[1]; j++){
            update_F(i,j,0) = F(i,j,0) - ((G(i,j,0)/(G(i,j,0) * G(i,j,0) + G(i,j,1) * G(i,j,1))) * (V(i,j) + (F(i,j,0) * G(i,j,0) + F(i,j,1) * G(i,j,1))));
            update_F(i,j,1) = F(i,j,1) - ((G(i,j,1)/(G(i,j,0) * G(i,j,0) + G(i,j,1) * G(i,j,1))) * (V(i,j) + (F(i,j,0) * G(i,j,0) + F(i,j,1) * G(i,j,1))));
        }
    }
    F = (1-weight_FG)*F + lr * weight_FG * update_F;
}

void update_G_from_F(Tensor<float,3>& G, Tensor<float,2>& V, Tensor<float,3>& F, float lr, float weight_GF){

    const auto& dimensions = G.dimensions();
    Tensor<float,3> update_G(dimensions);
    for (int i = 0; i<dimensions[0]; i++){
        for (int j = 0; j<dimensions[1]; j++){
            update_G(i,j,0) = (i,j,0) - ((F(i,j,0)/(F(i,j,0) * F(i,j,0) + F(i,j,1) * F(i,j,1))) * (V(i,j) + (G(i,j,0) * F(i,j,0) + G(i,j,1) * F(i,j,1))));
            update_G(i,j,1) = G(i,j,1) - ((F(i,j,1)/(F(i,j,0) * F(i,j,0) + F(i,j,1) * F(i,j,1))) * (V(i,j) + (G(i,j,0) * F(i,j,0) + G(i,j,1) * F(i,j,1))));
        }
    }
    G = (1-weight_GF)*G + lr * weight_GF * update_G;
}

void update_G_from_I(Tensor<float,3> &G, Tensor<float,3> &I_gradient, float weight_GI){
    G = (1 - weight_GI) * G + weight_GI*I_gradient;
}

void update_I_from_V(Tensor<float,2> &I, Tensor<float,2> &cum_V, float weight_IV=0.5, float time_step=0.05){
    I = (1 - weight_IV)*I + weight_IV*cum_V;
    const auto& dimensions = I.dimensions();
    for (int i = 0; i<dimensions[0]; i++){
        for (int j = 0; j<dimensions[1]; j++){
            if (I(i,j)>0){
                if (I(i,j)>time_step){
                    I(i,j) -= time_step;
                }
                else{
                    I(i,j) = 0;
                }
            }
            if (I(i,j)<0){
                if (I(i,j)<time_step){
                    I(i,j) += time_step;
                }
                else{
                    I(i,j) = 0;
                }
            }
        }
    }
    // I = jnp.where(I > 0, jnp.maximum(I - time_step, 0), I) # decay to neutral value
    // I = jnp.where(I < 0, jnp.minimum(I + time_step, 0), I) # decay to neutral value

}

void update_I_from_G(Tensor<float,2> &I, Tensor<float,3> &I_gradient, Tensor<float,3> &G, float weight_IG=0.5){
    const auto& dimensions = I.dimensions();
    Tensor<float,3> temp_map = G - I_gradient;
    Tensor<float,2> x_update(dimensions);
    Tensor<float,2> y_update(dimensions);
    for (int i = 0; i<dimensions[0]-1; i++){
        for (int j = 0; j<dimensions[1]-1; j++){
            if (i==0){
                x_update(i,j) = temp_map(i,j,0);
            }
            else{
                x_update(i,j) = temp_map(i,j,0) - temp_map(i-1,j,0);
            }
            if (j==0){
                y_update(i,j) = temp_map(i,j,1);
            }
            else{
                x_update(i,j) = temp_map(i,j,1) - temp_map(i,j-1,1);
            }
        }
    }
    I = (1 - weight_IG)*I + weight_IG*(I - x_update - y_update);
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
}

void update_F_from_R(Tensor<float,3>& F, Tensor<float,3>& CCM, Tensor<float,3>& Cx, Tensor<float,3>& Cy, Tensor<float,1> R, float weight_FR){
    Tensor<float,3> cross(F.dimensions());
    const auto& dimensions = F.dimensions();
    Tensor<float,3> update(dimensions[0], dimensions[1], dimensions[2]);
    crossProductTensors(R, CCM, cross);
    m32(update, Cx, Cy, update);
    F = (1 - weight_FR) * F + weight_FR * update;
}

void update_R_from_F(Tensor<float,1> R, Tensor<float,3>& F, Tensor<float,3>& C, Tensor<float,3>& Cx, Tensor<float,3>& Cy, float weight_RF, int N) {
    Tensor<float,3> transformed_F;
    m23(F, Cx, Cy, transformed_F);
    Tensor<float,3> points;
    crossProductTensors(C, transformed_F, points);

    Eigen::SparseMatrix<double> SpMat(N*3, N+3);
    Eigen::VectorXd x(N+3);
    x.setZero();

    Eigen::array<Eigen::DenseIndex, 1> points_reshaper({N*3});
    Eigen::Tensor<float,1> reshaped_points = points.reshape(points_reshaper); // reshaped_points need to be Eigen::Vector for solver
    Eigen::VectorXf points_vector = Tensor2Vector(reshaped_points);
    
    Eigen::array<Eigen::DenseIndex, 1> C_reshaper({N*3});
    Tensor<float,1> reshaped_C = C.reshape(C_reshaper);

    std::vector<Eigen::Triplet<double>> tripletList;
    tripletList.reserve(6 * N); // Assuming on average 2 non-zero entries per row
    int j = 3;
    for (int i = 0; i < N*3; i++) {
        tripletList.emplace_back(i, i%3, 1.0); // Diagonal element
        tripletList.emplace_back(i, j, reshaped_C[i]);
        if (i%3 == 2){
            j++;
        }
    }
    
    SpMat.setFromTriplets(tripletList.begin(), tripletList.end());
    
    // Perform the computation and measure the time
    // Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<double>> solver;
    Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<double>, Eigen::LeastSquareDiagonalPreconditioner<double>> solver;
    // solver.setTolerance(1e-6);
    solver.setTolerance(1e-4);

    solver.compute(SpMat);
    if(solver.info() != Eigen::Success) {
        std::cerr << "Decomposition failed during update_R_from_C" << std::endl;
    }

    // x = solver.solve(b);
    x = solver.solveWithGuess(points_vector.cast<double>(), x);
    if(solver.info() != Eigen::Success) {
        std::cerr << "Solving failed during update_R_from_C " << std::endl;
    }
}

// void interacting_maps_step(Tensor<float,2>& V, Tensor<float,2>& cum_V, Tensor<float,2>& I, Tensor<float,3>& F, Tensor<float,3>& G, Tensor<float,1>& R, Tensor<float,3>& CCM, Tensor<float,3>& dCdx, Tensor<float,3>& dCdy, std::unordered_map<std::string,float> weights, std::vector<int> permutation, int N){
//     Eigen::array<Eigen::Index, 2> dimensions = I.dimensions();
//     Eigen::Tensor<float,3> delta_I(dimensions[0], dimensions[1], 2);
//     // delta_I.chip(0,2) = Matrix2Tensor(gradient_x(Tensor2Matrix(I)));
//     delta_I.chip(1,2) = Matrix2Tensor(gradient_y(Tensor2Matrix(I)));

//     for (auto & element : permutation){
//         switch(element) {
//             case 0:
//             update_F_from_G(F, V, G, weights["lr"], weights["weight_FG"]);

//             case 1:
//             update_F_from_R(F, CCM, dCdx, dCdy, R, weights["weight_FR"]);

//             case 2:
//             update_G_from_F(G, V, F, weights["lr"], weights["weight_FG"]);

//             case 3:
//             update_G_from_I(G, delta_I, weights["weight_GI"]);

//             case 4:
//             update_I_from_G(I, delta_I, G, weights["weight_IG"]);
//             delta_I.chip(0,2) = Matrix2Tensor(gradient_x(Tensor2Matrix(I)));
//             delta_I.chip(1,2) = Matrix2Tensor(gradient_y(Tensor2Matrix(I)));

//             case 5:
//             update_I_from_V(I, V, weights["weight_IV"], weights["timestep"]);
//             delta_I.chip(0,2) = Matrix2Tensor(gradient_x(Tensor2Matrix(I)));
//             delta_I.chip(1,2) = Matrix2Tensor(gradient_y(Tensor2Matrix(I)));

//             case 6:
//             update_R_from_F(R, F, CCM, dCdx, dCdy, weights["weight_RF"], N);
//         }
//     }
// }


int main() {

    // TEST PARAMETERS
    int n = 5;
    int m = 7;
    int N = 50;
    int M = 70;
    int NM = N*M;

    // Camera calibration matrix (C/CCM) and dCdx/dCdy
    Tensor<float,3> CCM(n,m,3);
    CCM.setZero();
    Tensor<float,3> dCdx(n,m,3);
    dCdx.setZero();
    Tensor<float,3> dCdy(n,m,3);
    dCdy.setZero();

    // Optic flow F, temporal derivative V, spatial derivative G
    Tensor<float,3> F(n,m,2);
    F.setConstant(1.0);
    Tensor<float,2> V(n,m);
    V.setRandom();
    Tensor<float,3> G(n,m,2);
    G.setConstant(3.0);

    // Intesity I
    Tensor<float,2> I(n,m);
    I.setRandom();
    Tensor<float,3> I_gradient(n,m,2);
    I_gradient.setRandom();

    // Rotation Vector R
    Tensor<float,1> R(3);
    R.setRandom();

    // Vector distance function
    Tensor<float,2> distance;
    distance.setZero();

    // Test Tensor casts
    Tensor<float,2> T2M (N,M);
    Tensor<float,1> T2V (N);
    Tensor<float,2> M2T_res (N,M);
    Tensor<float,1> V2T_res (N);
    T2M.setConstant(1.0);
    T2V.setConstant(1.0);

    Eigen::MatrixXf M2T (N,M);
    Eigen::VectorXf V2T (N);
    Eigen::MatrixXf T2M_res (N,M);
    Eigen::VectorXf T2V_res (N);
    M2T.setConstant(2.0);
    V2T.setConstant(2.0);

    T2M_res = Tensor2Matrix(T2M);
    T2V_res = Tensor2Vector(T2V);
    M2T_res = Matrix2Tensor(M2T);
    V2T_res = Vector2Tensor(V2T);

    std::cout << "Implemented Tensor/Matrix/Vector casts" << std::endl;

    // Create results_folder
    std::string folder_name = "results";
    std::string folder_path = create_folder_and_update_gitignore(folder_name);
    std::cout << "Implemented Folder creation" << std::endl;

    // Read calibration file
    std::string calib_path = "../res/shapes_rotation/calib.txt";
    std::vector<float> calibration_data;
    read_calib(calib_path, calibration_data);
    std::cout << "Implemented calibration data readout" << std::endl;

    // Read events file
    std::string event_path = "../res/shapes_rotation/eventsshort.txt";
    std::vector<Event> event_data;
    read_events(event_path, event_data, 0.0, 1.0);
    std::cout << "Implemented events readout" << std::endl;

    // Bin events
    std::vector<std::vector<Event>> bucketed_events;
    bucketed_events = bin_events(event_data, 0.05);

    // Create frames
    std::vector<Tensor<float,2>> frames;
    create_frames(bucketed_events, frames, 180, 240);
    std::cout << "Implemented event binning and event frame creation" << std::endl;

    // Test Sparsematrix creation
    SpMat sparse_m(NM*3,NM+3);
    Tensor<float,2> F_flat(NM,3);
    F_flat.setRandom();
    create_sparse_matrix(NM, F_flat, sparse_m);
    std::cout << "Implemented sparse matrix creation for update_R" << std::endl;

    // Camera calibration map
    find_C(n, m, 3.1415/4, 3.1415/4, 1.0f, CCM, dCdx, dCdy);
    std::cout << "Implemented find_C function with autodiff" << std::endl;
    std::cout << "C" << std::endl;
    std::cout << CCM << std::endl;
    std::cout << "dCdx" << std::endl;
    std::cout << dCdx << std::endl;
    std::cout << "dCdy" << std::endl;
    std::cout << dCdy << std::endl;

    // Update F/G from G/F
    update_F_from_G(F, V, G, 1.0, 0.5);
    std::cout << "Implemented update_F_from_G/update_G_from_F" << std::endl;
    std::cout << "F after update" << std::endl;
    std::cout << F << std::endl;

    // Update I from G
    update_I_from_G(I, I_gradient, G, 0.2f);
    std::cout << "Implemented update_I_from_G" << std::endl;
    std::cout << "I after update" << std::endl;
    std::cout << I << std::endl;

    // Update I from V
    update_I_from_V(I, V, 0.2f, 0.05f);
    std::cout << "Implemented update_I_from_V" << std::endl;
    std::cout << "I after update" << std::endl;
    std::cout << I << std::endl;

    // Update G from I
    update_G_from_I(G, I_gradient, 0.2f);
    std::cout << "Implemented update_G_from_I" << std::endl;
    std::cout << "G after update" << std::endl;
    std::cout << G << std::endl;

    // // Tensor cross product
    // // Define the dimensions of the tensors
    Eigen::array<Eigen::Index, 3> dimensions = {n, m, 3};

    // // Initialize tensors A and B with random values
    Eigen::Tensor<float, 3> A(dimensions);
    Eigen::Tensor<float, 3> B(dimensions);
    Eigen::Tensor<float, 3> D(dimensions);
    A.setRandom();
    B.setRandom();

    int iterations = 1000;

    // Timing the chip-based implementation
    std::cout << "Timing cross product for tensors implementations " << std::endl;
    auto start_chip = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        crossProductTensors(A, B, D);
    }
    auto end_chip = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_chip = end_chip - start_chip;
    std::cout << "Time for chip-based implementation: " << duration_chip.count()/iterations << " seconds\n";

    // Timing the loop-based implementation
    auto start_loop = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        crossProductTensors_loop(A, B, D);
    }
    auto end_loop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_loop = end_loop - start_loop;
    std::cout << "Time for loop-based implementation: " << duration_loop.count()/iterations << " seconds\n";
    std::cout << "Implemented cross product for Tensors" << std::endl;

    // Define the resulting tensor D with shape (m, n)
    Eigen::Tensor<float, 2> E(n, m);
    // Time the dot product computation using nested loops
    std::cout << "Timing dot product computation with loops:" << std::endl;
    timeDotProductComputation(computeDotProductWithLoops, A, B, E, 1000);
    std::cout << "Implemented dot product for tensors" << std::endl;

    // // Time the dot product computation using .chip() (CHATGPT USES CHIP JUST FOR FANCY INDEXING)
    // std::cout << "Timing dot product computation with .chip():" << std::endl;
    // timeDotProductComputation(computeDotProductWithChip, A, B, E, 1000);

    // Test Vector distance function needed for M32
    vector_distance(dCdx, dCdy, distance);
    std::cout << "Implemented vector distance function for m32" << std::endl;
    std::cout << distance << std::endl;

    // M32 Test    
    Tensor<float,3> A2(n,m,2);
    m32(A, dCdx, dCdy, A2);
    std::cout << "Implemented transformation function m32" << std::endl;
    std::cout << A << std::endl;
    std::cout << A2 << std::endl;

    std::unordered_map<std::string,float> weights;
    weights["weight_FG"] = 0.2;
    weights["weight_FR"] = 0.2;
    weights["weight_GF"] = 0.2;
    weights["weight_GI"] = 0.2;
    weights["weight_IG"] = 0.2;
    weights["weight_IV"] = 0.2;
    weights["weight_RF"] = 0.2;
    weights["lr"] = 0.9;
    weights["timestep"] = 0.05f;

    std::vector<int> permutation {0,1,2,3,4,5,6};

    // interacting_maps_step(V, V, I, F, G, R, CCM, dCdx, dCdy, weights, permutation, NM);
    // std::cout << "Implemented interacting maps update step" << std::endl;

    // TESTING OPENCV CONVERSION
    Eigen::MatrixXf eigen_matrix = Eigen::MatrixXf::Constant(3, 3, 2.0);
    cv::Mat mat = eigenToCvMat(eigen_matrix);
    std::cout << "Implemented Eigen to CV map" << std::endl << mat << std::endl;

    cv::Mat mat2(3, 3, CV_32F, cv::Scalar::all(1));
    Eigen::MatrixXf eigen_matrix2 = cvMatToEigenMap(mat2);
    std::cout << "Implemented Eigen to CV map" << std::endl << eigen_matrix2 << std::endl;

    // Create an example Eigen matrix
    Eigen::MatrixXf grayscale_test = Eigen::MatrixXf::Random(100,100);

    // Convert to grayscale image
    cv::Mat grayscale_image = frame2grayscale(grayscale_test);

    // Display the result
    cv::imshow("Grayscale Image", grayscale_image);
    cv::waitKey(0);

    // cv::Mat sample_image = (cv::Mat_<float>(3, 3) << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9);

    // TESTING UNDISTORT IMAGE
    // Camera matrix (3x3)
    Eigen::MatrixXf camera_matrix_eigen(3, 3);
    camera_matrix_eigen << 800, 0, 320,
                            0, 800, 240,
                            0, 0, 1;
    cv::Mat camera_matrix_cv = eigenToCvMat(camera_matrix_eigen);

    // Distortion coefficients (1x5)
    Eigen::MatrixXf distortion_coefficients_eigen(1, 5);
    distortion_coefficients_eigen << 0.1, -0.05, 0, 0, 0; // Simple distortion
    cv::Mat distortion_coefficients_cv = eigenToCvMat(distortion_coefficients_eigen);

    // Convert Eigen matrices to cv::Mat and back to Eigen
    // Test undistort_image function
    cv::Mat undistorted_image = undistort_image(grayscale_image, camera_matrix_cv, distortion_coefficients_cv);

    // Print result
    cv::imshow("Undistorted Image", undistorted_image);
    cv::waitKey(0);

    // V to Image
    Eigen::MatrixXf V2image_eigen = Eigen::MatrixXf::Random(100,100);
    cv::Mat V2image_cv = V2image(V2image_eigen);
    cv::imshow("V Image", V2image_cv);
    cv::waitKey(0);

    // Vectorfield to Image
    Eigen::Tensor<float, 3> vector_field(1000,1000,2);
    // Eigen::Tensor<float, 2> x_values(1000,1000);
    // x_values.setConstant(-1.0);
    // Eigen::Tensor<float, 2> y_values(1000,1000);
    // y_values.setConstant(1.0);
    // vector_field.chip(0,2) = x_values;
    // vector_field.chip(1,2) = y_values;
    vector_field.setRandom();
    cv::Mat image = vector_field2image(vector_field);
    cv::imshow("Vector Field Image", image);
    cv::waitKey(0);



    // Example usage of the functions
    Eigen::MatrixXf Vimage = Eigen::MatrixXf::Random(100, 100);  // Example data
    Eigen::MatrixXf IImage = Eigen::MatrixXf::Random(100, 100);  // Example data
    Eigen::Tensor<float, 3> Fimage(100, 100, 2);          // Example data
    Eigen::Tensor<float, 3> Gimage(100, 100, 2);          // Example data

    // Fill example tensor data (normally you would have real data)
    for (int i = 0; i < 100; ++i) {
        for (int j = 0; j < 100; ++j) {
            F(i, j, 0) = std::sin(i * 0.1);
            F(i, j, 1) = std::cos(j * 0.1);
            G(i, j, 0) = std::cos(i * 0.1);
            G(i, j, 1) = std::sin(j * 0.1);
        }
    }

    cv::Mat result = create_VIFG(Vimage, IImage, Fimage, Gimage, "output.png", true);
    cv::imshow("Result", result);
    cv::waitKey(0);
}