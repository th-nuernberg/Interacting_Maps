#include <interacting_maps.h>
#include <iostream>
#include <filesystem>
#include <string>
#include <sstream>
#include <cassert>
#include <numeric>
#include "Instrumentor.h"
#include <cmath>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

#ifdef PROFILING
#define PROFILE_SCOPE(name) InstrumentationTimer timer##__LINE__(name)
#define PROFILE_FUNCTION() PROFILE_SCOPE(__FUNCTION__)
// #define PROFILE_FUNCTION() PROFILE_SCOPE(__FUNCSIG__) (Includes call attributes, whole signature of function)
#define PROFILE_MAIN(name)
#else
#define PROFILE_SCOPE(name)
#define PROFILE_FUNCTION()
#define PROFILE_MAIN(name) InstrumentationTimer timer##__LINE__(name)
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  STRING OPERATIONS  /////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Enables use of Event in out stream
 * @param os out stream to add Event to
 * @param e Event to add
 * @return new out stream
 */
std::ostream& operator << (std::ostream &os, const Event &e) {
    return (os << "Time: " << e.time << " Coords: " << e.coordinates[0] << " " << e.coordinates[1] << " Polarity: " << e.polarity);
}

//std::string Event::toString() const {
//    std::stringstream ss;
//    ss << (*this);
//    return ss.str();
//}

///**
// * Splits a string stream at a provided delimiter. Delimiter is removed
// * @param sstream string stream to be split
// * @param delimiter The delimiter, can be any char
// * @return Vector of split string
// */
//std::vector<std::string> split_string(std::stringstream sstream, char delimiter){
//    std::string segment;
//    std::vector<std::string> seglist;
//    while(std::getline(sstream, segment, delimiter))
//    {
//        seglist.push_back(segment);
//    }
//    return seglist;
//}

//  GRADIENT CALCULATIONS  /////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Calculates the gradient of 2-Tensor via central differences at a location (x,y) in x and y direction.
 * Reuses first/last value at border (effectively using forward or backward difference)
 * @param data Input of shape NxM
 * @param gradients 3-Tensor holding gradients
 * @param y up and down position of gradient of interest
 * @param x left and right position of gradient of interest
 */
void computeGradient(const Tensor2f &data, Tensor3f &gradients, int y, int x) {
    PROFILE_FUNCTION();
    // Compute gradient for update_IG
    const auto& ddimensions = data.dimensions();
    int rows = static_cast<int>(ddimensions[0]);
    int cols = static_cast<int>(ddimensions[1]);
    assert(ddimensions[0] == gdimensions[0]);
    assert(ddimensions[1] == gdimensions[1]);
    assert(y < rows);
    assert(x < cols);

    // Compute gradient along columns (down-up, y-direction)
    if (y == 0) {
        gradients(y, x, 0) = (data(y, x) - data(y + 1, x)) / 2.0f; // Central difference with replicate border
    } else if (y == rows - 1) {
        gradients(y, x, 0) = (data(y - 2, x) - data(y - 1, x)) / 2.0f; // Central difference with replicate border
    } else {
        gradients(y, x, 0) = (data(y - 1, x) - data(y + 1, x)) / 2.0f;
    }
    // Compute gradient along rows (left-right, x-direction)
    if (x == 0) {
        gradients(y, x, 1) = (data(y, x + 1) - data(y, x)) / 2.0f; // Central difference with replicate border
    } else if (x == cols - 1) {
        gradients(y, x, 1) = (data(y, x - 1) - data(y, x - 2)) / 2.0f; // Central difference with replicate border
    } else {
        gradients(y, x, 1) = (data(y, x + 1) - data(y, x - 1)) / 2.0f;
    }
}

/**
 * Calculates the gradient of 3-Tensor via central differences at a location (x,y) in x and y direction.
 * Reuses first/last value at border (effectively using forward or backward difference). Input should have a depth
 * of 2. Gradient in up and down direction is determined for depth 0; Gradient in left and right direction
 * is determined for depth 1.
 * @param data 3-Tensor containing the data of shape NxMxD
 * @param gradients 3-Tensor holding results
 * @param y up and down position of gradient of interest
 * @param x left and right position of gradient of interest
 */
void computeGradient(const Tensor3f &data, Tensor3f &gradients, int y, int x) {
    // Compute gradient for update_IG
    const auto& dimensions = data.dimensions();
    int rows = static_cast<int>(dimensions[0]);
    int cols = static_cast<int>(dimensions[1]);
    assert(y < rows);
    assert(x < cols);
    assert(static_cast<int>(dimensions[2]) == 2);
    // Compute gradient along columns (down-up, y-direction)
    if (y == 0) {
        gradients(y, x, 0) = (data(y, x, 0) - data(y + 1, x, 0)) / 2.0f; // Central difference with replicate border
    } else if (y == rows - 1) {
        gradients(y, x, 0) = (data(y - 2, x, 0) - data(y - 1, x, 0)) / 2.0f; // Central difference with replicate border
    } else {
        gradients(y, x, 0) = (data(y - 1, x, 0) - data(y + 1, x, 0)) / 2.0f;
    }
    // Compute gradient along rows (left-right, x-direction)
    if (x == 0) {
        gradients(y, x, 1) = (data(y, x + 1, 1) - data(y, x, 1)) / 2.0f; // Central difference with replicate border
    } else if (x == cols - 1) {
        gradients(y, x, 1) = (data(y, x - 1, 1) - data(y, x - 2, 1)) / 2.0f; // Central difference with replicate border
    } else {
        gradients(y, x, 1) = (data(y, x + 1, 1) - data(y, x - 1, 1)) / 2.0f;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  INTERACTING MAPS HELPER FUNCTIONS  /////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Take a vector of events and sort them into a collection of bins. Sorting happens according to the time dimension of
 * the events. Size of the bins is given in seconds.
 * @param events std::vector of Events
 * @param bin_size float
 * @return vector of vectors of binned events. First bin contains first timespan, last bin the last timespan.
 */
std::vector<std::vector<Event>> bin_events(std::vector<Event> &events, float bin_size = 0.05){
    std::vector<std::vector<Event>> bins;
    if (events.empty()) {
        return bins;  // Return empty if the input vector is empty
    }

    Event minVal = events.front();  // The lowest number in the sorted vector
    float currentBinStart = minVal.time;

    std::vector<Event> currentBin;

    for (const Event &event : events) {
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

/**
 * Convert a collection of binned events to a collection of frames (i.e. matrices representing an image)
 * @param bucketed_events collection of events sorted into bins based on time
 * @param frames returning collection of frames
 * @param camera_height height of the frame
 * @param camera_width width of the frame
 * @param eventContribution scales the contribution of a single event to the frame: polarity*eventContribution = intensity
 */
void create_frames(const std::vector<std::vector<Event>> &bucketed_events, std::vector<Tensor2f> &frames, const int camera_height, const int camera_width, float eventContribution){
    int i = 0;
    Tensor2f frame(camera_height, camera_width);
    Tensor2f cum_frame(camera_height, camera_width);
    for (const std::vector<Event> &event_vector : bucketed_events){
        frame.setZero();
        cum_frame.setZero();
        for (Event event : event_vector){

            frame(event.coordinates.at(0), event.coordinates.at(1)) = (float) event.polarity * eventContribution;
        }
        frames[i] = frame;
        i++;
    }
}

/**
 * Compares two float 3-Tensors on approximate equality. Threshold can be set with precision parameter.
 * @param t1 First float 3-Tensor
 * @param t2 Second float 3-Tensor
 * @param precision comparison precision
 * @return
 */
bool isApprox(Tensor3f &t1, Tensor3f &t2, const float precision = 1e-8){
    Map<VectorXf> mt1(t1.data(), t1.size());
    Map<VectorXf> mt2(t2.data(), t2.size());
    return mt1.isApprox(mt2, precision);
}

/**
 * Compares two float 2-Tensors on approximate equality. Threshold can be set with precision parameter.
 * @param t1 First float 3-Tensor
 * @param t2 Second float 3-Tensor
 * @param precision comparison precision
 * @return
 */
bool isApprox(Tensor2f &t1, Tensor2f &t2, const float precision = 1e-8){
    Map<VectorXf> mt1(t1.data(), t1.size());
    Map<VectorXf> mt2(t2.data(), t2.size());
    return mt1.isApprox(mt2, precision);
}

/**
 * Calculates the euclidean norm on entries of a 3-Tensor. The 3-Tensor is considered as a collection of vectors spread
 * over a 2D array. Norms are calculated along the vector dimension, which is the last tensor dimension. Results in a
 * 2-Tensor of norm values
 * @param T 3-Tensor of shape NxMxD, where D is the length of the vectors, over which the norms are calculated
 * @param norm 2-Tensor of norm values of shape NxM
 */
void norm_tensor_along_dim3(const Tensor3f &T, Tensor2f &norm){
    array<int,1> dims({2});
    norm = T.square().sum(dims).sqrt();
}

/**
 * Calculates intermediate values of the calibration matrix for the Interacting Maps algorithm.
 * @param x x-Position of the pixel
 * @param y y-Position of the pixel
 * @param N_x width of the image in pixels
 * @param N_y height of the image in pixels
 * @param height height of the image in realworld meters
 * @param width width of the image in realworld meters
 * @param rs
 * @return 3-Vector of pixel coordinates mapped onto sphere
 */
autodiff::Vector3real C_star(autodiff::real x, autodiff::real y, int N_x, int N_y, float height, float width, float rs) {
    autodiff::Vector3real result;
    result << height * (1 - (2 * y) / (N_y - 1)),
              width * (-1 + (2 * x) / (N_x - 1)),
              rs;
    return result;
}

/**
 * Maps pixels of image onto a 3d sphere with the camera at the center
 * @param x x-Position of the pixel
 * @param y y-Position of the pixel
 * @param N_x width of the image in pixels
 * @param N_y height of the image in pixels
 * @param height height of the image in real world meters
 * @param width width of the image in real world meters
 * @param rs
 * @return Normed 3-Vector of pixel coordinates mapped onto sphere
 */
autodiff::Vector3real C(autodiff::real x, autodiff::real y, int N_x, int N_y, float height, float width, float rs) {
    autodiff::Vector3real c_star = C_star(x, y, N_x, N_y, height, width, rs);
    autodiff::real norm = sqrt(c_star.squaredNorm());
    return c_star / norm;
}

/**
 * Calculates the camera calibration matrix which connects 2D image coordinates (of pixels) to 3D world coordinates
 * @param N_x width of the image in pixels
 * @param N_y height of the image in pixels
 * @param view_angle_x horizontal viewing angle captured by the camera
 * @param view_angle_y vertical viewing angle captured by the camera
 * @param rs
 * @param CCM resulting camera calibration matrix
 * @param C_x derivative of the CMM in x direction
 * @param C_y derivative of the CMM in x direction
 */
void find_C(int N_x, int N_y, float view_angle_x, float view_angle_y, float rs, Tensor3f &CCM, Tensor3f &C_x, Tensor3f &C_y) {
    float height = tan(view_angle_y / 2);
    float width = tan(view_angle_x / 2);
    MatrixXf XX(N_y, N_x);
    MatrixXf YY(N_y, N_x);
    for (int i = 0; i < N_y; ++i) {
        for (int j = 0; j < N_x; ++j) {
            XX(i, j) = float(j);
            YY(i, j) = float(i);
        }
    }
    for (int i = 0; i < N_y; ++i) {
        for (int j = 0; j < N_x; ++j) {
            autodiff::real x = XX(i, j);
            autodiff::real y = YY(i, j);

            // Compute the function value
            autodiff::Vector3real c_val = C(x, y, N_x, N_y, height, width, rs);
            CCM(i,j,0) = static_cast<float>(c_val(0)); // y
            CCM(i,j,1) = static_cast<float>(c_val(1)); // x
            CCM(i,j,2) = static_cast<float>(c_val(2)); // z
            // Compute the jacobians
            autodiff::VectorXreal F;

            // NEEDS TO STAY D O U B L E
            VectorXd dCdx;
            autodiff::jacobian(C, wrt(x), at(x,y,N_x, N_y, height, width, rs), F, dCdx);
            VectorXd dCdy;
            autodiff::jacobian(C, wrt(y), at(x,y,N_x, N_y, height, width, rs), F, dCdy);

            // C_x = dCdx
            C_x(i,j,0) = (float) dCdx(0); // y
            C_x(i,j,1) = (float) dCdx(1); // x
            C_x(i,j,2) = (float) dCdx(2); // z

            // C_y = -dCdy
            C_y(i,j,0) = (float) -dCdy(0); // y
            C_y(i,j,1) = (float) -dCdy(1); // x
            C_y(i,j,2) = (float) -dCdy(2); // z
        }
    }
}

/**
 * Calculates the cross product for two 3-Tensors, where each Tensor describes a collection of Vectors distributed over
 * a 2D array. Vector dimension is the last Tensor dimension (depth).
 * @param A First input Tensor of shape NxMxD
 * @param B Second input vector of shape NxMxD
 * @param C Output Tensor of shape NxMxD
 */
void crossProduct3x3(const Tensor3f &A, const Tensor3f &B, Tensor3f &C) {
    const auto& dims = A.dimensions();
    long rows = dims[0]; // height
    long cols = dims[1]; // width
    for (long i = 0; i < rows; ++i){
        for (long j = 0; j < cols; ++j){
            C(i, j, 0) = A(i, j, 2) * B(i, j, 1) - A(i, j, 1) * B(i, j, 2);  // y
            C(i, j, 1) = A(i, j, 0) * B(i, j, 2) - A(i, j, 2) * B(i, j, 0);  // x
            C(i, j, 2) = A(i, j, 1) * B(i, j, 0) - A(i, j, 0) * B(i, j, 1);  // z
        }
    }
}

/**
 * Calculates the cross product for two 3-Tensors, where each Tensor describes a collection of Vectors distributed over
 * a 2D array. Vector dimension is the last Tensor dimension (depth). Only calculates value of cross product at positions
 * [y,x,:]
 * @param A First input Tensor of shape NxMxD
 * @param B Second input vector of shape NxMxD
 * @param C Output Tensor of shape NxMxD
 */
void crossProduct3x3(const Tensor3f &A, const Vector3f &B, Vector3f &C, int y, int x) {
    C(0) = A(y, x, 2) * B(1) - A(y, x, 1) * B(2);  // y
    C(1) = A(y, x, 0) * B(2) - A(y, x, 2) * B(0);  // x
    C(2) = A(y, x, 1) * B(0) - A(y, x, 0) * B(1);  // z
}

/**
 * Calculates cross product between a 3-Tensor and a vector, where the tensor describes a collection of Vectors
 * distributed over a 2D array. Vector dimension is the last Tensor dimension (depth).
 * @param A vector as a 1-Tensor
 * @param B collection of vectors as a 3-Tensor
 * @param C Resulting collection of cross product vectors as a 3-Tensor
 */
void crossProduct1x3(const Tensor<float,1> &A, const Tensor3f &B, Tensor3f &C){
    const auto& dimensions = B.dimensions();
    for (long i = 0; i < dimensions[0]; ++i){
        for (long j = 0; j < dimensions[1]; ++j){
            C(i, j, 0) = A(2) * B(i, j, 1) - A(1) * B(i, j, 2);  // y
            C(i, j, 1) = A(0) * B(i, j, 2) - A(2) * B(i, j, 0);  // x
            C(i, j, 2) = A(1) * B(i, j, 0) - A(0) * B(i, j, 1);  // z
        }
    }

}

/**
 * Calculates the distances between two collection of vectors. For a pair of vectors V,W from each collection
 * the formula |V+W|/|W| is used. Collection is a 3-Tensor, where the tensor describes a collection of Vectors
 * distributed over a 2D array. Vector dimension is the last Tensor dimension (depth).
 * @param vec1 First collection of vectors in form of a 3-Tensor.
 * @param vec2 Second collection of vectors in form of a 3-Tensor.
 * @param distance Collection of distances as 2-Tensor
 */
void vector_distance(const Tensor3f &vec1, const Tensor3f &vec2, Tensor2f &distance){
    PROFILE_FUNCTION();
    const auto& dimensions = vec1.dimensions();
    Tensor3f cross_product(dimensions);
    Tensor2f norm(dimensions[0], dimensions[1]);
    Tensor2f norm2(dimensions[0], dimensions[1]);
    crossProduct3x3(vec1, vec2, cross_product);
    norm.setZero();
    norm2.setZero();
    norm_tensor_along_dim3(cross_product, norm);
    norm_tensor_along_dim3(vec2, norm2);
    distance = norm/norm2;
}

/**
 * Simple sign function for floats. Returns 0.0 for 0.0f
 * @param x floating point number
 * @return Sign of the number (-1.0,1.0)
 */
float sign_func(float x){
    if (x > 0)
        return +1.0;
    else if (x == 0)
        return 0.0;
    else
        return -1.0;
}

/**
 * Calculates the dot product for two 3-Tensors, where each Tensor describes a collection of Vectors distributed over
 * a 2D array. Vector dimension is the last Tensor dimension (depth). Returns a 2-Tensor of products.
 * @param A First input Tensor of shape NxMxD
 * @param B Second input vector of shape NxMxD
 * @param D Output Tensor of shape NxM
 */
void computeDotProductWithLoops(const Tensor3f &A, const Tensor3f &B, Tensor2f &D) {
    PROFILE_FUNCTION();
    const int height = (int) A.dimension(0);
    const int width = (int) A.dimension(1);
    const int depth = (int) A.dimension(2);

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

/**
 * Maps a 3 dimensional world vectors to 2 dimensional image vectors. Expects vectors as a 3-Tensor. Vector dimension is
 * last dimension (depth)
 * @param In 3-Tensor of vectors distributed over an array. Shape: NxMx3
 * @param C_x x-Derivative (width) of camera calibration matrix
 * @param C_y y-Derivative (height) of camera calibration matrix
 * @param Out Resulting collection of vectors as 3-Tensor of shape NxMx2
 */
void m32(const Tensor3f &In, const Tensor3f &C_x, const Tensor3f &C_y, Tensor3f &Out){
    const auto& dimensions = In.dimensions();
    Tensor3f C1(dimensions);
    Tensor3f C2(dimensions);
    Tensor2f dot(dimensions[0], dimensions[1]);
    Tensor2f sign(dimensions[0], dimensions[1]);
    Tensor2f distance1(dimensions[0], dimensions[1]);
    Tensor2f distance2(dimensions[0], dimensions[1]);

    crossProduct3x3(C_x,C_y,C1);
    crossProduct3x3(C_y,C1,C2);
    computeDotProductWithLoops(In,C2,dot);
    sign = dot.unaryExpr(std::ptr_fun(sign_func));
    vector_distance(In, C_y, distance1);
    vector_distance(C_x, C_y, distance2);
    Out.chip(1,2) = sign * distance1/distance2;

    crossProduct3x3(C_y,C_x,C1);
    crossProduct3x3(C_x,C1,C2);
    computeDotProductWithLoops(In,C2,dot);
    sign = dot.unaryExpr(std::ptr_fun(sign_func));
    vector_distance(In, C_x, distance1);
    vector_distance(C_y, C_x, distance2);
    Out.chip(0,2) = sign * distance1/distance2;
}

/**
 * Maps 2 dimensional image vectors to 3 dimensional world vectors. Expects vectors as a 3-Tensor. Vector dimension is
 * last dimension (depth). Only calculates mapping for vector at position [y,x,:]
 * @param In 3-Tensor of vectors distributed over an array. Shape: NxMx2
 * @param C_x x-Derivative (width) of camera calibration matrix
 * @param C_y y-Derivative (height) of camera calibration matrix
 * @param Out Resulting 3-Vector
 */
void m23(const Tensor3f &In, const Tensor3f &Cx, const Tensor3f &Cy, Vector3f &Out, int y, int x) {
    Out(0) = In(y, x, 1) * Cx(y, x, 0) + In(y, x, 0) * Cy(y, x, 0);
    Out(1) = In(y, x, 1) * Cx(y, x, 1) + In(y, x, 0) * Cy(y, x, 1);
    Out(2) = In(y, x, 1) * Cx(y, x, 2) + In(y, x, 0) * Cy(y, x, 2);
}

/**
 * Setup for the update function which updates R from F (and C). It pre calculates a matrix A of a linear system which
 * needs to be solved for each update. Matrix A does not change between updates which is why it gets precaluculated. Also prepares RHS vector B
 * @param CCM Camera calibration matrix connection world and image coordinates
 * @param A Lefthand side matrix
 * @param B Righthand side vector
 * @param Identity_minus_outerProducts Terms needed for final calculation of RHS at each update of R
 * @param points
 */
void setup_R_update(const Tensor3f &CCM, Matrix3f &A, Vector3f &B, std::vector<std::vector<Matrix3f>> &Identity_minus_outerProducts, std::vector<std::vector<Vector3f>> &points){
    PROFILE_FUNCTION();
    const auto &dimensions = CCM.dimensions();
    int rows = (int) dimensions[0];
    int cols = (int) dimensions[1];
    Matrix3f Identity = Matrix3f::Identity();
    Vector3f d;
    B.setZero();

    for (size_t i = 0; i < rows; ++i){
        for (size_t j = 0; j < cols; ++j){
            d(0) = CCM((int) i, (int) j, 0);
            d(1) = CCM((int) i, (int) j, 1);
            d(2) = CCM((int) i, (int) j, 2);
            Identity_minus_outerProducts[i][j] = Identity - d * d.transpose();
            A += Identity_minus_outerProducts[i][j];
            points[i][j].setZero();
        }
    }
}

/**
 * Checks how close the dot product of F and G are to -V, using the infinity norm.
 * @param V Temporal gradient (often approximated by agglomerating Events to a frame), 2-Tensor
 * @param F Optical flow 3-Tensor
 * @param G Spatial gradient 3-Tensor
 * @param precision Currently unused
 * @return
 */
float VFG_check(const Tensor2f &V, const Tensor3f &F, const Tensor3f &G){
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

/**
 * Update the optical flow F at a pixel based on the spatial gradient G and the temporal gradient V (for example given by Events
 * from an Event camera)
 * @param F Optical Flow, 3-Tensor, shape NxMx2, collection of 2-Vectors
 * @param V Temporal Gradients, float
 * @param G Spatial gradients, 3-Tensor, shape NxMx2, collection of 2-Vectors
 * @param y vertical coordinate of pixel
 * @param x horizontal coordinate of pixel
 * @param lr learning rate, currently fixed to 1.0
 * @param weight_FG weight of the update for the convex combination with the old F value. 0.0 = no update
 * @param eps round F values with a magnitude below eps to 0.0
 * @param gamma cap F values with a greater magnitude than gamma to gamma.
 */
void update_FG(Tensor3f &F, const float V, const Tensor3f &G, int y, int x, const float lr, const float weight_FG, float eps=1e-8, float gamma=255.0){
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
            F(y, x, 0) = 0.0;
        }
        if (std::abs(F(y, x, 1)) < eps){
            F(y, x, 1) = 0.0;
        }
    }
}

/**
 * Update the spatial gradient G at a pixel based on the optical flow F and the temporal gradient V (for example given by Events
 * from an Event camera)
 * @param G Optical Flow, 3-Tensor, shape NxMx2, collection of 2-Vectors
 * @param V Temporal Gradients, Events, 2-Tensor
 * @param F Spatial gradients, 3-Tensor, shape NxMx2, collection of 2-Vectors
 * @param y vertical coordinate of pixel
 * @param x horizontal coordinate of pixel
 * @param lr learning rate, currently fixed to 1.0
 * @param weight_FG weight of the update for the convex combination with the old F value. 0.0 = no update
 * @param eps round F values below eps to 0.0
 * @param gamma cap F values with a greater magnitude than gamma to gamma.
 */
void update_GF(Tensor3f &G, float V, const Tensor3f &F, int y, int x, const float lr, const float weight_GF, float eps=1e-8, float gamma=255.0){
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
            G(y, x, 0) = 0.0;
        }
        if (std::abs(G(y, x, 1)) < eps){
            G(y, x, 1) = 0.0;
        }
    }
}

/**
 * Update the spatial gradient G based on the Image intensity I with central difference gradient calculation. Only done
 * at pixel (y,x) (height, width)
 * @param G spatial gradient, shape NxMx2
 * @param I_gradient gradient of I, shape NxMx2
 * @param y vertical position
 * @param x horizontal position
 * @param weight_GI weight for convex combination with old G value, 0 = no update
 * @param eps G values lower get rounded to 0.0
 * @param gamma G values with magnitude larger get rounded to gamma (or -gamma)
 */
void update_GI(Tensor3f &G, const Tensor3f &I_gradient, int y, int x, float weight_GI, float eps, float gamma){
    PROFILE_FUNCTION();
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
}

void contribute(Tensor2f &I, float V, int y, int x, float minPotential, float maxPotential){
    I(y, x) = std::min(std::max(I(y, x) + V, minPotential), maxPotential);
}

void globalDecay(Tensor2f &I, Tensor2f &decayTimeSurface, Tensor2f &nP, Tensor2f &t, Tensor2f &dP) {
    I = (I - nP) * (-(t - decayTimeSurface) / dP).exp() + nP;
    decayTimeSurface = t;
}

//void decay(Tensor2f &I, Tensor2f &decayTimeSurface, const int y, const int x, const float time, const float neutralPotential, const float decayParam){
//    float newIntensity = (I(y, x) - neutralPotential) * expf(-(time - decayTimeSurface(y, x)) / decayParam) + neutralPotential;
//    I(y, x) = newIntensity;
//    decayTimeSurface(y, x) = time;
//}

//void update_IV(Tensor2f &I, const float V, Tensor2f &decayTimeSurface, const int y, const int x, const float time, const float minPotential, const float maxPotential, const float neutralPotential, const float decayParam){
//    PROFILE_FUNCTION();
//    contribute(I, V, y, x, minPotential, maxPotential);
//}

void update_IV(Tensor2f &I, const float V, const int y, const int x, const float minPotential, const float maxPotential){
    PROFILE_FUNCTION();
    contribute(I, V, y, x, minPotential, maxPotential);
}

void updateGIDiffGradient(Tensor3f &G, Tensor3f &I_gradient, Tensor3f &GIDiff, Tensor3f &GIDiffGradient, int y, int x){
    PROFILE_FUNCTION();
    GIDiff(y, x, 0) = G(y, x, 0) - I_gradient(y, x, 0);
    GIDiff(y, x, 1) = G(y, x, 1) - I_gradient(y, x, 1);
    computeGradient(GIDiff, GIDiffGradient, y, x);
}

/**
 * Updates image intensity I based on spatial gradient G
 * @param I
 * @param GIDiffGradient
 * @param y
 * @param x
 * @param weight_IG
 */
void update_IG(Tensor2f &I, const Tensor3f &GIDiffGradient, int y, int x, float weight_IG){
    PROFILE_FUNCTION();
    I(y, x) = I(y, x) + weight_IG * (- GIDiffGradient(y, x, 0) - GIDiffGradient(y, x, 1));
}

/**
 * updates F based on R and C
 * @param F
 * @param CCM
 * @param Cx
 * @param Cy
 * @param R
 * @param weight_FR
 * @param eps
 * @param gamma
 */
void update_FR(Tensor3f &F, const Tensor3f &CCM, const Tensor3f &Cx, const Tensor3f &Cy, const Tensor<float,1> &R, const float weight_FR, float eps=1e-8, float gamma=255.0){
    PROFILE_FUNCTION();
    Tensor3f cross(CCM.dimensions());
    const auto& dimensions = F.dimensions();
    Tensor3f update(F.dimensions());
    {
        PROFILE_SCOPE("FR CROSS PRODUCT");
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

//void update_RF(Tensor<float,1> &R, const Tensor3f &F, const Tensor3f &C, const Tensor3f &Cx, const Tensor3f &Cy, const Matrix3f &A, Vector3f &B, const std::vector<std::vector<Matrix3f>> &Identity_minus_outerProducts, std::vector<std::vector<Vector3f>> &old_points, const float weight_RF, const std::vector<Event> &frameEvents) {
//    PROFILE_FUNCTION();
//    const auto &dimensions = F.dimensions();
//    Vector3f transformed_F(3);
//    Vector3f point(3);
//    Vector3f solution(3);
//    {
//        PROFILE_SCOPE("RF Pre");
//        for (auto event : frameEvents){
//            // Transform F from 2D image space to 3D world space with C
//            m23(F, Cx, Cy, transformed_F, event.coordinates[0], event.coordinates[1]);
//            // calculate cross product between world space F and calibration matrix.
//            // this gives us the point on which the line stands
//            crossProduct3x3(C, transformed_F, point, event.coordinates[0], event.coordinates[1]);
//            // right hand side B consists of a sum of a points
//            // subtract the contribution of the old_point at y,x and add the contribution of the new point
//            B = B - Identity_minus_outerProducts[event.coordinates[0]][event.coordinates[1]]*old_points[event.coordinates[0]][event.coordinates[1]] + Identity_minus_outerProducts[event.coordinates[0]][event.coordinates[1]]*point;
//            // new point is now old
//            old_points[event.coordinates[0]][event.coordinates[1]] = point;
//        }
//    }
//    // solve for the new rotation vector
//    solution = A.partialPivLu().solve(B);
//    R(0) = (1 - weight_RF) * R(0) + weight_RF * solution(0);
//    R(1) = (1 - weight_RF) * R(1) + weight_RF * solution(1);
//    R(2) = (1 - weight_RF) * R(2) + weight_RF * solution(2);
//}


/**
 * Updates rotational velocity vector R based on F and C
 * @param R
 * @param F
 * @param C
 * @param Cx
 * @param Cy
 * @param A
 * @param B
 * @param Identity_minus_outerProducts
 * @param old_points
 * @param weight_RF
 * @param y
 * @param x
 */
void update_RF(Tensor<float,1> &R, const Tensor3f &F, const Tensor3f &C, const Tensor3f &Cx, const Tensor3f &Cy, const Matrix3f &A, Vector3f &B, const std::vector<std::vector<Matrix3f>> &Identity_minus_outerProducts, std::vector<std::vector<Vector3f>> &old_points, float weight_RF, int y, int x) {
    PROFILE_FUNCTION();
    Vector3f transformed_F(3);
    Vector3f point(3);
    Vector3f solution(3);
    {
        PROFILE_SCOPE("RF Pre");
        // Transform F from 2D image space to 3D world space with C
        m23(F, Cx, Cy, transformed_F, y, x);
        // calculate cross product between world space F and calibration matrix.
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

void event_step(const float V, Tensor2f &MI, Tensor3f &delta_I, Tensor3f &GIDiff, Tensor3f &GIDiffGradient, Tensor3f &F, Tensor3f &G, Tensor<float,1> &R, const Tensor3f &CCM, const Tensor3f &dCdx, const Tensor3f &dCdy, const Matrix3f &A, Vector3f &B, const std::vector<std::vector<Matrix3f>> &Identity_minus_outerProducts, std::vector<std::vector<Vector3f>> &old_points, std::unordered_map<std::string,float> &parameters, std::vector<int> &permutation, int y, int x){
    PROFILE_FUNCTION();
    array<Index, 2> dimensions = MI.dimensions();
    update_IV(MI, V, y, x, parameters["minPotential"], parameters["maxPotential"]);
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

    update_IG(MI, GIDiffGradient, y, x, parameters["weight_IG"]);
    computeGradient(MI, delta_I, y, x);

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

int main(int argc, char* argv[]) {

    // Define the command-line options
    po::options_description desc("Allowed options");
    desc.add_options()
            ("help,h", "Produce help message")
            ("startTime,f", po::value<float>()->default_value(20), "Where to start with event consideration")
            ("endTime,f", po::value<float>()->default_value(30), "Where to end with event consideration")
            ("timeStep,f", po::value<float>()->default_value(0.005), "Size of the event frames")
            ("resourceDirectory,s", po::value<std::string>()->default_value("poster_rotation"), "Which dataset to use, searches in res directory")
            ("resultsDirectory,s", po::value<std::string>()->default_value("poster_rotation"), "Where to store the results, located in output directory")
            ("addTime,b", po::value<bool>()->default_value(false), "Add time to output folder?")
            ("startIndex,i", po::value<int>()->default_value(4002), "With what index to start for the images");

    // Parse command-line arguments
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    // Display help message if requested
    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 0;
    }

    // Retrieve values (using defaults if not provided)
    float startTime = vm["startTime"].as<float>();
    float endTime = vm["endTime"].as<float>();
    float timeStep = vm["timeStep"].as<float>();
    bool addTime = vm["addTime"].as<bool>();
    int startIndex = vm["startIndex"].as<int>();
    std::string resourceDirectory = vm["resourceDirectory"].as<std::string>();
    std::string resultsDirectory = vm["resultsDirectory"].as<std::string>();

    std::cout << "Parsed startTime: " << startTime << "\n";
    std::cout << "Parsed endTime: " << endTime << "\n";
    std::cout << "Parsed timeStep: " << timeStep << "\n";
    std::cout << "Parsed resourceDirectory: " << resourceDirectory << "\n";
    std::cout << "Parsed resultsDirectory: " << resultsDirectory << "\n";

    auto start = std::chrono::high_resolution_clock::now();

    //##################################################################################################################
    // Create results_folder

    std::string folder_name;
    if (addTime) {
        auto clock_time = std::chrono::system_clock::now();
        std::time_t time = std::chrono::system_clock::to_time_t(clock_time);
        folder_name = resultsDirectory + " " + std::ctime(&time);
    }
    else{
        folder_name = resultsDirectory;
    }
    fs::path folder_path = create_folder_and_update_gitignore(folder_name);
    std::cout << "Created Folder " << folder_name << std::endl;

    std::string profiler_name = "Profiler.json";
    fs::path profiler_path = folder_path / profiler_name;
    Instrumentor::Get().BeginSession("Interacting Maps", profiler_path);
    std::cout << "Setup Profiler" << std::endl;

    std::string calibrationPath = "../res/" + resourceDirectory + "/calib.txt";
    std::string eventPath = "../res/" + resourceDirectory + "/events.txt";
    std::string settingsPath = "../res/" + resourceDirectory + "/settings.txt";

    std::cout << "Parsed calibrationPath: " << calibrationPath << "\n";
    std::cout << "Parsed eventPath: " << eventPath << "\n";
    std::cout << "Parsed settingsPath: " << settingsPath << "\n";

    std::unordered_map<std::string,float> parameters;
    parameters["start_time"] = startTime;                                   // in seconds
    parameters["end_time"] = endTime;                                       // in seconds
    parameters["time_step"] = timeStep;                                     // in seconds
    parameters["weight_FG"] = 0.2;                                          // [0-1]
    parameters["weight_FR"] = 0.8;                                          // [0-1]
    parameters["weight_GF"] = 0.2;                                          // [0-1]
    parameters["weight_GI"] = 0.2;                                          // [0-1]
    parameters["weight_IG"] = 0.0;                                          // [0-1]
    parameters["weight_IV"] = 1.0;                                          // [0-1]
    parameters["weight_RF"] = 0.8;                                          // [0-1]
    parameters["lr"] = 1.0;                                                 // [0-1]
    parameters["eventContribution"] = 25;                                   // mainly important for the visibility of the intensity image
    parameters["eps"] = 0.00001;                                            // lowest value allowed for F, G,...
    parameters["gamma"] = 255;                                              // highest value allowed for F, G,...
    parameters["decayParam"] = 1e-2;                                        // exp: 1e6; linear: 0.000001
    parameters["minPotential"] = 0.0;                                       // minimum Value for Image
    parameters["maxPotential"] = 255;                                       // maximal Value for Image
    parameters["neutralPotential"] = 128;                                   // base value where image decays back to
    parameters["fps"] = 1.0f/parameters["time_step"];                       // how often shown images are update
    parameters["FR_updates_per_second"] = 1.0f/parameters["time_step"];     // how often the FR update is performed; It is not done after every event
    parameters["updateIterationsFR"] = 10;                                  // more iterations -> F captures general movement of scene/camera better but significantly more computation time

    // Plotting
    float cutoff = 0.1;

    // Read resolution from file
    std::vector<float> settings;
    read_single_line_txt(settingsPath, settings);

    // Set sizes according to read settings
    int height = int(settings[0])+1; // in pixels
    int rows = int(settings[0])+1; // in pixels
    int width = int(settings[1])+1; // in pixels
    int cols = int(settings[1])+1; // in pixels

    // iterations are done after event calculations for a frame are done
    std::vector<int> permutation {0,2,3}; // Which update steps to take; 1 is not needed
    std::random_device myRandomDevice;
    unsigned seed = myRandomDevice();
    std::default_random_engine rng(seed);

    //##################################################################################################################
    // Optic flow F, temporal derivative V, spatial derivative G, intensity I, rotation vector R
    Tensor2f V_Vis(height, width);
    V_Vis.setZero();
    float V;

    // Initialize optical flow
    Tensor3f F(height, width, 2);
    Tensor3f F1(height, width, 2);
    Tensor3f F2(height, width, 2);
    Tensor3f F3(height, width, 2);
    F1.setConstant(0.0);
    F2.setConstant(2.0);
    F3.setConstant(0.01);
    F.setRandom();
    F = F*F2 - F1;
    F = F * F3;

    // Initialize spatial gradient G
    Tensor3f G(height, width, 2);
    G.setZero();
    Tensor3f delta_I(height, width,2);
    delta_I.setZero();

    // Initialize intensity image I
    Tensor2f I(height, width);
    I.setConstant(128.0);

    // For the image we want to decay the image intensity. We save for each pixel how old the
    // information is.
    Tensor2f decayTimeSurface(height, width);
    decayTimeSurface.setConstant(parameters["start_time"]);

    // For the "I from G" update rule we need helper values.
    Tensor3f GIDiff(height, width,2);
    GIDiff.setRandom();
    Tensor3f GIDiffGradient(height, width,2);
    GIDiffGradient.setRandom();

    // Initialize rotational velocity to a random vector with values between -1 and 1
    Tensor<float,1> R(3);
    Tensor<float,1> R2(3);
    Tensor<float,1> R3(3);
    R.setRandom(); // between 0 and 1
    R2.setConstant(2);
    R3.setConstant(1);
    R = R*R2 - R3; // between -1 and 1

    //##################################################################################################################
    // Read calibration file
    std::vector<float> raw_calibration_data;
    read_single_line_txt(calibrationPath, raw_calibration_data);
    Calibration_Data calibration_data = get_calibration_data(raw_calibration_data, height, width);
    std::cout << "Readout calibration file at " << calibrationPath << std::endl;

    //##################################################################################################################
    // Read events file
    std::vector<Event> event_data;
    read_events(eventPath, event_data, parameters["start_time"], parameters["end_time"], INT32_MAX);
    std::cout << "Readout events at " << eventPath << std::endl;

    //##################################################################################################################
    // Bin events
    std::vector<std::vector<Event>> binned_events;
    binned_events = bin_events(event_data, parameters["time_step"]);
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

    // For keeping track of the current Event
    int y;
    int x;
    std::vector<Event> frameEvents;

    // Tensors for Image decay
    Tensor2f nP(I.dimensions());    // neutralPotential
    Tensor2f t(I.dimensions());     // time
    Tensor2f dP(I.dimensions());    // decayParameter

    auto start_realtime = std::chrono::high_resolution_clock::now();

    int vis_counter = 0;
    int FR_update_counter = 0;

    nP.setConstant(parameters["neutralPotential"]);
    dP.setConstant(parameters["decayParam"]);

    for (Event event : event_data) {
        // Shuffle the order of operations for the interacting maps operations
        std::shuffle(std::begin(permutation), std::end(permutation), rng);

        // Get the current event
        y = event.coordinates[0];
        x = event.coordinates[1];
        V = (float) event.polarity;

        decayTimeSurface(y,x) = event.time;

        // For Showing the events as an image increase the intensity
        V_Vis(y, x) = (float) event.polarity * parameters["eventContribution"];

        frameEvents.push_back(event);

        // Perform an update step for the current event for I G R and F
        for (int i = 0; i < 2; ++i) {
            event_step(V, MI, delta_I, GIDiff, GIDiffGradient, F, G, R, CCM, dCdx, dCdy, A, B,
                       Identity_minus_outerProducts, old_points, parameters, permutation, y, x);
        }

        if (parameters["start_time"] + (float) FR_update_counter * (float) 1 / parameters["FR_updates_per_second"] <
            event.time) {
            t.setConstant(event.time);
            for (int i = 0; i < (int) parameters["updateIterationsFR"]; ++i) {
                update_FR(F, CCM, dCdx, dCdy, R, parameters["weight_FR"], parameters["eps"], parameters["gamma"]);
            }
            globalDecay(MI, decayTimeSurface, nP, t, dP);
        }

        // Starting from the start time we count up if the current time (event.time)
        // Reaches the time of the next "frame" we want to save to disk
        if (parameters["start_time"] + (float) vis_counter * (float) 1 / parameters["fps"] < event.time) {
            vis_counter++;
            std::cout << "Frame " << startIndex+vis_counter << "/"
                      << int((parameters["end_time"] - parameters["start_time"]) * parameters["fps"]) << std::endl;

            {
                PROFILE_SCOPE("BETWEEN FRAMES");
                //writeToFile(CCM, folder_path / ("C" + std::to_string(counter) + ".txt"));
                //writeToFile(V_Vis, folder_path / ("V" + std::to_string(counter) + ".txt"));
                //writeToFile(MI, folder_path / ("MI" + std::to_string(counter)  + ".txt"));
                //writeToFile(I, folder_path / ("I" + std::to_string(counter)  + ".txt"));
                //writeToFile(delta_I, folder_path / ("I_gradient" + std::to_string(counter)  + ".txt"));
                //writeToFile(F, folder_path / ("F" + std::to_string(counter)  + ".txt"));
                //writeToFile(G, folder_path / ("G" + std::to_string(counter)  + ".txt"));

            }
#ifdef IMAGES
            //float loss = VFG_check(V_Vis, F, G);
            //std::cout << "VFG Check: " << loss << std::endl;
            std::string image_name = "VIGF_" + std::to_string(int(startIndex + vis_counter)) + ".png";
            fs::path image_path = folder_path / image_name;
            //create_VIGF(Tensor2Matrix(V_Vis), Tensor2Matrix(MI), G, F, image_path, true, cutoff);
            //image_name = "VvsFG" + std::to_string(int(counter)) + ".png";
            //image_path = folder_path / image_name;
            //plot_VvsFG(Tensor2Matrix(V_Vis), F, G, image_path, true);
            cv::Mat VIGF;
            create_VIGF(Tensor2Matrix(V_Vis), Tensor2Matrix(MI), G, F, image_path, true, cutoff);
            // cv::imshow("VIGF", VIGF);
            // // Press 'q' to exit
            // if (cv::waitKey(1) == 'q') {
            //     break;
            // }
            V_Vis.setZero();
            F.setRandom();
            F = F * F2 - F1;
            F = F * F3;
            G.setZero();
#endif
        }

        if (parameters["start_time"] + (float) FR_update_counter * (float) 1 / parameters["FR_updates_per_second"] <event.time) {
            FR_update_counter++;

        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> elapsed = end - start;
    std::chrono::duration<float> elapsed_realtime = end - start_realtime;
    std::stringstream ss;
    std::stringstream ssrt;
    ss << "Time elapsed: " << elapsed.count() << " seconds" << std::endl;
    ssrt << "Time elapsed: " << elapsed_realtime.count() << " seconds" << std::endl;
    writeToFile(ss.str(), folder_path / "time_complete.txt");
    writeToFile(ssrt.str(), folder_path / "time_realtime.txt");
    std::cout << "Algorithm took: " << elapsed_realtime.count() << "seconds/ Real elapsed time: " << parameters["end_time"] - parameters["start_time"] << std::endl;


    std::string outputFile = "output.mp4";

#ifdef IMAGES
    VideoCreator::createMP4Video(folder_path, folder_path / outputFile, int(parameters["fps"]));
#endif

    Instrumentor::Get().EndSession();
}