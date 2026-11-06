//
// Created by root on 7/25/25.
//

#ifndef UPDATE_H
#define UPDATE_H

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

#include <datatypes.h>
#include <conversions.h>
#include "contrast.h"
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <Eigen/src/LU/PartialPivLU.h>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  INTERACTING MAPS HELPER FUNCTIONS  /////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Compares two float 3-Tensors on approximate equality. Threshold can be set with precision parameter.
 * @param t1 First float 3-Tensor
 * @param t2 Second float 3-Tensor
 * @param precision comparison precision
 * @return
 */
bool isApprox(Tensor3f &t1, Tensor3f &t2, float precision);

/**
 * Compares two float 2-Tensors on approximate equality. Threshold can be set with precision parameter.
 * @param t1 First float 3-Tensor
 * @param t2 Second float 3-Tensor
 * @param precision comparison precision
 * @return
 */
bool isApprox(Tensor2f &t1, Tensor2f &t2, float precision);

/**
 * Calculates the euclidean norm on entries of a 3-Tensor. The 3-Tensor is considered as a collection of vectors spread
 * over a 2D array. Norms are calculated along the vector dimension, which is the last tensor dimension. Results in a
 * 2-Tensor of norm values
 * @param T 3-Tensor of shape NxMxD, where D is the length of the vectors, over which the norms are calculated
 * @param norm 2-Tensor of norm values of shape NxM
 */
void norm_tensor_along_dim3(const Tensor3f &T, Tensor2f &norm);

/**
 * Unnormalised viewing direction of a pixel for a pinhole camera. Camera frame x right, y down, z forward
 * (right-handed), stored as (y, x, z).
 * @param x x-Position of the pixel (column)
 * @param y y-Position of the pixel (row)
 * @param fx focal length along x in pixels
 * @param fy focal length along y in pixels
 * @param cx principal point x in pixels
 * @param cy principal point y in pixels
 * @return 3-Vector ((y - cy) / fy, (x - cx) / fx, 1)
 */
autodiff::Vector3real C_star(autodiff::real x, autodiff::real y, float fx, float fy, float cx, float cy);

/**
 * Maps pixels of image onto the unit sphere with the camera at the center
 * @param x x-Position of the pixel (column)
 * @param y y-Position of the pixel (row)
 * @param fx focal length along x in pixels
 * @param fy focal length along y in pixels
 * @param cx principal point x in pixels
 * @param cy principal point y in pixels
 * @return Normed 3-Vector, stored as (y, x, z)
 */
autodiff::Vector3real C(autodiff::real x, autodiff::real y, float fx, float fy, float cx, float cy);

/**
 * Calculates the unit viewing direction of every pixel and its derivatives along x and y from pinhole intrinsics.
 * Lens distortion is not modelled.
 * @param N_x width of the image in pixels
 * @param N_y height of the image in pixels
 * @param fx focal length along x in pixels
 * @param fy focal length along y in pixels
 * @param cx principal point x in pixels
 * @param cy principal point y in pixels
 * @param CCM resulting viewing directions
 * @param C_x derivative of CCM in x direction
 * @param C_y derivative of CCM in y (row, downward) direction
 */
void find_C(int N_x, int N_y, float fx, float fy, float cx, float cy, Tensor3f &CCM, Tensor3f &C_x, Tensor3f &C_y);

/**
 * Calculates the cross product for two 3-Tensors, where each Tensor describes a collection of Vectors distributed over
 * a 2D array. Vector dimension is the last Tensor dimension (depth).
 * @param A First input Tensor of shape NxMxD
 * @param B Second input vector of shape NxMxD
 * @param C Output Tensor of shape NxMxD
 */
void crossProduct3x3(const Tensor3f &A, const Tensor3f &B, Tensor3f &C);

/**
 * Calculates the cross product for two 3-Tensors, where each Tensor describes a collection of Vectors distributed over
 * a 2D array. Vector dimension is the last Tensor dimension (depth). Only calculates value of cross product at positions
 * [y,x,:]
 * @param A First input Tensor of shape NxMxD
 * @param B Second input vector of shape NxMxD
 * @param C Output Tensor of shape NxMxD
 */
void crossProduct3x3(const Tensor3f &A, const Vector3f &B, Vector3f &C, int y, int x);

/**
 * Calculates cross product between a 3-Tensor and a vector, where the tensor describes a collection of Vectors
 * distributed over a 2D array. Vector dimension is the last Tensor dimension (depth).
 * @param A vector as a 1-Tensor
 * @param B collection of vectors as a 3-Tensor
 * @param C Resulting collection of cross product vectors as a 3-Tensor
 */
void crossProduct1x3(const Tensor1f &A, const Tensor3f &B, Tensor3f &C);

/**
 * Calculates the distances between two collection of vectors. For a pair of vectors V,W from each collection
 * the formula |V+W|/|W| is used. Collection is a 3-Tensor, where the tensor describes a collection of Vectors
 * distributed over a 2D array. Vector dimension is the last Tensor dimension (depth).
 * @param vec1 First collection of vectors in form of a 3-Tensor.
 * @param vec2 Second collection of vectors in form of a 3-Tensor.
 * @param distance Collection of distances as 2-Tensor
 */
void vector_distance(const Tensor3f &vec1, const Tensor3f &vec2, Tensor2f &distance);

/**
 * Simple sign function for floats. Returns 0.0 for 0.0f
 * @param x floating point number
 * @return Sign of the number (-1.0,1.0)
 */
float sign_func(float x);

/**
 * Calculates the dot product for two 3-Tensors, where each Tensor describes a collection of Vectors distributed over
 * a 2D array. Vector dimension is the last Tensor dimension (depth). Returns a 2-Tensor of products.
 * @param A First input Tensor of shape NxMxD
 * @param B Second input vector of shape NxMxD
 * @param D Output Tensor of shape NxM
 */
void computeDotProductWithLoops(const Tensor3f &A, const Tensor3f &B, Tensor2f &D);
/**
 * Maps a 3 dimensional world vectors to 2 dimensional image vectors. Expects vectors as a 3-Tensor. Vector dimension is
 * last dimension (depth)
 * @param In 3-Tensor of vectors distributed over an array. Shape: NxMx3
 * @param C_x x-Derivative (width) of camera calibration matrix
 * @param C_y y-Derivative (height) of camera calibration matrix
 * @param Out Resulting collection of vectors as 3-Tensor of shape NxMx2
 */
void m32(const Tensor3f &In, const Tensor3f &C_x, const Tensor3f &C_y, Tensor3f &Out);

/**
 * Maps 2 dimensional image vectors to 3 dimensional world vectors. Expects vectors as a 3-Tensor. Vector dimension is
 * last dimension (depth). Only calculates mapping for vector at position [y,x,:]
 * @param In 3-Tensor of vectors distributed over an array. Shape: NxMx2
 * @param C_x x-Derivative (width) of camera calibration matrix
 * @param C_y y-Derivative (height) of camera calibration matrix
 * @param Out Resulting 3-Vector
 */
void m23(const Tensor3f &In, const Tensor3f &Cx, const Tensor3f &Cy, Vector3f &Out, int y, int x);

/**
 * Calculates the gradient of 2-Tensor via central differences at a location (x,y) in x and y direction.
 * Reuses first/last value at border (effectively using forward or backward difference)
 * @param data Input of shape NxM
 * @param gradients 3-Tensor holding gradients
 * @param y up and down position of gradient of interest
 * @param x left and right position of gradient of interest
 */
void computeGradient(const Tensor2f &data, Tensor3f &gradients, int y, int x);

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
void computeGradient(const Tensor3f &data, Tensor3f &gradients, int y, int x);

/**
 * Checks how close the dot product of F and G are to -V, using the infinity norm.
 * @param V Temporal gradient (often approximated by agglomerating Events to a frame), 2-Tensor
 * @param F Optical flow 3-Tensor
 * @param G Spatial gradient 3-Tensor
 * @param precision Currently unused
 * @return
 */
float VFG_check(const Tensor2f &V, const Tensor3f &F, const Tensor3f &G);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  INTERACTING MAPS UPDATE FUNCTIONS  /////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Setup for the update function which updates R from F (and C). It pre calculates a matrix A of a linear system which
 * needs to be solved for each update. Matrix A does not change between updates which is why it gets precaluculated. Also prepares RHS vector B
 * @param CCM Camera calibration matrix connection world and image coordinates
 * @param A Lefthand side matrix
 * @param B Righthand side vector
 * @param Identity_minus_outerProducts Terms needed for final calculation of RHS at each update of R
 * @param points
 */
void setup_R_update(const Tensor3f &CCM, Matrix3f &A, Vector3f &B, std::vector<std::vector<Matrix3f>> &Identity_minus_outerProducts, std::vector<std::vector<Vector3f>> &points);

/**
 * Recomputes the right-hand side B of the F -> R least squares and the per-pixel points C x m23(F) from the current
 * flow at every pixel. update_RF only refreshes the event pixel, so this must run after anything that changes F
 * globally (initialisation, update_FR).
 * @param F optical flow
 * @param C viewing directions
 * @param Cx derivative of C along x
 * @param Cy derivative of C along y (down)
 * @param B Righthand side vector, overwritten
 * @param Identity_minus_outerProducts I - C C^T per pixel, from setup_R_update
 * @param points per-pixel points C x m23(F), overwritten
 */
void rebuild_R_sums(const Tensor3f &F, const Tensor3f &C, const Tensor3f &Cx, const Tensor3f &Cy, Vector3f &B, const std::vector<std::vector<Matrix3f>> &Identity_minus_outerProducts, std::vector<std::vector<Vector3f>> &points);

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
void update_FG(Tensor3f &F, float V, const Tensor3f &G, int y, int x, float lr, float weight_FG, float eps, float gamma);

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
void update_GF(Tensor3f &G, float V, const Tensor3f &F, int y, int x, float lr, float weight_GF, float eps, float gamma);

/**
 * Update the spatial gradient G based on the Image intensity I with central difference gradient calculation. Only done
 * at pixel (y,x) (height, width)
 * @param G spatial gradient, shape NxMx2
 * @param I_gradient gradient of I, shape NxMx2
 * @param y vertical coordinate of the updated pixel
 * @param x horizontal coordinate of the updated pixel
 * @param weight_GI weight for convex combination with old G value, 0 = no update
 * @param eps G values lower get rounded to 0.0
 * @param gamma G values with magnitude larger get rounded to gamma (or -gamma)
 */
void update_GI(Tensor3f &G, const Tensor3f &I_gradient, int y, int x, float weight_GI, float eps=1e-8, float gamma=255.0);

/**
 * Calculate the gradient of the difference between the spatial gradient value G and the spatial gradient calculated on
 * the basis of I
 * @param G spatial gradient, calculated based on old I values and V and F
 * @param I_gradient spatial gradient, calculated solely based on I
 * @param GIDiff difference between G and I_gradient; refreshed at (y, x) and its 4 neighbours
 * @param GIDiffGradient gradient of GIDiff, written at (y, x)
 * @param y vertical coordinate of the updated pixel
 * @param x horizontal coordinate of the updated pixel
 */
void updateGIDiffGradient(Tensor3f &G, Tensor3f &I_gradient, Tensor3f &GIDiff, Tensor3f &GIDiffGradient, int y, int x);

/**
 * Updates image intensity I based on spatial gradient G
 * @param I image intensity, gray scale 0-255
 * @param GIDiffGradient gradient of the difference between G and the actual current gradient of I in y and x direction
 * @param y vertical coordinate of the updated pixel
 * @param x horizontal coordinate of the updated pixel
 * @param weight_IG weight for convex combination with old I value, 0 = no update
 */
void update_IG(Tensor2f &I, const Tensor3f &GIDiffGradient, int y, int x, float weight_IG=0.5);

/**
 * Let the temporal gradient recorded by the event camera locally contribute to the image intensity
 * @param I image intensity
 * @param V temporal gradient
 * @param y vertical coordinate of the updated pixel
 * @param x horizontal coordinate of the updated pixel
 * @param minPotential lower bound for values in I, 0
 * @param maxPotential upper bound for values in I, 255
 * @param weight_IV weight for convex combination with old I value, 0 = no update
 */
void contribute(Tensor2f &I, float V, int y, int x, float minPotential, float maxPotential, const float weight_IV);

/**
 * Exponentially decay all image intensity pixels at the same time according to the the time since last update
 * @param I image intensity
 * @param decayTimeSurface latest update time for each pixel
 * @param nP neutral potential to decay to
 * @param t current time
 * @param dP decay parameter
 */
void globalDecay(Tensor2f &I, Tensor2f &decayTimeSurface, Tensor2f &nP, Tensor2f &t, Tensor2f &dP);

/**
 * Linearly decay a single pixel according
 * @param I iamge intensity
 * @param decayTimeSurface time since last decay
 * @param y vertical coordinate of the updated pixel
 * @param x horizontal coordinate of the updated pixel
 * @param time curent time
 * @param neutralPotential neutral potential
 * @param decayParam decay parameter
 */
void linearDecay(Tensor2f &I, Tensor2f &decayTimeSurface, const int y, const int x, const float time, const float neutralPotential, const float decayParam);

/**
 * Exponentially decay a single pixel
 * @param I iamge intensity
 * @param decayTimeSurface time since last decay
 * @param y vertical coordinate of the updated pixel
 * @param x horizontal coordinate of the updated pixel
 * @param time curent time
 * @param neutralPotential neutral potential
 * @param decayParam decay parameter
 */
void exponentialDecay(Tensor2f &I, Tensor2f &decayTimeSurface, const int y, const int x, const float time, const float neutralPotential, const float decayParam);

/**
 * Exponentially decay every pixel that event_step reads through a gradient stencil around (y, x), i.e. all pixels
 * within Manhattan distance 2, so that gradients never mix decayed and stale potentials
 * @param I image intensity
 * @param decayTimeSurface time of the last decay for each pixel
 * @param y vertical coordinate of the event pixel
 * @param x horizontal coordinate of the event pixel
 * @param time current time
 * @param neutralPotential neutral potential
 * @param decayParam decay time constant
 */
void decayNeighbourhood(Tensor2f &I, Tensor2f &decayTimeSurface, const int y, const int x, const float time, const float neutralPotential, const float decayParam);

/**
 * Integrate what the paper applies at a pixel without events over its silent time (no event since
 * GTimeSurface(y, x), i.e. V = 0): G <- I pulls G towards grad I, G <- F with V = 0 removes G's component along F.
 * Closed form, exact for grad I and F constant over the silent time.
 * @param G spatial gradient, shape NxMx2
 * @param F optical flow, shape NxMx2
 * @param I_gradient current gradient of I, shape NxMx2; read at (y, x)
 * @param GTimeSurface time up to which G is relaxed, per pixel; set to time
 * @param y vertical coordinate of the pixel
 * @param x horizontal coordinate of the pixel
 * @param time current time
 * @param rate_GI relaxation rate of G <- I in 1/s, see Parameters::rate_GI_silent
 * @param rate_GF relaxation rate of G <- F (V = 0) in 1/s, see Parameters::rate_GF_silent
 */
void relaxSilentG(Tensor3f &G, const Tensor3f &F, const Tensor3f &I_gradient, Tensor2f &GTimeSurface, const int y, const int x, const float time, const float rate_GI, const float rate_GF);

/**
 * relaxSilentG at (y, x) and its 4 neighbours, the pixels whose G event_step reads and writes. Computes grad I there
 * first, so I must already be decayed to time.
 */
void relaxSilentNeighbourhood(Tensor3f &G, const Tensor3f &F, const Tensor2f &I, Tensor3f &I_gradient, Tensor2f &GTimeSurface, const int y, const int x, const float time, const float rate_GI, const float rate_GF);

/**
 * Copy of G with every pixel relaxed to the given time against grad I (I already decayed to time); for display and
 * costs only
 */
Tensor3f relaxedSilentCopy(const Tensor3f &G, const Tensor3f &F, const Tensor2f &I, const Tensor2f &GTimeSurface, const float time, const float rate_GI, const float rate_GF);

/**
 * G <- I as its definition (Parameters::silentRule = "gradient"): G = grad I at (y, x) and its 4 neighbours, the pixels
 * event_step reads. The rate_GI -> infinity, rate_GF = 0 limit of relaxSilentG; I must already be decayed to time.
 */
void followGradientNeighbourhood(Tensor3f &G, const Tensor2f &I, Tensor3f &I_gradient, int y, int x);

/**
 * grad I at every pixel (I already decayed to the display time); the display copy of G for silentRule "gradient"
 */
Tensor3f gradientCopy(const Tensor2f &I);

/**
 * F, polarity -> G: brightness constancy in sign. G has two sources: grad I, and the sign evidence of events, kept as
 * a correction dG that forgets with I's time constant tau (it describes the same intensity pattern).
 * readCorrectedGradient: G at (y, x) = grad I + dG decayed to time (I already decayed to time).
 */
Vector2f readCorrectedGradient(const Tensor2f &I, Tensor3f &I_gradient, Tensor3f &dG, Tensor2f &dGTime, int y, int x, float time, float tau);

/**
 * G = grad I + dG at (y, x) and its 4 neighbours (silentRule "gradient" with the sign correction)
 */
void followCorrectedGradientNeighbourhood(Tensor3f &G, const Tensor2f &I, Tensor3f &I_gradient, Tensor3f &dG, Tensor2f &dGTime, int y, int x, float time, float tau);

/**
 * The event's polarity is the sign of V = -G . F. With G's component along F uncertain (sigmaG) and the polarity right
 * with probability `reliability`, the posterior mean of that component is
 *     g + s sigmaG phi(z) (2 rho - 1) / (rho Phi(z) + (1 - rho)(1 - Phi(z))),   s = -polarity, z = s g / sigmaG
 * dG at (y, x) takes the difference; the component across F is unchanged. Nothing changes where F = 0.
 */
void correctGradientSign(Tensor3f &dG, const Vector2f &G_read, const Vector2f &F, int polarity, int y, int x, float sigmaG, float reliability);

/**
 * grad I + dG at every pixel, dG decayed to time; display copy of G with the sign correction
 */
Tensor3f correctedGradientCopy(const Tensor2f &I, const Tensor3f &dG, const Tensor2f &dGTime, float time, float tau);

/**
 * Silence as a bound on G (Parameters::silentRule = "bound"). No event at (y, x) for silentTime means the change of I
 * there stayed between the OFF threshold (-1 step) and the ON threshold (+onStep). With brightness constancy,
 * dI = -(G . F) dt, so G's component along F is bounded:
 *     -(onStep + slack) / (|F| dt)  <  G_par  <  (1 + slack) / (|F| dt)
 * G is set to grad I (its definition) and its component along F is then held to that interval.
 * @param silentTime time without events, after the refractory time (s); <= 0 means no bound
 * @param onStep c_ON / c_OFF at this pixel
 * @param slack widens both thresholds (steps)
 */
void boundSilentG(Tensor3f &G, const Tensor3f &F, const Tensor3f &I_gradient, int y, int x, float silentTime, float onStep, float slack);

/**
 * boundSilentG at (y, x) and its 4 neighbours, each with its own silence since EventTimeSurface (minus the refractory
 * time) and its own contrast step. I must already be decayed to time.
 */
void boundSilentNeighbourhood(Tensor3f &G, const Tensor3f &F, const Tensor2f &I, Tensor3f &I_gradient, const Tensor2f &EventTimeSurface, const ContrastSteps &contrast, int y, int x, float time, float refractoryTime, float slack);

/**
 * Copy of G with every pixel bounded by its silence up to time (I already decayed to time); for display only
 */
Tensor3f boundedSilentCopy(const Tensor3f &F, const Tensor2f &I, const Tensor2f &EventTimeSurface, const ContrastSteps &contrast, float time, float refractoryTime, float slack);

/**
 * Copy of the image intensity with every pixel decayed to the given time; for display and costs only
 * @param I image intensity
 * @param decayTimeSurface time of the last decay for each pixel
 * @param time current time
 * @param neutralPotential neutral potential
 * @param decayParam decay time constant
 * @return decayed copy of I
 */
Tensor2f decayedCopy(const Tensor2f &I, const Tensor2f &decayTimeSurface, const float time, const float neutralPotential, const float decayParam);

/**
 * Locally update Image Intensity I based on temporal gradient V
 * @param I image intensity, gray scale 0-255
 * @param V temporal gradient,
 * @param y vertical coordinate of the updated pixel
 * @param x horizontal coordinate of the updated pixel
 * @param minPotential lower bound for values in I, 0
 * @param maxPotential upper bound for values in I, 255
 * @param weight_IV weighting of the update, 0 = no update, 1.0 = overwrite old value fully.
 */
void update_IV(Tensor2f &I, float  V, int y, int x, float minPotential, float maxPotential, float weight_IV);

/**
 * Update the image intensity based on camera recorded image intensity
 * @param I image intensity approximated by the network
 * @param realImage recorded image intensity
 * @param weight_Ifusion weighting of the update, 0 = no update, 1.0 = overwrite old value fully.
 */
void update_Ifusion(Tensor2f &I, const cv::Mat &realImage, const float weight_Ifusion);

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
void update_FR(Tensor3f &F, const Tensor3f &CCM, const Tensor3f &Cx, const Tensor3f &Cy, const Tensor1f &R, float weight_FR, float eps, float gamma);

// void update_RF(Tensor1f &R, const Tensor3f &F, const Tensor3f &C, const Tensor3f &Cx, const Tensor3f &Cy, const Matrix3f &A, Vector3f &B, const std::unique_ptr<Matrix3f[]> &Identity_minus_outerProducts, Vector3f &old_point, int y, int x, float weight_RF, const std::vector<Event> &frameEvents);

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
void update_RF(Tensor1f &R, const Tensor3f &F, const Tensor3f &C, const Tensor3f &Cx, const Tensor3f &Cy, const Matrix3f &A, Vector3f &B, const std::vector<std::vector<Matrix3f>> &Identity_minus_outerProducts, std::vector<std::vector<Vector3f>> &old_points, float weight_RF, int y, int x);

/**
 * Replace the contribution of pixel (y, x) to the F -> R right-hand side B with its current flow (B += (I - C C^T)
 * (new point - old point)). Used after an event step so that every change of F reaches B.
 */
void register_R_point(const Tensor3f &F, const Tensor3f &C, const Tensor3f &Cx, const Tensor3f &Cy, Vector3f &B, const std::vector<std::vector<Matrix3f>> &Identity_minus_outerProducts, std::vector<std::vector<Vector3f>> &old_points, int y, int x);

/**
 * m32 at a single pixel: coefficients (channel 0 on Cy, channel 1 on Cx) of a tangent-plane vector v
 */
Vector2f m32_pixel(const Vector3f &v, const Tensor3f &Cx, const Tensor3f &Cy, int y, int x);

/**
 * Event-driven F <- R, global part: B relaxes towards A R at rate_FR over the time since BTime, as the sum of all
 * pixels relaxing their F towards m32(R x C). O(1) instead of a pass over all pixels.
 * @param B F -> R right-hand side, sum over pixels of (I - C C^T)(C x m23(F))
 * @param BTime time up to which B is relaxed; set to time
 * @param A sum over pixels of (I - C C^T)
 * @param R rotation, (y, x, z) storage
 * @param time current time
 * @param rate_FR relaxation rate in 1/s, see Parameters::rate_FR
 */
void relaxB(Vector3f &B, float &BTime, const Matrix3f &A, const Tensor1f &R, const float time, const float rate_FR);

/**
 * Event-driven F <- R, local part: F at (y, x) relaxes towards m32(R x C) over its time since FTimeSurface(y, x),
 * with the current R. Refreshes old_points(y, x) without touching B (relaxB already accounts for it).
 */
void relaxFR(Tensor3f &F, const Tensor3f &C, const Tensor3f &Cx, const Tensor3f &Cy, const Tensor1f &R, Tensor2f &FTimeSurface, std::vector<std::vector<Vector3f>> &old_points, const int y, const int x, const float time, const float rate_FR);

/**
 * relaxFR at (y, x) and its 4 neighbours, the pixels whose F the event step and the silent G relaxation read
 */
void relaxFRNeighbourhood(Tensor3f &F, const Tensor3f &C, const Tensor3f &Cx, const Tensor3f &Cy, const Tensor1f &R, Tensor2f &FTimeSurface, std::vector<std::vector<Vector3f>> &old_points, const int y, const int x, const float time, const float rate_FR);

/**
 * Copy of F with every pixel relaxed to the given time; for display and costs only
 */
Tensor3f relaxedFRCopy(const Tensor3f &F, const Tensor3f &C, const Tensor3f &Cx, const Tensor3f &Cy, const Tensor1f &R, const Tensor2f &FTimeSurface, const float time, const float rate_FR);

/**
 * Update R from a gyro reading. The reading (x, y, z, IMU frame) is rotated into the camera frame, scaled to R's units,
 * negated (R is minus the camera rotation, see docs/cpp-implementation.md) and stored as (y, x, z).
 * @param R rotation estimated by the Interacting Maps network, stored as (y, x, z)
 * @param rotVelIMU angular velocity measured by the gyro, (x, y, z) in the IMU frame
 * @param weight_RIMU weighting between both values; 1.0 = 100% gyro, 0.0 = 100% R
 * @param R_cam_imu rotation from the IMU frame to the camera frame
 * @param imuScale factor from the gyro's angular velocity to R's units (e.g. seconds per bin)
 */
void update_RIMU(Tensor1f &R, const std::vector<float> &rotVelIMU, float weight_RIMU, const Matrix3f &R_cam_imu, float imuScale);

/**
 * Fuse a real image from a camera with the approximation of the Network
 * @param I approxmation of the network
 * @param realImage Image recorded by a camera, assumed to be grayscale 8-bit
 * @param weight_Ifusion how the images should be weighted relative to each other, 1 = 100% new image.
 */
void update_Ifusion(Tensor2f &I, const cv::Mat &realImage, const float weight_Ifusion);
#endif //UPDATE_H
