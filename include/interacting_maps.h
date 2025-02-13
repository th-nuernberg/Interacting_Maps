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

#include "conversions.h"
#include "datatypes.h"
#include "imaging.h"
#include "file_operations.h"
#include "video.h"


namespace fs = std::filesystem;
using namespace Eigen;

// Define DEBUG_LOG macro that logs with function name in debug mode
#ifdef DEBUG
#define DEBUG_LOG(message) \
        std::cout << "DEBUG (" << __func__ << "): " << message << std::endl << \
        "###########################################" << std::endl;
#else
#define DEBUG_LOG(message) // No-op in release mode
#endif


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  STRING OPERATIONS  /////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::ostream &operator << (std::ostream &os, Event &e);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  GRADIENT CALCULATIONS  /////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void computeGradient(const Tensor2f &data, Tensor3f &gradients, int y, int x);

void computeGradient(const Tensor3f &data, Tensor3f &gradients, int y, int x);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// EVENT HANDLING  //////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<std::vector<Event>> bin_events(std::vector<Event> &events, float bin_size);

//void create_frames(std::vector<std::vector<Event>> &bucketed_events, std::vector<Tensor2f> &frames, int camera_height, int camera_width);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  INTERACTING MAPS HELPER FUNCTIONS  /////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool isApprox(Tensor3f &t1, Tensor3f &t2, float precision);

bool isApprox(Tensor2f &t1, Tensor2f &t2, float precision);

void norm_tensor_along_dim3(const Tensor3f &T, Tensor2f &norm);

// Function to compute C_star
autodiff::Vector3real C_star(autodiff::real x, autodiff::real y, int N_x, int N_y, float height, float width, float rs);

// Function to compute C
autodiff::Vector3real C(autodiff::real x, autodiff::real y, int N_x, int N_y, float height, float width, float rs);

// Jacobian for x value tested by hand
void find_C(int N_x, int N_y, float view_angle_x, float view_angle_y, float rs, Tensor3f &CCM, Tensor3f &C_x, Tensor3f &C_y);

// Function to compute the cross product of two 3-tensors
void crossProduct3x3(const Tensor3f &A, const Tensor3f &B, Tensor3f &C);

// Function to compute the cross product of two 3-tensors
void crossProduct3x3(const Tensor3f &A, const Vector3f &B, Vector3f &C, int y, int x);

void crossProduct1x3(const Tensor<float,1> &A, const Tensor3f &B, Tensor3f &C);

void vector_distance(const Tensor3f &vec1, const Tensor3f &vec2, Tensor2f &distance);

float sign_func(float x);

// Function to time the performance of a given dot product computation function
//template<typename Func>
//void timeDotProductComputation(Func func, Tensor3f &A, Tensor3f &B, Tensor2f &D, int iterations);

// Function using nested loops to compute the dot product
void computeDotProductWithLoops(const Tensor3f &A, const Tensor3f &B, Tensor2f &D);

void m32(const Tensor3f &In, const Tensor3f &C_x, const Tensor3f &C_y, Tensor3f &Out);

void m23(const Tensor3f &In, const Tensor3f &Cx, const Tensor3f &Cy, Vector3f &Out, int y, int x);

void setup_R_update(const Tensor3f &CCM, Matrix3f &A, Vector3f &B, std::vector<std::vector<Matrix3f>> &Identity_minus_outerProducts, std::vector<std::vector<Vector3f>> &points);


float VFG_check(const Tensor2f &V, const Tensor3f &F, const Tensor3f &G);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  INTERACTING MAPS UPDATE FUNCTIONS  /////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void update_FG(Tensor3f &F, float V, const Tensor3f &G, int y, int x, float lr, float weight_FG, float eps, float gamma);

void update_GF(Tensor3f &G, float V, const Tensor3f &F, int y, int x, float lr, float weight_GF, float eps, float gamma);

void update_GI(Tensor3f &G, const Tensor3f &I_gradient, int y, int x, float weight_GI, float eps=1e-8, float gamma=255.0);

void update_IV(Tensor2f &I, float  V, int y, int x, float minPotential, float maxPotential);

void updateGIDiffGradient(Tensor3f &G, Tensor3f &I_gradient, Tensor3f &GIDiff, Tensor3f &GIDiffGradient, int y, int x);

void update_IG(Tensor2f &I, const Tensor3f &GIDiffGradient, int y, int x, float weight_IG=0.5);

void update_FR(Tensor3f &F, const Tensor3f &CCM, const Tensor3f &Cx, const Tensor3f &Cy, const Tensor<float,1> &R, float weight_FR, float eps, float gamma);

// void update_RF(Tensor<float,1> &R, const Tensor3f &F, const Tensor3f &C, const Tensor3f &Cx, const Tensor3f &Cy, const Matrix3f &A, Vector3f &B, const std::unique_ptr<Matrix3f[]> &Identity_minus_outerProducts, Vector3f &old_point, int y, int x, float weight_RF, const std::vector<Event> &frameEvents);

void update_RF(Tensor<float,1> &R, const Tensor3f &F, const Tensor3f &C, const Tensor3f &Cx, const Tensor3f &Cy, const Matrix3f &A, Vector3f &B, const std::vector<std::vector<Matrix3f>> &Identity_minus_outerProducts, std::vector<std::vector<Vector3f>> &old_points, float weight_RF, int y, int x);


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  INTERACTING MAPS MAIN FUNCTION  ////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//void event_step(float V, Tensor2f &MI, Tensor2f &I, Tensor3f &delta_I, Tensor3f &GIDiff, Tensor3f &GIDiffGradient, Tensor3f &F, Tensor3f &G, Tensor<float,1> &R, Tensor3f &CCM, Tensor3f &dCdx, Tensor3f &dCdy, Matrix3f &A, Vector3f &B, std::unique_ptr<Matrix3f[]> &Identity_minus_outerProducts, Vector3f &old_point, std::unordered_map<std::string,float> &weights, std::vector<int> &permutation, int y, int x);