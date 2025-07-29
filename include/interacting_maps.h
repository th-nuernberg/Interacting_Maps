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
#include "events.h"
#include "update.h"

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

void randomInit(Tensor1f &T, float lower, float upper);

void randomInit(Tensor3f &T, float lower, float upper);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  STRING OPERATIONS  /////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::ostream &operator << (std::ostream &os, Event &e);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  INTERACTING MAPS MAIN FUNCTION  ////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void event_step(float V, Tensor2f &MI, Tensor2f &I, Tensor3f &delta_I, Tensor3f &GIDiff, Tensor3f &GIDiffGradient, Tensor3f &F, Tensor3f &G, Tensor1f &R, Tensor3f &CCM, Tensor3f &dCdx, Tensor3f &dCdy, Matrix3f &A, Vector3f &B, std::unique_ptr<Matrix3f[]> &Identity_minus_outerProducts, Vector3f &old_point, std::unordered_map<std::string,float> &weights, std::vector<int> &permutation, int y, int x);