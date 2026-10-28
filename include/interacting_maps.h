#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/CXX11/Tensor>
#include <opencv2/core.hpp>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <filesystem>


#include "datatypes.h"

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

struct Parameters
{
    // Timing
    float startTime;                    // in seconds
    float endTime;                      // in seconds
    float time_step;                    // in seconds
    float fps;
    float FR_updates_per_second;
    int   updateIterationsFR = 4;
    int   event_steps = 1;

    // Weights [0-1]
    float weight_FG    = 0.2f;
    float weight_FR    = 0.8f;
    float weight_GF    = 0.2f;
    float weight_GI    = 0.2f;
    float weight_IG    = 0.2f;
    float weight_IV    = 1.0f;
    float weight_RF    = 0.8f;
    float weight_RIMU = 0.0f;
    float weight_Ifusion = 0.0f;
    float lr           = 1.0f;
    int   FR_update_counter = 100;

    // Image
    float eventContribution = 10.0f;
    float minPotential      = 0.0f;
    float maxPotential      = 255.0f;
    float neutralPotential  = 128.0f;
    int   vis_counter = -1;

    // Bounds
    float eps   = 0.00001f;
    float gamma = 255.0f;

    // Decay
    float decayParam = 1e-1f;
};

struct OpticalFlowState
{
    Tensor2f V_Vis;
    float V = 0.0f;
    cv::Mat VIGF;

    Tensor3f F;               // Optical flow
    Tensor3f G;               // Spatial gradient
    Tensor3f delta_I;

    Tensor2f I;               // Intensity image
    Tensor2f decayTimeSurface;

    Tensor3f GIDiff;          // Helper values for "I from G" update rule
    Tensor3f GIDiffGradient;

    Tensor1f R;               // Rotational velocity

    // Camera calibration
    Tensor3f CCM;
    Tensor3f dCdx;
    Tensor3f dCdy;

    // R update helpers
    Matrix3f A;
    Vector3f B;
    std::vector<std::vector<Matrix3f>> Identity_minus_outerProducts;
    std::vector<std::vector<Vector3f>> old_points;

    // Memory image
    Tensor2f MI;
    Tensor2f decayBase;
    Tensor2f expDecay;
    Tensor2f neutralPotential;
    Tensor2f time;
    Tensor2f decayParam;

    // Tensors for Image decay

};

Parameters initParameters(float startTime, float endTime, float timeStep);

OpticalFlowState initOpticalFlowState(int height, int width, const Parameters& parameters,
                                      const Calibration_Data& calibration_data);

void event_step(float V, Tensor2f &MI, Tensor3f &delta_I, Tensor3f &GIDiff, Tensor3f &GIDiffGradient, Tensor3f &F, Tensor3f &G, Tensor1f &R, const Tensor3f &CCM, const Tensor3f &dCdx, const Tensor3f &dCdy, const Matrix3f &A, Vector3f &B, const std::vector<std::vector<Matrix3f>> &Identity_minus_outerProducts, std::vector<std::vector<Vector3f>> &old_points, Parameters &parameters, std::vector<int> &permutation, int y, int x);

std::vector<float> splitTimeInterval(float startTime, float endTime, float maxIntervalLength = 0.01f);
