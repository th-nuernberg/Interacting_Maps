#include <interacting_maps.h>
#include <iostream>
#include <filesystem>
#include <string>
#include <sstream>
#include <cassert>
#include <memory>
#include "Instrumentor.h"
#include <cmath>
#include <update.h>
#include <cost.h>

#include <boost/program_options.hpp>
#include "file_operations.h"
#include "imaging.h"

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
//  INTERACTING MAPS STATE AND EVENT STEP  /////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Parameters initParameters(float startTime, float endTime, float timeStep)
{
    Parameters p;

    p.startTime             = startTime;
    p.endTime               = endTime;
    p.time_step             = timeStep;
    p.fps                   = 1.0f / timeStep;
    p.FR_updates_per_second = 1.0f / timeStep;

    return p;
}

OpticalFlowState initOpticalFlowState(int height, int width, const Parameters& parameters,
                                      const Calibration_Data& calibration_data)
{
    OpticalFlowState s;

    s.V_Vis = Tensor2f(height, width);
    s.V_Vis.setZero();

    s.F = Tensor3f(height, width, 2);
    // One stream per randomly initialised map, so adding maps does not shift existing draws.
    std::seed_seq seed_F{parameters.seed, 0u};
    std::mt19937 rng_F(seed_F);
    randomInit(s.F, -1, 1, rng_F);

    s.G = Tensor3f(height, width, 2);
    s.G.setZero();
    s.GTimeSurface = Tensor2f(height, width);
    s.GTimeSurface.setConstant(parameters.startTime);
    s.FTimeSurface = Tensor2f(height, width);
    s.FTimeSurface.setConstant(parameters.startTime);
    s.BTime = parameters.startTime;

    s.delta_I = Tensor3f(height, width, 2);
    s.delta_I.setZero();

    s.I = Tensor2f(height, width);
    s.I.setConstant(parameters.neutralPotential);

    s.decayTimeSurface = Tensor2f(height, width);
    s.decayTimeSurface.setConstant(parameters.startTime);

    s.GIDiff = Tensor3f(height, width, 2);
    s.GIDiff.setZero();

    s.GIDiffGradient = Tensor3f(height, width, 2);
    s.GIDiffGradient.setZero();

    s.R = Tensor1f(3);
    std::seed_seq seed_R{parameters.seed, 1u};
    std::mt19937 rng_R(seed_R);
    randomInit(s.R, -10, 10, rng_R);

    // Camera calibration
    s.CCM  = Tensor3f(height, width, 3);  s.CCM.setZero();
    s.dCdx = Tensor3f(height, width, 3);  s.dCdx.setZero();
    s.dCdy = Tensor3f(height, width, 3);  s.dCdy.setZero();
    const MatrixXf &K = calibration_data.camera_matrix;
    find_C(width, height, K(0, 0), K(1, 1), K(0, 2), K(1, 2), s.CCM, s.dCdx, s.dCdy);

    // R update helpers
    s.A = Matrix3f::Zero();
    s.B = Vector3f::Zero();
    s.Identity_minus_outerProducts.resize(height, std::vector<Matrix3f>(width));
    s.old_points.resize(height, std::vector<Vector3f>(width));
    setup_R_update(s.CCM, s.A, s.B, s.Identity_minus_outerProducts, s.old_points);
    rebuild_R_sums(s.F, s.CCM, s.dCdx, s.dCdy, s.B, s.Identity_minus_outerProducts, s.old_points);

    // Memory image
    s.MI = Tensor2f(height, width);
    s.MI.setConstant(parameters.neutralPotential);

    s.decayBase = Tensor2f(height, width);
    s.decayBase.setConstant(parameters.neutralPotential);

    s.expDecay = Tensor2f(height, width);
    s.expDecay.setConstant(1.0f);

    s.neutralPotential(s.I.dimensions());    // neutralPotential
    s.time(s.I.dimensions());     // time
    s.decayParam(s.I.dimensions());    // decayParameter
    s.neutralPotential.setConstant(parameters.neutralPotential);
    s.decayParam.setConstant(parameters.decayParam);

    return s;
}

void event_step(const float V, Tensor2f &MI, Tensor3f &delta_I, Tensor3f &GIDiff, Tensor3f &GIDiffGradient, Tensor3f &F, Tensor3f &G, Tensor1f &R, const Tensor3f &CCM, const Tensor3f &dCdx, const Tensor3f &dCdy, const Matrix3f &A, Vector3f &B, const std::vector<std::vector<Matrix3f>> &Identity_minus_outerProducts, std::vector<std::vector<Vector3f>> &old_points, Parameters &parameters, std::vector<int> &permutation, int y, int x){
    PROFILE_FUNCTION();
    array<Index, 2> dimensions = MI.dimensions();
    update_IV(MI, V, y, x, parameters.minPotential, parameters.maxPotential, parameters.weight_IV);
    // Image (MI) got changed through update by V. we need to update all surrounding gradient values. Because of the change at this pixel
    {
        PROFILE_SCOPE("GRADIENTS");
        if (y>0){
            computeGradient(MI, delta_I, y-1, x);
            update_GI(G, delta_I, y-1, x, parameters.weight_GI, parameters.zeroThreshold(), parameters.clampBound());
        }
        if (x>0){
            computeGradient(MI, delta_I, y, x-1);
            update_GI(G, delta_I, y, x-1, parameters.weight_GI, parameters.zeroThreshold(), parameters.clampBound());
        }
        if (y<dimensions[0]-1){
            computeGradient(MI, delta_I, y+1, x);
            update_GI(G, delta_I, y+1, x, parameters.weight_GI, parameters.zeroThreshold(), parameters.clampBound());
        }
        if (x<dimensions[1]-1){
            computeGradient(MI, delta_I, y, x+1);
            update_GI(G, delta_I, y, x+1, parameters.weight_GI, parameters.zeroThreshold(), parameters.clampBound());
        }
    }

    computeGradient(MI, delta_I, y, x);
    update_GI(G, delta_I, y, x, parameters.weight_GI, parameters.zeroThreshold(), parameters.clampBound());
    updateGIDiffGradient(G, delta_I, GIDiff, GIDiffGradient, y, x);
    update_IG(MI, GIDiffGradient, y, x, parameters.weight_IG);
    //computeGradient(MI, delta_I, y, x);

    for (const auto& element : permutation){
        switch( element ){
            case 0:
                update_FG(F, V, G, y, x, parameters.lr, parameters.weight_FG, parameters.zeroThreshold(), parameters.clampBound());
                break;
            case 1:
                // Gets called separately because we do not want to do an update of F based on R with every event since this update is global
                update_FR(F, CCM, dCdx, dCdy, R, parameters.weight_FR, parameters.zeroThreshold(), parameters.clampBound());
                break;
            case 2:
                update_GF(G, V, F, y, x, parameters.lr, parameters.weight_GF, parameters.zeroThreshold(), parameters.clampBound());
                break;
            case 3:
                update_RF(R, F, CCM, dCdx, dCdy, A, B, Identity_minus_outerProducts, old_points, parameters.weight_RF, y, x);
                break;
            default:
                std::cout << "Unknown number in permutation" << std::endl;
                break;
        }
    }
}

void randomInit(Tensor3f &T, const float lower, const float upper, std::mt19937 &rng) {
    std::uniform_real_distribution<float> uniform(lower, upper);
    for (Index i = 0; i < T.size(); ++i) {
        T.data()[i] = uniform(rng);
    }
}

void softReset(Tensor3f &T, const float lower, const float upper) {
    const auto &dimensions = T.dimensions();
    Tensor3f T1(dimensions[0], dimensions[1], dimensions[2]);
    Tensor3f T2(dimensions[0], dimensions[1], dimensions[2]);
    Tensor3f T3(dimensions[0], dimensions[1], dimensions[2]);
    T1.setRandom();
    T2.setConstant(lower);
    T3.setConstant(upper - lower);
    T = T + T1*T3 + T2;
}

void randomInit(Tensor1f &T, const float lower, const float upper, std::mt19937 &rng) {
    std::uniform_real_distribution<float> uniform(lower, upper);
    for (Index i = 0; i < T.size(); ++i) {
        T.data()[i] = uniform(rng);
    }
}

std::vector<float> splitTimeInterval(float startTime, float endTime, float maxIntervalLength)
{
    std::vector<float> intervals = {startTime, endTime};

    if (endTime - startTime > maxIntervalLength) {
        intervals = {startTime};
        float currentTime = startTime;

        while (currentTime + maxIntervalLength < endTime) {
            currentTime += maxIntervalLength;
            intervals.push_back(currentTime);
        }
        intervals.push_back(endTime);
    }

    return intervals;
}
