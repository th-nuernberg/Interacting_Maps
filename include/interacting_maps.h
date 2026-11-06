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

#include <limits>
#include <random>

void randomInit(Tensor1f &T, float lower, float upper, std::mt19937 &rng);

void randomInit(Tensor3f &T, float lower, float upper, std::mt19937 &rng);

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

    // Legacy pipeline (rotationFromTiming = false, --legacy): F <- G, V, G <- F, V, the F <-> R loop and gyro fusion. V has no
    // time unit there, so F and R have none. Its values below (FR_updates_per_second, updateIterationsFR, event_steps,
    // weight_FG/GF/FR/RF, rate_FR, imuScale, lr) are inherited and are not being grounded; timing mode replaces them.
    // I <- V, I <-> G and G's rules are shared by both pipelines.
    //
    // Weights [0-1] of the per-update convex combination old -> candidate. 1 solves the relation's local
    // constraint exactly (no damping); kept at 1 until there is a reason for another value (sweep, M2).
    float weight_FG    = 1.0f;
    float weight_FR    = 1.0f;   // synchronous F <- R only (asyncFR = false); the event-driven path uses rate_FR
    float weight_GF    = 1.0f;
    float weight_GI    = 1.0f;
    float weight_IG    = 1.0f;
    float weight_IV    = 1.0f;
    float weight_RF    = 1.0f;
    // Inputs, 0 = off. Not set to 1: that would clamp R to the gyro / I to the frame (rule 5, weak inputs).
    float weight_RIMU = 0.0f;
    Matrix3f R_cam_imu = Matrix3f::Identity();   // IMU frame -> camera frame
    float imuScale     = 1.0f;                   // gyro rad/s -> R units; open, see docs/cpp-implementation.md
    float weight_Ifusion = 0.0f;
    float lr           = 1.0f;
    int   FR_update_counter = 100;

    // Image. I is log intensity in units of the contrast threshold c: an event is a step of +-1 (delta log I = +-c).
    // The offset of log I is unobservable, so I starts at 0; no bounds (never clamp). The former 10 / 128 / [0, 255]
    // only mapped I onto 8-bit grey values; that mapping now lives in the display (displayGreyPerEvent in main.cpp).
    float eventContribution = 1.0f;
    float minPotential      = -std::numeric_limits<float>::infinity();
    float maxPotential      = std::numeric_limits<float>::infinity();
    float neutralPotential  = 0.0f;
    // ON and OFF events may stand for different contrasts (c_ON != c_OFF; docs/assumptions.md, W3). I is in units of
    // c_OFF; an ON event adds c_ON / c_OFF. "equal", "fixed" (contrastRatio), "global" or "pixel" (include/contrast.h).
    std::string contrastSteps = "pixel";   // per-pixel thresholds differ (experiments/reports/forgetting.md)
    double      contrastRatio = 1.0;     // c_ON / c_OFF for "fixed"
    double      contrastPrior = 100.0;   // "pixel": pseudo-events of the global ratio
    int   vis_counter = -1;

    // Deviations from the paper (docs/cpp-implementation.md). Defaults keep the established behaviour.
    // V -> I injection is controlled by weight_IV (0 disables it).
    bool useDecay        = true;    // decay MI towards neutralPotential between events
    bool clampMaps       = false;   // clip F and G to +-gamma and zero entries below eps (old display-range bound)
    bool resetGEachFrame = false;   // IMAGES builds only: zero G after writing each output frame (old hack)

    // Pixels without events carry V = 0 (paper: every pixel is updated every iteration). G <- I and G <- F (V = 0)
    // are integrated lazily per pixel over its silent time instead of per time bin. Rates in 1/s. Continuous-time
    // rates have no neutral value (weight 1 per step would be an infinite rate, i.e. a hard constraint); these are
    // the former -ln(1 - 0.2) / 10 ms and are open, to be swept (M2).
    bool  silentRelaxation = true;
    float rate_GI_silent   = 22.314355f;
    float rate_GF_silent   = 22.314355f;
    // What G does between events (docs/assumptions.md G4, experiments/reports/silent-bound.md):
    // "gradient": G = grad I whenever it is read, its definition; no rates (default).
    // "rate": relax towards grad I and remove the component along F at rate_GI_silent / rate_GF_silent (former default;
    // same result in timing mode). "bound": no event for dt means the change of I
    // stayed between the OFF and ON thresholds, so with dI = -G.F dt, G's component along F is bounded by the
    // thresholds / (|F| dt). G follows grad I immediately. The silence counts from refractoryTime after the last
    // event (the pixel cannot fire before; 1.2 ms is the smallest interval on shapes_rotation); silentSlack widens
    // both thresholds (steps). Falsified at edge fronts: silence bounds G before the edge arrived, not the gradient
    // that is arriving now, and applying it to the current grad I removes that edge (0.850 -> 0.752).
    std::string silentRule  = "gradient";  // "gradient", "rate" or "bound"
    float refractoryTime    = 1.2e-3f;     // s
    float silentSlack       = 0.0f;        // steps

    // Event-driven F <- R: instead of updateIterationsFR global passes per time step, B relaxes in closed form at
    // every event and F is relaxed lazily where it is read. rate_FR [1/s] = -n ln(1 - w) / dt reproduces n synchronous
    // passes of weight w per dt; the value is the former 4 x 0.8 per 10 ms and is open (M2).
    bool  asyncFR = true;
    float rate_FR = 643.775165f;

    // Rotation from event timing (include/timing.h, docs/cpp-implementation.md). Replaces F <- G, V, G <- F, V and
    // the F <-> R loop: F gets evidence from the arrival times of edges, R (rad/s) is fitted to that evidence, and F's
    // stored value combines its evidence with R's prediction. All values are open (sweep on the simulator, M1).
    bool        rotationFromTiming  = true;    // timing mode (default); false is the legacy pipeline (--legacy)
    std::string onsetRule           = "depression";   // "latest", "gap", "depression"
    double      onsetTime           = 0.02;           // s: depression recovery time constant, or the hard burst gap
    double      onsetUse            = 1.0;            // depression: fraction of the resource an event consumes
    int         normalFlowRadius    = 2;              // neighbourhood (2 r + 1)^2
    double      normalFlowHorizon   = 0.03;           // s
    double      normalFlowTimeFloor = 1e-4;           // s
    float       normalFlowMaxSpeed  = 5000.0f;        // px/s
    double      evidenceLifetime    = 0.01;           // s
    bool        evidenceAccumulate  = false;          // false: a new measurement replaces the pixel's evidence
    double      flowPriorSigma      = 27.0;           // px/s: how closely F follows R where evidence is weak
    double      gyroSigma           = 0.05;           // rad/s: gyro as a measurement of R (with fuseR)
    // G, polarity -> F evidence (timing::directionProbability): "off", "gate" (drop a measurement whose direction
    // disagrees with polarity and G where |G . n| > gradientSigma) or "weight" (scale its information by the
    // probability that its direction is right). gradientSigma: noise of G, sqrt(sigma^2 rate tau / 4) = 0.5 steps/px
    // with sigma^2 = 1.9 steps^2 per event, 8.8 events/s and tau = 0.06 s (experiments/reports/forgetting.md).
    // flipPrior: share of flipped directions among all measurements (9.5 % on shapes_rotation).
    std::string polaritySign        = "weight";   // experiments/reports/polarity-sign.md
    double      gradientSigma       = 0.5;            // steps/px
    double      flipPrior           = 0.1;
    // F, polarity -> G (correctGradientSign): the polarity moves G's component along F to its posterior mean, kept as
    // a correction that forgets with I's decay time. polarityReliability: P(polarity = sign of the true V), 0.96 from
    // the frame gradient (experiments/reports/polarity-check.md).
    bool        signCorrectG        = true;    // experiments/reports/delay-and-sign.md
    double      polarityReliability = 0.96;
    // R now instead of its average over the evidence's lifetime: measurements dated at their onsets, R and its rate
    // of change fitted together (RotationFromFlow::rotationAt). rotationSolveInterval: computational, not physical.
    bool        predictRotation     = true;    // experiments/reports/delay-and-sign.md
    double      rotationSolveInterval = 1e-3;     // s

    // Bounds
    float eps   = 0.00001f;
    float gamma = 255.0f;
    float clampBound() const { return clampMaps ? gamma : std::numeric_limits<float>::infinity(); }
    float zeroThreshold() const { return clampMaps ? eps : 0.0f; }

    // Decay: I forgets with time constant decayParam (s). tau* = 1 / (sigma^2 rate): forget before the error that every
    // event adds (sigma^2 = 1.9 steps^2 per event) reaches one step^2 in a neighbour difference; 0.06 s at the median
    // 8.8 events/s of active pixels on shapes_rotation, and the optimum of a sweep (experiments/reports/forgetting.md).
    float decayParam = 0.06f;

    // Reproducibility: random initialisation and the update-order shuffle derive from this seed
    unsigned seed = 0;
    // Start state. F = 0 and R = 0 encode "no information yet". The former random start (F uniform in [-1, 1],
    // R uniform in [-10, 10]) is inherited without a reason and kept for comparison only.
    bool randomInitialState = false;
};

struct OpticalFlowState
{
    Tensor2f V_Vis;
    float V = 0.0f;
    cv::Mat VIGF;

    Tensor3f F;               // Optical flow
    Tensor3f G;               // Spatial gradient
    Tensor2f GTimeSurface;    // time up to which G's silent-time relaxation has been applied
    Tensor2f EventTimeSurface;   // time of each pixel's latest event (start time if none)
    Tensor3f GSign;              // dG: G's correction from the sign evidence of events (F, polarity -> G)
    Tensor2f GSignTime;          // time up to which dG has been decayed
    Tensor2f FTimeSurface;    // time up to which F's relaxation towards m32(R x C) has been applied
    float    BTime = 0.0f;    // time up to which B's relaxation towards A R has been applied
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
