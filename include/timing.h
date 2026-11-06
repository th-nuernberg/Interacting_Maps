#pragma once
// Rotation from event timing (docs/cpp-implementation.md, "Rotation from event timing").
//
//   events --(OnsetRule)--> O --(fitNormalFlow)--> N --(FlowEvidence)--> F evidence --(RotationFromFlow)--> R
//                                                                         F <--(readFlow: evidence + MotionModel R)
//
// Every map is its own struct and every relation a function or method that takes only the maps it reads and
// writes, so a relation can be swapped (e.g. another OnsetRule) or another one added (e.g. a translation model next
// to MotionModel::rotation) without touching the others. Conventions as in the rest of the code: pixel (y, x),
// 2-vectors (F, n) stored as (y, x), R stored as (y, x, z) with R = -(camera angular velocity), here in rad/s.

#include <Eigen/Dense>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "datatypes.h"

namespace timing {

// ------------------------------------------------------------------------------------------------------------------
// O: arrival time of the current edge, per pixel and polarity

struct OnsetMap {
    int height, width;
    std::vector<double> stamp;       // arrival time
    std::vector<double> lastEvent;   // time of the latest event, -inf if none
    std::vector<double> resource;    // depression state, 1 = fully recovered

    OnsetMap(int height, int width);
    int index(int y, int x, int polarity) const { return (y * width + x) * 2 + (polarity > 0 ? 1 : 0); }
};

// events -> O. An onset rule updates the stamp of the event's pixel and polarity.
class OnsetRule {
public:
    virtual ~OnsetRule() = default;
    virtual void update(OnsetMap &onset, int y, int x, int polarity, double t) const = 0;
    virtual std::string name() const = 0;
};

// O = time of the latest event (no burst handling)
class LatestEventOnset : public OnsetRule {
public:
    void update(OnsetMap &onset, int y, int x, int polarity, double t) const override;
    std::string name() const override { return "latest"; }
};

// O = first event of the current burst; a gap longer than `gap` starts a new burst
class HardGapOnset : public OnsetRule {
public:
    explicit HardGapOnset(double gap) : gap(gap) {}
    void update(OnsetMap &onset, int y, int x, int polarity, double t) const override;
    std::string name() const override { return "gap"; }
    double gap;
};

// Soft onset by short-term synaptic depression (Tsodyks-Markram): the resource r recovers towards 1 with time
// constant tau; an event moves O towards t by r (how much it counts as news) and consumes the fraction `use` of r.
//   r_before = 1 - (1 - r) exp(-dt / tau),   O <- O + r_before (t - O),   r <- (1 - use) r_before
class DepressionOnset : public OnsetRule {
public:
    DepressionOnset(double tau, double use) : tau(tau), use(use) {}
    void update(OnsetMap &onset, int y, int x, int polarity, double t) const override;
    std::string name() const override { return "depression"; }
    double tau, use;
};

// "latest", "gap" (parameter: gap in s) or "depression" (parameter: tau in s; use = 1)
std::unique_ptr<OnsetRule> makeOnsetRule(const std::string &name, double parameter, double use = 1.0);

// ------------------------------------------------------------------------------------------------------------------
// O -> N: normal flow from the arrival times of the neighbours

struct NormalFlow {
    bool valid = false;
    Eigen::Vector2f n = Eigen::Vector2f::Zero();   // unit direction of motion, (y, x)
    float speed = 0.0f;                            // px/s along n
    float variance = 0.0f;                         // of speed, (px/s)^2
    int neighbours = 0;
    double time = 0.0;                             // s; when the measured motion happened: mean of the onsets fitted
};

struct NormalFlowSettings {
    int radius = 2;              // neighbourhood (2 * radius + 1)^2
    double horizon = 0.03;       // s; neighbours whose latest event is older are ignored (slowest speed ~ radius/horizon)
    double timeFloor = 1e-4;     // s; lower bound of the fit residual
    float maxSpeed = 5000.0f;    // px/s; faster fits are rejected
};

// Fits stamp_q - stamp_e = a dx + b dy over the same-polarity neighbours q of the event pixel. Arrival times grow
// along the motion, so the edge moves along (a, b) / |(a, b)| with speed 1 / |(a, b)|.
NormalFlow fitNormalFlow(const OnsetMap &onset, int y, int x, int polarity, double t, const NormalFlowSettings &settings);

// ------------------------------------------------------------------------------------------------------------------
// Motion model: F at a pixel as a linear function of the global motion parameters

struct MotionModel {
    int height = 0, width = 0;
    std::vector<Eigen::Matrix<double, 2, 3>> M;   // F_p = M_p R

    // Rotation only, through the lens model (radial-tangential distortion from the calibration): M_p R is the
    // pixel's image velocity in px/s for R = -(angular velocity in rad/s). Without distortion this equals
    // m32(R x C) at the pixel.
    static MotionModel rotation(const Calibration_Data &calibration, int height, int width);
    const Eigen::Matrix<double, 2, 3> &at(int y, int x) const { return M[y * width + x]; }
};

// ------------------------------------------------------------------------------------------------------------------
// F evidence: what the events measured about F, per pixel, in information form, fading with `lifetime`

struct EvidenceChange {
    Eigen::Matrix2d dInformation = Eigen::Matrix2d::Zero();   // added - removed
    Eigen::Vector2d dVector = Eigen::Vector2d::Zero();
    // The same change split by the time each part describes (for relations that model motion over time)
    Eigen::Matrix2d removedInformation = Eigen::Matrix2d::Zero(), addedInformation = Eigen::Matrix2d::Zero();
    Eigen::Vector2d removedVector = Eigen::Vector2d::Zero(), addedVector = Eigen::Vector2d::Zero();
    double removedTime = 0.0, addedTime = 0.0;
};

struct FlowEvidence {
    int height, width;
    double lifetime;     // s
    bool accumulate;     // false: a new measurement replaces the pixel's evidence; true: it is added
    std::vector<Eigen::Matrix2d> information;   // Lambda = sum w n n^T
    std::vector<Eigen::Vector2d> vector;        // eta = sum w n s
    std::vector<double> time;                   // time up to which the pixel's evidence has been faded
    std::vector<double> measured;               // time the pixel's evidence describes (replace mode)

    FlowEvidence(int height, int width, double lifetime, bool accumulate);
    // Fades the pixel's evidence to time t (exactly; all evidence fades with the same rate).
    void fadeTo(int y, int x, double t);
    // N -> F evidence. Returns the change, for relations that keep sums over the evidence.
    EvidenceChange add(int y, int x, const NormalFlow &measurement, double t);
};

// ------------------------------------------------------------------------------------------------------------------
// F evidence -> R: least squares of the motion model against all evidence, as running sums
//   H = sum_p M_p^T Lambda_p M_p,  b = sum_p M_p^T eta_p,  R = H^-1 b
// All evidence fades with the same lifetime, so the sums fade as a whole in O(1). R is solved only when the sums
// changed since the last solve and R is read.
//
// rotationAt(t): every measurement describes the motion at its own time t_i (the onsets it was fitted from), which
// lies in the past. Modelling R(t_i) = R(t) + Rdot (t_i - t), the least squares over [R(t), Rdot] needs the moments
// sum t_i^k M^T Lambda M (k = 0, 1, 2) and sum t_i^k M^T eta (k = 0, 1), which also fade as a whole. R is then the
// rotation now instead of an average over the evidence's past. Times are kept relative to `epoch`.

struct RotationFromFlow {
    double lifetime;
    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    Eigen::Vector3d b = Eigen::Vector3d::Zero();
    double time = -std::numeric_limits<double>::infinity();
    Eigen::Vector3f R = Eigen::Vector3f::Zero();
    bool dirty = false;
    long solves = 0;
    // Moments for rotationAt (H and b are the 0th)
    double epoch = 0.0;
    Eigen::Matrix3d H1 = Eigen::Matrix3d::Zero(), H2 = Eigen::Matrix3d::Zero();
    Eigen::Vector3d b1 = Eigen::Vector3d::Zero();
    double solveInterval = 1e-3;   // s; rotationAt solves at most this often and extrapolates with Rdot in between
    bool dirtyAt = false;          // rotationAt's own flag: new evidence since its last solve
    double solvedAt = -std::numeric_limits<double>::infinity();
    Eigen::Vector3d Rsolved = Eigen::Vector3d::Zero(), Rdot = Eigen::Vector3d::Zero();

    explicit RotationFromFlow(double lifetime) : lifetime(lifetime) {}
    void fadeTo(double t);
    void apply(const Eigen::Matrix<double, 2, 3> &M, const EvidenceChange &change);
    // A direct measurement of R (e.g. the gyro) with information `weight` per component, describing time `measured`,
    // fading like the evidence.
    void addMeasurement(const Eigen::Vector3f &R_measured, double weight, double measured = 0.0);
    // R as a weighted average over the evidence (what the evidence says on average over its lifetime)
    const Eigen::Vector3f &rotation();
    // R at time t, with the rate of change estimated from when each measurement happened
    const Eigen::Vector3f &rotationAt(double t);

private:
    void addMoments(const Eigen::Matrix3d &P, const Eigen::Vector3d &q, double measured, double sign);
};

// ------------------------------------------------------------------------------------------------------------------
// G, polarity -> F evidence: brightness constancy in sign. V's size is unknown, but its sign is the event's polarity,
// and V = -G . F. A normal flow whose direction n gives sign(-G . n) != polarity is therefore suspicious, if G is
// clearly non-zero along n:
//   q = Phi(|G . n| / sigmaG)                      probability that G's sign along n is right (sigmaG: noise of G)
//   P(n right | agrees)    = q (1 - flip) / (q (1 - flip) + (1 - q) flip)
//   P(n right | disagrees) = (1 - q)(1 - flip) / ((1 - q)(1 - flip) + q flip)
// with flip the prior probability that a measured direction is flipped.
double directionProbability(const Eigen::Vector2f &G, const Eigen::Vector2f &n, int polarity, double sigmaG,
                            double flip);

// ------------------------------------------------------------------------------------------------------------------
// R -> F: F's value from its evidence and the motion model's prediction (prior information kappa)
//   F = (Lambda + kappa I)^-1 (eta + kappa M R)
// Writes the value into the stored F map (channels y, x) and returns it. The prior never enters the evidence.

Eigen::Vector2f readFlow(FlowEvidence &evidence, const MotionModel &model, const Eigen::Vector3f &R, double kappa,
                         int y, int x, double t, Tensor3f &F);
// readFlow at (y, x) and its 4 neighbours, the pixels whose F an event step reads
void readFlowNeighbourhood(FlowEvidence &evidence, const MotionModel &model, const Eigen::Vector3f &R, double kappa,
                           int y, int x, double t, Tensor3f &F);
void readFlowAll(FlowEvidence &evidence, const MotionModel &model, const Eigen::Vector3f &R, double kappa, double t,
                 Tensor3f &F);

}  // namespace timing
