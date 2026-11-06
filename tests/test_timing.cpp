#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <random>

#include <file_operations.h>
#include <interacting_maps.h>
#include <timing.h>
#include <update.h>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using namespace timing;

namespace {
Calibration_Data calibration(int height, int width, std::vector<float> distortion = {0, 0, 0, 0, 0}) {
    std::vector<float> raw = {199.0f, 198.8f, width / 2.0f + 12.0f, height / 2.0f + 20.7f};
    raw.insert(raw.end(), distortion.begin(), distortion.end());
    return get_calibration_data(raw, height, width);
}
}

// ------------------------------------------------------------------------------------------------------------------
// events -> O

TEST_CASE("Hard-gap onset keeps the first event of a burst and restarts after a gap", "[timing][onset]") {
    OnsetMap onset(3, 3);
    HardGapOnset rule(0.01);
    for (double t : {0.100, 0.103, 0.106}) rule.update(onset, 1, 1, 1, t);
    REQUIRE(onset.stamp[onset.index(1, 1, 1)] == 0.100);
    rule.update(onset, 1, 1, 1, 0.200);
    REQUIRE(onset.stamp[onset.index(1, 1, 1)] == 0.200);
    REQUIRE(std::isinf(onset.lastEvent[onset.index(1, 1, -1)]));   // polarities are separate maps
}

TEST_CASE("Depression onset lies between latest event (tau -> 0) and first onset (tau -> inf)", "[timing][onset]") {
    const std::vector<double> times = {0.100, 0.102, 0.104, 0.106};
    auto run = [&](const OnsetRule &rule) {
        OnsetMap onset(1, 1);
        for (double t : times) rule.update(onset, 0, 0, 1, t);
        return onset.stamp[onset.index(0, 0, 1)];
    };
    REQUIRE_THAT(run(DepressionOnset(1e-9, 1.0)), WithinAbs(0.106, 1e-12));
    REQUIRE_THAT(run(DepressionOnset(1e9, 1.0)), WithinAbs(0.100, 1e-9));
    const double soft = run(DepressionOnset(0.02, 1.0));
    REQUIRE(soft > 0.100);
    REQUIRE(soft < 0.103);   // events in quick succession barely move the arrival time
}

TEST_CASE("Depression onset resets after a long silence, also when an event keeps resource (use < 1)", "[timing][onset]") {
    OnsetMap onset(1, 1);
    DepressionOnset rule(0.02, 0.5);
    rule.update(onset, 0, 0, 1, 0.1);
    rule.update(onset, 0, 0, 1, 5.0);
    REQUIRE_THAT(onset.stamp[onset.index(0, 0, 1)], WithinAbs(5.0, 1e-9));
}

TEST_CASE("makeOnsetRule builds the named rules and rejects unknown names", "[timing][onset]") {
    REQUIRE(makeOnsetRule("latest", 0)->name() == "latest");
    REQUIRE(makeOnsetRule("gap", 0.01)->name() == "gap");
    REQUIRE(makeOnsetRule("depression", 0.02)->name() == "depression");
    REQUIRE_THROWS(makeOnsetRule("other", 0));
}

// ------------------------------------------------------------------------------------------------------------------
// O -> N

TEST_CASE("fitNormalFlow recovers direction and speed of a plane wave of arrival times", "[timing][normal-flow]") {
    const double speed = 200.0, angle = 0.6;   // px/s, direction of motion (x right, y down)
    const double cx = std::cos(angle), cy = std::sin(angle);
    OnsetMap onset(9, 9);
    const double t0 = 1.0, now = 1.02;
    for (int y = 0; y < 9; ++y) {
        for (int x = 0; x < 9; ++x) {
            const int i = onset.index(y, x, 1);
            onset.stamp[i] = t0 + ((x - 4) * cx + (y - 4) * cy) / speed;
            onset.lastEvent[i] = now;
        }
    }
    const NormalFlow nf = fitNormalFlow(onset, 4, 4, 1, now, NormalFlowSettings{});
    REQUIRE(nf.valid);
    REQUIRE_THAT(nf.speed, WithinRel(200.0f, 1e-4f));
    REQUIRE_THAT(nf.n(0), WithinAbs(cy, 1e-5));   // (y, x) storage
    REQUIRE_THAT(nf.n(1), WithinAbs(cx, 1e-5));
    REQUIRE(nf.neighbours == 24);
}

TEST_CASE("fitNormalFlow needs four recent neighbours and reports more variance for noisy times", "[timing][normal-flow]") {
    OnsetMap onset(5, 5);
    const double now = 1.0;
    std::mt19937 rng(0);
    std::normal_distribution<double> noise(0.0, 1e-3);
    auto fill = [&](double sigma) {
        for (int y = 0; y < 5; ++y)
            for (int x = 0; x < 5; ++x) {
                const int i = onset.index(y, x, 1);
                onset.stamp[i] = 0.99 + (x - 2) / 100.0 + sigma * noise(rng) / 1e-3;
                onset.lastEvent[i] = now;
            }
    };
    fill(0.0);
    const NormalFlow clean = fitNormalFlow(onset, 2, 2, 1, now, NormalFlowSettings{});
    fill(2e-3);
    const NormalFlow noisy = fitNormalFlow(onset, 2, 2, 1, now, NormalFlowSettings{});
    REQUIRE(clean.valid);
    REQUIRE(noisy.valid);
    REQUIRE(noisy.variance > 10 * clean.variance);

    NormalFlowSettings settings;
    settings.horizon = 0.001;
    for (int i = 0; i < 5 * 5 * 2; ++i) onset.lastEvent[i] = -1.0;   // all neighbours too old
    REQUIRE_FALSE(fitNormalFlow(onset, 2, 2, 1, now, settings).valid);
}

// ------------------------------------------------------------------------------------------------------------------
// Motion model

TEST_CASE("Rotation model without distortion equals F = m32(R x C) of the network", "[timing][model][conventions]") {
    const int H = 180, W = 240;
    const Calibration_Data cal = calibration(H, W);
    const MotionModel model = MotionModel::rotation(cal, H, W);
    Tensor3f C(H, W, 3), Cx(H, W, 3), Cy(H, W, 3), RxC(H, W, 3), F(H, W, 2);
    const MatrixXf &K = cal.camera_matrix;
    find_C(W, H, K(0, 0), K(1, 1), K(0, 2), K(1, 2), C, Cx, Cy);
    Tensor1f R(3);
    R.setValues({0.3f, -0.5f, 0.8f});
    crossProduct1x3(R, C, RxC);
    m32(RxC, Cx, Cy, F);
    const Eigen::Vector3d r(0.3, -0.5, 0.8);
    for (const auto &px : std::vector<std::array<int, 2>>{{90, 120}, {10, 15}, {170, 230}, {40, 200}}) {
        const Eigen::Vector2d f = model.at(px[0], px[1]) * r;
        INFO("pixel " << px[0] << "," << px[1]);
        REQUIRE_THAT(f(0), WithinRel(static_cast<double>(F(px[0], px[1], 0)), 1e-3));
        REQUIRE_THAT(f(1), WithinRel(static_cast<double>(F(px[0], px[1], 1)), 1e-3));
    }
}

TEST_CASE("Barrel distortion slows the predicted motion towards the border, not at the centre", "[timing][model]") {
    const int H = 180, W = 240;
    const MotionModel pinhole = MotionModel::rotation(calibration(H, W), H, W);
    const MotionModel barrel = MotionModel::rotation(calibration(H, W, {-0.37f, 0.15f, 0, 0, 0}), H, W);
    const Eigen::Vector3d r(0.0, 1.0, 0.0);   // R about y (pan)
    const int cy = 110, cx = 132;              // principal point
    REQUIRE_THAT((barrel.at(cy, cx) * r).norm(), WithinRel((pinhole.at(cy, cx) * r).norm(), 1e-3));
    REQUIRE((barrel.at(5, 5) * r).norm() < 0.95 * (pinhole.at(5, 5) * r).norm());
}

// ------------------------------------------------------------------------------------------------------------------
// N -> F evidence -> R

namespace {
NormalFlow measurement(const Eigen::Vector2f &n, float speed, float variance) {
    NormalFlow m;
    m.valid = true;
    m.n = n.normalized();
    m.speed = speed;
    m.variance = variance;
    return m;
}

void checkSumsMatch(bool accumulate) {
    const int H = 12, W = 16;
    const MotionModel model = MotionModel::rotation(calibration(H, W), H, W);
    FlowEvidence evidence(H, W, 0.01, accumulate);
    RotationFromFlow rotation(0.01);
    std::mt19937 rng(1);
    std::uniform_int_distribution<int> py(0, H - 1), px(0, W - 1);
    std::uniform_real_distribution<float> u(-1, 1);
    double t = 0.0;
    for (int k = 0; k < 400; ++k) {
        t += 1e-4;
        const int y = py(rng), x = px(rng);
        rotation.fadeTo(t);
        rotation.apply(model.at(y, x), evidence.add(y, x, measurement({u(rng), u(rng)}, 100 + 50 * u(rng), 4 + u(rng)), t));
    }
    t += 0.003;
    rotation.fadeTo(t);
    Eigen::Matrix3d Hb = Eigen::Matrix3d::Zero();
    Eigen::Vector3d bb = Eigen::Vector3d::Zero();
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            evidence.fadeTo(y, x, t);
            Hb += model.at(y, x).transpose() * evidence.information[y * W + x] * model.at(y, x);
            bb += model.at(y, x).transpose() * evidence.vector[y * W + x];
        }
    REQUIRE((rotation.H - Hb).norm() <= 1e-9 * Hb.norm());
    REQUIRE((rotation.b - bb).norm() <= 1e-9 * bb.norm());
}
}

TEST_CASE("Running sums equal the faded evidence of every pixel (replace and accumulate)", "[timing][evidence]") {
    checkSumsMatch(false);
    checkSumsMatch(true);
}

TEST_CASE("RotationFromFlow recovers R from normal-flow measurements generated by R", "[timing][rotation]") {
    const int H = 60, W = 80;
    const MotionModel model = MotionModel::rotation(calibration(H, W), H, W);
    FlowEvidence evidence(H, W, 0.01, false);
    RotationFromFlow rotation(0.01);
    const Eigen::Vector3d R_true(0.7, -1.2, 2.5);   // rad/s, (y, x, z)
    std::mt19937 rng(2);
    std::uniform_real_distribution<float> angle(0.0f, 6.2831853f);
    double t = 0.0;
    for (int y = 0; y < H; y += 3)
        for (int x = 0; x < W; x += 3) {
            t += 1e-5;
            const float a = angle(rng);
            const Eigen::Vector2f n(std::sin(a), std::cos(a));
            const double s = n.cast<double>().dot(model.at(y, x) * R_true);
            const Eigen::Vector2f dir = s >= 0 ? n : Eigen::Vector2f(-n);   // speed is positive along n
            rotation.fadeTo(t);
            rotation.apply(model.at(y, x), evidence.add(y, x, measurement(dir, static_cast<float>(std::abs(s)), 1.0f), t));
        }
    const Eigen::Vector3f R = rotation.rotation();
    for (int k = 0; k < 3; ++k) REQUIRE_THAT(R(k), WithinAbs(R_true(k), 1e-3));
}

TEST_CASE("R is solved only when new evidence arrived and R is read", "[timing][rotation]") {
    const int H = 20, W = 20;
    const MotionModel model = MotionModel::rotation(calibration(H, W), H, W);
    FlowEvidence evidence(H, W, 0.01, false);
    RotationFromFlow rotation(0.01);
    std::mt19937 rng(3);
    std::uniform_real_distribution<float> u(-1, 1);
    double t = 0.0;
    for (int k = 0; k < 200; ++k) {
        t += 1e-5;
        const int y = k % H, x = (7 * k) % W;
        rotation.fadeTo(t);
        rotation.apply(model.at(y, x), evidence.add(y, x, measurement({u(rng), u(rng)}, 100, 1), t));
    }
    rotation.rotation();
    const long solves = rotation.solves;
    rotation.rotation();
    rotation.fadeTo(t + 0.005);   // fading alone leaves R unchanged
    rotation.rotation();
    REQUIRE(rotation.solves == solves);
    rotation.apply(model.at(3, 3), evidence.add(3, 3, measurement({1, 0}, 50, 1), t + 0.005));
    rotation.rotation();
    REQUIRE(rotation.solves == solves + 1);
}

TEST_CASE("Gyro-like measurements of R alone determine R", "[timing][rotation]") {
    RotationFromFlow rotation(0.01);
    rotation.fadeTo(0.0);
    rotation.addMeasurement(Eigen::Vector3f(0.1f, -0.2f, 0.3f), 100.0);
    const Eigen::Vector3f R = rotation.rotation();
    REQUIRE_THAT(R(0), WithinAbs(0.1, 1e-6));
    REQUIRE_THAT(R(1), WithinAbs(-0.2, 1e-6));
    REQUIRE_THAT(R(2), WithinAbs(0.3, 1e-6));
}

// ------------------------------------------------------------------------------------------------------------------
// R -> F

TEST_CASE("readFlow gives R's prediction without evidence and follows strong evidence along n", "[timing][flow]") {
    const int H = 10, W = 10;
    const MotionModel model = MotionModel::rotation(calibration(H, W), H, W);
    FlowEvidence evidence(H, W, 0.01, false);
    Tensor3f F(H, W, 2);
    F.setZero();
    const Eigen::Vector3f R(0.2f, -0.4f, 0.6f);
    const double kappa = 1.0 / (27.0 * 27.0);

    const Eigen::Vector2f prior = readFlow(evidence, model, R, kappa, 4, 5, 1.0, F);
    const Eigen::Vector2d expected = model.at(4, 5) * R.cast<double>();
    REQUIRE_THAT(prior(0), WithinRel(static_cast<float>(expected(0)), 1e-5f));
    REQUIRE_THAT(prior(1), WithinRel(static_cast<float>(expected(1)), 1e-5f));
    REQUIRE(F(4, 5, 0) == prior(0));   // the stored F holds the value

    evidence.add(4, 5, measurement({0, 1}, 300.0f, 1e-2f), 1.0);   // very precise: x component 300 px/s
    const Eigen::Vector2f measured = readFlow(evidence, model, R, kappa, 4, 5, 1.0, F);
    REQUIRE_THAT(measured(1), WithinAbs(300.0, 0.1));
    REQUIRE_THAT(measured(0), WithinRel(static_cast<float>(expected(0)), 1e-4f));   // along the edge: R's prediction
}

TEST_CASE("directionProbability: polarity and G confirm or contradict a normal flow's direction", "[timing][polarity]") {
    const Eigen::Vector2f n(0.0f, 1.0f);                  // moving along +x
    const Eigen::Vector2f bright_left(0.0f, -2.0f);       // brighter on the left
    // Moving along +x brings the brighter side over the pixel: V = -G . n > 0, so an ON event agrees, an OFF event
    // contradicts
    const double agree = timing::directionProbability(bright_left, n, +1, 0.5, 0.1);
    const double contradict = timing::directionProbability(bright_left, n, -1, 0.5, 0.1);
    REQUIRE(agree > 0.99);
    REQUIRE(contradict < 0.01);
    // G = 0 says nothing: the prior
    REQUIRE_THAT(timing::directionProbability(Eigen::Vector2f::Zero(), n, -1, 0.5, 0.1), WithinAbs(0.9, 1e-9));
    // G across n says nothing either
    REQUIRE_THAT(timing::directionProbability(Eigen::Vector2f(3.0f, 0.0f), n, -1, 0.5, 0.1), WithinAbs(0.9, 1e-6));
    // weak G (one sigma): between prior and certainty; q = Phi(1) = 0.8413
    const double q = 0.841344746;
    REQUIRE_THAT(timing::directionProbability(Eigen::Vector2f(0.0f, -0.5f), n, +1, 0.5, 0.1),
                 WithinAbs(q * 0.9 / (q * 0.9 + (1 - q) * 0.1), 1e-6));
    REQUIRE_THAT(timing::directionProbability(Eigen::Vector2f(0.0f, -0.5f), n, -1, 0.5, 0.1),
                 WithinAbs((1 - q) * 0.9 / ((1 - q) * 0.9 + q * 0.1), 1e-6));
}

TEST_CASE("Normal flow is weighted by its polarity check by default", "[deviations][polarity]") {
    const Parameters p = initParameters(0.0f, 1.0f, 0.01f);
    REQUIRE(p.polaritySign == "weight");
    REQUIRE_THAT(p.gradientSigma, WithinAbs(0.5, 1e-12));
    REQUIRE_THAT(p.flipPrior, WithinAbs(0.1, 1e-12));
}
