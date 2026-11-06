#include "timing.h"

#include <cmath>
#include <stdexcept>

namespace timing {

// ------------------------------------------------------------------------------------------------------------------
// O

OnsetMap::OnsetMap(int height, int width)
    : height(height), width(width),
      stamp(static_cast<size_t>(height) * width * 2, 0.0),
      lastEvent(static_cast<size_t>(height) * width * 2, -std::numeric_limits<double>::infinity()),
      resource(static_cast<size_t>(height) * width * 2, 1.0) {}

void LatestEventOnset::update(OnsetMap &onset, int y, int x, int polarity, double t) const {
    const int i = onset.index(y, x, polarity);
    onset.stamp[i] = t;
    onset.lastEvent[i] = t;
}

void HardGapOnset::update(OnsetMap &onset, int y, int x, int polarity, double t) const {
    const int i = onset.index(y, x, polarity);
    if (t - onset.lastEvent[i] > gap) {
        onset.stamp[i] = t;
    }
    onset.lastEvent[i] = t;
}

void DepressionOnset::update(OnsetMap &onset, int y, int x, int polarity, double t) const {
    const int i = onset.index(y, x, polarity);
    if (std::isinf(onset.lastEvent[i])) {
        onset.stamp[i] = t;
        onset.resource[i] = 1.0 - use;
    } else {
        const double recovered = 1.0 - (1.0 - onset.resource[i]) * std::exp(-(t - onset.lastEvent[i]) / tau);
        onset.stamp[i] += recovered * (t - onset.stamp[i]);
        onset.resource[i] = (1.0 - use) * recovered;
    }
    onset.lastEvent[i] = t;
}

std::unique_ptr<OnsetRule> makeOnsetRule(const std::string &name, double parameter, double use) {
    if (name == "latest") return std::make_unique<LatestEventOnset>();
    if (name == "gap") return std::make_unique<HardGapOnset>(parameter);
    if (name == "depression") return std::make_unique<DepressionOnset>(parameter, use);
    throw std::invalid_argument("unknown onset rule: " + name);
}

// ------------------------------------------------------------------------------------------------------------------
// O -> N

NormalFlow fitNormalFlow(const OnsetMap &onset, int y, int x, int polarity, double t, const NormalFlowSettings &settings) {
    const double reference = onset.stamp[onset.index(y, x, polarity)];
    double sxx = 0, sxy = 0, syy = 0, sxt = 0, syt = 0, stt = 0;
    int n = 0;
    for (int dy = -settings.radius; dy <= settings.radius; ++dy) {
        for (int dx = -settings.radius; dx <= settings.radius; ++dx) {
            if (dx == 0 && dy == 0) continue;
            const int yq = y + dy, xq = x + dx;
            if (yq < 0 || yq >= onset.height || xq < 0 || xq >= onset.width) continue;
            const int q = onset.index(yq, xq, polarity);
            const double last = onset.lastEvent[q];
            if (!(last <= t && last > t - settings.horizon)) continue;
            const double d = onset.stamp[q] - reference;
            sxx += dx * dx; sxy += dx * dy; syy += dy * dy;
            sxt += dx * d; syt += dy * d; stt += d * d;
            ++n;
        }
    }
    NormalFlow result;
    result.neighbours = n;
    const double det = sxx * syy - sxy * sxy;
    if (n < 4 || det <= 1e-9) return result;
    const double a = (syy * sxt - sxy * syt) / det;   // d(time)/dx
    const double b = (sxx * syt - sxy * sxt) / det;   // d(time)/dy
    const double g2 = a * a + b * b;
    if (g2 <= 0) return result;
    const double g = std::sqrt(g2);
    const double speed = 1.0 / g;
    if (speed >= settings.maxSpeed) return result;
    const double residual = std::max(stt - a * sxt - b * syt, 0.0);
    const double sr2 = std::max(residual / std::max(n - 2, 1), settings.timeFloor * settings.timeFloor);
    const double nx = a / g, ny = b / g;
    const double varSlope = sr2 * (nx * nx * syy - 2 * nx * ny * sxy + ny * ny * sxx) / det;
    result.valid = true;
    result.n = Eigen::Vector2f(static_cast<float>(ny), static_cast<float>(nx));
    result.speed = static_cast<float>(speed);
    result.variance = static_cast<float>(varSlope * speed * speed * speed * speed);   // sigma_s = sigma_g s^2
    return result;
}

// ------------------------------------------------------------------------------------------------------------------
// Motion model

namespace {
struct Lens {
    double fx, fy, cx, cy, k1, k2, p1, p2, k3;

    void distort(double xn, double yn, double &u, double &v) const {
        const double r2 = xn * xn + yn * yn;
        const double radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
        const double xd = xn * radial + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn);
        const double yd = yn * radial + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn;
        u = fx * xd + cx;
        v = fy * yd + cy;
    }

    void undistort(double u, double v, double &xn, double &yn) const {
        const double xd = (u - cx) / fx, yd = (v - cy) / fy;
        xn = xd; yn = yd;
        for (int i = 0; i < 20; ++i) {
            const double r2 = xn * xn + yn * yn;
            const double radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
            const double dx = 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn);
            const double dy = p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn;
            xn = (xd - dx) / radial;
            yn = (yd - dy) / radial;
        }
    }
};
}  // namespace

MotionModel MotionModel::rotation(const Calibration_Data &calibration, int height, int width) {
    const auto &K = calibration.camera_matrix;
    const auto &d = calibration.distortion_coefficients;
    auto coefficient = [&](size_t i) { return i < d.size() ? static_cast<double>(d[i]) : 0.0; };
    const Lens lens{K(0, 0), K(1, 1), K(0, 2), K(1, 2), coefficient(0), coefficient(1), coefficient(2),
                    coefficient(3), coefficient(4)};

    MotionModel model;
    model.height = height;
    model.width = width;
    model.M.resize(static_cast<size_t>(height) * width);
    const double step = 1e-6;   // rad; small enough for the linearisation, large enough for double precision
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double xn, yn;
            lens.undistort(x, y, xn, yn);
            const Eigen::Vector3d P(xn, yn, 1.0);
            Eigen::Matrix<double, 2, 3> M;
            for (int k = 0; k < 3; ++k) {
                // R basis vector in (y, x, z) storage -> camera angular velocity omega = -R in (x, y, z)
                Eigen::Vector3d R_yxz = Eigen::Vector3d::Zero();
                R_yxz(k) = 1.0;
                const Eigen::Vector3d omega = -Eigen::Vector3d(R_yxz(1), R_yxz(0), R_yxz(2));
                // static points move as dP/dt = -omega x P in the camera frame
                const Eigen::Vector3d P2 = P - omega.cross(P) * step;
                double u2, v2;
                lens.distort(P2(0) / P2(2), P2(1) / P2(2), u2, v2);
                M(0, k) = (v2 - y) / step;   // channel 0: y
                M(1, k) = (u2 - x) / step;   // channel 1: x
            }
            model.M[y * width + x] = M;
        }
    }
    return model;
}

// ------------------------------------------------------------------------------------------------------------------
// F evidence

FlowEvidence::FlowEvidence(int height, int width, double lifetime, bool accumulate)
    : height(height), width(width), lifetime(lifetime), accumulate(accumulate),
      information(static_cast<size_t>(height) * width, Eigen::Matrix2d::Zero()),
      vector(static_cast<size_t>(height) * width, Eigen::Vector2d::Zero()),
      time(static_cast<size_t>(height) * width, -std::numeric_limits<double>::infinity()) {}

void FlowEvidence::fadeTo(int y, int x, double t) {
    const int p = y * width + x;
    if (t <= time[p]) return;
    const double keep = std::isinf(time[p]) ? 0.0 : std::exp(-(t - time[p]) / lifetime);
    information[p] *= keep;
    vector[p] *= keep;
    time[p] = t;
}

EvidenceChange FlowEvidence::add(int y, int x, const NormalFlow &measurement, double t) {
    EvidenceChange change;
    if (!measurement.valid) return change;
    fadeTo(y, x, t);
    const int p = y * width + x;
    const Eigen::Vector2d n = measurement.n.cast<double>();
    const double w = 1.0 / std::max(static_cast<double>(measurement.variance), 1e-12);
    const Eigen::Matrix2d newInformation = w * n * n.transpose();
    const Eigen::Vector2d newVector = w * n * static_cast<double>(measurement.speed);
    if (accumulate) {
        change.dInformation = newInformation;
        change.dVector = newVector;
        information[p] += newInformation;
        vector[p] += newVector;
    } else {
        change.dInformation = newInformation - information[p];
        change.dVector = newVector - vector[p];
        information[p] = newInformation;
        vector[p] = newVector;
    }
    return change;
}

// ------------------------------------------------------------------------------------------------------------------
// F evidence -> R

void RotationFromFlow::fadeTo(double t) {
    if (t <= time) return;
    if (!std::isinf(time)) {
        // H and b fade by the same factor, so R = H^-1 b is unchanged: no new solve needed
        const double keep = std::exp(-(t - time) / lifetime);
        H *= keep;
        b *= keep;
    }
    time = t;
}

void RotationFromFlow::apply(const Eigen::Matrix<double, 2, 3> &M, const EvidenceChange &change) {
    H += M.transpose() * change.dInformation * M;
    b += M.transpose() * change.dVector;
    dirty = true;
}

void RotationFromFlow::addMeasurement(const Eigen::Vector3f &R_measured, double weight) {
    H += weight * Eigen::Matrix3d::Identity();
    b += weight * R_measured.cast<double>();
    dirty = true;
}

const Eigen::Vector3f &RotationFromFlow::rotation() {
    if (!dirty) return R;
    dirty = false;
    // Solve only when the evidence constrains all three components; otherwise keep the previous R.
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen(H, Eigen::EigenvaluesOnly);
    const double largest = eigen.eigenvalues().maxCoeff();
    if (largest <= 0 || eigen.eigenvalues().minCoeff() <= 1e-9 * largest) return R;
    R = H.ldlt().solve(b).cast<float>();
    ++solves;
    return R;
}

// ------------------------------------------------------------------------------------------------------------------
// R -> F

Eigen::Vector2f readFlow(FlowEvidence &evidence, const MotionModel &model, const Eigen::Vector3f &R, double kappa,
                         int y, int x, double t, Tensor3f &F) {
    evidence.fadeTo(y, x, t);
    const int p = y * evidence.width + x;
    const Eigen::Vector2d prior = model.at(y, x) * R.cast<double>();
    const Eigen::Matrix2d A = evidence.information[p] + kappa * Eigen::Matrix2d::Identity();
    const Eigen::Vector2f value = A.ldlt().solve(evidence.vector[p] + kappa * prior).cast<float>();
    F(y, x, 0) = value(0);
    F(y, x, 1) = value(1);
    return value;
}

void readFlowAll(FlowEvidence &evidence, const MotionModel &model, const Eigen::Vector3f &R, double kappa, double t,
                 Tensor3f &F) {
    for (int y = 0; y < evidence.height; ++y) {
        for (int x = 0; x < evidence.width; ++x) {
            readFlow(evidence, model, R, kappa, y, x, t, F);
        }
    }
}

}  // namespace timing
