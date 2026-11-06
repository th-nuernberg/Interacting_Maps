#include "rotation_plot.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>

#include <opencv2/imgproc.hpp>

namespace {
// Dark-surface steps of the categorical palette (BGR): slot 1 blue for the reference, slot 2 orange for the network
const cv::Scalar SURFACE(25, 26, 26), GRID(58, 60, 60), TEXT(236, 239, 240), TEXT_2(183, 194, 195);
const cv::Scalar REFERENCE(229, 135, 57), ESTIMATE(38, 89, 217);
const float NaN = std::numeric_limits<float>::quiet_NaN();

float median(std::vector<float> values) {
    if (values.empty()) return NaN;
    const auto mid = values.begin() + values.size() / 2;
    std::nth_element(values.begin(), mid, values.end());
    return *mid;
}

void text(cv::Mat &img, const std::string &s, cv::Point at, const cv::Scalar &colour, double scale = 0.38) {
    cv::putText(img, s, at, cv::FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv::LINE_AA);
}

std::string fixed(double v, int digits) {
    std::ostringstream os;
    os.setf(std::ios::fixed);
    os.precision(digits);
    os << v;
    return os.str();
}

// One strip: a frame with a reference line, y range [lo, hi], x range [t0, t1].
struct Strip {
    cv::Rect area;
    double t0, t1, lo, hi;
    cv::Point point(double t, double v) const {
        const double fx = (t - t0) / (t1 - t0), fy = (v - lo) / (hi - lo);
        return {area.x + static_cast<int>(std::lround(fx * area.width)),
                area.y + area.height - static_cast<int>(std::lround(std::clamp(fy, 0.0, 1.0) * area.height))};
    }
    void draw(cv::Mat &img, const std::string &label, double reference) const {
        cv::rectangle(img, area, GRID, 1);
        const int yRef = point(t0, reference).y;
        cv::line(img, {area.x, yRef}, {area.x + area.width, yRef}, GRID, 1, cv::LINE_AA);
        text(img, label, {4, area.y + 11}, TEXT_2);
        text(img, fixed(hi, hi < 10 ? 1 : 0), {area.x + area.width + 3, area.y + 9}, TEXT_2, 0.32);
        text(img, fixed(lo, lo > -10 ? 1 : 0), {area.x + area.width + 3, area.y + area.height}, TEXT_2, 0.32);
    }
    void series(cv::Mat &img, const std::vector<cv::Point> &points, const cv::Scalar &colour) const {
        if (points.size() < 2) return;
        cv::Mat roi = img(area);
        std::vector<cv::Point> local;
        local.reserve(points.size());
        for (const auto &p : points) local.emplace_back(p.x - area.x, p.y - area.y);
        cv::polylines(roi, local, false, colour, 1, cv::LINE_AA);
    }
};
}  // namespace

// ------------------------------------------------------------------------------------------------------------------

GyroReference GyroReference::load(const std::filesystem::path &path, const Eigen::Matrix3f &R_cam_imu) {
    GyroReference ref;
    std::ifstream file(path);
    double t, a1, a2, a3, g1, g2, g3;
    while (file >> t >> a1 >> a2 >> a3 >> g1 >> g2 >> g3) {
        ref.time.push_back(t);
        ref.omega.push_back(R_cam_imu * Eigen::Vector3f(static_cast<float>(g1), static_cast<float>(g2), static_cast<float>(g3)));
    }
    return ref;
}

Eigen::Vector3f GyroReference::at(double t) const {
    if (time.empty()) return Eigen::Vector3f::Zero();
    if (t <= time.front()) return omega.front();
    if (t >= time.back()) return omega.back();
    const size_t j = static_cast<size_t>(std::upper_bound(time.begin(), time.end(), t) - time.begin());
    const double w = (t - time[j - 1]) / (time[j] - time[j - 1]);
    return static_cast<float>(1.0 - w) * omega[j - 1] + static_cast<float>(w) * omega[j];
}

// ------------------------------------------------------------------------------------------------------------------

RotationPlot::RotationPlot(double startTime, double endTime, double window, double statsWindow, float minReferenceSpeed)
    : startTime_(startTime), endTime_(endTime), window_(window), statsWindow_(statsWindow),
      minReferenceSpeed_(minReferenceSpeed) {}

void RotationPlot::add(double t, const Eigen::Vector3f &estimate, const Eigen::Vector3f &reference) {
    Sample s{t, estimate, reference, NaN, NaN, NaN, NaN};
    const float ne = estimate.norm(), nr = reference.norm();
    if (nr > minReferenceSpeed_ && ne > 0) {
        s.cos = estimate.dot(reference) / (ne * nr);
        s.ratio = ne / nr;
    }
    samples_.push_back(s);
    std::vector<float> cosines, ratios;
    for (auto it = samples_.rbegin(); it != samples_.rend() && it->t > t - statsWindow_; ++it) {
        if (!std::isnan(it->cos)) {
            cosines.push_back(it->cos);
            ratios.push_back(it->ratio);
        }
    }
    if (cosines.size() >= 10) {
        samples_.back().rollingCos = median(cosines);
        samples_.back().rollingRatio = median(ratios);
    }
}

cv::Mat RotationPlot::render(int width, int height) const {
    cv::Mat img(height, width, CV_8UC3, SURFACE);
    const int left = 44, right = 34, header = 22, gap = 6, labelRow = 14;
    const int plotWidth = width - left - right;
    // three axis strips, a time-label row, two statistics strips, a time-label row
    const int available = height - header - 4 * gap - 2 * labelRow;
    const int axisHeight = available * 2 / 9;
    const int statHeight = (available - 3 * axisHeight) / 2;

    // Legend and current values
    cv::line(img, {6, 13}, {26, 13}, REFERENCE, 2, cv::LINE_AA);
    text(img, "gyro (reference only)", {30, 17}, TEXT);
    cv::line(img, {190, 13}, {210, 13}, ESTIMATE, 2, cv::LINE_AA);
    text(img, "network, events only (rad/s)", {214, 17}, TEXT);
    if (!samples_.empty() && !std::isnan(samples_.back().rollingCos)) {
        text(img, "cos " + fixed(samples_.back().rollingCos, 3) + "  |w| ratio " + fixed(samples_.back().rollingRatio, 2),
             {width - 190, 17}, TEXT);
    }
    if (samples_.empty()) return img;

    // Axes: last `window` seconds, symmetric range from the visible values
    const double now = samples_.back().t, t0 = std::max(startTime_, now - window_);
    auto first = std::lower_bound(samples_.begin(), samples_.end(), t0,
                                  [](const Sample &s, double t) { return s.t < t; });
    const char *names[3] = {"wx", "wy", "wz"};
    for (int k = 0; k < 3; ++k) {
        double range = 1.0;
        for (auto it = first; it != samples_.end(); ++it) {
            range = std::max({range, static_cast<double>(std::abs(it->estimate(k))), static_cast<double>(std::abs(it->reference(k)))});
        }
        range = std::ceil(range * 1.1);
        const Strip strip{cv::Rect(left, header + k * (axisHeight + gap), plotWidth, axisHeight),
                          t0, std::max(now, t0 + 1e-3), -range, range};
        strip.draw(img, names[k], 0.0);
        std::vector<cv::Point> ref, est;
        for (auto it = first; it != samples_.end(); ++it) {
            ref.push_back(strip.point(it->t, it->reference(k)));
            est.push_back(strip.point(it->t, it->estimate(k)));
        }
        strip.series(img, ref, REFERENCE);
        strip.series(img, est, ESTIMATE);
    }
    const int axesBottom = header + 3 * axisHeight + 2 * gap;
    text(img, fixed(t0, 1) + " s", {left, axesBottom + 11}, TEXT_2, 0.32);
    text(img, fixed(now, 1) + " s", {left + plotWidth - 34, axesBottom + 11}, TEXT_2, 0.32);
    text(img, "last " + fixed(window_, 0) + " s", {left + plotWidth / 2 - 20, axesBottom + 11}, TEXT_2, 0.32);

    // Statistics over the whole run so far (1 s rolling medians)
    const int statsTop = axesBottom + labelRow + gap;
    const Strip cosStrip{cv::Rect(left, statsTop, plotWidth, statHeight), startTime_, endTime_, 0.8, 1.0};
    const Strip ratioStrip{cv::Rect(left, statsTop + statHeight + gap, plotWidth, statHeight), startTime_, endTime_, 0.0, 2.0};
    cosStrip.draw(img, "cos", 1.0);
    ratioStrip.draw(img, "ratio", 1.0);
    std::vector<cv::Point> cosPoints, ratioPoints;
    for (const auto &s : samples_) {
        if (std::isnan(s.rollingCos)) {   // break the line where the reference is too slow
            cosStrip.series(img, cosPoints, ESTIMATE);
            ratioStrip.series(img, ratioPoints, ESTIMATE);
            cosPoints.clear();
            ratioPoints.clear();
            continue;
        }
        cosPoints.push_back(cosStrip.point(s.t, s.rollingCos));
        ratioPoints.push_back(ratioStrip.point(s.t, s.rollingRatio));
    }
    cosStrip.series(img, cosPoints, ESTIMATE);
    ratioStrip.series(img, ratioPoints, ESTIMATE);
    const int bottom = statsTop + 2 * statHeight + gap;
    text(img, fixed(startTime_, 0) + " s", {left, bottom + 11}, TEXT_2, 0.32);
    text(img, "whole run, 1 s medians", {left + plotWidth / 2 - 60, bottom + 11}, TEXT_2, 0.32);
    text(img, fixed(endTime_, 0) + " s", {left + plotWidth - 24, bottom + 11}, TEXT_2, 0.32);
    return img;
}
