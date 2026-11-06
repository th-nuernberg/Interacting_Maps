#include "contrast.h"

#include <stdexcept>

ContrastSteps::ContrastSteps(const std::string &mode, int height, int width, double ratio, double prior)
    : mode_(mode), width_(width), fixedRatio_(ratio), prior_(prior) {
    if (mode != "equal" && mode != "fixed" && mode != "global" && mode != "pixel") {
        throw std::invalid_argument("unknown contrast step mode: " + mode);
    }
    if (mode == "pixel") {
        on_.assign(static_cast<size_t>(height) * width, 0.0f);
        off_.assign(static_cast<size_t>(height) * width, 0.0f);
    }
}

double ContrastSteps::globalRatio() const {
    return (offTotal_ + 1.0) / (onTotal_ + 1.0);   // one pseudo-event each: 1 before any data
}

double ContrastSteps::ratio(int y, int x) const {
    if (mode_ == "equal") return 1.0;
    if (mode_ == "fixed") return fixedRatio_;
    if (mode_ == "global") return globalRatio();
    const size_t i = static_cast<size_t>(y) * width_ + x;
    return (off_[i] + prior_ * globalRatio()) / (on_[i] + prior_);
}

float ContrastSteps::step(int y, int x, int polarity) {
    const float s = polarity > 0 ? static_cast<float>(ratio(y, x)) : -1.0f;
    (polarity > 0 ? onTotal_ : offTotal_) += 1.0;
    if (mode_ == "pixel") {
        const size_t i = static_cast<size_t>(y) * width_ + x;
        (polarity > 0 ? on_[i] : off_[i]) += 1.0f;
    }
    return s;
}
