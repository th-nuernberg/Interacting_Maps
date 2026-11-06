#pragma once
// Contrast thresholds: how much log intensity one event stands for. I is kept in units of the OFF threshold c_OFF:
// an OFF event is a step of -1, an ON event a step of +c_ON / c_OFF.
//
// The sensor's real thresholds are unknown (docs/assumptions.md, W3). The ratio can be given (e.g. from a threshold
// calibration or the bias settings) or estimated from the events, assuming each pixel's log intensity returns to
// where it started: then c_ON * (ON events) = c_OFF * (OFF events).

#include <string>
#include <vector>

class ContrastSteps {
public:
    // mode "equal":  c_ON / c_OFF = 1 (the former behaviour)
    //      "fixed":  c_ON / c_OFF = ratio
    //      "global": OFF count / ON count over all pixels so far
    //      "pixel":  per pixel, (OFF + prior * global) / (ON + prior): `prior` pseudo-events of the global ratio
    ContrastSteps(const std::string &mode, int height, int width, double ratio = 1.0, double prior = 100.0);

    // The step this event adds to I, from the counts before it; then counts the event (causal).
    float step(int y, int x, int polarity);
    // c_ON / c_OFF at this pixel with the current counts
    double ratio(int y, int x) const;
    double globalRatio() const;
    const std::string &mode() const { return mode_; }

private:
    std::string mode_;
    int width_;
    double fixedRatio_, prior_;
    double onTotal_ = 0.0, offTotal_ = 0.0;
    std::vector<float> on_, off_;
};
