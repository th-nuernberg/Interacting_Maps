#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <stdexcept>

#include "contrast.h"

using Catch::Matchers::WithinAbs;

TEST_CASE("Equal contrast steps are +-1, a fixed ratio scales only ON events", "[contrast]") {
    ContrastSteps equal("equal", 2, 2);
    REQUIRE(equal.step(0, 0, 1) == 1.0f);
    REQUIRE(equal.step(0, 0, -1) == -1.0f);
    ContrastSteps fixed("fixed", 2, 2, 1.4);
    REQUIRE_THAT(fixed.step(1, 1, 1), WithinAbs(1.4, 1e-6));
    REQUIRE(fixed.step(1, 1, -1) == -1.0f);
}

TEST_CASE("Global contrast ratio is the OFF/ON count ratio, used causally", "[contrast]") {
    ContrastSteps c("global", 2, 2);
    REQUIRE(c.step(0, 0, 1) == 1.0f);            // no data yet: ratio 1
    for (int i = 0; i < 2999; ++i) c.step(0, 1, -1);
    for (int i = 0; i < 1999; ++i) c.step(1, 0, 1);
    // 2000 ON, 2999 OFF so far: (2999 + 1) / (2000 + 1)
    REQUIRE_THAT(c.step(1, 1, 1), WithinAbs(3000.0 / 2001.0, 1e-5));
}

TEST_CASE("Per-pixel contrast ratio moves from the global ratio to the pixel's own counts", "[contrast]") {
    ContrastSteps c("pixel", 1, 2, 1.0, 10.0);
    for (int i = 0; i < 1000; ++i) { c.step(0, 0, 1); c.step(0, 0, -1); }   // pixel (0,0): ratio 1
    for (int i = 0; i < 1000; ++i) { c.step(0, 1, 1); c.step(0, 1, -1); c.step(0, 1, -1); }   // pixel (0,1): 2
    const double g = c.globalRatio();
    REQUIRE_THAT(g, WithinAbs(3001.0 / 2001.0, 1e-9));
    REQUIRE_THAT(c.ratio(0, 0), WithinAbs((1000 + 10 * g) / (1000 + 10), 1e-5));
    REQUIRE_THAT(c.ratio(0, 1), WithinAbs((2000 + 10 * g) / (1000 + 10), 1e-5));
}

TEST_CASE("Unknown contrast step modes are rejected", "[contrast]") {
    REQUIRE_THROWS_AS(ContrastSteps("adaptive", 1, 1), std::invalid_argument);
}
