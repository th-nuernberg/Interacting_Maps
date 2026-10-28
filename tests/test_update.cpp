#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <interacting_maps.h>
#include <update.h>

using Catch::Matchers::WithinAbs;

TEST_CASE("crossProduct3x3 matches the right-handed cross product in (y, x, z) storage", "[conventions]") {
    // Storage order is (y, x, z). e_x x e_y = e_z.
    Tensor3f A(1, 1, 3), B(1, 1, 3), C(1, 1, 3);
    A.setValues({{{0.f, 1.f, 0.f}}});   // e_x
    B.setValues({{{1.f, 0.f, 0.f}}});   // e_y
    crossProduct3x3(A, B, C);
    REQUIRE_THAT(C(0, 0, 0), WithinAbs(0.f, 1e-6));
    REQUIRE_THAT(C(0, 0, 1), WithinAbs(0.f, 1e-6));
    REQUIRE_THAT(C(0, 0, 2), WithinAbs(1.f, 1e-6));
}

#include <file_operations.h>
#include <filesystem>

TEST_CASE("create_results_folder creates output/<name> without touching .gitignore", "[io]") {
    namespace fs = std::filesystem;
    const fs::path previous = fs::current_path();
    const fs::path sandbox = fs::temp_directory_path() / "im_tests_results_folder";
    fs::remove_all(sandbox);
    fs::create_directories(sandbox);
    fs::current_path(sandbox);

    const fs::path folder = create_results_folder("run_a/nested");

    const bool created = fs::is_directory(folder);
    const bool gitignore_written = fs::exists(sandbox / ".gitignore");
    fs::current_path(previous);
    fs::remove_all(sandbox);

    REQUIRE(created);
    REQUIRE_FALSE(gitignore_written);
}

TEST_CASE("read_single_line_txt keeps fractional and negative values", "[io]") {
    namespace fs = std::filesystem;
    const fs::path file = fs::temp_directory_path() / "im_tests_calib.txt";
    {
        std::ofstream out(file);
        out << "199.092366542 198.82882047 132.192071378 110.712660011 -0.368436311798 0.0\n";
    }
    std::vector<float> values;
    read_single_line_txt(file.string(), values);
    fs::remove(file);

    REQUIRE(values.size() == 6);
    REQUIRE_THAT(values[0], WithinAbs(199.092366542f, 1e-4));
    REQUIRE_THAT(values[3], WithinAbs(110.712660011f, 1e-4));
    REQUIRE_THAT(values[4], WithinAbs(-0.368436311798f, 1e-6));
}

namespace {
// Candidate of the flow constraint -V = A . B for A, projected onto the constraint line.
Vector2f flowConstraintCandidate(const Vector2f &A, const Vector2f &B, float V) {
    return A - B * (V + A.dot(B)) / B.squaredNorm();
}
}

TEST_CASE("update_GF blends each component of G towards its own candidate", "[update][flow]") {
    Tensor3f G(1, 1, 2), F(1, 1, 2);
    G.setValues({{{0.3f, -0.7f}}});
    F.setValues({{{1.5f, 0.5f}}});
    const float V = 0.4f, w = 0.5f;
    const Vector2f g0(G(0, 0, 0), G(0, 0, 1)), f(F(0, 0, 0), F(0, 0, 1));
    const Vector2f expected = (1 - w) * g0 + w * flowConstraintCandidate(g0, f, V);

    update_GF(G, V, F, 0, 0, 1.0f, w, 0.0f, 1e6f);

    REQUIRE_THAT(G(0, 0, 0), WithinAbs(expected(0), 1e-6));
    REQUIRE_THAT(G(0, 0, 1), WithinAbs(expected(1), 1e-6));
}

TEST_CASE("update_FG blends each component of F towards its own candidate", "[update][flow]") {
    Tensor3f F(1, 1, 2), G(1, 1, 2);
    F.setValues({{{0.3f, -0.7f}}});
    G.setValues({{{1.5f, 0.5f}}});
    const float V = 0.4f, w = 0.5f;
    const Vector2f f0(F(0, 0, 0), F(0, 0, 1)), g(G(0, 0, 0), G(0, 0, 1));
    const Vector2f expected = (1 - w) * f0 + w * flowConstraintCandidate(f0, g, V);

    update_FG(F, V, G, 0, 0, 1.0f, w, 0.0f, 1e6f);

    REQUIRE_THAT(F(0, 0, 0), WithinAbs(expected(0), 1e-6));
    REQUIRE_THAT(F(0, 0, 1), WithinAbs(expected(1), 1e-6));
}

TEST_CASE("splitTimeInterval returns slice boundaries including both ends", "[io]") {
    const std::vector<float> single = splitTimeInterval(0.0f, 0.5f, 1.0f);
    REQUIRE(single == std::vector<float>{0.0f, 0.5f});

    const std::vector<float> split = splitTimeInterval(0.0f, 2.5f, 1.0f);
    REQUIRE(split == std::vector<float>{0.0f, 1.0f, 2.0f, 2.5f});
}

namespace {
Calibration_Data smallCalibration(int height, int width, float f = 10.0f) {
    const std::vector<float> raw = {f, f, (width - 1) / 2.0f, (height - 1) / 2.0f, 0, 0, 0, 0, 0};
    return get_calibration_data(raw, height, width);
}

void runEventStep(OpticalFlowState &s, Parameters &p, std::vector<int> order, float V, int y, int x) {
    event_step(V, s.MI, s.delta_I, s.GIDiff, s.GIDiffGradient, s.F, s.G, s.R, s.CCM, s.dCdx, s.dCdy, s.A, s.B,
               s.Identity_minus_outerProducts, s.old_points, p, order, y, x);
}
}

TEST_CASE("event_step ignores unknown update ids instead of running update_FG", "[event_step]") {
    Parameters p = initParameters(0.0f, 1.0f, 0.1f);
    OpticalFlowState s = initOpticalFlowState(6, 6, p, smallCalibration(6, 6));
    s.G.setConstant(1.0f);
    const Tensor3f F_before = s.F;

    runEventStep(s, p, {7}, 10.0f, 3, 3);

    for (int i = 0; i < s.F.size(); ++i) {
        REQUIRE(s.F.data()[i] == F_before.data()[i]);
    }
}

TEST_CASE("Vector2Tensor keeps every element of the vector", "[conversions]") {
    VectorXf v(3);
    v << 1.0f, 2.0f, 3.0f;
    const Tensor1f t = Vector2Tensor(v);
    REQUIRE(t.dimension(0) == 3);
    REQUIRE(t(2) == 3.0f);
}

TEST_CASE("computeGradient uses replicate borders on all four sides", "[update][gradient]") {
    // Channel 0 is d/d(up) = I(y-1) - I(y+1) over 2, channel 1 is d/dx. Replicate border: the
    // missing neighbour is replaced by the border pixel itself.
    const int n = 5;
    Tensor2f Iy(n, n), Ix(n, n);
    Tensor3f gy(n, n, 2), gx(n, n, 2);
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            Iy(y, x) = float(y * y);
            Ix(y, x) = float(x * x);
        }
    }
    for (int i = 0; i < n; ++i) {
        computeGradient(Iy, gy, i, 2);
        computeGradient(Ix, gx, 2, i);
    }
    auto sq = [](int v) { return float(v * v); };
    REQUIRE_THAT(gy(0, 2, 0), WithinAbs((sq(0) - sq(1)) / 2.0f, 1e-6));
    REQUIRE_THAT(gy(2, 2, 0), WithinAbs((sq(1) - sq(3)) / 2.0f, 1e-6));
    REQUIRE_THAT(gy(n - 1, 2, 0), WithinAbs((sq(n - 2) - sq(n - 1)) / 2.0f, 1e-6));
    REQUIRE_THAT(gx(2, 0, 1), WithinAbs((sq(1) - sq(0)) / 2.0f, 1e-6));
    REQUIRE_THAT(gx(2, 2, 1), WithinAbs((sq(3) - sq(1)) / 2.0f, 1e-6));
    REQUIRE_THAT(gx(2, n - 1, 1), WithinAbs((sq(n - 1) - sq(n - 2)) / 2.0f, 1e-6));
}

TEST_CASE("computeGradient on a 3-tensor differentiates channel 0 along y and channel 1 along x", "[update][gradient]") {
    const int n = 5;
    Tensor3f D(n, n, 2), g(n, n, 2);
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            D(y, x, 0) = float(y * y);
            D(y, x, 1) = float(x * x);
        }
    }
    computeGradient(D, g, n - 1, n - 1);
    REQUIRE_THAT(g(n - 1, n - 1, 0), WithinAbs(float((n - 2) * (n - 2) - (n - 1) * (n - 1)) / 2.0f, 1e-6));
    REQUIRE_THAT(g(n - 1, n - 1, 1), WithinAbs(float((n - 1) * (n - 1) - (n - 2) * (n - 2)) / 2.0f, 1e-6));
}

TEST_CASE("G-I residual divergence vanishes when G equals the gradient of I", "[update][gradient]") {
    // GIDiff starts with garbage; the divergence at (y, x) reads residuals of the neighbours, so
    // those must be refreshed too, not only the event pixel.
    const int n = 5;
    Tensor3f G(n, n, 2), I_gradient(n, n, 2), GIDiff(n, n, 2), GIDiffGradient(n, n, 2);
    G.setRandom();
    I_gradient = G;
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            GIDiff(y, x, 0) = float(3 * y + x);   // stale, non-uniform residuals
            GIDiff(y, x, 1) = float(x * x - y);
        }
    }
    GIDiffGradient.setZero();

    updateGIDiffGradient(G, I_gradient, GIDiff, GIDiffGradient, 2, 2);

    REQUIRE_THAT(GIDiffGradient(2, 2, 0), WithinAbs(0.0f, 1e-6));
    REQUIRE_THAT(GIDiffGradient(2, 2, 1), WithinAbs(0.0f, 1e-6));
}

TEST_CASE("event_step moves G at the event pixel towards the current gradient of I", "[event_step]") {
    Parameters p = initParameters(0.0f, 1.0f, 0.1f);
    p.weight_GI = 1.0f;
    OpticalFlowState s = initOpticalFlowState(6, 6, p, smallCalibration(6, 6));
    for (int y = 0; y < 6; ++y) {
        for (int x = 0; x < 6; ++x) {
            s.MI(y, x) = 100.0f + 2.0f * float(x);   // d/dx = 2 everywhere inside
        }
    }

    runEventStep(s, p, {}, 0.0f, 3, 3);

    REQUIRE_THAT(s.G(3, 3, 1), WithinAbs(2.0f, 1e-5));
}

namespace {
// B = sum over all pixels of (I - C C^T) (C x m23(F)), computed independently of the running sums.
Vector3f expectedRSum(const OpticalFlowState &s) {
    Vector3f B = Vector3f::Zero();
    for (int y = 0; y < s.F.dimension(0); ++y) {
        for (int x = 0; x < s.F.dimension(1); ++x) {
            const Vector3f c(s.CCM(y, x, 0), s.CCM(y, x, 1), s.CCM(y, x, 2));
            const Vector3f cx(s.dCdx(y, x, 0), s.dCdx(y, x, 1), s.dCdx(y, x, 2));
            const Vector3f cy(s.dCdy(y, x, 0), s.dCdy(y, x, 1), s.dCdy(y, x, 2));
            const Vector3f f3 = s.F(y, x, 1) * cx + s.F(y, x, 0) * cy;
            // (y, x, z) storage: cross product of the underlying (x, y, z) vectors
            const Vector3f p(c(2) * f3(1) - c(1) * f3(2), c(0) * f3(2) - c(2) * f3(0), c(1) * f3(0) - c(0) * f3(1));
            B += (Matrix3f::Identity() - c * c.transpose()) * p;
        }
    }
    return B;
}
}

TEST_CASE("F->R running sum covers every pixel's current flow after initialisation", "[update][rotation]") {
    Parameters p = initParameters(0.0f, 1.0f, 0.1f);
    OpticalFlowState s = initOpticalFlowState(8, 8, p, smallCalibration(8, 8));
    const Vector3f expected = expectedRSum(s);
    REQUIRE(s.B.isApprox(expected, 1e-4f));
}

TEST_CASE("F->R recovers R from a flow field generated by R after a global F<-R pass", "[update][rotation]") {
    Parameters p = initParameters(0.0f, 1.0f, 0.1f);
    OpticalFlowState s = initOpticalFlowState(8, 8, p, smallCalibration(8, 8));
    Tensor1f R_true(3);
    R_true.setValues({0.01f, -0.02f, 0.03f});

    update_FR(s.F, s.CCM, s.dCdx, s.dCdy, R_true, 1.0f, 0.0f, 1e6f);
    rebuild_R_sums(s.F, s.CCM, s.dCdx, s.dCdy, s.B, s.Identity_minus_outerProducts, s.old_points);
    REQUIRE(s.B.isApprox(expectedRSum(s), 1e-4f));

    s.R.setZero();
    update_RF(s.R, s.F, s.CCM, s.dCdx, s.dCdy, s.A, s.B, s.Identity_minus_outerProducts, s.old_points, 1.0f, 4, 4);
    for (int i = 0; i < 3; ++i) {
        REQUIRE_THAT(s.R(i), WithinAbs(R_true(i), 1e-4));
    }
}

TEST_CASE("find_C uses the principal point and per-axis focal lengths", "[calibration]") {
    const int W = 8, H = 6;
    const float fx = 4.0f, fy = 3.0f, cx = 3.0f, cy = 2.0f;
    Tensor3f C(H, W, 3), Cx(H, W, 3), Cy(H, W, 3);
    find_C(W, H, fx, fy, cx, cy, C, Cx, Cy);

    // Principal point looks straight ahead.
    REQUIRE_THAT(C(2, 3, 0), WithinAbs(0.0f, 1e-6));
    REQUIRE_THAT(C(2, 3, 1), WithinAbs(0.0f, 1e-6));
    REQUIRE_THAT(C(2, 3, 2), WithinAbs(1.0f, 1e-6));
    // fx pixels to the right of it: 45 degrees in x.
    REQUIRE_THAT(C(2, 7, 1), WithinAbs(C(2, 7, 2), 1e-6));
    REQUIRE_THAT(C(2, 7, 0), WithinAbs(0.0f, 1e-6));
    // fy rows below it: 45 degrees in y (stored y points up, so negative).
    REQUIRE_THAT(C(5, 3, 0), WithinAbs(-C(5, 3, 2), 1e-6));
    REQUIRE_THAT(C(5, 3, 1), WithinAbs(0.0f, 1e-6));
}

TEST_CASE("get_calibration_data pairs the vertical view angle with fy", "[calibration]") {
    const Calibration_Data d = get_calibration_data({100.0f, 50.0f, 0, 0, 0, 0, 0, 0, 0}, 100, 200);
    REQUIRE_THAT(d.view_angles[0], WithinAbs(2.0f * std::atan(100.0f / (2.0f * 50.0f)), 1e-6));
    REQUIRE_THAT(d.view_angles[1], WithinAbs(2.0f * std::atan(200.0f / (2.0f * 100.0f)), 1e-6));
}
