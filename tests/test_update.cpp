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
    // Channel 0 is d/dy (down) = I(y+1) - I(y-1) over 2, channel 1 is d/dx. Replicate border: the
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
    REQUIRE_THAT(gy(0, 2, 0), WithinAbs((sq(1) - sq(0)) / 2.0f, 1e-6));
    REQUIRE_THAT(gy(2, 2, 0), WithinAbs((sq(3) - sq(1)) / 2.0f, 1e-6));
    REQUIRE_THAT(gy(n - 1, 2, 0), WithinAbs((sq(n - 1) - sq(n - 2)) / 2.0f, 1e-6));
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
    REQUIRE_THAT(g(n - 1, n - 1, 0), WithinAbs(float((n - 1) * (n - 1) - (n - 2) * (n - 2)) / 2.0f, 1e-6));
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
    // fy rows below it: 45 degrees in y (y points down).
    REQUIRE_THAT(C(5, 3, 0), WithinAbs(C(5, 3, 2), 1e-6));
    REQUIRE_THAT(C(5, 3, 1), WithinAbs(0.0f, 1e-6));
}

TEST_CASE("get_calibration_data pairs the vertical view angle with fy", "[calibration]") {
    const Calibration_Data d = get_calibration_data({100.0f, 50.0f, 0, 0, 0, 0, 0, 0, 0}, 100, 200);
    REQUIRE_THAT(d.view_angles[0], WithinAbs(2.0f * std::atan(100.0f / (2.0f * 50.0f)), 1e-6));
    REQUIRE_THAT(d.view_angles[1], WithinAbs(2.0f * std::atan(200.0f / (2.0f * 100.0f)), 1e-6));
}

#include <Eigen/Geometry>

namespace {
// Pixel motion of the static scene point seen at pixel (row, col) when the camera rotates by angle about
// axis (camera frame x right, y down, z forward), from explicit pinhole projection.
Vector2f projectedPixelMotion(float row, float col, const Vector3f &axis, float angle,
                              float fx, float fy, float cx, float cy) {
    const Vector3f P((col - cx) / fx, (row - cy) / fy, 1.0f);   // (x, y, z)
    // The camera rotates by +angle, so the point rotates by -angle in camera coordinates.
    const Vector3f P2 = AngleAxisf(-angle, axis.normalized()) * P;
    const float col2 = fx * P2.x() / P2.z() + cx;
    const float row2 = fy * P2.y() / P2.z() + cy;
    return {row2 - row, col2 - col};   // (dy, dx), matching F's channel order
}
}

TEST_CASE("F = m32(R x C) with R = -(camera rotation) reproduces projected pixel motion", "[conventions]") {
    const int W = 21, H = 15;
    const float fx = 20.0f, fy = 18.0f, cx = 10.0f, cy = 7.0f, angle = 1e-3f;
    Tensor3f C(H, W, 3), Cx(H, W, 3), Cy(H, W, 3), F(H, W, 2), RxC(H, W, 3);
    find_C(W, H, fx, fy, cx, cy, C, Cx, Cy);

    const std::vector<Vector3f> axes = {Vector3f::UnitX(), Vector3f::UnitY(), Vector3f::UnitZ(), Vector3f(1, -2, 0.5f)};
    for (const Vector3f &axis_xyz : axes) {
        const Vector3f rot = -angle * axis_xyz.normalized();
        Tensor1f R(3);
        R.setValues({rot.y(), rot.x(), rot.z()});   // (y, x, z) storage
        crossProduct1x3(R, C, RxC);
        m32(RxC, Cx, Cy, F);

        for (const auto &pixel : std::vector<std::array<int, 2>>{{7, 10}, {2, 3}, {12, 18}}) {
            const int row = pixel[0], col = pixel[1];
            const Vector2f expected = projectedPixelMotion(float(row), float(col), axis_xyz, angle, fx, fy, cx, cy);
            INFO("axis " << axis_xyz.transpose() << " pixel " << row << "," << col);
            REQUIRE_THAT(F(row, col, 0), WithinAbs(expected(0), 2e-4));
            REQUIRE_THAT(F(row, col, 1), WithinAbs(expected(1), 2e-4));
        }
    }
}

TEST_CASE("Timing mode is the default; the legacy pipeline is opt-in", "[deviations]") {
    REQUIRE(initParameters(0.0f, 1.0f, 0.1f).rotationFromTiming);
}

TEST_CASE("F and R start at zero by default", "[determinism][deviations]") {
    const Parameters p = initParameters(0.0f, 1.0f, 0.1f);
    REQUIRE_FALSE(p.randomInitialState);
    const OpticalFlowState s = initOpticalFlowState(6, 6, p, smallCalibration(6, 6));
    for (int i = 0; i < s.F.size(); ++i) REQUIRE(s.F.data()[i] == 0.0f);
    for (int i = 0; i < 3; ++i) REQUIRE(s.R(i) == 0.0f);
}

TEST_CASE("initOpticalFlowState is reproducible for a given seed", "[determinism]") {
    Parameters p = initParameters(0.0f, 1.0f, 0.1f);
    p.randomInitialState = true;
    p.seed = 42;
    const OpticalFlowState a = initOpticalFlowState(6, 6, p, smallCalibration(6, 6));
    const OpticalFlowState b = initOpticalFlowState(6, 6, p, smallCalibration(6, 6));
    p.seed = 43;
    const OpticalFlowState c = initOpticalFlowState(6, 6, p, smallCalibration(6, 6));

    bool any_difference = false;
    for (int i = 0; i < a.F.size(); ++i) {
        REQUIRE(a.F.data()[i] == b.F.data()[i]);
        REQUIRE(a.F.data()[i] >= -1.0f);
        REQUIRE(a.F.data()[i] <= 1.0f);
        any_difference |= a.F.data()[i] != c.F.data()[i];
    }
    for (int i = 0; i < 3; ++i) {
        REQUIRE(a.R(i) == b.R(i));
    }
    REQUIRE(any_difference);
}

TEST_CASE("clampMaps switches the F/G clamping off", "[event_step][deviations]") {
    for (const bool clamp : {true, false}) {
        Parameters p = initParameters(0.0f, 1.0f, 0.1f);
        p.clampMaps = clamp;
        p.weight_IV = 0.0f;
        p.weight_GI = 0.0f;
        p.weight_IG = 0.0f;
        p.weight_FG = 1.0f;
        OpticalFlowState s = initOpticalFlowState(6, 6, p, smallCalibration(6, 6));
        s.F.setZero();
        s.G.setZero();
        s.G(3, 3, 1) = 0.01f;   // flat region: the projected flow candidate is huge

        runEventStep(s, p, {0}, 10.0f, 3, 3);

        // candidate F_x = -V / G_x = -1000
        REQUIRE_THAT(s.F(3, 3, 1), WithinAbs(clamp ? -p.gamma : -1000.0f, 1e-2));
    }
}

TEST_CASE("update_RIMU turns a gyro reading into the R that reproduces the projected flow", "[conventions][imu]") {
    const int W = 21, H = 15;
    const float fx = 20.0f, fy = 18.0f, cx = 10.0f, cy = 7.0f;
    const float dt = 1e-3f;                                      // R is rotation per dt in this test
    Tensor3f C(H, W, 3), Cx(H, W, 3), Cy(H, W, 3), F(H, W, 2), RxC(H, W, 3);
    find_C(W, H, fx, fy, cx, cy, C, Cx, Cy);

    const Matrix3f R_cam_imu = AngleAxisf(0.5f * float(M_PI), Vector3f::UnitZ()).toRotationMatrix();
    const Vector3f omega_cam(0.3f, -0.6f, 0.9f);                 // rad/s, camera frame (x, y, z)
    const Vector3f omega_imu = R_cam_imu.transpose() * omega_cam;  // what the gyro reports

    Tensor1f R(3);
    R.setZero();
    update_RIMU(R, {omega_imu.x(), omega_imu.y(), omega_imu.z()}, 1.0f, R_cam_imu, dt);

    crossProduct1x3(R, C, RxC);
    m32(RxC, Cx, Cy, F);
    const float angle = omega_cam.norm() * dt;
    for (const auto &pixel : std::vector<std::array<int, 2>>{{7, 10}, {2, 3}, {12, 18}}) {
        const Vector2f expected = projectedPixelMotion(float(pixel[0]), float(pixel[1]), omega_cam, angle, fx, fy, cx, cy);
        REQUIRE_THAT(F(pixel[0], pixel[1], 0), WithinAbs(expected(0), 2e-4));
        REQUIRE_THAT(F(pixel[0], pixel[1], 1), WithinAbs(expected(1), 2e-4));
    }
}

TEST_CASE("decayNeighbourhood brings every pixel read by event_step's gradients to the event time", "[update][decay]") {
    // A uniform potential set at t = 0 decays uniformly, so its gradient must stay zero.
    // Decaying only the event pixel (the old behaviour) produced spurious gradients around it.
    Parameters p = initParameters(0.0f, 1.0f, 0.1f);
    OpticalFlowState s = initOpticalFlowState(9, 9, p, smallCalibration(9, 9));
    s.MI.setConstant(200.0f);
    s.decayTimeSurface.setConstant(0.0f);
    const float t = p.decayParam;

    decayNeighbourhood(s.MI, s.decayTimeSurface, 4, 4, t, p.neutralPotential, p.decayParam);

    const float expected = p.neutralPotential + (200.0f - p.neutralPotential) * std::exp(-1.0f);
    REQUIRE_THAT(s.MI(4, 6), WithinAbs(expected, 1e-3));   // distance 2: decayed
    REQUIRE_THAT(s.MI(5, 5), WithinAbs(expected, 1e-3));
    REQUIRE_THAT(s.MI(4, 7), WithinAbs(200.0f, 1e-6));     // distance 3: not read, left alone
    const int stencil[5][2] = {{4, 4}, {3, 4}, {5, 4}, {4, 3}, {4, 5}};
    for (const auto &c : stencil) {
        computeGradient(s.MI, s.delta_I, c[0], c[1]);
        REQUIRE_THAT(s.delta_I(c[0], c[1], 0), WithinAbs(0.0f, 1e-4));
        REQUIRE_THAT(s.delta_I(c[0], c[1], 1), WithinAbs(0.0f, 1e-4));
    }
}

TEST_CASE("decayedCopy decays every pixel and leaves the state untouched", "[update][decay]") {
    Parameters p = initParameters(0.0f, 1.0f, 0.1f);
    OpticalFlowState s = initOpticalFlowState(4, 4, p, smallCalibration(4, 4));
    s.MI.setConstant(200.0f);
    s.decayTimeSurface.setConstant(0.0f);

    const Tensor2f shown = decayedCopy(s.MI, s.decayTimeSurface, p.decayParam, p.neutralPotential, p.decayParam);

    const float expected = p.neutralPotential + (200.0f - p.neutralPotential) * std::exp(-1.0f);
    REQUIRE_THAT(shown(0, 0), WithinAbs(expected, 1e-3));
    REQUIRE_THAT(shown(3, 3), WithinAbs(expected, 1e-3));
    REQUIRE_THAT(s.MI(3, 3), WithinAbs(200.0f, 1e-6));
    REQUIRE_THAT(s.decayTimeSurface(3, 3), WithinAbs(0.0f, 1e-9));
}

TEST_CASE("relaxSilentG without G <- I equals repeated V = 0 update_GF steps, one per time step", "[update][silent]") {
    // The asynchronous closed form must reproduce what the paper's synchronous update does at a pixel
    // without events: n time steps of update_GF with V = 0.
    Parameters p = initParameters(0.0f, 1.0f, 0.01f);
    Tensor3f F(1, 1, 2), G_steps(1, 1, 2), G_lazy(1, 1, 2), g(1, 1, 2);
    F.setValues({{{0.6f, 0.8f}}});
    G_steps.setValues({{{2.0f, -1.0f}}});
    G_lazy = G_steps;
    g.setZero();
    const int n = 7;
    const float w = 0.2f;
    for (int i = 0; i < n; ++i) {
        update_GF(G_steps, 0.0f, F, 0, 0, p.lr, w, 0.0f, std::numeric_limits<float>::infinity());
    }
    Tensor2f GTime(1, 1);
    GTime.setConstant(0.0f);

    relaxSilentG(G_lazy, F, g, GTime, 0, 0, n * p.time_step, 0.0f, -std::log(1.0f - w) / p.time_step);

    REQUIRE_THAT(G_lazy(0, 0, 0), WithinAbs(G_steps(0, 0, 0), 1e-5));
    REQUIRE_THAT(G_lazy(0, 0, 1), WithinAbs(G_steps(0, 0, 1), 1e-5));
    REQUIRE_THAT(GTime(0, 0), WithinAbs(n * p.time_step, 1e-7));
}

TEST_CASE("relaxSilentG without G <- F equals repeated update_GI steps, one per time step", "[update][silent]") {
    Parameters p = initParameters(0.0f, 1.0f, 0.01f);
    Tensor3f F(1, 1, 2), G_steps(1, 1, 2), G_lazy(1, 1, 2), g(1, 1, 2);
    F.setValues({{{0.6f, 0.8f}}});
    G_steps.setValues({{{2.0f, -1.0f}}});
    G_lazy = G_steps;
    g.setValues({{{0.5f, 0.25f}}});
    const int n = 7;
    const float w = 0.2f;
    for (int i = 0; i < n; ++i) {
        update_GI(G_steps, g, 0, 0, w, 0.0f, std::numeric_limits<float>::infinity());
    }
    Tensor2f GTime(1, 1);
    GTime.setConstant(0.0f);

    relaxSilentG(G_lazy, F, g, GTime, 0, 0, n * p.time_step, -std::log(1.0f - w) / p.time_step, 0.0f);

    REQUIRE_THAT(G_lazy(0, 0, 0), WithinAbs(G_steps(0, 0, 0), 1e-5));
    REQUIRE_THAT(G_lazy(0, 0, 1), WithinAbs(G_steps(0, 0, 1), 1e-5));
}

TEST_CASE("relaxSilentG: V = 0 removes only G's component along F", "[update][silent]") {
    Parameters p = initParameters(0.0f, 1.0f, 0.01f);
    Tensor3f F(1, 1, 2), G(1, 1, 2), g(1, 1, 2);
    F.setValues({{{0.0f, 3.0f}}});    // flow along x
    G.setValues({{{1.5f, 2.0f}}});
    g.setZero();
    Tensor2f GTime(1, 1);
    GTime.setConstant(0.0f);

    relaxSilentG(G, F, g, GTime, 0, 0, 1.0f, 0.0f, p.rate_GF_silent);

    REQUIRE_THAT(G(0, 0, 0), WithinAbs(1.5f, 1e-6));                                     // perpendicular: untouched
    REQUIRE_THAT(G(0, 0, 1), WithinAbs(2.0f * std::exp(-p.rate_GF_silent * 1.0f), 1e-6));  // parallel: decays
}

TEST_CASE("relaxSilentG: stale G at a pixel an edge has left goes to zero", "[update][silent]") {
    // An edge passed, left G behind, and the pixel is now flat (grad I = 0) and silent.
    Parameters p = initParameters(0.0f, 1.0f, 0.01f);
    Tensor3f F(1, 1, 2), G(1, 1, 2), g(1, 1, 2);
    F.setValues({{{0.6f, 0.8f}}});
    G.setValues({{{-3.0f, 4.0f}}});
    g.setZero();
    Tensor2f GTime(1, 1);
    GTime.setConstant(0.0f);

    relaxSilentG(G, F, g, GTime, 0, 0, 1.0f, p.rate_GI_silent, p.rate_GF_silent);

    REQUIRE_THAT(G(0, 0, 0), WithinAbs(0.0f, 1e-6));
    REQUIRE_THAT(G(0, 0, 1), WithinAbs(0.0f, 1e-6));
}

TEST_CASE("I is in event steps without bounds, and F and G are not clamped by default", "[deviations]") {
    const Parameters p = initParameters(0.0f, 1.0f, 0.01f);
    REQUIRE(p.eventContribution == 1.0f);   // delta log I = +-c per event
    REQUIRE(p.neutralPotential == 0.0f);    // the offset of log I is unobservable
    REQUIRE(std::isinf(p.minPotential));
    REQUIRE(std::isinf(p.maxPotential));
    REQUIRE_FALSE(p.clampMaps);
    REQUIRE(std::isinf(p.clampBound()));
}

TEST_CASE("I uses per-pixel contrast steps and forgets with the derived time constant by default", "[deviations][contrast]") {
    const Parameters p = initParameters(0.0f, 1.0f, 0.01f);
    REQUIRE(p.contrastSteps == "pixel");   // c_ON / c_OFF differs between pixels
    REQUIRE(p.useDecay);
    REQUIRE_THAT(p.decayParam, WithinAbs(0.06f, 1e-6));   // 1 / (sigma^2 rate), experiments/reports/forgetting.md
}

// Silence as a bound: 1x1 maps, grad I given directly
namespace {
Tensor3f vec(float y, float x) {
    Tensor3f t(1, 1, 2);
    t(0, 0, 0) = y;
    t(0, 0, 1) = x;
    return t;
}
}

TEST_CASE("boundSilentG: without motion or silence G is grad I", "[update][silent][bound]") {
    Tensor3f G = vec(5, 5);
    boundSilentG(G, vec(0, 0), vec(0.3f, -2.0f), 0, 0, 0.5f, 1.0f, 0.0f);   // F = 0: no bound
    REQUIRE(G(0, 0, 0) == 0.3f);
    REQUIRE(G(0, 0, 1) == -2.0f);
    boundSilentG(G, vec(0, 100), vec(0.3f, -2.0f), 0, 0, -0.001f, 1.0f, 0.0f);   // inside the refractory time
    REQUIRE(G(0, 0, 1) == -2.0f);
}

TEST_CASE("boundSilentG holds G's component along F to the thresholds over the distance travelled", "[update][silent][bound]") {
    // F = 100 px/s along x, silent 0.05 s: 5 px travelled. OFF threshold 1 -> G_x < 0.2, ON step 1.5 -> G_x > -0.3
    Tensor3f G(1, 1, 2);
    boundSilentG(G, vec(0, 100), vec(0.7f, 2.0f), 0, 0, 0.05f, 1.5f, 0.0f);
    REQUIRE_THAT(G(0, 0, 1), WithinAbs(0.2f, 1e-6));
    REQUIRE_THAT(G(0, 0, 0), WithinAbs(0.7f, 1e-6));   // across F: untouched
    boundSilentG(G, vec(0, 100), vec(0.7f, -2.0f), 0, 0, 0.05f, 1.5f, 0.0f);
    REQUIRE_THAT(G(0, 0, 1), WithinAbs(-0.3f, 1e-6));
    boundSilentG(G, vec(0, 100), vec(0.7f, 0.1f), 0, 0, 0.05f, 1.5f, 0.0f);   // inside: grad I unchanged
    REQUIRE_THAT(G(0, 0, 1), WithinAbs(0.1f, 1e-6));
    boundSilentG(G, vec(0, 100), vec(0.7f, 2.0f), 0, 0, 0.05f, 1.5f, 1.0f);   // slack 1 step: (1 + 1) / 5
    REQUIRE_THAT(G(0, 0, 1), WithinAbs(0.4f, 1e-6));
}

TEST_CASE("boundSilentG: the bound is the change brightness constancy allows during the silence", "[update][silent][bound]") {
    // A diagonal F: the bounded G predicts dI = -(G . F) dt exactly at the OFF threshold
    Tensor3f G(1, 1, 2);
    const float dt = 0.02f;
    boundSilentG(G, vec(300, 400), vec(3.0f, 4.0f), 0, 0, dt, 1.0f, 0.0f);
    const float dI = -(G(0, 0, 0) * 300 + G(0, 0, 1) * 400) * dt;
    REQUIRE_THAT(dI, WithinAbs(-1.0f, 1e-5));
}

TEST_CASE("followGradientNeighbourhood is relaxSilentG in the limit of an infinite G <- I rate", "[update][silent]") {
    const int n = 5;
    Tensor2f I(n, n);
    for (int y = 0; y < n; ++y) for (int x = 0; x < n; ++x) I(y, x) = float(y * y) - 0.5f * float(x);
    Tensor3f G_follow(n, n, 2), G_relax(n, n, 2), grad(n, n, 2), F(n, n, 2);
    G_follow.setConstant(7.0f);
    G_relax.setConstant(7.0f);
    F.setConstant(1.0f);
    Tensor2f times(n, n);
    times.setZero();
    followGradientNeighbourhood(G_follow, I, grad, 2, 2);
    relaxSilentNeighbourhood(G_relax, F, I, grad, times, 2, 2, 1.0f, 1e9f, 0.0f);
    for (const auto &[y, x] : std::vector<std::pair<int, int>>{{2, 2}, {1, 2}, {3, 2}, {2, 1}, {2, 3}}) {
        REQUIRE_THAT(G_follow(y, x, 0), WithinAbs(G_relax(y, x, 0), 1e-5));
        REQUIRE_THAT(G_follow(y, x, 1), WithinAbs(G_relax(y, x, 1), 1e-5));
    }
    REQUIRE(G_follow(0, 0, 0) == 7.0f);   // outside the neighbourhood: untouched
    const Tensor3f all = gradientCopy(I);
    REQUIRE_THAT(all(2, 2, 0), WithinAbs((9.0f - 1.0f) / 2.0f, 1e-6));
    REQUIRE_THAT(all(2, 2, 1), WithinAbs(-0.5f, 1e-6));
}

TEST_CASE("G follows grad I between events by default; the bound uses the measured refractory time", "[deviations][silent]") {
    const Parameters p = initParameters(0.0f, 1.0f, 0.01f);
    REQUIRE(p.silentRule == "gradient");
    REQUIRE_THAT(p.refractoryTime, WithinAbs(1.2e-3f, 1e-9));
    REQUIRE(p.silentSlack == 0.0f);
}

TEST_CASE("G reset is off and silent relaxation on by default", "[deviations][silent]") {
    const Parameters p = initParameters(0.0f, 1.0f, 0.01f);
    REQUIRE_FALSE(p.resetGEachFrame);
    REQUIRE(p.silentRelaxation);
    REQUIRE_THAT(p.rate_GI_silent, WithinAbs(-std::log(0.8f) / 0.01f, 1e-3));   // former values, open (M2)
    REQUIRE_THAT(p.rate_GF_silent, WithinAbs(-std::log(0.8f) / 0.01f, 1e-3));
}

TEST_CASE("load_events_txt skips a header and slice_events keeps read_events' inclusive bounds", "[io]") {
    namespace fs = std::filesystem;
    const fs::path previous = fs::current_path();
    const fs::path sandbox = fs::temp_directory_path() / "im_tests_load_events";
    fs::remove_all(sandbox);
    fs::create_directories(sandbox);
    fs::current_path(sandbox);
    {
        std::ofstream f("events.txt");
        f << "time x y p\n0.000000000 33 39 1\n0.5 7 3 0\n1.0 5 6 1\n1.25 2 1 0\n";
    }
    std::vector<RawCameraEvent> all;
    const bool loaded = load_events_txt("events.txt", all);
    fs::current_path(previous);
    fs::remove_all(sandbox);

    REQUIRE(loaded);
    REQUIRE(all.size() == 4);
    REQUIRE(all[0].x == 33);
    REQUIRE(all[0].y == 39);
    REQUIRE(all[0].polarity == 1);
    REQUIRE(all[1].polarity == -1);

    std::vector<std::shared_ptr<Event>> slice;
    slice_events(all, 0.5f, 1.0f, slice);
    REQUIRE(slice.size() == 2);   // both boundaries included, as in read_events
    const auto *first = dynamic_cast<CameraEvent *>(slice[0].get());
    REQUIRE(first != nullptr);
    REQUIRE(first->coordinates == std::vector<int>{3, 7});   // (y, x)
    REQUIRE(first->polarity == -1);
}

TEST_CASE("m32_pixel equals m32 for R x C", "[update][rotation][async]") {
    Parameters p = initParameters(0.0f, 1.0f, 0.1f);
    OpticalFlowState s = initOpticalFlowState(8, 8, p, smallCalibration(8, 8));
    Tensor1f R(3);
    R.setValues({0.01f, -0.02f, 0.03f});
    Tensor3f RxC(s.CCM.dimensions()), F(s.F.dimensions());
    crossProduct1x3(R, s.CCM, RxC);
    m32(RxC, s.dCdx, s.dCdy, F);
    for (const auto &px : std::vector<std::array<int, 2>>{{0, 0}, {3, 5}, {7, 7}}) {
        const Vector3f v(RxC(px[0], px[1], 0), RxC(px[0], px[1], 1), RxC(px[0], px[1], 2));
        const Vector2f f = m32_pixel(v, s.dCdx, s.dCdy, px[0], px[1]);
        REQUIRE_THAT(f(0), WithinAbs(F(px[0], px[1], 0), 1e-5));
        REQUIRE_THAT(f(1), WithinAbs(F(px[0], px[1], 1), 1e-5));
    }
}

TEST_CASE("Event-driven F <- R over one time step equals the synchronous passes", "[update][rotation][async]") {
    Parameters p = initParameters(0.0f, 1.0f, 0.01f);
    OpticalFlowState s = initOpticalFlowState(8, 8, p, smallCalibration(8, 8));
    s.R.setValues({0.01f, -0.02f, 0.03f});
    Tensor3f F_sync = s.F;
    const float w = 0.8f;
    for (int i = 0; i < p.updateIterationsFR; ++i) {
        update_FR(F_sync, s.CCM, s.dCdx, s.dCdy, s.R, w, 0.0f, std::numeric_limits<float>::infinity());
    }
    const float rate = -static_cast<float>(p.updateIterationsFR) * std::log(1.0f - w) / p.time_step;

    const Tensor3f F_async = relaxedFRCopy(s.F, s.CCM, s.dCdx, s.dCdy, s.R, s.FTimeSurface, p.time_step, rate);

    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
            REQUIRE_THAT(F_async(y, x, 0), WithinAbs(F_sync(y, x, 0), 1e-5));
            REQUIRE_THAT(F_async(y, x, 1), WithinAbs(F_sync(y, x, 1), 1e-5));
        }
    }
}

TEST_CASE("relaxB matches the running sum of every pixel relaxed with relaxFR", "[update][rotation][async]") {
    // The global closed form for B and the lazy per-pixel F relaxation must describe the same state.
    Parameters p = initParameters(0.0f, 1.0f, 0.01f);
    OpticalFlowState s = initOpticalFlowState(8, 8, p, smallCalibration(8, 8));
    s.R.setValues({0.01f, -0.02f, 0.03f});
    const float t = 0.002f;   // partially relaxed, so both the transient and the target matter

    relaxB(s.B, s.BTime, s.A, s.R, t, p.rate_FR);
    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
            relaxFR(s.F, s.CCM, s.dCdx, s.dCdy, s.R, s.FTimeSurface, s.old_points, y, x, t, p.rate_FR);
        }
    }

    REQUIRE(s.B.isApprox(expectedRSum(s), 1e-4f));
    REQUIRE_THAT(s.BTime, WithinAbs(t, 1e-9));
}

TEST_CASE("register_R_point brings B up to date with a changed F", "[update][rotation][async]") {
    Parameters p = initParameters(0.0f, 1.0f, 0.01f);
    OpticalFlowState s = initOpticalFlowState(8, 8, p, smallCalibration(8, 8));
    s.F(3, 4, 0) += 0.5f;
    s.F(3, 4, 1) -= 0.25f;

    register_R_point(s.F, s.CCM, s.dCdx, s.dCdy, s.B, s.Identity_minus_outerProducts, s.old_points, 3, 4);

    REQUIRE(s.B.isApprox(expectedRSum(s), 1e-4f));
}

TEST_CASE("Event-driven F <- R is on by default", "[deviations][async]") {
    const Parameters p = initParameters(0.0f, 1.0f, 0.01f);
    REQUIRE(p.asyncFR);
    REQUIRE_THAT(p.rate_FR, WithinAbs(-4.0f * std::log(0.2f) / 0.01f, 1e-2));   // former value, open (M2)
}

TEST_CASE("All relation weights default to 1, inputs stay off", "[deviations]") {
    const Parameters p = initParameters(0.0f, 1.0f, 0.01f);
    for (const float w : {p.weight_FG, p.weight_FR, p.weight_GF, p.weight_GI, p.weight_IG, p.weight_IV, p.weight_RF}) {
        REQUIRE(w == 1.0f);
    }
    REQUIRE(p.weight_RIMU == 0.0f);
    REQUIRE(p.weight_Ifusion == 0.0f);
}

TEST_CASE("read_imu keeps the file's (x, y, z) axis order", "[io][imu]") {
    namespace fs = std::filesystem;
    const fs::path previous = fs::current_path();
    const fs::path sandbox = fs::temp_directory_path() / "im_tests_read_imu";
    fs::remove_all(sandbox);
    fs::create_directories(sandbox);
    fs::current_path(sandbox);
    {
        std::ofstream f("imu.txt");
        f << "0.5 1 2 3 0.1 0.2 0.3\n";
    }
    std::vector<std::shared_ptr<Event>> events;
    read_imu("imu.txt", events, 0.0f, 1.0f, INT32_MAX);
    fs::current_path(previous);
    fs::remove_all(sandbox);

    REQUIRE(events.size() == 1);
    const auto *imu = dynamic_cast<IMUEvent *>(events[0].get());
    REQUIRE(imu != nullptr);
    REQUIRE(imu->lin_acceleration == std::vector<float>{1.0f, 2.0f, 3.0f});
    REQUIRE(imu->ang_velocity == std::vector<float>{0.1f, 0.2f, 0.3f});
}
