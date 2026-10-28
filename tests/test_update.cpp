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
