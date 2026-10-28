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
