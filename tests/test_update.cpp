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
