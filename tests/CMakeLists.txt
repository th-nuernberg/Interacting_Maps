# Catch2 unit tests for the update rules and helpers.
# tests.cpp is the older hand-rolled harness written against the xtensor API of the FrameBased
# branch; it does not compile against the Eigen code on this branch and is not built here.
add_executable(im_tests
        test_update.cpp
        test_timing.cpp
        test_rotation_plot.cpp
)

target_link_libraries(im_tests PRIVATE interacting_maps_lib Catch2::Catch2WithMain)

add_test(
        NAME im_tests
        COMMAND im_tests
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)
