add_executable(my_tests tests.cpp)

#[[add_library(interacting_maps_lib
        interacting_maps.cpp
        imaging.cpp
        file_operations.cpp
        conversions.cpp
        video.cpp
        update.cpp
        events.cpp
        datatypes.cpp
        cost.cpp
)

target_link_libraries(interacting_maps_lib
        PUBLIC
        autodiff
        Boost::asio
        Boost::container
        Boost::filesystem
        Boost::log
        Boost::program_options
        Eigen3::Eigen
        ${OpenCV_LIBS}
        opencv_core
        opencv_imgproc
        opencv_calib3d  # <-- Add this line (required for cv::undistort)
        opencv_highgui
)]]

target_link_libraries(my_tests PUBLIC interacting_maps_lib)

add_test(
        NAME GeneralTest
        COMMAND my_tests
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

