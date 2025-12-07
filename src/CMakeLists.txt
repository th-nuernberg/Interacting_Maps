add_executable(InteractingMaps interacting_maps.cpp)

# Common library (all reusable sources)
add_library(interacting_maps_lib
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
        ${BLAS_LIBRARIES}
        ${LAPACK_LIBRARIES}
        Eigen3::Eigen
        ${OpenCV_LIBS}
        autodiff
        Boost::asio
        Boost::container
        Boost::filesystem
        Boost::log
        Boost::program_options
        opencv_core
        opencv_imgproc
        opencv_calib3d  # <-- Add this line (required for cv::undistort)
        opencv_highgui
        xtensor
        xtensor-blas
)


# === Executables ===
# Example 2: Main program
target_link_libraries(InteractingMaps PUBLIC interacting_maps_lib)

# Set output directory for the executable (optional, since global is already set)
set_target_properties(InteractingMaps PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
)

add_executable(xTensorTest xtensorTest.cpp)
target_link_libraries(xTensorTest PUBLIC interacting_maps_lib)
set_target_properties(xTensorTest PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
)

# Add compiler options
foreach(exe IN ITEMS InteractingMaps)
    # Compiler options
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-g)
    elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3 -march=native)
    endif()
    if(PROFILING)
        target_compile_definitions(${exe} PRIVATE PROFILING=1)
    endif()

    if(IMAGES)
        target_compile_definitions(${exe} PRIVATE IMAGES=1)
    endif()
endforeach()
