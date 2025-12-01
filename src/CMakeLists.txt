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
)

# === Executables ===
# Example 2: Main program
target_link_libraries(InteractingMaps PUBLIC interacting_maps_lib)

# Set output directory for the executable (optional, since global is already set)
set_target_properties(InteractingMaps PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
)

# Create a new executable file from source file
add_executable(DV_1 01_EventStreamSlicer.cpp)
# Link libraries
target_link_libraries(DV_1 ${OpenCV_LIBS} dv::processing)
# Set the output directory for the executable
set_target_properties(DV_1 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/output)

add_executable(DV_0 00_CheckingCamera.cpp)
target_link_libraries(DV_0 ${OpenCV_LIBS} dv::processing)
set_target_properties(DV_0 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/output)

add_executable(DV_2 02_Accumulator.cpp)
target_link_libraries(DV_2 ${OpenCV_LIBS} dv::processing)
set_target_properties(DV_2 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/output)

add_executable(DV_3 03_Visualise.cpp)
target_link_libraries(DV_3 ${OpenCV_LIBS} dv::processing)
set_target_properties(DV_3 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/output)

add_executable(DV_4 04_EdgeAccumulation.cpp)
target_link_libraries(DV_4 ${OpenCV_LIBS} dv::processing)
set_target_properties(DV_4 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/output)

add_executable(DV_5 05_NoiseFilter.cpp)
target_link_libraries(DV_5 ${OpenCV_LIBS} dv::processing)
set_target_properties(DV_5 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/output)

add_executable(DV_6 06_CameraCalibration.cpp)
target_link_libraries(DV_6 ${OpenCV_LIBS} dv::processing)
set_target_properties(DV_6 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/output)

add_executable(DV_7 07_Undistort.cpp)
target_link_libraries(DV_7 ${OpenCV_LIBS} dv::processing)
set_target_properties(DV_7 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/output)

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
