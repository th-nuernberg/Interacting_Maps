# List all source files
set(SOURCES
        conversions.cpp
        events.cpp
        imaging.cpp
        update.cpp
        video.cpp
        file_operations.cpp
        interacting_maps.cpp
)

# Create the executable
add_executable(InteractingMaps ${SOURCES})

# Include directories for the executable
target_include_directories(InteractingMaps
        PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/external
)

# Link libraries to the executable
target_link_libraries(InteractingMaps
        PRIVATE
        autodiff
        Eigen3::Eigen
        Boost::container
        Boost::filesystem
        Boost::log
        Boost::program_options
        ${OpenCV_LIBS}
        opencv_core
        opencv_imgproc
        opencv_calib3d  # <-- Add this line (required for cv::undistort)
        opencv_highgui
)

# Set output directory for the executable
set_target_properties(InteractingMaps PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

# Compiler options
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(InteractingMaps PRIVATE -g)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(InteractingMaps PRIVATE -O3)
endif()
if(PROFILING)
    target_compile_definitions(InteractingMaps PRIVATE PROFILING=1)
endif()
if(IMAGES)
    target_compile_definitions(InteractingMaps PRIVATE IMAGES=1)
endif()
