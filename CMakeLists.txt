cmake_minimum_required(VERSION 3.15)
project(InteractingMaps)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(
        ${PROJECT_SOURCE_DIR}/external/eigen
        ${PROJECT_SOURCE_DIR}/external/autodiff
        ${PROJECT_SOURCE_DIR}/include
)

# Find OpenCV package
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Add source files
set(SOURCES
        ${PROJECT_SOURCE_DIR}/src/interacting_maps.cpp
        include/Instrumentor.h
        # Add more source files here if needed
)

# Add executable
add_executable(Executable ${SOURCES})

# Link libraries
target_link_libraries(Executable ${OpenCV_LIBS})

# Set the output directory for the executable
set_target_properties(Executable PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/output
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(Executable PRIVATE -g)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(Executable PRIVATE -O3)
endif()