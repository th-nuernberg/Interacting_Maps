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
        ${OpenCV_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
)

# Find packages
find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS program_options REQUIRED)
find_package(HDF5 NAMES hdf5)# Add the src directory
add_subdirectory(src)

# Compiler options
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-g -lhdf5 -lhdf5_cpp)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(Executable PRIVATE -O3 -march=native -lhdf5 -lhdf5_cpp)
endif()

if (PROFILING)
    add_definitions(-DPROFILING=1)
endif (PROFILING)

if (IMAGES)
    add_definitions(-DIMAGES=1)
endif (IMAGES)
