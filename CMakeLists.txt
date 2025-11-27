cmake_minimum_required(VERSION 3.26)
project(InteractingMaps LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include CPM for dependency management
include(cmake/CPM.cmake)

# Dependencies
CPMAddPackage(
        NAME Boost
        VERSION 1.86.0
        URL https://github.com/boostorg/boost/releases/download/boost-1.86.0/boost-1.86.0-cmake.tar.xz
        URL_HASH SHA256=2c5ec5edcdff47ff55e27ed9560b0a0b94b07bd07ed9928b476150e16b0efc57
        OPTIONS
        "BOOST_ENABLE_CMAKE ON"
        "BOOST_SKIP_INSTALL_RULES ON"
        "BUILD_SHARED_LIBS OFF"
        "BOOST_INCLUDE_LIBRARIES container\\\;asio\\\;filesystem\\\;regex\\\;log\\\;program_options"
)

CPMAddPackage(
        NAME Eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4
        OPTIONS "EIGEN_BUILD_DOC OFF" "EIGEN_BUILD_PKGCONFIG OFF"
)

CPMAddPackage(
        NAME Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.9.1
)

CPMAddPackage(
        NAME pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v3.0.0
)

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)
find_package(HDF5 NAMES hdf5)# Add the src directory


# Include directories for the entire project
include_directories(
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/external
        ${PROJECT_SOURCE_DIR}/external/autodiff
        ${PROJECT_SOURCE_DIR}/external/eigen
        ${OpenCV_INCLUDE_DIRS}
)

# Add the src subdirectory
add_subdirectory(src)
add_subdirectory(external/autodiff)