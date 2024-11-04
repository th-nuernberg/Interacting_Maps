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
find_package(dv-processing REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${dv-processing_INCLUDE_DIRS})
# Find installed dv-processing.
#find_package(dv-processing)

# Add the src directory
add_subdirectory(src)

# Add source files
set(SOURCES
        ${PROJECT_SOURCE_DIR}/src/interacting_maps.cpp
        include/Instrumentor.h
        include/dv.h
        # Add more source files here if needed
)

# Add executable
add_executable(Executable ${SOURCES})

# Link libraries (include dv-processing and required dependencies)
target_link_libraries(Executable
        ${OpenCV_LIBS}
        dv::processing
        #        dv-processing
#        OpenSSL::SSL
#        OpenSSL::Crypto
#        Boost::nowide
)

# Set the output directory for the executable
set_target_properties(Executable PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/output
)

# Add compiler options based on build type
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(Executable PRIVATE -g)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(Executable PRIVATE -O3)
endif()
