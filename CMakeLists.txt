cmake_minimum_required(VERSION 3.15)
project(InteractingMaps)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(
        ${PROJECT_SOURCE_DIR}/external/eigen
        ${PROJECT_SOURCE_DIR}/external/autodiff
        ${PROJECT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)

# Find OpenCV package
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
FIND_PACKAGE(Boost COMPONENTS program_options REQUIRED)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
#TARGET_LINK_LIBRARIES(target ${Boost_LIBRARIES})
#find_package(dv-processing REQUIRED)


#include_directories(${dv-processing_INCLUDE_DIRS})
# Find installed dv-processing.
#find_package(dv-processing)

# Add the src directory
add_subdirectory(src)

set(PROJECT_HEADERS
        ${PROJECT_SOURCE_DIR}/include/Instrumentor.h
        ${PROJECT_SOURCE_DIR}/include/dv.h
        ${PROJECT_SOURCE_DIR}/include/datatypes.h
        ${PROJECT_SOURCE_DIR}/include/interacting_maps.h
        ${PROJECT_SOURCE_DIR}/include/imaging.h
        ${PROJECT_SOURCE_DIR}/include/file_operations.h
        ${PROJECT_SOURCE_DIR}/include/conversions.h
        ${PROJECT_SOURCE_DIR}/include/video.h
        ${PROJECT_SOURCE_DIR}/include/update.h
        ${PROJECT_SOURCE_DIR}/include/events.h
        include/testSIMD.h
)

set(PROJECT_HEADERS_SIMD
        ${PROJECT_SOURCE_DIR}/include/Instrumentor.h
        ${PROJECT_SOURCE_DIR}/include/dv.h
        ${PROJECT_SOURCE_DIR}/include/datatypes.h
        ${PROJECT_SOURCE_DIR}/include/imaging.h
        ${PROJECT_SOURCE_DIR}/include/file_operations.h
        ${PROJECT_SOURCE_DIR}/include/conversions.h
        ${PROJECT_SOURCE_DIR}/include/video.h
        ${PROJECT_SOURCE_DIR}/include/update.h
        ${PROJECT_SOURCE_DIR}/include/events.h
        include/testSIMD.h
)

# Add source files
set(SOURCES
        ${PROJECT_SOURCE_DIR}/src/interacting_maps.cpp
        ${PROJECT_SOURCE_DIR}/src/imaging.cpp
        ${PROJECT_SOURCE_DIR}/src/file_operations.cpp
        ${PROJECT_SOURCE_DIR}/src/conversions.cpp
        ${PROJECT_SOURCE_DIR}/src/video.cpp
        ${PROJECT_SOURCE_DIR}/src/datatypes.cpp
        ${PROJECT_SOURCE_DIR}/src/update.cpp
        ${PROJECT_SOURCE_DIR}/src/events.cpp
        src/testSIMD.cpp
)

# Add source files
set(SOURCES_SIMD
        ${PROJECT_SOURCE_DIR}/src/imaging.cpp
        ${PROJECT_SOURCE_DIR}/src/file_operations.cpp
        ${PROJECT_SOURCE_DIR}/src/conversions.cpp
        ${PROJECT_SOURCE_DIR}/src/video.cpp
        ${PROJECT_SOURCE_DIR}/src/datatypes.cpp
        ${PROJECT_SOURCE_DIR}/src/update.cpp
        ${PROJECT_SOURCE_DIR}/src/events.cpp
        src/testSIMD.cpp
)

# Add executable
add_executable(InteractingMaps ${SOURCES} ${PROJECT_HEADERS})

# Add executable
add_executable(TestSIMD ${SOURCES_SIMD} ${PROJECT_HEADERS_SIMD})

# Link libraries (include dv-processing and required dependencies)
target_link_libraries(InteractingMaps
        ${OpenCV_LIBS}
        ${Boost_LIBRARIES}
#        dv::processing
)

# Link libraries (include dv-processing and required dependencies)
target_link_libraries(TestSIMD
        ${OpenCV_LIBS}
        ${Boost_LIBRARIES}
        #        dv::processing
)

# Set the output directory for the executable
set_target_properties(InteractingMaps PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/output
)

# Set the output directory for the executable
set_target_properties(TestSIMD PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/output
)

# Add compiler options based on build type
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(InteractingMaps PRIVATE -g)
    target_compile_options(TestSIMD PRIVATE -g)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(InteractingMaps PRIVATE -O3)
    target_compile_options(TestSIMD PRIVATE -O3)
endif()
if (PROFILING)
    add_definitions(-DPROFILING=1)
endif (PROFILING)

if (IMAGES)
    add_definitions(-DIMAGES=1)
endif (IMAGES)
