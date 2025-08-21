
# Common library (all reusable sources)
add_library(interacting_maps_lib
        conversions.cpp
        events.cpp
        imaging.cpp
        update.cpp
        video.cpp
        file_operations.cpp
        interacting_maps.cpp
)

target_include_directories(interacting_maps_lib
        PUBLIC ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(interacting_maps_lib
        PUBLIC
        ${OpenCV_LIBS}
        ${Boost_LIBRARIES}
        Eigen3::Eigen
        autodiff
)

# target_link_libraries(example  Boost::asio Boost::container  Boost::filesystem Boost::log)
# === Executables ===

# Example 1: TestSIMD
add_executable(TestSIMD testSIMD.cpp)
target_link_libraries(TestSIMD PRIVATE interacting_maps_lib Eigen3::Eigen Boost::program_options)
set_target_properties(TestSIMD
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
)


# Example 2: Main program
add_executable(InteractingMaps interacting_maps.cpp)
target_link_libraries(InteractingMaps PRIVATE interacting_maps_lib Eigen3::Eigen Boost::program_options)
set_target_properties(InteractingMaps
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
)

# Add compiler options
foreach(exe IN ITEMS TestSIMD InteractingMaps)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(${exe} PRIVATE -g)
    elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(${exe} PRIVATE -O3)
    endif()

    if(PROFILING)
        target_compile_definitions(${exe} PRIVATE PROFILING=1)
    endif()

    if(IMAGES)
        target_compile_definitions(${exe} PRIVATE IMAGES=1)
    endif()
endforeach()
