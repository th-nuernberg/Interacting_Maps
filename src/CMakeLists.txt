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

#target_include_directories(interacting_maps_lib
#        PUBLIC ${PROJECT_SOURCE_DIR}/include
#)



# target_link_libraries(example  Boost::asio Boost::container Boost::filesystem Boost::log)
# === Executables ===

# Example 1: TestSIMD
#add_executable(TestSIMD testSIMD.cpp)
#target_link_libraries(TestSIMD PRIVATE interacting_maps_lib Eigen3::Eigen Boost::program_options)
#set_target_properties(TestSIMD
#        PROPERTIES
#        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
#)


# Example 2: Main program
add_executable(InteractingMaps interacting_maps.cpp)
target_link_libraries(InteractingMaps
        PRIVATE
        interacting_maps_lib
        Eigen3::Eigen
        Boost::program_options
        Boost::container
        Boost::filesystem
        Boost::log
        ${OpenCV_LIBS}
        opencv_core
        opencv_imgproc
        opencv_calib3d  # <-- Add this line (required for cv::undistort)
        opencv_highgui
        autodiff
        #${Boost_LIBRARIES}
)


set_target_properties(InteractingMaps
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
)

# Add compiler options
foreach(exe IN ITEMS InteractingMaps)
    # Compiler options
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-g)
    elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(Executable PRIVATE -O3 -march=native)
    endif()
    if(PROFILING)
        target_compile_definitions(${exe} PRIVATE PROFILING=1)
    endif()

    if(IMAGES)
        target_compile_definitions(${exe} PRIVATE IMAGES=1)
    endif()
endforeach()
