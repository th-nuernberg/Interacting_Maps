# Define the executable for test1
add_executable(13_Gradient 13_gradient.cpp)

add_executable(14_OMP_Collapse 14_OMP_Collapse.cpp)

add_executable(15_DV 15_dv-processing_test.cpp)

add_executable(16_DV 16_dv_accumulator.cpp)

# Link libraries
target_link_libraries(15_DV ${OpenCV_LIBS} dv::processing)

# Link libraries
target_link_libraries(16_DV ${OpenCV_LIBS} dv::processing)

# Set the output directory for the executable
set_target_properties(15_DV PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/output
)

# Set the output directory for the executable
set_target_properties(16_DV PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/output
)