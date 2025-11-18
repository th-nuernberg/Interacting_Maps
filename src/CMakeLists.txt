# List all source files
set(SOURCES
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

# List headers (optional, for IDE support)
set(HEADERS
        ../include/Instrumentor.h
        ../include/dv.h
        ../include/datatypes.h
        ../include/interacting_maps.h
        ../include/imaging.h
        ../include/file_operations.h
        ../include/conversions.h
        ../include/video.h
        ../include/update.h
        ../include/events.h
        ../include/cost.h
)


# Create the executable
add_executable(Executable ${SOURCES} ${HEADERS})

# Link libraries
target_link_libraries(Executable
        ${OpenCV_LIBS}
        ${Boost_LIBRARIES}
        ${HDF5_LIBRARIES}
        ${HDF5_CXX_LIBRARIES})


# Set output directory
set_target_properties(Executable PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
)
