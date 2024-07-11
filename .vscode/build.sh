#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Variables
CXX="/usr/bin/g++"
CXXFLAGS="-fdiagnostics-color=always -g -Wall -Wextra -O3 -march=native -fopenmp"
INCLUDES="-I${workspaceFolder}/eigen/Eigen $(python3.10-config --includes) $(python3.10 -m pybind11 --includes)"
LDFLAGS="-L/usr/lib/x86_64-linux-gnu/ -lopenblas -lpthread -llapacke"

# Arguments
SRC_FILE=$1
WORKSPACE_DIR=$(dirname $(dirname ${SRC_FILE}))
OUTPUT_DIR="${WORKSPACE_DIR}/output"
OUTPUT_FILE="${OUTPUT_DIR}/$(basename ${SRC_FILE%.*})"

# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

# Compile command
$CXX $CXXFLAGS $INCLUDES $SRC_FILE -o $OUTPUT_FILE $LDFLAGS

# Print success message
echo "Build successful: $OUTPUT_FILE"
