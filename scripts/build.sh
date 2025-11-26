#!/usr/bin/env bash

# Source the PetaLinux SDK environment (puts cross-compilers in PATH)
. /opt/petalinux/2022.2/environment-setup-cortexa72-cortexa53-xilinx-linux

set -euo pipefail

# Clean and create build dir
rm -rf build
mkdir -p build

cmake -S . -B build -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=toolchain-gcc13.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
  -D BUILD_SHARED_LIBS=OFF \
  -D WITH_GTK=OFF \
  -D WITH_QT=OFF \
  -D BUILD_opencv_videoio=OFF \
  -D WITH_FFMPEG=OFF \
  -D IMAGES=ON \
  -D AUTODIFF_BUILD_PYTHON=OFF  # <-- Add this line to disable Python bindings

cmake --build build

# Quick sanity check on the built binary
echo "---- build complete ----"
file ./InteractingMaps || true
