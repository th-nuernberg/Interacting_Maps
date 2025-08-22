# toolchain-gcc13.cmake - cross compile for AArch64 SoC

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# Sysroot from PetaLinux SDK
set(CMAKE_SYSROOT /opt/petalinux/2022.2/sysroots/cortexa72-cortexa53-xilinx-linux)

# Cross compilers provided by the environment script
set(CMAKE_C_COMPILER   aarch64-xilinx-linux-gcc)
set(CMAKE_CXX_COMPILER aarch64-xilinx-linux-g++)

set(CMAKE_FIND_ROOT_PATH ${CMAKE_SYSROOT})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

set(CMAKE_C_FLAGS "--sysroot=${CMAKE_SYSROOT} ${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "--sysroot=${CMAKE_SYSROOT} ${CMAKE_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "--sysroot=${CMAKE_SYSROOT} ${CMAKE_EXE_LINKER_FLAGS}")
