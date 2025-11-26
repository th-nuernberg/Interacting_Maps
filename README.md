

# Interacting Maps 
This is a C++ Implementation of the Interacting Maps algorithm.

The algorithm jointly estimates visual quantities like image intensity, optic flow from event data of a DVS

The implementation is done on an event basis in contrast to the original implementation which uses accumulated event frames.
As there is no code provided with the original paper, this implementation was done based on the descriptions in the paper.
The executable creates a folder of output frames for visual inspection, a time series of estimated rotational velocities

The events currently have to be provided with via a text file, called events.txt. Each event consists of a time stamp, a x and y coordinate and a polarity [0,1]

# Installation
```
mkdir external && cd external
```
## Install Eigen
```
cd external
git clone https://github.com/PX4/eigen.git
```
enough to use the libraries as eigen is header only
```
cd eigen
mkdir build && cd build
cmake ..
sudo make install
```
## Install boost
CMake
## Install Catch2
```
git clone https://github.com/catchorg/Catch2.git
cd Catch2
git checkout v3.9.1
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr/local ..
```
## Install pybind11
```
cd external
git clone https://github.com/pybind/pybind11
cd pybind11/
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr/local ..
```
## Install autodiff
```
cd external
git clone https://github.com/autodiff/autodiff
mkdir .build && cd .build
cmake ..
cmake --build . --target install
```
-D AUTODIFF_BUILD_PYTHON=OFF  # <-- Add this line to the ```cmake -S . -B build -G Ninja...``` command to disable Python bindings
pybind11 is then probably not needed. Otherwise python makes trouble

## Acknowledgements

 - M. Cook, L. Gugelmann, F. Jug, C. Krautz and A. Steger, "Interacting maps for fast visual interpretation," The 2011 International Joint Conference on Neural Networks, San Jose, CA, USA, 2011, pp. 770-776, doi: 10.1109/IJCNN.2011.6033299.


## Authors

- [@Daniel Pommer](https://www.github.com/DanielBanana)

