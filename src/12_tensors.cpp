#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/CXX11/src/Tensor/Tensor.h>
#include <iostream>
// #include "../Eigen/Eigen"
using ::Eigen::Tensor;

int main(){
    Tensor<float, 3> t_3d(2,3,4);
    // std::cout << "NumRows: " << t_3d.dimension(0) << " NumCols: " << t_3d.dimension(1) << std::endl;
}