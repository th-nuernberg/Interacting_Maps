#include <iostream>
#include "../Eigen/Dense"
 
int main()
{
    Eigen::Vector3d v(1,2,3);
    Eigen::Vector3d w(0,1,2);

    std::cout << "Dot product: " << v.dot(w) << std::endl;
    double dp = v.adjoint()*w; // automatic conversion of the inner product to a scalar
    std::cout << "Dot product via a matrix product: " << dp << std::endl;
    std::cout << "Cross product:\n" << v.cross(w) << std::endl;

    Eigen::Matrix2d mat;
    mat << 1, 2,
            3, 4;
    std::cout << "Here is mat:             " << std::endl << mat<< std::endl;     
    std::cout << "Here is mat.sum():       " << mat.sum()       << std::endl;
    std::cout << "Here is mat.prod():      " << mat.prod()      << std::endl;
    std::cout << "Here is mat.mean():      " << mat.mean()      << std::endl;
    std::cout << "Here is mat.minCoeff():  " << mat.minCoeff()  << std::endl;
    std::cout << "Here is mat.maxCoeff():  " << mat.maxCoeff()  << std::endl;
    std::cout << "Here is mat.trace():     " << mat.trace()     << std::endl;


    Eigen::Matrix3f m = Eigen::Matrix3f::Random();
    std::ptrdiff_t i, j;
    float minOfM = m.minCoeff(&i,&j);
    std::cout << "Here is the matrix m:\n" << m << std::endl;
    std::cout << "Its minimum coefficient (" << minOfM 
        << ") is at position (" << i << "," << j << ")\n\n";
    
    Eigen::RowVector4i vvv = Eigen::RowVector4i::Random();
    int maxOfV = vvv.maxCoeff(&i);
    std::cout << "Here is the vector v: " << vvv << std::endl;
    std::cout << "Its maximum coefficient (" << maxOfV 
        << ") is at position " << i << std::endl;
}
