// https://eigen.tuxfamily.org/dox/group__TutorialAdvancedInitialization.html
#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;

int main(){
    RowVectorXd vec1(3);
    vec1 << 1, 2, 3;
    std::cout << "vec1 = " << vec1 << std::endl;
    
    RowVectorXd vec2(4);
    vec2 << 1, 4, 9, 16;
    std::cout << "vec2 = " << vec2 << std::endl;
    
    RowVectorXd joined(7);
    joined << vec1, vec2;
    std::cout << "joined = " << joined << std::endl;

    std::cout << "We can use the same technique to initialize matrices with a block structure." << std::endl;

    MatrixXf matA(2, 2);
    matA << 1, 2, 3, 4;
    MatrixXf matB(4, 4);
    matB << matA, matA/10, matA/10, matA;
    std::cout << matB << std::endl;

    std::cout << "The comma initializer can also be used to fill block expressions such as m.row(i). Here is a more complicated way to get the same result as in the first example above:" << std::endl;

    Matrix3f m;
    m.row(0) << 1, 2, 3;
    m.block(1,0,2,2) << 4, 5, 7, 8;
    m.col(2).tail(2) << 6, 9;
    std::cout << m << std::endl;

    std::cout << "A fixed-size array:\n";
    Array33f a1 = Array33f::Zero();
    std::cout << a1 << "\n\n";
    
    
    std::cout << "A one-dimensional dynamic-size array:\n";
    ArrayXf a2 = ArrayXf::Zero(3);
    std::cout << a2 << "\n\n";
    
    
    std::cout << "A two-dimensional dynamic-size array:\n";
    ArrayXXf a3 = ArrayXXf::Zero(3, 4);
    std::cout << a3 << "\n";

}