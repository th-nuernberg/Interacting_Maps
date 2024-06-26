// https://eigen.tuxfamily.org/dox/group__TutorialMatrixArithmetic.html

#include <iostream>
#include "../Eigen/Dense"

int main(){
    Eigen::Matrix2f a;
    a << 1, 2,
         3, 4;
    
    Eigen::Matrix2f b {{5,6},{7,8}};

    std::cout << "Adding a and b" << std::endl << a+b << std::endl;

    std::cout << "Subtracting a and b" << std::endl << a-b << std::endl;

    a+=b;
    std::cout << "Compounding a with b gives a = " << std::endl << a << std::endl;

    Eigen::Vector3f v = {1.4, 2.3, 3.1};
    Eigen::Vector3f w = {10.5, 23.3, 1.3};
    std::cout << "v - w + v" << std::endl << v - w + v << std::endl;

    std::cout << "Scalar multiplication of b with 2.5" << std::endl << 2.5 * b << std::endl;
    std::cout << "Scalar multiplication of v with 0.1" << std::endl << 0.1 * v << std::endl;

    Eigen::MatrixXcf c = Eigen::MatrixXcf::Random(2,2);
    std::cout << "Here is the matrix c\n" << c << std::endl;
    
    std::cout << "Here is the matrix c^T\n" << c.transpose() << std::endl;
    

    std::cout << "Here is the conjugate of c\n" << c.conjugate() << std::endl;
    
    
    std::cout << "Here is the matrix c^*\n" << c.adjoint() << std::endl;

    Eigen::Matrix2i d; d << 1, 2, 3, 4;
    std::cout << "Here is the matrix d:\n" << d << std::endl;
    
    //d = d.transpose(); // !!! do NOT do this !!!

    d.transposeInPlace(); // USE THIS INSTEAD
    std::cout << "using transposition in place:\n" << d << std::endl;

    Eigen::Matrix2d mat;
    mat << 1, 2,
            3, 4;
    Eigen::Vector2d u(-1,1), t(2,0);
    std::cout << "Here is mat*mat:\n" << mat*mat << std::endl;
    std::cout << "Here is mat*u:\n" << mat*u << std::endl;
    std::cout << "Here is u^T*mat:\n" << u.transpose()*mat << std::endl;
    std::cout << "Here is u^T*v:\n" << u.transpose()*t << std::endl;
    std::cout << "Here is u*v^T:\n" << u*t.transpose() << std::endl;
    std::cout << "Let's multiply mat by itself" << std::endl;
    mat = mat*mat;
    std::cout << "Now mat is mat:\n" << mat << "\n\n" << std::endl;

    //If you know your matrix product can be safely evaluated into the destination matrix without aliasing issue, then you can use the noalias() function to avoid the temporary, e.g.:
    //c.noalias() += a * b;


    Eigen::MatrixXf o(2,2);
    Eigen::MatrixXf p(2,2);
    Eigen::MatrixXf r(2,2);

    o << 1,2,3,4;
    p << 5,6,7,8;

    r = o * p;
    std::cout << "-- Matrix o*p: --\n" << r << "\n\n";
    r = o.array() * p.array();
    std::cout << "-- Array o*p: --\n" << r << "\n\n";
    r = o.cwiseProduct(p);
    std::cout << "-- With cwiseProduct: --\n" << r << "\n\n";
    r = o.array() + 4;
    std::cout << "-- Array m + 4: --\n" << r << "\n\n";

    Eigen::MatrixXf m(2,2);
    Eigen::MatrixXf n(2,2);
    Eigen::MatrixXf result(2,2);
    
    m << 1,2,
        3,4;
    n << 5,6,
        7,8;
    
    result = (m.array() + 4).matrix() * m;
    std::cout << "-- Combination 1: --\n" << result << "\n\n";
    result = (m.array() * n.array()).matrix() * m;
    std::cout << "-- Combination 2: --\n" << result << "\n\n";

}