#include <iostream>
#include <vector>
#include "../Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

int main(int argc, char *argv[]){
    MatrixXd m(2,2);
    m(0,0) = 3;
    m(1,0) = 2.5;
    m(0,1) = -1;
    m(1,1) = m(1,0) + m(0,1);
    std::cout << m << std::endl;

    MatrixXd m2 = MatrixXd::Random(3,3);
    m2 = (m2 + MatrixXd::Constant(3,3,1.2)) * 50;
    std::cout << "m =" << std::endl << m2 << std::endl;
    VectorXd v(3);
    v << 1, 2, 3;
    std::cout << "m * v =" << std::endl << m2 * v << std::endl;
}