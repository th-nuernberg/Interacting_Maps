#include "../Eigen/Dense"
#include <iostream>

int main(){
    Eigen::Vector4f v(1,2,3,4);
    Eigen::RowVector3f vv = {4,5,6};
    Eigen::Matrix2f m {{1,2},{3,4}};

    std::cout << v << std::endl;
    std::cout << vv << std::endl;
    std::cout << "Matrix m" << std::endl << m << std::endl;

    m(0,0) = m(0,0) + 100.;

    std::cout << "Matrix m after addition " << std::endl << m << std::endl;

    Eigen::MatrixXd mm(2,6);
    mm.resize(3,4);
      std::cout << "The matrix mm is of size "
            << mm.rows() << "x" << mm.cols() << std::endl;
    std::cout << "It has " << mm.size() << " coefficients" << std::endl;
}

//When should one use fixed sizes (e.g. Matrix4f), and when should one prefer dynamic sizes (e.g. MatrixXf)? 
//The simple answer is: use fixed sizes for very small sizes where you can, and use dynamic sizes for larger sizes or where you have to.
//For small sizes, especially for sizes smaller than (roughly) 16, using fixed sizes is hugely beneficial to performance, as it allows Eigen to avoid dynamic memory allocation and to unroll loops

//The limitation of using fixed sizes, of course, is that this is only possible when you know the sizes at compile time.
//Also, for large enough sizes, say for sizes greater than (roughly) 32, the performance benefit of using fixed sizes becomes negligible.
//Worse, trying to create a very large matrix using fixed sizes inside a function could result in a stack overflow,
//since Eigen will try to allocate the array automatically as a local variable, and this is normally done on the stack.
//Finally, depending on circumstances, Eigen can also be more aggressive trying to vectorize (use SIMD instructions) when dynamic sizes are used, see Vectorization.