// https://eigen.tuxfamily.org/dox/group__TutorialReductionsVisitorsBroadcasting.html

#include <iostream>
#include "../Eigen/Dense"

int main()
{
    // Basic Reductions
    Eigen::Matrix2d mat;
    mat << 1, 2,
            3, 4;
    std::cout << "Here is mat.sum():       " << mat.sum()       << std::endl;
    std::cout << "Here is mat.prod():      " << mat.prod()      << std::endl;
    std::cout << "Here is mat.mean():      " << mat.mean()      << std::endl;
    std::cout << "Here is mat.minCoeff():  " << mat.minCoeff()  << std::endl;
    std::cout << "Here is mat.maxCoeff():  " << mat.maxCoeff()  << std::endl;
    std::cout << "Here is mat.trace():     " << mat.trace()     << std::endl;

    // Matrix and Vector norms
    Eigen::VectorXf v(2);
    Eigen::MatrixXf m(2,2), n(2,2);

    v << -1,
        2;

    m << 1,-2,
        -3,4;

    std::cout << "v.squaredNorm() = " << v.squaredNorm() << std::endl;
    std::cout << "v.norm() = " << v.norm() << std::endl;
    std::cout << "v.lpNorm<1>() = " << v.lpNorm<1>() << std::endl;
    std::cout << "v.lpNorm<Infinity>() = " << v.lpNorm<Eigen::Infinity>() << std::endl;

    std::cout << std::endl;
    std::cout << "m.squaredNorm() = " << m.squaredNorm() << std::endl;
    std::cout << "m.norm() = " << m.norm() << std::endl;
    std::cout << "m.lpNorm<1>() = " << m.lpNorm<1>() << std::endl;
    std::cout << "m.lpNorm<Infinity>() = " << m.lpNorm<Eigen::Infinity>() << std::endl;

    // Matrix Operator norm
    Eigen::MatrixXf p(2,2);
    p << 1,-2,
        -3,4;

    std::cout << "1-norm(p)     = " << p.cwiseAbs().colwise().sum().maxCoeff()
            << " == "             << p.colwise().lpNorm<1>().maxCoeff() << std::endl;

    std::cout << "infty-norm(p) = " << p.cwiseAbs().rowwise().sum().maxCoeff()
            << " == "             << p.rowwise().lpNorm<1>().maxCoeff() << std::endl;

    // Boolean Reduction
    Eigen::ArrayXXf a(2,2);
  
    a << 1,2,
        3,4;

    std::cout << "(a > 0).all()   = " << (a > 0).all() << std::endl;
    std::cout << "(a > 0).any()   = " << (a > 0).any() << std::endl;
    std::cout << "(a > 0).count() = " << (a > 0).count() << std::endl;
    std::cout << std::endl;
    std::cout << "(a > 2).all()   = " << (a > 2).all() << std::endl;
    std::cout << "(a > 2).any()   = " << (a > 2).any() << std::endl;
    std::cout << "(a > 2).count() = " << (a > 2).count() << std::endl;

    // Visitors are useful when one wants to obtain the location of a coefficient inside a Matrix or Array.
    // The simplest examples are maxCoeff(&x,&y) and minCoeff(&x,&y), which can be used to find the location of the greatest or smallest coefficient in a Matrix or Array.
    // The arguments passed to a visitor are pointers to the variables where the row and column position are to be stored. These variables should be of type Index , as shown below:
    Eigen::MatrixXf b(2,2);
  
    b << 1, 2,
        3, 4;
    
    //get location of maximum
    Eigen::Index maxRow, maxCol;
    float max = b.maxCoeff(&maxRow, &maxCol);
    
    //get location of minimum
    Eigen::Index minRow, minCol;
    float min = b.minCoeff(&minRow, &minCol);
    
    std::cout << "Max: " << max <<  ", at: " <<
        maxRow << "," << maxCol << std::endl;
    std:: cout << "Min: " << min << ", at: " <<
        minRow << "," << minCol << std::endl;

    // Partial Reductions
      Eigen::MatrixXf mat2(2,4);
    mat2 << 1, 2, 6, 9,
            3, 1, 7, 2;
    
    std::cout << "Column's maximum: " << std::endl
    << mat2.colwise().maxCoeff() << std::endl;

    Eigen::MatrixXf mat3(2,4);
    mat3 << 1, 2, 6, 9,
            3, 1, 7, 2;
    
    std::cout << "Row's maximum: " << std::endl
    << mat3.rowwise().maxCoeff() << std::endl;

    // Combining partials with other operations
    Eigen::MatrixXf mat4(2,4);
    mat4 << 1, 2, 6, 9,
            3, 1, 7, 2;
    
    Eigen::Index   maxIndex;
    float maxNorm = mat4.colwise().sum().maxCoeff(&maxIndex);
    
    std::cout << "Maximum sum at position " << maxIndex << std::endl;
    
    std::cout << "The corresponding vector is: " << std::endl;
    std::cout << mat4.col( maxIndex ) << std::endl;
    std::cout << "And its sum is is: " << maxNorm << std::endl;

    // Broadcasting
    Eigen::MatrixXf mat5(2,4);
    Eigen::VectorXf w(2);
    
    mat5 << 1, 2, 6, 9,
            3, 1, 7, 2;
            
    w << 0,
        1;
        
    //add v to each column of m
    mat5.colwise() += w;
    
    std::cout << "Broadcasting result: " << std::endl;
    std::cout << mat5 << std::endl;

    Eigen::MatrixXf mat6(2,4);
    Eigen::VectorXf u(4);
    
    mat6 << 1, 2, 6, 9,
            3, 1, 7, 2;
            
    u << 0,1,2,3;
        
    //add v to each row of m
    mat6.rowwise() += u.transpose();
    
    std::cout << "Broadcasting result: " << std::endl;
    std::cout << mat6 << std::endl;

    // Combinination: Find nearest Neighbour of a Vector in a Matrix
    Eigen::MatrixXf mat7(2,4);
    Eigen::VectorXf x(2);

    mat7 << 1, 23, 6, 9,
        3, 11, 7, 2;
        
    x << 2,
        3;

    Eigen::Index index;
    // find nearest neighbour
    (mat7.colwise() - x).colwise().squaredNorm().minCoeff(&index);

    std::cout << "Nearest neighbour is column " << index << ":" << std::endl;
    std::cout << mat7.col(index) << std::endl;
}