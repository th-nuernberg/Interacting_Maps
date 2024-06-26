#include <iostream>
#include "../Eigen/Dense"

int main()
{
  Eigen::ArrayXXf  m(2,2);
  
  // assign some values coefficient by coefficient
  m(0,0) = 1.0; m(0,1) = 2.0;
  m(1,0) = 3.0; m(1,1) = m(0,1) + m(1,0);
  
  // print values to standard output
  std::cout << m << std::endl << std::endl;
 
  // using the comma-initializer is also allowed
  m << 1.0,2.0,
       3.0,4.0;
     
  // print values to standard output
  std::cout << m << std::endl;

  Eigen::ArrayXXf a(3,3);
  Eigen::ArrayXXf b(3,3);
  a << 1,2,3,
       4,5,6,
       7,8,9;
  b << 1,2,3,
       1,2,3,
       1,2,3;
       
  // Adding two arrays
  std::cout << "a + b = " << std::endl << a + b << std::endl << std::endl;
 
  // Subtracting a scalar from an array
  std::cout << "a - 2 = " << std::endl << a - 2 << std::endl;

  // First of all, of course you can multiply an array by a scalar, this works in the same way as matrices.
  // Where arrays are fundamentally different from matrices, is when you multiply two together. 
  // Matrices interpret multiplication as matrix product and arrays interpret multiplication as coefficient-wise product.
  // Thus, two arrays can be multiplied if and only if they have the same dimensions.
  Eigen::ArrayXXf c(2,2);
  Eigen::ArrayXXf d(2,2);
  c << 1,2,
       3,4;
  d << 5,6,
       7,8;
  std::cout << "c * d = " << std::endl << c * d << std::endl;

  Eigen::ArrayXf e = Eigen::ArrayXf::Random(5);
  e *= 2;
  std::cout << "e =" << std::endl
            << e << std::endl;
  std::cout << "e.abs() =" << std::endl
            << e.abs() << std::endl;
  std::cout << "a.abs().sqrt() =" << std::endl
            << e.abs().sqrt() << std::endl;
  std::cout << "a.min(a.abs().sqrt()) =" << std::endl
            << e.min(e.abs().sqrt()) << std::endl;
}