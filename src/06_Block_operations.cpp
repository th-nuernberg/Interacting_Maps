// https://eigen.tuxfamily.org/dox/group__TutorialBlockOperations.html

// On the topic of performance, all what matters is that you give Eigen as much information as possible at compile time.
// For example, if your block is a single whole column in a matrix, using the specialized .col() function described below lets Eigen know that, which can give it optimization opportunities.
// Individual columns and rows are special cases of blocks. Eigen provides methods to easily address them: .col() and .row().
// Eigen also provides special methods for blocks that are flushed against one of the corners or sides of a matrix or array.
// For instance, .topLeftCorner() can be used to refer to a block in the top-left corner of a matrix.
// Eigen also provides a set of block operations designed specifically for the special case of vectors and one-dimensional arrays:

#include "../Eigen/Dense"
#include <iostream>

using namespace std;

int main()
{
  Eigen::MatrixXf m(4,4);
  m <<  1, 2, 3, 4,
        5, 6, 7, 8,
        9,10,11,12,
       13,14,15,16;
  cout << "Block in the middle" << endl;
  cout << m.block<2,2>(1,1) << endl << endl;
  for (int i = 1; i <= 3; ++i)
  {
    cout << "Block of size " << i << "x" << i << endl;
    cout << m.block(0,0,i,i) << endl << endl;
  }


  Eigen::Array22f n;
  n << 1,2,
       3,4;
  Eigen::Array44f a = Eigen::Array44f::Constant(0.6);
  cout << "Here is the array a:\n" << a << "\n\n";
  a.block<2,2>(1,1) = n;
  cout << "Here is now a with n copied into its central 2x2 block:\n" << a << "\n\n";
  a.block(0,0,2,3) = a.block(2,1,2,3);
  cout << "Here is now a with bottom-right 2x3 block copied into top-left 2x3 block:\n" << a << "\n\n";


  Eigen::MatrixXf p(3,3);
  p << 1,2,3,
       4,5,6,
       7,8,9;
  cout << "Here is the matrix p:" << endl << p << endl;
  cout << "2nd Row: " << p.row(1) << endl;
  p.col(2) += 3 * p.col(0);
  cout << "After adding 3 times the first column into the third column, the matrix p is:\n";
  cout << p << endl;


  Eigen::Matrix4f q;
  q << 1, 2, 3, 4,
       5, 6, 7, 8,
       9, 10,11,12,
       13,14,15,16;
  cout << "q.leftCols(2) =" << endl << q.leftCols(2) << endl << endl;
  cout << "q.bottomRows<2>() =" << endl << q.bottomRows<2>() << endl << endl;
  q.topLeftCorner(1,3) = q.bottomRightCorner(3,1).transpose();
  cout << "After assignment, q = " << endl << q << endl;


  Eigen::ArrayXf v(6);
  v << 1, 2, 3, 4, 5, 6;
  cout << "v.head(3) =" << endl << v.head(3) << endl << endl;
  cout << "v.tail<3>() = " << endl << v.tail<3>() << endl << endl;
  v.segment(1,4) *= 2;
  cout << "after 'v.segment(1,4) *= 2', v =" << endl << v << endl;
}