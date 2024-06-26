// https://eigen.tuxfamily.org/dox/group__TutorialSlicingIndexing.html

#include <iostream>
#include "../Eigen/Dense"

using namespace std;
using namespace Eigen;

int main(){
    std::vector<int> ind{4,2,5,5,3};
    MatrixXi A = MatrixXi::Random(4,6);
    cout << "Initial matrix A:\n" << A << "\n\n";
    cout << "A(all,ind):\n" << A(Eigen::placeholders::all,ind) << "\n\n";

    cout << "A(all,{4,2,5,5,3}):\n" << A(Eigen::placeholders::all,{4,2,5,5,3}) << "\n\n";

    ArrayXi ind_(5); 
    ind_<<4,2,5,5,3;
    cout << "A(all,ind-1):\n" << A(Eigen::placeholders::all,ind_-1) << "\n\n";

    // This means you can easily build your own fancy sequence generator and pass it to operator().
    // Here is an example enlarging a given matrix while padding the additional first rows and columns through repetition:

    struct pad {
        Index size() const { return out_size; }
        Index operator[] (Index i) const { return std::max<Index>(0,i-(out_size-in_size)); }
        Index in_size, out_size;
    };
    
    Matrix3i B;
    B.reshaped() = VectorXi::LinSpaced(9,1,9);
    cout << "Initial matrix A:\n" <<B << "\n\n";
    MatrixXi C(5,5);
    C = B(pad{3,5}, pad{3,5});
    cout << "A(pad{3,N}, pad{3,N}):\n" << C << "\n\n";
}