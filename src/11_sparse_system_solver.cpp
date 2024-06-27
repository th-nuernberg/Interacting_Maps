#include "../Eigen/Sparse"
using namespace Eigen;
typedef Triplet<double> T;


int main(){
    // ...
    int size = 300;

    SparseMatrix<double> A(size,size);
    std::vector<T> coefficients;            // list of non-zeros coefficients
    Eigen::VectorXd b(size);
    Eigen::VectorXd x(size);

    

    LeastSquaresConjugateGradient<SparseMatrix<double>> solver;
    solver.compute(A);
    if(solver.info()!=Success) {
    // decomposition failed
        return 1;
    }
    x = solver.solve(b);
    if(solver.info()!=Success) {
    // solving failed
        return 2;
    }
    // solve for another right hand side:
    std::cout << x <<
}
