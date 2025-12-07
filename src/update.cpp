//
// Created by root on 7/29/25.
//
#include <update.h>
#include <Eigen/Core>
#include <xtensor.hpp>
#include <xtensor-blas/xblas.hpp>
#include <xtensor-blas/xlinalg.hpp>


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  INTERACTING MAPS HELPER FUNCTIONS  /////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool isApprox(Tensor3f &t1, Tensor3f &t2, const float precision = 1e-8){
    const Map<VectorXf> mt1(t1.data(), t1.size());
    const Map<VectorXf> mt2(t2.data(), t2.size());
    return mt1.isApprox(mt2, precision);
}

bool isApprox(Tensor2f &t1, Tensor2f &t2, const float precision = 1e-8){
    const Map<VectorXf> mt1(t1.data(), t1.size());
    const Map<VectorXf> mt2(t2.data(), t2.size());
    return mt1.isApprox(mt2, precision);
}

void norm_tensor_along_dim3(const Tensor3f &T, Tensor2f &norm){
    constexpr array<int,1> dims({2});
    norm = T.square().sum(dims).sqrt();
}

void norm_tensor_along_dim3(const xt::xtensor<float, 3> &T, xt::xtensor<float, 2> &norm){
    auto Tshape = T.shape();
    auto normShape = norm.shape();
    xt::xarray<float> array(T);
    xt::xarray<float> array2 = xt::zeros_like(norm);
    norm = xt::norm_l2(array, {2});
}

autodiff::Vector3real C_star(autodiff::real x, autodiff::real y, int N_x, int N_y, float height, float width, float rs) {
    autodiff::Vector3real result;
    result << height * (1 - (2 * y) / (N_y - 1)),
              width * (-1 + (2 * x) / (N_x - 1)),
              rs;
    return result;
}

autodiff::Vector3real C(autodiff::real x, autodiff::real y, int N_x, int N_y, float height, float width, float rs) {
    autodiff::Vector3real c_star = C_star(x, y, N_x, N_y, height, width, rs);
    autodiff::real norm = sqrt(c_star.squaredNorm());
    return c_star / norm;
}

void find_C(int N_x, int N_y, float view_angle_x, float view_angle_y, float rs, Tensor3f &CCM, Tensor3f &C_x, Tensor3f &C_y) {
    float height = tan(view_angle_y / 2);
    float width = tan(view_angle_x / 2);
    MatrixXf XX(N_y, N_x);
    MatrixXf YY(N_y, N_x);
    for (int i = 0; i < N_y; ++i) {
        for (int j = 0; j < N_x; ++j) {
            XX(i, j) = float(j);
            YY(i, j) = float(i);
        }
    }
    for (int i = 0; i < N_y; ++i) {
        for (int j = 0; j < N_x; ++j) {
            autodiff::real x = XX(i, j);
            autodiff::real y = YY(i, j);

            // Compute the function value
            autodiff::Vector3real c_val = C(x, y, N_x, N_y, height, width, rs);
            CCM(i,j,0) = static_cast<float>(c_val(0)); // y
            CCM(i,j,1) = static_cast<float>(c_val(1)); // x
            CCM(i,j,2) = static_cast<float>(c_val(2)); // z
            // Compute the jacobians
            autodiff::VectorXreal F;

            // NEEDS TO STAY D O U B L E
            VectorXd dCdx;
            autodiff::jacobian(C, wrt(x), at(x,y,N_x, N_y, height, width, rs), F, dCdx);
            VectorXd dCdy;
            autodiff::jacobian(C, wrt(y), at(x,y,N_x, N_y, height, width, rs), F, dCdy);

            // C_x = dCdx
            C_x(i,j,0) = static_cast<float>(dCdx(0)); // y
            C_x(i,j,1) = static_cast<float>(dCdx(1)); // x
            C_x(i,j,2) = static_cast<float>(dCdx(2)); // z

            // C_y = -dCdy
            C_y(i,j,0) = static_cast<float>(-dCdy(0)); // y
            C_y(i,j,1) = static_cast<float>(-dCdy(1)); // x
            C_y(i,j,2) = static_cast<float>(-dCdy(2)); // z
        }
    }
}

void find_C(size_t N_x, size_t N_y, float view_angle_x, float view_angle_y, float rs, xt::xtensor<float, 3> &CCM, xt::xtensor<float, 3> &C_x, xt::xtensor<float, 3> &C_y) {
    float height = tan(view_angle_y / 2);
    float width = tan(view_angle_x / 2);
    std::vector<size_t> shape = {N_y, N_x};
    xt::xarray<int> X = xt::arange(0, static_cast<int>(N_x), 1);
    xt::xarray<int> Y = xt::arange(0, static_cast<int>(N_y), 1);
    xt::xtensor<int, 2> XX;
    xt::xtensor<int, 2> YY;
    std::tie(YY, XX) = xt::meshgrid(Y, X);
    /*for (int i = 0; i < N_y; ++i) {
        for (int j = 0; j < N_x; ++j) {
            XX.at(i, j) = static_cast<float>(j);
            YY(i, j) = static_cast<float>(i);
        }
    }*/
    std::cout << XX << std::endl;
    std::cout << YY << std::endl;
    for (int i = 0; i < N_y; ++i) {
        for (int j = 0; j < N_x; ++j) {
            autodiff::real x = XX[{i,j}];
            std::cout << x << std::endl;
            std::cout << XX[{i,j}] << std::endl;
            autodiff::real y = YY[{i,j}];
            std::cout << y << std::endl;
            std::cout << YY[{i,j}] << std::endl;

            // Compute the function value
            autodiff::Vector3real c_val = C(x, y, N_x, N_y, height, width, rs);
            std::cout << c_val << std::endl;
            CCM[{i,j,0}] = static_cast<float>(c_val(0)); // y
            CCM[{i,j,1}] = static_cast<float>(c_val(1)); // x
            CCM[{i,j,2}] = static_cast<float>(c_val(2)); // z
            // Compute the jacobians
            autodiff::VectorXreal F;

            // NEEDS TO STAY D O U B L E
            VectorXd dCdx;
            autodiff::jacobian(C, wrt(x), at(x,y,N_x, N_y, height, width, rs), F, dCdx);
            VectorXd dCdy;
            autodiff::jacobian(C, wrt(y), at(x,y,N_x, N_y, height, width, rs), F, dCdy);
            std::cout << dCdx << std::endl;
            std::cout << dCdy << std::endl;
            // C_x = dCdx
            C_x[{i,j,0}] = static_cast<float>(dCdx(0)); // y
            C_x[{i,j,1}] = static_cast<float>(dCdx(1)); // x
            C_x[{i,j,2}] = static_cast<float>(dCdx(2)); // z

            // C_y = -dCdy
            C_y[{i,j,0}] = static_cast<float>(-dCdy(0)); // y
            C_y[{i,j,1}] = static_cast<float>(-dCdy(1)); // x
            C_y[{i,j,2}] = static_cast<float>(-dCdy(2)); // z
        }
    }
}

void crossProduct3x3(const Tensor3f &A, const Tensor3f &B, Tensor3f &C) {
    // Extract slices for each channel
    const auto A0 = A.chip<2>(0);  // A[:,:,0]
    const auto A1 = A.chip<2>(1);  // A[:,:,1]
    const auto A2 = A.chip<2>(2);  // A[:,:,2]

    const auto B0 = B.chip<2>(0);  // B[:,:,0]
    const auto B1 = B.chip<2>(1);  // B[:,:,1]
    const auto B2 = B.chip<2>(2);  // B[:,:,2]

    // Compute cross product components using broadcasting
    C.chip<2>(0) = A2 * B1 - A1 * B2;  // C[:,:,0] = A[:,:,2] * B[:,:,1] - A[:,:,1] * B[:,:,2]
    C.chip<2>(1) = A0 * B2 - A2 * B0;  // C[:,:,1] = A[:,:,0] * B[:,:,2] - A[:,:,2] * B[:,:,0]
    C.chip<2>(2) = A1 * B0 - A0 * B1;  // C[:,:,2] = A[:,:,1] * B[:,:,0] - A[:,:,0] * B[:,:,1]
}

void crossProduct3x3(const xt::xtensor<float, 3>& A,
                     const xt::xtensor<float, 3>& B,
                     xt::xtensor<float, 3>& C) {
    // Extract views for each channel (last dimension)
    const xt::xtensor<float, 2> A0 = xt::view(A, xt::all(), xt::all(), 0);  // A[:,:,0]
    const xt::xtensor<float, 2> A1 = xt::view(A, xt::all(), xt::all(), 1);  // A[:,:,1]
    const xt::xtensor<float, 2> A2 = xt::view(A, xt::all(), xt::all(), 2);  // A[:,:,2]

    const xt::xtensor<float, 2> B0 = xt::view(B, xt::all(), xt::all(), 0);  // B[:,:,0]
    const xt::xtensor<float, 2> B1 = xt::view(B, xt::all(), xt::all(), 1);  // B[:,:,1]
    const xt::xtensor<float, 2> B2 = xt::view(B, xt::all(), xt::all(), 2);  // B[:,:,2]

    // Compute cross product components using broadcasting
    xt::view(C, xt::all(), xt::all(), 0) = A1 * B2 - A2 * B1;  // C[:,:,0] = A[:,:,2] * B[:,:,1] - A[:,:,1] * B[:,:,2]
    xt::view(C, xt::all(), xt::all(), 1) = A2 * B0 - A0 * B2;  // C[:,:,1] = A[:,:,0] * B[:,:,2] - A[:,:,2] * B[:,:,0]
    xt::view(C, xt::all(), xt::all(), 2) = A0 * B1 - A1 * B0;  // C[:,:,2] = A[:,:,1] * B[:,:,0] - A[:,:,0] * B[:,:,1]
}

void crossProduct3x3(const Tensor3f &A, const Vector3f &B, Vector3f &C, int y, int x) {
    C(0) = A(y, x, 2) * B(1) - A(y, x, 1) * B(2);  // y
    C(1) = A(y, x, 0) * B(2) - A(y, x, 2) * B(0);  // x
    C(2) = A(y, x, 1) * B(0) - A(y, x, 0) * B(1);  // z
}

void crossProduct1x3(const Tensor1f &A, const Tensor3f &B, Tensor3f &C){
    // Extract slices for each channel of B
    const auto B0 = B.chip<2>(0);  // B[:,:,0]
    const auto B1 = B.chip<2>(1);  // B[:,:,1]
    const auto B2 = B.chip<2>(2);  // B[:,:,2]

    // Compute cross product components using broadcasting
    C.chip<2>(0) = A(2) * B1 - A(1) * B2;  // C[:,:,0] = A(2) * B[:,:,1] - A(1) * B[:,:,2]
    C.chip<2>(1) = A(0) * B2 - A(2) * B0;  // C[:,:,1] = A(0) * B[:,:,2] - A(2) * B[:,:,0]
    C.chip<2>(2) = A(1) * B0 - A(0) * B1;  // C[:,:,2] = A(1) * B[:,:,0] - A(0) * B[:,:,1]
}

void crossProduct1x3(const xt::xtensor<float, 1>& A,
                     const xt::xtensor<float, 3>& B,
                     xt::xtensor<float, 3>& C) {
    // Extract slices for each channel of B (last dimension)
    const xt::xtensor<float, 2> B0 = xt::view(B, xt::all(), xt::all(), 0);  // B[:,:,0]
    const xt::xtensor<float, 2> B1 = xt::view(B, xt::all(), xt::all(), 1);  // B[:,:,1]
    const xt::xtensor<float, 2> B2 = xt::view(B, xt::all(), xt::all(), 2);  // B[:,:,2]

    // Compute cross product components using broadcasting
    xt::view(C, xt::all(), xt::all(), 0) = A(1) * B2 - A(2) * B1;  // C[:,:,0] = A(2) * B[:,:,1] - A(1) * B[:,:,2]
    xt::view(C, xt::all(), xt::all(), 1) = A(2) * B0 - A(0) * B2;  // C[:,:,1] = A(0) * B[:,:,2] - A(2) * B[:,:,0]
    xt::view(C, xt::all(), xt::all(), 2) = A(0) * B1 - A(1) * B0;  // C[:,:,2] = A(1) * B[:,:,0] - A(0) * B[:,:,1]
}

void vector_distance(const Tensor3f &vec1, const Tensor3f &vec2, Tensor2f &distance){
    PROFILE_FUNCTION();
    const auto& dimensions = vec1.dimensions();
    Tensor3f cross_product(dimensions);
    Tensor2f norm(dimensions[0], dimensions[1]);
    Tensor2f norm2(dimensions[0], dimensions[1]);
    crossProduct3x3(vec1, vec2, cross_product);
    norm.setZero();
    norm2.setZero();
    norm_tensor_along_dim3(cross_product, norm);
    norm_tensor_along_dim3(vec2, norm2);
    distance = norm/norm2;
}

void vector_distance(const xt::xtensor<float, 3> &vec1, const xt::xtensor<float, 3> &vec2, xt::xtensor<float, 2> &distance){
    PROFILE_FUNCTION();
    const shape_type3&shape = vec1.shape();
    const shape_type2&shape2 = distance.shape();
    xt::xtensor<float, 3> cross_product(shape);
    xt::xtensor<float, 2> norm(shape2);
    xt::xtensor<float, 2> norm2(shape2);
    crossProduct3x3(vec1, vec2, cross_product);
    norm_tensor_along_dim3(cross_product, norm);
    norm_tensor_along_dim3(vec2, norm2);
    distance = norm/norm2;
}

float sign_func(const float x){
    if (x > 0)
        return +1.0;
    else if (x == 0)
        return 0.0;
    else
        return -1.0;
}

void computeDotProductWithLoops(const Tensor3f &A, const Tensor3f &B, Tensor2f &D) {
    PROFILE_FUNCTION();
    const int height = (int) A.dimension(0);
    const int width = (int) A.dimension(1);
    const int depth = (int) A.dimension(2);

    // #pragma omp parallel for collapse(2)
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            float dotProduct = 0.0f; // Initialize the dot product for position (i, j)
            for (int k = 0; k < depth; ++k) {
                dotProduct += A(i, j, k) * B(i, j, k);
            }
            D(i, j) = dotProduct; // Store the result in tensor D
        }
    }
}

void computeDotProduct(const Tensor3f& A, const Tensor3f& B, Tensor2f& D) {
    // Compute the element-wise product of A and B
    const Tensor3f elementwiseProduct = A * B;
    // Sum along the depth dimension (k) to get the dot product for each (i, j)
    D = elementwiseProduct.sum(Eigen::array<int, 1>{2});
}

void computeDotProduct(const xt::xtensor<float, 3>& A, const xt::xtensor<float, 3>& B, xt::xtensor<float, 2>& D) {
    // Compute the element-wise product of A and B
    const xt::xtensor<float, 3> elementwiseProduct = A * B;
    // Sum along the depth dimension (k) to get the dot product for each (i, j)
    D = xt::sum(elementwiseProduct, {2});
}

void m32(const Tensor3f &In, const Tensor3f &C_x, const Tensor3f &C_y, Tensor3f &Out){
    const auto& dimensions = In.dimensions();
    Tensor3f C1(dimensions);
    Tensor3f C2(dimensions);
    Tensor2f dot(dimensions[0], dimensions[1]);
    Tensor2f sign(dimensions[0], dimensions[1]);
    Tensor2f distance1(dimensions[0], dimensions[1]);
    Tensor2f distance2(dimensions[0], dimensions[1]);

    crossProduct3x3(C_x,C_y,C1);
    crossProduct3x3(C_y,C1,C2);
    computeDotProductWithLoops(In,C2,dot);
    sign = dot.unaryExpr(std::function(sign_func));
    vector_distance(In, C_y, distance1);
    vector_distance(C_x, C_y, distance2);
    Out.chip(1,2) = sign * distance1/distance2;

    crossProduct3x3(C_y,C_x,C1);
    crossProduct3x3(C_x,C1,C2);
    computeDotProductWithLoops(In,C2,dot);
    sign = dot.unaryExpr(std::function(sign_func));
    vector_distance(In, C_x, distance1);
    vector_distance(C_y, C_x, distance2);
    Out.chip(0,2) = sign * distance1/distance2;
}

void m32(const xt::xtensor<float, 3> &In, const xt::xtensor<float, 3> &C_x, const xt::xtensor<float, 3> &C_y, xt::xtensor<float, 3> &Out){
    const shape_type3&shape = In.shape();
    const shape_type3&shape2 = Out.shape();
    xt::xtensor<float, 3> C1(shape);
    xt::xtensor<float, 3> C2(shape);
    xt::xtensor<float, 2> dot({shape[0], shape[1]});
    xt::xtensor<float, 2> sign({shape[0], shape[1]});
    xt::xtensor<float, 2> distance1({shape[0], shape[1]});
    xt::xtensor<float, 2> distance2({shape[0], shape[1]});

    crossProduct3x3(C_x,C_y,C1);
    crossProduct3x3(C_y,C1,C2);
    computeDotProduct(In,C2,dot);
    sign = xt::sign(dot);
    vector_distance(In, C_y, distance1);
    vector_distance(C_x, C_y, distance2);
    xt::view(Out, xt::all(), xt::all(), 1) = sign * distance1/distance2;

    crossProduct3x3(C_y,C_x,C1);
    crossProduct3x3(C_x,C1,C2);
    computeDotProduct(In,C2,dot);
    sign = xt::sign(dot);
    vector_distance(In, C_x, distance1);
    vector_distance(C_y, C_x, distance2);
    xt::view(Out, xt::all(), xt::all(), 0) = sign * distance1/distance2;
}

void m23(const Tensor3f &In, const Tensor3f &Cx, const Tensor3f &Cy, Vector3f &Out, int y, int x) {
    Out(0) = In(y, x, 1) * Cx(y, x, 0) + In(y, x, 0) * Cy(y, x, 0);
    Out(1) = In(y, x, 1) * Cx(y, x, 1) + In(y, x, 0) * Cy(y, x, 1);
    Out(2) = In(y, x, 1) * Cx(y, x, 2) + In(y, x, 0) * Cy(y, x, 2);
}

void m23(const Tensor3f &In, const Tensor3f &Cx, const Tensor3f &Cy, Tensor3f &Out) {
    Out.chip<2>(0) = In.chip<2>(1) * Cx.chip<2>(0) + In.chip<2>(0) * Cy.chip<2>(0);
    Out.chip<2>(1) = In.chip<2>(1) * Cx.chip<2>(1) + In.chip<2>(0) * Cy.chip<2>(1);
    Out.chip<2>(2) = In.chip<2>(1) * Cx.chip<2>(2) + In.chip<2>(0) * Cy.chip<2>(2);
}

void m23(const xt::xtensor<float, 3>& In,
         const xt::xtensor<float, 3>& Cx,
         const xt::xtensor<float, 3>& Cy,
         xt::xtensor<float, 3>& Out) {
    // Use xt::view to access each channel (last dimension)
    xt::view(Out, xt::all(), xt::all(), 0) = xt::view(In, xt::all(), xt::all(), 1) * xt::view(Cx, xt::all(), xt::all(), 0) +
                                                 xt::view(In, xt::all(), xt::all(), 0) * xt::view(Cy, xt::all(), xt::all(), 0);

    xt::view(Out, xt::all(), xt::all(), 1) = xt::view(In, xt::all(), xt::all(), 1) * xt::view(Cx, xt::all(), xt::all(), 1) +
                                                 xt::view(In, xt::all(), xt::all(), 0) * xt::view(Cy, xt::all(), xt::all(), 1);

    xt::view(Out, xt::all(), xt::all(), 2) = xt::view(In, xt::all(), xt::all(), 1) * xt::view(Cx, xt::all(), xt::all(), 2) +
                                                 xt::view(In, xt::all(), xt::all(), 0) * xt::view(Cy, xt::all(), xt::all(), 2);
}
void computeGradient(const Tensor2f &data, Tensor3f &gradients, int y, int x) {
    PROFILE_FUNCTION();
    // Compute gradient for update_IG
    const auto& gdimensions = data.dimensions();
    int rows = static_cast<int>(gdimensions[0]);
    int cols = static_cast<int>(gdimensions[1]);
    assert(y < rows);
    assert(x < cols);

    // Compute gradient along columns (down-up, y-direction)
    if (y == 0) {
        gradients(y, x, 0) = (data(y, x) - data(y + 1, x)) / 2.0f; // Central difference with replicate border
    } else if (y == rows - 1) {
        gradients(y, x, 0) = (data(y - 2, x) - data(y - 1, x)) / 2.0f; // Central difference with replicate border
    } else {
        gradients(y, x, 0) = (data(y - 1, x) - data(y + 1, x)) / 2.0f;
    }
    // Compute gradient along rows (left-right, x-direction)
    if (x == 0) {
        gradients(y, x, 1) = (data(y, x + 1) - data(y, x)) / 2.0f; // Central difference with replicate border
    } else if (x == cols - 1) {
        gradients(y, x, 1) = (data(y, x - 1) - data(y, x - 2)) / 2.0f; // Central difference with replicate border
    } else {
        gradients(y, x, 1) = (data(y, x + 1) - data(y, x - 1)) / 2.0f;
    }
}

void computeGradient(const Tensor2f& data, Tensor3f& gradients) {
    const int rows = static_cast<int>(data.dimension(0));
    const int cols = static_cast<int>(data.dimension(1));

    // Compute gradient along columns (y-direction)
    // Central difference for interior points
    gradients.slice(Eigen::array<Eigen::Index, 3> {1,0,0}, Eigen::array<Eigen::Index, 3> {rows-2,cols,1}).chip<2>(0) =
        (data.slice(Eigen::array<int, 2>{2, 0}, Eigen::array<int, 2>{rows - 2, cols}) -
         data.slice(Eigen::array<int, 2>{0, 0}, Eigen::array<int, 2>{rows - 2, cols})) / 2.0f;

    // Forward difference for top border (y = 0)
    gradients.chip<2>(0).chip<0>(0) =
        (data.chip<0>(1) - data.chip<0>(0));

    // Backward difference for bottom border (y = rows - 1)
    gradients.chip<2>(0).chip<0>(rows - 1) =
        (data.chip<0>(rows - 1) - data.chip<0>(rows - 2));

    // Compute gradient along rows (x-direction)
    // Central difference for interior points
    gradients.slice(Eigen::array<Eigen::Index, 3> {0,1,1}, Eigen::array<Eigen::Index, 3> {rows,cols-2,1}).chip<2>(0) =
        (data.slice(Eigen::array<int, 2>{0, 2}, Eigen::array<int, 2>{rows, cols-2}) -
         data.slice(Eigen::array<int, 2>{0, 0}, Eigen::array<int, 2>{rows, cols-2})) / 2.0f;

    // Forward difference for top border (x = 0)
    gradients.chip<2>(1).chip<1>(0) =
        (data.chip<1>(1) - data.chip<1>(0));

    // Backward difference for bottom border (x = cols - 1)
    gradients.chip<2>(1).chip<1>(cols - 1) =
        (data.chip<1>(cols - 1) - data.chip<1>(cols - 2));
}

void computeGradient(const xt::xtensor<float, 2>& data, xt::xtensor<float, 3>& gradients) {
    const size_t rows = data.shape()[0];
    const size_t cols = data.shape()[1];

    // Resize gradients if necessary (assuming it's already sized correctly)
    // gradients.resize({rows, cols, 2});

    // Compute gradient along columns (y-direction)
    // Central difference for interior points
    xt::view(gradients, xt::range(1, rows - 1), xt::all(), 0) =
        (xt::view(data, xt::range(2, rows), xt::all()) -
         xt::view(data, xt::range(0, rows - 2), xt::all())) / 2.0f;

    // Forward difference for top border (y = 0)
    xt::view(gradients, 0, xt::all(), 0) =
        xt::view(data, 1, xt::all()) - xt::view(data, 0, xt::all());

    // Backward difference for bottom border (y = rows - 1)
    xt::view(gradients, rows - 1, xt::all(), 0) =
        xt::view(data, rows - 1, xt::all()) - xt::view(data, rows - 2, xt::all());

    // Compute gradient along rows (x-direction)
    // Central difference for interior points
    xt::view(gradients, xt::all(), xt::range(1, cols - 1), 1) =
        (xt::view(data, xt::all(), xt::range(2, cols)) -
         xt::view(data, xt::all(), xt::range(0, cols - 2))) / 2.0f;

    // Forward difference for left border (x = 0)
    xt::view(gradients, xt::all(), 0, 1) =
        xt::view(data, xt::all(), 1) - xt::view(data, xt::all(), 0);

    // Backward difference for right border (x = cols - 1)
    xt::view(gradients, xt::all(), cols - 1, 1) =
        xt::view(data, xt::all(), cols - 1) - xt::view(data, xt::all(), cols - 2);
}

void computeGradient(const Tensor3f &data, Tensor3f &gradients, int y, int x) {
    // Compute gradient for update_IG
    const auto& dimensions = data.dimensions();
    int rows = static_cast<int>(dimensions[0]);
    int cols = static_cast<int>(dimensions[1]);
    assert(y < rows);
    assert(x < cols);
    assert(static_cast<int>(dimensions[2]) == 2);
    // Compute gradient along columns (down-up, y-direction)
    if (y == 0) {
        gradients(y, x, 0) = (data(y, x, 0) - data(y + 1, x, 0)) / 2.0f; // Central difference with replicate border
    } else if (y == rows - 1) {
        gradients(y, x, 0) = (data(y - 2, x, 0) - data(y - 1, x, 0)) / 2.0f; // Central difference with replicate border
    } else {
        gradients(y, x, 0) = (data(y - 1, x, 0) - data(y + 1, x, 0)) / 2.0f;
    }
    // Compute gradient along rows (left-right, x-direction)
    if (x == 0) {
        gradients(y, x, 1) = (data(y, x + 1, 1) - data(y, x, 1)) / 2.0f; // Central difference with replicate border
    } else if (x == cols - 1) {
        gradients(y, x, 1) = (data(y, x - 1, 1) - data(y, x - 2, 1)) / 2.0f; // Central difference with replicate border
    } else {
        gradients(y, x, 1) = (data(y, x + 1, 1) - data(y, x - 1, 1)) / 2.0f;
    }
}

void computeGradient(const xt::xtensor<float, 3>& data, xt::xtensor<float, 3>& gradients) {
    // Get dimensions
    auto shape = data.shape();
    size_t rows = shape[0];
    size_t cols = shape[1];

    // Pad the data with replicate borders
    xt::xtensor<float, 3> padded_data = xt::pad(data, {{1, 1}, {1, 1}, {0, 0}}, xt::pad_mode::reflect);

    // Compute gradient along columns (y-direction)
    // Central difference for all points using padded data
    xt::view(gradients, xt::all(), xt::all(), 0) =
        (xt::view(padded_data, xt::range(0, rows), xt::range(1, cols+1), 0) -
         xt::view(padded_data, xt::range(2, rows + 2), xt::range(1, cols+1), 0)) / 2.0f;

    // Compute gradient along rows (x-direction)
    // Central difference for all points using padded data
    xt::view(gradients, xt::all(), xt::all(), 1) =
        (xt::view(padded_data, xt::range(1, rows+1), xt::range(2, cols + 2), 1) -
         xt::view(padded_data, xt::range(1, rows+1), xt::range(0, cols), 1)) / 2.0f;
}

float VFG_check(const Tensor2f &V, const Tensor3f &F, const Tensor3f &G){
    const auto& dimensions = F.dimensions();
    MatrixXfRowMajor dot(dimensions[0], dimensions[1]);
    MatrixXfRowMajor diff(dimensions[0], dimensions[1]);

    for (int i = 0; i<dimensions[0]; i++){
        for (int j = 0; j<dimensions[1]; j++){
            dot(i,j) = -(F(i,j,0)*G(i,j,0) + F(i,j,1)*G(i,j,1));
            diff(i,j) = (V(i,j) - dot(i,j));
        }
    }
    return diff.lpNorm<Infinity>();
}

float VFG_check(const xt::xtensor<float, 2>& V,
                const xt::xtensor<float, 3>& F,
                const xt::xtensor<float, 3>& G) {
    // Compute dot product for all (i,j) using vectorized operations
    auto dot = -(xt::view(F, xt::all(), xt::all(), 0) * xt::view(G, xt::all(), xt::all(), 0) +
                 xt::view(F, xt::all(), xt::all(), 1) * xt::view(G, xt::all(), xt::all(), 1));

    // Compute difference
    auto diff = V - dot;

    // Return infinity norm of the difference
    return xt::amax(xt::abs(diff))();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  INTERACTING MAPS UPDATE FUNCTIONS  /////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void setup_R_update(const Tensor3f &CCM, Matrix3f &A, Vector3f &B, std::vector<std::vector<Matrix3f>> &Identity_minus_outerProducts, std::vector<std::vector<Vector3f>> &points){
    PROFILE_FUNCTION();
    const auto &dimensions = CCM.dimensions();
    int rows = (int) dimensions[0];
    int cols = (int) dimensions[1];
    Matrix3f Identity = Matrix3f::Identity();
    Vector3f d;
    B.setZero();

    for (size_t i = 0; i < rows; ++i){
        for (size_t j = 0; j < cols; ++j){
            d(0) = CCM((int) i, (int) j, 0);
            d(1) = CCM((int) i, (int) j, 1);
            d(2) = CCM((int) i, (int) j, 2);
            Identity_minus_outerProducts[i][j] = Identity - d * d.transpose();
            A += Identity_minus_outerProducts[i][j];
            points[i][j].setZero();
        }
    }
}

void setup_R_update(const xt::xtensor<float, 3>& CCM,
                    xt::xtensor<float, 2>& A,
                    xt::xtensor<float, 1>& B,
                    xt::xtensor<float, 4>& Identity_minus_outerProducts,
                    xt::xtensor<float, 3>& points) {
    auto dimensions = CCM.shape();
    size_t rows = dimensions[0];
    size_t cols = dimensions[1];

    // Create a 3x3 identity matrix
    xt::xtensor<float, 2> Identity = xt::eye<float>(3);

    // Initialize B to zero
    B.fill(0);

    // Initialize A to zero
    A.fill(0);

    // Temporary vector for d
    xt::xtensor<float, 1> d = xt::zeros<float>({3});

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            // Extract d from CCM
            d(0) = CCM(i, j, 0);
            d(1) = CCM(i, j, 1);
            d(2) = CCM(i, j, 2);

            // Compute Identity - d*d^T and store in 4D tensor
            xt::view(Identity_minus_outerProducts, i, j, xt::all(), xt::all()) = Identity - xt::linalg::outer(d, d);

            // Accumulate to A
            A += xt::view(Identity_minus_outerProducts, i, j, xt::all(), xt::all());

            // Set points[i][j] to zero in 3D tensor
            xt::view(points, i, j, xt::all()) = xt::zeros<float>({3});
        }
    }
}


void update_FG(Tensor3f &F, const float V, const Tensor3f &G, int y, int x, const float lr, const float weight_FG, float eps=1e-8, float gamma=255.0){
    PROFILE_FUNCTION();
    Vector2f update_F;
    update_F.setZero();
    float norm = std::abs((G(y, x, 0) * G(y, x, 0) + G(y, x, 1) * G(y, x, 1)));
    if (norm != 0.0) {
        update_F(0) = F(y, x, 0) - ((G(y, x, 0) / norm) * (V + (F(y, x, 0) * G(y, x, 0) + F(y, x, 1) * G(y, x, 1))));
        update_F(1) = F(y, x, 1) - ((G(y, x, 1) / norm) * (V + (F(y, x, 0) * G(y, x, 0) + F(y, x, 1) * G(y, x, 1))));
        F(y, x, 0) = (1 - weight_FG) * F(y, x, 0) + lr * weight_FG * update_F(0);
        F(y, x, 1) = (1 - weight_FG) * F(y, x, 1) + lr * weight_FG * update_F(1);
        if (F(y, x, 0) > gamma){
            F(y, x, 0) = gamma;
        }
        if (F(y, x, 1) > gamma){
            F(y, x, 1) = gamma;
        }
        if (F(y, x, 0) < -gamma){
            F(y, x, 0) = -gamma;
        }
        if (F(y, x, 1) < -gamma){
            F(y, x, 1) = -gamma;
        }
        if (std::abs(F(y, x, 0)) < eps){
            F(y, x, 0) = 0.0;
        }
        if (std::abs(F(y, x, 1)) < eps){
            F(y, x, 1) = 0.0;
        }
    }
}

void update_FG(xt::xtensor<float, 3> &F,
               const xt::xtensor<float, 2> &V,
               const xt::xtensor<float, 3> &G,
               const float lr,
               const float weight_FG,
               const float eps,
               const float gamma) {
    // Compute the norm of G for each (i,j)

    xt::xtensor<float, 3> G_square = xt::square(G);
    xt::xtensor<float, 2> norm = xt::sum(G_square, {2});  // Sum over the last dimension (z)
    //std::cout << norm << std::endl;

    // Avoid division by zero
    xt::xtensor<float, 2> safe_norm = xt::where(xt::equal(norm, 0.0f), 1e-8f, norm);
    //std::cout << safe_norm << std::endl;

    // Compute the dot product of F and G for each (i,j)
    xt::xtensor<float, 3> F_dot_G = F * G;
    //std::cout << F_dot_G << std::endl;
    xt::xtensor<float, 2> F_dot_G_sum = xt::sum(F_dot_G, {2});  // Sum over the last dimension (z)
    //std::cout << F_dot_G_sum << std::endl;

    // Compute the update term
    xt::xtensor<float, 2> update_term = (V + F_dot_G_sum) / safe_norm;
    //std::cout << update_term << std::endl;

    // Compute update_F for each (i,j)
    auto dimensions = F.shape();
    int rows = dimensions[0];
    int cols = dimensions[1];

    xt::xtensor<float, 3> update_term2 = xt::broadcast(update_term, {2,rows,cols});
    //std::cout << update_term2 << std::endl;

    update_term2 = xt::transpose(update_term2, {1,2,0});
    //std::cout << update_term2 << std::endl;

    xt::xtensor<float, 3> update_term3 = G * update_term2;
    //std::cout << update_term3 << std::endl;

    xt::xtensor<float, 3> update_F = F - update_term3;
    //std::cout << update_F << std::endl;


    // Update F using the update rule
    F = (1.0f - weight_FG) * F + lr * weight_FG * update_F;
    //std::cout << F << std::endl;

    // Clamp F to [-gamma, gamma]
    F = xt::clip(F, -gamma, gamma);

    /*// Set near-zero values to zero
    F = xt::where(xt::abs(F) < eps, 0.0f, F);*/
}

void update_GF(Tensor3f &G, float V, const Tensor3f &F, int y, int x, const float lr, const float weight_GF, float eps=1e-8, float gamma=255.0){
    PROFILE_FUNCTION();
    Vector2f update_G;
    update_G.setZero();
    float norm = std::abs((F(y, x, 0) * F(y, x, 0) + F(y, x, 1) * F(y, x, 1)));
    if (norm != 0.0) {
        update_G(0) = G(y, x, 0) - ((F(y, x, 0) / norm) * (V + (G(y, x, 0) * F(y, x, 0) + G(y, x, 1) * F(y, x, 1))));
        update_G(1) = G(y, x, 1) - ((F(y, x, 1) / norm) * (V + (G(y, x, 0) * F(y, x, 0) + G(y, x, 1) * F(y, x, 1))));
        G(y, x, 0) = (1 - weight_GF) * G(y, x, 0) + lr * weight_GF * update_G(0);
        G(y, x, 1) = (1 - weight_GF) * G(y, x, 0) + lr * weight_GF * update_G(1);
        if (G(y, x, 0) > gamma){
            G(y, x, 0) = gamma;
        }
        if (G(y, x, 1) > gamma){
            G(y, x, 1) = gamma;
        }
        if (G(y, x, 0) < -gamma){
            G(y, x, 0) = -gamma;
        }
        if (G(y, x, 1) < -gamma){
            G(y, x, 1) = -gamma;
        }
        if (std::abs(G(y, x, 0)) < eps){
            G(y, x, 0) = 0.0;
        }
        if (std::abs(G(y, x, 1)) < eps){
            G(y, x, 1) = 0.0;
        }
    }
}

void update_GI(Tensor3f &G, const Tensor3f &I_gradient, int y, int x, float weight_GI, float eps, float gamma){
    PROFILE_FUNCTION();
    G(y, x, 0) = (1 - weight_GI) * G(y, x, 0) + weight_GI*I_gradient(y, x, 0);
    G(y, x, 1) = (1 - weight_GI) * G(y, x, 1) + weight_GI*I_gradient(y, x, 1);
    if (G(y, x, 0) > gamma){
        G(y, x, 0) = gamma;
    }
    if (G(y, x, 1) > gamma){
        G(y, x, 1) = gamma;
    }
    if (G(y, x, 0) < -gamma){
        G(y, x, 0) = -gamma;
    }
    if (G(y, x, 1) < -gamma){
        G(y, x, 1) = -gamma;
    }
    if (std::abs(G(y, x, 0)) < eps){
        G(y, x, 0) = 0;
    }
    if (std::abs(G(y, x, 1)) < eps){
        G(y, x, 1) = 0;
    }
}

void update_GI(xt::xtensor<float, 3>& G,
               const xt::xtensor<float, 3>& I_gradient,
               float weight_GI,
               float eps,
               float gamma) {
    // Update G using the update rule for all elements
    G = (1.0f - weight_GI) * G + weight_GI * I_gradient;

    // Clamp G to [-gamma, gamma] for all elements
    G = xt::clip(G, -gamma, gamma);
}

void updateGIDiffGradient(Tensor3f &G, Tensor3f &I_gradient, Tensor3f &GIDiff, Tensor3f &GIDiffGradient, int y, int x){
    PROFILE_FUNCTION();
    GIDiff(y, x, 0) = G(y, x, 0) - I_gradient(y, x, 0);
    GIDiff(y, x, 1) = G(y, x, 1) - I_gradient(y, x, 1);
    computeGradient(GIDiff, GIDiffGradient, y, x);
}

void updateGIDiffGradient(xt::xtensor<float, 3>& G,
                          xt::xtensor<float, 3>& I_gradient,
                          xt::xtensor<float, 3>& GIDiff,
                          xt::xtensor<float, 3>& GIDiffGradient) {
    // Compute the difference between G and I_gradient for all elements
    GIDiff = G - I_gradient;

    // Compute the gradient of GIDiff for all elements
    computeGradient(GIDiff, GIDiffGradient);
}

void update_IG(Tensor2f &I, const Tensor3f &GIDiffGradient, int y, int x, float weight_IG){
    PROFILE_FUNCTION();
    I(y, x) = I(y, x) + weight_IG * (- GIDiffGradient(y, x, 0) - GIDiffGradient(y, x, 1));
}

void update_IG(xt::xtensor<float, 2>& I,
               const xt::xtensor<float, 3>& GIDiffGradient,
               float weight_IG) {
    // Sum the last dimension of GIDiffGradient (sum of x and y components)
    const xt::xtensor<float, 2> gradient_sum = xt::sum(GIDiffGradient, {2});

    // Update I using the update rule for all elements
    I = (1-weight_IG) * I + weight_IG * (-gradient_sum);
}

void contribute(Tensor2f &I, float V, int y, int x, float minPotential, float maxPotential, const float weight_IV){
    I(y, x) = std::min(std::max(I(y, x) + weight_IV * V, minPotential), maxPotential);
}

void globalDecay(Tensor2f &I, Tensor2f &decayTimeSurface, Tensor2f &nP, Tensor2f &t, Tensor2f &dP) {
    const Tensor2f lastPotential = I;
    I = (lastPotential - nP) * (-(t - decayTimeSurface) / dP).exp() + nP;
    decayTimeSurface = t;
}

void linearDecay(Tensor2f &I, Tensor2f &decayTimeSurface, const int y, const int x, const float time, const float neutralPotential, const float decayParam){
    const float lastDecayTime = decayTimeSurface(y,x);
    const float lastPotential = I(y,x);
    I(y,x)
        = (lastPotential >= neutralPotential)
            ? std::max(lastPotential - (time - lastDecayTime) * decayParam, neutralPotential)
            : std::min(lastPotential + (time - lastDecayTime) * decayParam, neutralPotential);
    decayTimeSurface(y, x) = time;
}

void exponentialDecay(Tensor2f &I, Tensor2f &decayTimeSurface, const int y, const int x, const float time, const float neutralPotential, const float decayParam){
    const float lastDecayTime = decayTimeSurface(y,x);
    const float lastPotential = I(y,x);
    I(y,x)
        = (lastPotential - neutralPotential) * expf(-(time-lastDecayTime) / decayParam)
        + neutralPotential;
    decayTimeSurface(y,x) = time;
    //float newIntensity = (I(y, x) - neutralPotential) * expf(-(time - decayTimeSurface(y, x)) / decayParam) + neutralPotential;
    //I(y, x) = newIntensity;
    //decayTimeSurface(y, x) = time;
}

void update_IV(Tensor2f &I, const float V, const int y, const int x, const float minPotential, const float maxPotential, const float weight_IV){
    PROFILE_FUNCTION();
    contribute(I, V, y, x, minPotential, maxPotential, weight_IV);
}

void update_Ifusion(Tensor2f &I, const cv::Mat &realImage, const float weight_Ifusion) {
    Tensor2f lastPotential = I;
    I = (1-weight_Ifusion) * lastPotential + 255 * weight_Ifusion * Matrix2Tensor(cvMatToEigen(realImage));
}

void update_FR(Tensor3f &F, const Tensor3f &CCM, const Tensor3f &Cx, const Tensor3f &Cy, const Tensor1f &R, const float weight_FR, float eps=1e-8, float gamma=255.0){
    PROFILE_FUNCTION();
    Tensor3f cross(CCM.dimensions());
    const auto& dimensions = F.dimensions();
    Tensor3f update(F.dimensions());
    {
        PROFILE_SCOPE("FR CROSS PRODUCT");
        crossProduct1x3(R, CCM, cross);
    }
    {
        PROFILE_SCOPE("FR M32");
        m32(cross, Cx, Cy, update);
    }
    F = (1 - weight_FR)*F + weight_FR*update;
    for (int i = 0; i<dimensions[0]; i++) {
        for (int j = 0; j < dimensions[1]; j++) {
            if (F(i, j, 0) > gamma){
                F(i, j, 0) = gamma;
            }
            if (F(i, j, 1) > gamma){
                F(i, j, 1) = gamma;
            }
            if (F(i, j, 0) < -gamma){
                F(i, j, 0) = -gamma;
            }
            if (F(i, j, 1) < -gamma){
                F(i, j, 1) = -gamma;
            }
            if (std::abs(F(i,j,0)) < eps){
                F(i,j,0) = 0;
            }
            if (std::abs(F(i,j,1)) < eps){
                F(i,j,1) = 0;
            }
        }
    }
}

void update_FR(xt::xtensor<float, 3>& F,
              const xt::xtensor<float, 3>& CCM,
              const xt::xtensor<float, 3>& Cx,
              const xt::xtensor<float, 3>& Cy,
              const xt::xtensor<float, 1>& R,
              const float weight_FR,
              float eps = 1e-8f,
              float gamma = 255.0f) {
    // Create tensors for intermediate results
    xt::xtensor<float, 3> cross = xt::zeros<float>(CCM.shape());
    xt::xtensor<float, 3> update = xt::zeros<float>(F.shape());

    // Compute cross product
    crossProduct1x3(R, CCM, cross);

    // Compute m32
    m32(cross, Cx, Cy, update);

    // Update F using the update rule
    F = (1.0f - weight_FR) * F + weight_FR * update;

    // Clamp F to [-gamma, gamma] for all elements
    F = xt::clip(F, -gamma, gamma);

    // Set near-zero values to zero for all elements
    //F = xt::where(xt::abs(F) < eps, xt::zeros<float>(F.shape()), F);
}

//void update_RF(Tensor1f &R, const Tensor3f &F, const Tensor3f &C, const Tensor3f &Cx, const Tensor3f &Cy, const Matrix3f &A, Vector3f &B, const std::vector<std::vector<Matrix3f>> &Identity_minus_outerProducts, std::vector<std::vector<Vector3f>> &old_points, const float weight_RF, const std::vector<Event> &frameEvents) {
//    PROFILE_FUNCTION();
//    const auto &dimensions = F.dimensions();
//    Vector3f transformed_F(3);
//    Vector3f point(3);
//    Vector3f solution(3);
//    {
//        PROFILE_SCOPE("RF Pre");
//        for (auto event : frameEvents){
//            // Transform F from 2D image space to 3D world space with C
//            m23(F, Cx, Cy, transformed_F, event.coordinates[0], event.coordinates[1]);
//            // calculate cross product between world space F and calibration matrix.
//            // this gives us the point on which the line stands
//            crossProduct3x3(C, transformed_F, point, event.coordinates[0], event.coordinates[1]);
//            // right hand side B consists of a sum of a points
//            // subtract the contribution of the old_point at y,x and add the contribution of the new point
//            B = B - Identity_minus_outerProducts[event.coordinates[0]][event.coordinates[1]]*old_points[event.coordinates[0]][event.coordinates[1]] + Identity_minus_outerProducts[event.coordinates[0]][event.coordinates[1]]*point;
//            // new point is now old
//            old_points[event.coordinates[0]][event.coordinates[1]] = point;
//        }
//    }
//    // solve for the new rotation vector
//    solution = A.partialPivLu().solve(B);
//    R(0) = (1 - weight_RF) * R(0) + weight_RF * solution(0);
//    R(1) = (1 - weight_RF) * R(1) + weight_RF * solution(1);
//    R(2) = (1 - weight_RF) * R(2) + weight_RF * solution(2);
//}

void update_RF(Tensor1f &R, const Tensor3f &F, const Tensor3f &C, const Tensor3f &Cx, const Tensor3f &Cy, const Matrix3f &A, Vector3f &B, const std::vector<std::vector<Matrix3f>> &Identity_minus_outerProducts, std::vector<std::vector<Vector3f>> &old_points, float weight_RF, int y, int x) {
    PROFILE_FUNCTION();
    Vector3f transformed_F(3);
    Vector3f point(3);
    Vector3f solution(3);
    {
        PROFILE_SCOPE("RF Pre");
        // Transform F from 2D image space to 3D world space with C
        m23(F, Cx, Cy, transformed_F, y, x);
        // calculate cross product between world space F and calibration matrix.
        // this gives us the point on which the line stands
        crossProduct3x3(C, transformed_F, point, y, x);
        // right hand side B consists of a sum of a points
        // subtract the contribution of the old_point at y,x and add the contribution of the new point
        B = B - Identity_minus_outerProducts[y][x]*old_points[y][x] + Identity_minus_outerProducts[y][x]*point;
        // new point is now old
        old_points[y][x] = point;
    }
    // solve for the new rotation vector
    solution = A.partialPivLu().solve(B);
    R(0) = (1 - weight_RF) * R(0) + weight_RF * solution(0);
    R(1) = (1 - weight_RF) * R(1) + weight_RF * solution(1);
    R(2) = (1 - weight_RF) * R(2) + weight_RF * solution(2);
}

void update_RF(xt::xtensor<float, 1>& R,
               const xt::xtensor<float, 3>& F,
               const xt::xtensor<float, 3>& C,
               const xt::xtensor<float, 3>& Cx,
               const xt::xtensor<float, 3>& Cy,
               const xt::xtensor<float, 2>& A,
               xt::xtensor<float, 1>& B,
               const xt::xtensor<float, 4>& Identity_minus_outerProducts,
               xt::xtensor<float, 3>& old_points,
               float weight_RF) {
    auto dimensions = F.shape();
    int rows = dimensions[0];
    int cols = dimensions[1];

    // Create temporary tensors for intermediate results
    xt::xtensor<float, 3> transformed_F = xt::zeros<float>({rows, cols, 3});
    xt::xtensor<float, 3> points = xt::zeros_like(old_points);
    xt::xtensor<float, 1> solution = xt::zeros<float>({3});

    // Transform F from 2D image space to 3D world space with C
    m23(F, Cx, Cy, transformed_F);

    // Calculate cross product between world space F and calibration matrix
    crossProduct3x3(C, transformed_F, points);

    // Update B for all (y, x)
    // B = B - sum(Identity_minus_outerProducts * old_points) + sum(Identity_minus_outerProducts * point)
    // We need to compute this for all (y, x) pairs

    xt::xtensor<float, 1> new_B = xt::zeros<float>({3});
    for (size_t y = 0; y < rows; ++y) {
        for (size_t x = 0; x < cols; ++x) {
            // Get the current Identity_minus_outerProducts and points
            xt::xtensor<float, 2> Ip = xt::view(Identity_minus_outerProducts, y, x, xt::all(), xt::all());
            xt::xtensor<float, 1> old_p = xt::view(old_points, y, x, xt::all());

            // Update B: B = B - Ip * old_p + Ip * point(y, x, :)
            //new_B -= Ip * old_p;
            new_B += Ip * xt::view(points, y, x, xt::all());
        }
    }
    B = new_B;

    // Update old_points for all (y, x)
    old_points = points;

    // Solve for the new rotation vector
    solution = xt::linalg::solve(A, B);

    // Update R using the update rule
    R = (1.0f - weight_RF) * R + weight_RF * solution;
}

void update_RIMU(Tensor1f &R, const std::vector<float> &rotVelIMU, const float weight_RIMU) {
    R(0) = (1 - weight_RIMU)*R(0) + weight_RIMU*rotVelIMU[0];
    R(1) = (1 - weight_RIMU)*R(1) + weight_RIMU*rotVelIMU[1];
    R(2) = (1 - weight_RIMU)*R(2) + weight_RIMU*rotVelIMU[2];
}



