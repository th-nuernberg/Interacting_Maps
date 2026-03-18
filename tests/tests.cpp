//
// Created by Daniel Pommer on 02.12.25.
//
#include <iostream>
#include <cstdlib> // for EXIT_SUCCESS, EXIT_FAILURE
#include <update.h>
#include <cmath>

#include "boost/fusion/sequence/io/out.hpp"

/*void computeGradient(const Eigen::Tensor<float, 2>& data, Eigen::Tensor<float, 3>& gradients) {
    const int rows = static_cast<int>(data.dimension(0));
    const int cols = static_cast<int>(data.dimension(1));

    // Resize gradients tensor to match expected output shape
    gradients.resize(rows, cols, 2);

    // Compute gradient along columns (y-direction)
    // Central difference for interior points
    gradients.slice(Eigen::array<Eigen::Index, 3> {1,0,0}, Eigen::array<Eigen::Index, 3> {rows-2,cols,1}).chip<2>(0) =
        (data.slice(Eigen::array<int, 2>{0, 0}, Eigen::array<int, 2>{rows - 2, cols}) -
         data.slice(Eigen::array<int, 2>{2, 0}, Eigen::array<int, 2>{rows - 2, cols})) / 2.0f;

    // Forward difference for top border (y = 0)
    gradients.chip<2>(0).chip<0>(0) =
        (data.chip<0>(0) - data.chip<0>(1));

    // Backward difference for bottom border (y = rows - 1)
    gradients.chip<2>(0).chip<0>(rows - 1) =
        (data.chip<0>(rows - 1) - data.chip<0>(rows));

    // Compute gradient along rows (x-direction)
    // Central difference for interior points
    gradients.slice(Eigen::array<Eigen::Index, 3> {0,1,1}, Eigen::array<Eigen::Index, 3> {rows,cols-2,1}).chip<2>(1) =
        (data.slice(Eigen::array<int, 2>{0, 0}, Eigen::array<int, 2>{rows, cols-2}) -
         data.slice(Eigen::array<int, 2>{0, 2}, Eigen::array<int, 2>{rows, cols-2})) / 2.0f;

    // Forward difference for top border (x = 0)
    gradients.chip<2>(1).chip<1>(0) =
        (data.chip<1>(0) - data.chip<1>(1));

    // Backward difference for bottom border (x = cols - 1)
    gradients.chip<2>(1).chip<1>(cols - 1) =
        (data.chip<1>(cols - 1) - data.chip<1>(cols));
}*/

bool areTensorsEqual(const Tensor3f& A, const Tensor3f& B) {
    // Check if dimensions match
    if (A.dimensions() != B.dimensions()) {
        return false;
    }

    // Compare each element
    for (int i = 0; i < A.dimension(0); ++i) {
        for (int j = 0; j < A.dimension(1); ++j) {
            for (int k = 0; k < A.dimension(2); ++k) {
                if (!(abs(A(i, j, k) - B(i, j, k)) <= 1e-4)) {
                    return false;
                }
            }
        }
    }
    return true;
}

bool areTensorsEqual(const Tensor2f& A, const Tensor2f& B) {
    // Check if dimensions match
    if (A.dimensions() != B.dimensions()) {
        return false;
    }

    // Compare each element
    for (int i = 0; i < A.dimension(0); ++i) {
        for (int j = 0; j < A.dimension(1); ++j) {
            if (!(abs(A(i, j) - B(i, j)) <= 1e-4)) {
                return false;
            }
        }
    }
    return true;
}

bool areTensorsEqual(const xt::xarray<float>& A, const xt::xarray<float>& B) {
    // Check if shapes match
    if (A.shape() != B.shape()) {
        return false;
    }

    // Compare each element with a tolerance
    return xt::all(xt::abs(A - B) <= 1e-4f);
}

/*int test_gradient() {
    Tensor2f A(3,3);
    A.setValues({{1,2,3},{4,5,6},{7,8,9}});
    Tensor3f B(3,3,2);
    Tensor3f gradients(3,3,2);
    Tensor3f expected_gradients(3,3,2);
    expected_gradients.chip<2>(0).setConstant(3.0f);
    expected_gradients.chip<2>(1).setConstant(1.0f);
    computeGradient(A, gradients);
    if (!areTensorsEqual(gradients, expected_gradients)) {
        std::cerr << "Test failed: gradients != gradients\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}*/

int test_gradient() {
    // Create a 3x3 2D tensor (xtensor equivalent of Tensor2f)
    xt::xtensor<float, 2> A = {
        {1.f, 2.f, 3.f},
        {4.f, 5.f, 6.f},
        {7.f, 8.f, 9.f}
    };

    // Create a 3x3x2 3D tensor (xtensor equivalent of Tensor3f)
    xt::xtensor<float, 3> B({3, 3, 2});
    xt::xtensor<float, 3> gradients({3, 3, 2});
    xt::xtensor<float, 3> expected_gradients({3, 3, 2});

    // Set expected gradients: chip<2>(0) and chip<2>(1) become xview
    view(expected_gradients, xt::all(), xt::all(), 0) = 3.0f;
    view(expected_gradients, xt::all(), xt::all(), 1) = 1.0f;

    // Call your gradient computation function
    computeGradient(A, gradients);

    // Check if gradients match expected_gradients
    if (areTensorsEqual(A, gradients)) {
        std::cerr << "Test failed: gradients != expected_gradients\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

int testComputeDotProduct() {
    // Define input tensors A and B (3x3x2)
    xt::xtensor<float, 3> A = {
        {{1.f, 2.f}, {3.f, 4.f}, {5.f, 6.f}},
        {{7.f, 8.f}, {9.f, 10.f}, {11.f, 12.f}},
        {{13.f, 14.f}, {15.f, 16.f}, {17.f, 18.f}}
    };
    xt::xtensor<float, 3> B = {
        {{2.f, 3.f}, {4.f, 5.f}, {6.f, 7.f}},
        {{8.f, 9.f}, {10.f, 11.f}, {12.f, 13.f}},
        {{14.f, 15.f}, {16.f, 17.f}, {18.f, 19.f}}
    };

    // Expected result: sum of element-wise products along depth
    xt::xtensor<float, 2> expected = {
        {1*2 + 2*3, 3*4 + 4*5, 5*6 + 6*7},
        {7*8 + 8*9, 9*10 + 10*11, 11*12 + 12*13},
        {13*14 + 14*15, 15*16 + 16*17, 17*18 + 18*19}
    };

    // Compute result
    xt::xtensor<float, 2> result({3, 3});
    computeDotProduct(A, B, result);

    // Check if result matches expected
    if (!areTensorsEqual(result, expected)) {
        std::cerr << "Test failed: result does not match expected.\n";
        std::cerr << "Expected:\n" << expected << "\n";
        std::cerr << "Got:\n" << result << "\n";
        return EXIT_FAILURE;
    }

    std::cout << "Test passed!\n";
    return EXIT_SUCCESS;
}

int test_dot_() {
    // Define tensor dimensions
    const int height = 3;
    const int width = 3;
    // Create input tensors
    Tensor3f A(height, width, 2);
    Tensor3f B(height, width, 2);

    // Fill input tensors with known values
    // In: 3x3x2
    // Fill In with values 0 to 26
    A.setValues({{
        // Layer 0 (k=0)
        {0.0f, 9.0f},
        {1.0f, 10.0f},
        {2.0f, 11.0f},
        {3.0f, 12.0f},
        {4.0f, 13.0f},
        {5.0f, 14.0f},
        {6.0f, 15.0f},
        {7.0f, 16.0f},
        {8.0f, 17.0f},
    }});

    B.setValues({{
    // Layer 0 (k=0)
    {-0.0f, -9.0f},
    {-1.0f, -10.0f},
    {-2.0f, -11.0f},
    {-3.0f, -12.0f},
    {-4.0f, -13.0f},
    {-5.0f, -14.0f},
    {-6.0f, -15.0f},
    {-7.0f, -16.0f},
    {-8.0f, -17.0f},
    }});


    Tensor2f Out(3,3);
    computeDotProduct(A, B, Out);
    Tensor2f expected_Out(3,3);
    expected_Out.setValues({
        {-81, -101, -125},
        {-153, -185, -221},
        {-261, -305, -353},
    }
    );

    if (!areTensorsEqual(Out, expected_Out)) {
        std::cerr << "Test failed: gradients != gradients\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

int test_m23_() {
    // Define tensor dimensions
    const int height = 3;
    const int width = 3;
    // Create input tensors
    Tensor3f In(height, width, 2);
    Tensor3f Cx(height, width, 3);
    Tensor3f Cy(height, width, 3);
    Tensor3f Out(height, width, 3);

    // Fill input tensors with known values
    // In: 3x3x2
    // Fill In with values 0 to 26
    In.setValues({{
        // Layer 0 (k=0)
        {0.0f, 9.0f},
        {1.0f, 10.0f},
        {2.0f, 11.0f},
        {3.0f, 12.0f},
        {4.0f, 13.0f},
        {5.0f, 14.0f},
        {6.0f, 15.0f},
        {7.0f, 16.0f},
        {8.0f, 17.0f},
    }});

    // Cx: 3x3x3
    // Fill Cx with values 0.0 to 2.6 (0.1 increments)
    Cx.setValues({{
        // Layer 0 (k=0)
        {0.0f, 0.1f, 0.2f},
        {0.3f, 0.4f, 0.5f},
        {0.6f, 0.7f, 0.8f},
        // Layer 1 (k=1)
        {0.9f, 1.0f, 1.1f},
        {1.2f, 1.3f, 1.4f},
        {1.5f, 1.6f, 1.7f},
        // Layer 2 (k=2)
        {1.8f, 1.9f, 2.0f},
        {2.1f, 2.2f, 2.3f},
        {2.4f, 2.5f, 2.6f}
    }});

    // Cy: 3x3x3
    Cy.setConstant(-1);

    // Define the expected output
    Tensor3f expectedOut(height, width, 3);
    expectedOut.setValues({{
        // Layer 0 (k=0)
        {0.0f, 0.9f, 1.8f},
        {2.0f, 3.0f, 4.0f},
        {4.6f, 5.7f, 6.8f},
        // Layer 1 (k=1)
        {7.8f, 9.0f, 10.2f},
        {11.6f, 12.9f, 14.2f},
        {16.0f, 17.4f, 18.8f},
        // Layer 2 (k=2)
        {21.0f, 22.5f, 24.0f},
        {26.6f, 28.2f, 29.8f},
        {32.8f, 34.5f, 36.2f}
    }});

    // Call the function
    m23(In, Cx, Cy, Out);
    if (!areTensorsEqual(Out, expectedOut)) {
        std::cerr << "Test failed: m23 false\n";
        std::cerr << Out << std::endl;
        std::cerr << "Expected: \n" << expectedOut << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

int test_m23() {
    const int height = 3;
    const int width = 3;

    // In: 3x3x2
    xt::xtensor<float, 3> In = {
        {{0.0f, 9.0f}, {1.0f, 10.0f}, {2.0f, 11.0f}},
        {{3.0f, 12.0f}, {4.0f, 13.0f}, {5.0f, 14.0f}},
        {{6.0f, 15.0f}, {7.0f, 16.0f}, {8.0f, 17.0f}}
    };

    // Cx: 3x3x3
    xt::xtensor<float, 3> Cx = {
        {{0.0f, 0.1f, 0.2f}, {0.3f, 0.4f, 0.5f}, {0.6f, 0.7f, 0.8f}},
        {{0.9f, 1.0f, 1.1f}, {1.2f, 1.3f, 1.4f}, {1.5f, 1.6f, 1.7f}},
        {{1.8f, 1.9f, 2.0f}, {2.1f, 2.2f, 2.3f}, {2.4f, 2.5f, 2.6f}}
    };

    // Cy: 3x3x3, filled with -1
    xt::xtensor<float, 3> Cy = xt::xtensor<float,3>::from_shape({height, width, 3});
    Cy.fill(-1);

    // Expected output
    xt::xtensor<float, 3> expectedOut = {
        {{0.0f, 0.9f, 1.8f}, {2.0f, 3.0f, 4.0f}, {4.6f, 5.7f, 6.8f}},
        {{7.8f, 9.0f, 10.2f}, {11.6f, 12.9f, 14.2f}, {16.0f, 17.4f, 18.8f}},
        {{21.0f, 22.5f, 24.0f}, {26.6f, 28.2f, 29.8f}, {32.8f, 34.5f, 36.2f}}
    };

    // Result tensor
    xt::xtensor<float,3> Out = xt::xtensor<float,3>::from_shape({height, width, 3});

    // Call the function
    m23(In, Cx, Cy, Out);

    // Check result
    if (!areTensorsEqual(Out, expectedOut)) {
        std::cerr << "Test failed: m23 false\n";
        std::cerr << "Got:\n" << Out << "\n";
        std::cerr << "Expected:\n" << expectedOut << "\n";
        return EXIT_FAILURE;
    }

    std::cout << "Test passed!\n";
    return EXIT_SUCCESS;
}

int test_norm_tensor_along_dim3() {
    // Create a 3x3x2 3D tensor with known values
    xt::xtensor<float, 3> T = {
        {{1.f, 2.f}, {3.f, 4.f}, {5.f, 6.f}},
        {{7.f, 8.f}, {9.f, 10.f}, {11.f, 12.f}},
        {{13.f, 14.f}, {15.f, 16.f}, {17.f, 18.f}}
    };

    // Expected result: 2-norm along the third axis (depth)
    xt::xtensor<float, 2> expected = {
        {std::sqrt(1.f*1.f + 2.f*2.f), std::sqrt(3.f*3.f + 4.f*4.f), std::sqrt(5.f*5.f + 6.f*6.f)},
        {std::sqrt(7.f*7.f + 8.f*8.f), std::sqrt(9.f*9.f + 10.f*10.f), std::sqrt(11.f*11.f + 12.f*12.f)},
        {std::sqrt(13.f*13.f + 14.f*14.f), std::sqrt(15.f*15.f + 16.f*16.f), std::sqrt(17.f*17.f + 18.f*18.f)}
    };

    // Result tensor
    xt::xtensor<float, 2> result({3, 3});

    // Call the function
    norm_tensor_along_dim3(T, result);

    // Check result
    if (!areTensorsEqual(result, expected)) {
        std::cerr << "Test failed: result does not match expected.\n";
        std::cerr << "Got:\n" << result << "\n";
        std::cerr << "Expected:\n" << expected << "\n";
        return EXIT_FAILURE;
    }

    std::cout << "Test passed!\n";
    return EXIT_SUCCESS;
}

int test_crossProduct3x3() {
    // Create input tensors A and B (3x3x3)
    xt::xtensor<float, 3> A = {
        {{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}, {7.f, 8.f, 9.f}},
        {{10.f, 11.f, 12.f}, {13.f, 14.f, 15.f}, {16.f, 17.f, 18.f}},
        {{19.f, 20.f, 21.f}, {22.f, 23.f, 24.f}, {25.f, 26.f, 27.f}}
    };
    xt::xtensor<float, 3> B = {
        {{1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {0.f, 0.f, 1.f}},
        {{1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {0.f, 0.f, 1.f}},
        {{1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {0.f, 0.f, 1.f}}
    };

    // Expected result: cross product of A and B
    xt::xtensor<float, 3> expected = {
        {{ 0.f,   3.f,  -2.f},
         { -6.f,  0.f,   4.f},
         {  8.f,  -7.f,  0.f}},
        {{ 0.f,  12.f, -11.f},
         {-15.f,  0.f,  13.f},
         { 17.f, -16.f,  0.f}},
        {{ 0.f,  21.f, -20.f},
         {-24.f,  0.f,  22.f},
         { 26.f, -25.f,  0.f}}
    };

    // Result tensor
    xt::xtensor<float, 3> C({3, 3, 3});

    // Call the function
    crossProduct3x3(A, B, C);

    // Check result
    if (!areTensorsEqual(C, expected)) {
        std::cerr << "Test failed: result does not match expected.\n";
        std::cerr << "Got:\n" << C << "\n";
        std::cerr << "Expected:\n" << expected << "\n";
        return EXIT_FAILURE;
    }

    std::cout << "Test passed!\n";
    return EXIT_SUCCESS;
}

int test_crossProduct1x3() {
    // Create input vector A (3D vector)
    xt::xtensor<float, 1> A = {1.f, 2.f, 3.f};

    // Create input tensor B (3x3x3)
    xt::xtensor<float, 3> B = {
        {{1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {0.f, 0.f, 1.f}},
        {{1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {0.f, 0.f, 1.f}},
        {{1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {0.f, 0.f, 1.f}}
    };

    // Expected result: cross product of A and each vector in B
    xt::xtensor<float, 3> expected = {
        {{ 0.f,  3.f, -2.f},
         {-3.f,  0.f,  1.f},
         { 2.f, -1.f,  0.f}},
        {{ 0.f,  3.f, -2.f},
         {-3.f,  0.f,  1.f},
         { 2.f, -1.f,  0.f}},
        {{ 0.f,  3.f, -2.f},
         {-3.f,  0.f,  1.f},
         { 2.f, -1.f,  0.f}}
    };

    // Result tensor
    xt::xtensor<float, 3> C({3, 3, 3});

    // Call the function
    crossProduct1x3(A, B, C);

    // Check result
    if (!areTensorsEqual(C, expected)) {
        std::cerr << "Test failed: result does not match expected.\n";
        std::cerr << "Got:\n" << C << "\n";
        std::cerr << "Expected:\n" << expected << "\n";
        return EXIT_FAILURE;
    }

    std::cout << "Test passed!\n";
    return EXIT_SUCCESS;
}

int test_m32() {
    // Create input tensors (1x1x3)
    xt::xtensor<float, 3> In = {{{1.f, 2.f, 3.f}}};
    xt::xtensor<float, 3> C_x = {{{4.f, 5.f, 6.f}}};
    xt::xtensor<float, 3> C_y = {{{7.f, 8.f, 9.f}}};

    // Expected result: manually computed
    xt::xtensor<float, 3> expected({1, 1, 3});

    // --- First block ---
    xt::xtensor<float, 3> C1({1, 1, 3});
    xt::xtensor<float, 3> C2({1, 1, 3});
    xt::xtensor<float, 2> dot({1, 1});
    xt::xtensor<float, 2> sign({1, 1});
    xt::xtensor<float, 2> distance1({1, 1});
    xt::xtensor<float, 2> distance2({1, 1});

    crossProduct3x3(C_x, C_y, C1);
    crossProduct3x3(C_y, C1, C2);
    computeDotProduct(In, C2, dot);
    sign = xt::sign(dot);
    vector_distance(In, C_y, distance1);
    vector_distance(C_x, C_y, distance2);
    xt::view(expected, xt::all(), xt::all(), 1) = sign * distance1 / distance2;

    // --- Second block ---
    crossProduct3x3(C_y, C_x, C1);
    crossProduct3x3(C_x, C1, C2);
    computeDotProduct(In, C2, dot);
    sign = xt::sign(dot);
    vector_distance(In, C_x, distance1);
    vector_distance(C_y, C_x, distance2);
    xt::view(expected, xt::all(), xt::all(), 0) = sign * distance1 / distance2;

    // Result tensor
    xt::xtensor<float, 3> Out({1, 1, 3});

    // Call the function
    m32(In, C_x, C_y, Out);

    // Check result
    if (!areTensorsEqual(Out, expected)) {
        std::cerr << "Test failed: result does not match expected.\n";
        std::cerr << "Got:\n" << Out << "\n";
        std::cerr << "Expected:\n" << expected << "\n";
        return EXIT_FAILURE;
    }

    std::cout << "Test passed!\n";
    return EXIT_SUCCESS;
}

int main() {
    int result = EXIT_SUCCESS;
    result |= test_gradient();
    result |= test_m23();
    result |= testComputeDotProduct();
    result |= test_norm_tensor_along_dim3();
    result |= test_crossProduct3x3();
    result |= test_crossProduct1x3();
    result |= test_m32();
    return result;
}
