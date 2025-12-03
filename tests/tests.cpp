//
// Created by Daniel Pommer on 02.12.25.
//
#include <iostream>
#include <cstdlib> // for EXIT_SUCCESS, EXIT_FAILURE
#include <update.h>
#include <cmath>

#include "boost/fusion/sequence/io/out.hpp"

int test_addition() {
    if (1 + 1 != 2) {
        std::cerr << "Test failed: 1 + 1 != 2\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

int test_subtraction() {
    if (5 - 3 != 2) {
        std::cerr << "Test failed: 5 - 3 != 2\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

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

int test_gradient() {
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
}

int test_dot() {
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

int test_m23() {
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


int main() {
    int result = EXIT_SUCCESS;
    result |= test_addition();
    result |= test_subtraction();
    result |= test_gradient();
    result |= test_m23();
    result |= test_dot();
    return result;
}
