//
// Created by Daniel Pommer on 12.11.25.
//

#include "cost.h"
#include "update.h"

float costFR(const Tensor3f &F, const Tensor3f &CCM, const Tensor3f &Cx, const Tensor3f &Cy, const Tensor1f &R) {
    Tensor3f cross(CCM.dimensions());
    Tensor<float, 3> cross_colMajor;
    const auto& dimensions = F.dimensions();
    Eigen::array<int, 1> dims({2 /* dimension to reduce */});
    Eigen::array<int, 2> dims2({0,1});

    Tensor3f ref(dimensions);
    Tensor3f square(dimensions);
    Tensor<float, 2> sum(dimensions[0], dimensions[0]);
    {
        crossProduct1x3(R, CCM, cross);
    }
    {
        m32(cross, Cx, Cy, ref);
    }
    square = (F - ref).square();
    // Swap the layout and preserve the order of the dimensions
    array<int, 3> shuffle({2, 1, 0});
    cross_colMajor = cross.swap_layout().shuffle(shuffle);
    sum = cross_colMajor.sum(dims);
    float cost = static_cast<Eigen::Tensor<float, 0>>(sum.sum(dims2).eval())();
    return cost;
}

float costFG(const Tensor3f &F, const Tensor2f &V, const Tensor3f &G){
    const auto& dimensions = F.dimensions();
    Tensor2f dot(dimensions[0], dimensions[1]);
    Tensor<float, 2> dot_colMajor;
    Tensor<float, 2> V_colMajor;
    Tensor<float, 2> diff(dimensions[0], dimensions[1]);
    computeDotProductWithLoops(F,G,dot);
    array<int, 2> shuffle({1, 0});
    dot_colMajor = dot.swap_layout().shuffle(shuffle);
    V_colMajor = V.swap_layout().shuffle(shuffle);

    diff = (V_colMajor - dot_colMajor).square();
    float cost = static_cast<Eigen::Tensor<float, 0>>(diff.sum().eval())();
    return cost;
}

float costGI(Tensor3f &G, const Tensor3f &I_gradient){
    const auto& dimensions = G.dimensions();
    Eigen::array<int, 1> dims({2 /* dimension to reduce */});
    Tensor<float, 3> square(dimensions);
    Tensor<float, 2> sum(dimensions[0], dimensions[0]);

    Tensor<float, 3> G_colMajor;
    Tensor<float, 3> Igrad_colMajor;
    array<int, 3> shuffle({2, 1, 0});

    G_colMajor = G.swap_layout().shuffle(shuffle);
    Igrad_colMajor = I_gradient.swap_layout().shuffle(shuffle);

    square = (Igrad_colMajor - G_colMajor).square();
    sum = square.sum(dims);
    float cost = static_cast<Eigen::Tensor<float, 0>>(sum.sum().eval())();
    return cost;
}
