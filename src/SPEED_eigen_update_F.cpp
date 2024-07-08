#include <iostream>
#include "../eigen/unsupported/Eigen/CXX11/Tensor"
#include <chrono>

using Eigen::Tensor;
using Eigen::array;
using namespace std::chrono;

const int SIZE = 128;

void update_ForG(Tensor<double, 3> &ForG, const Tensor<double, 2> &V, const Tensor<double, 3> &GorF, double lr, double weight_ForG) {
    // Calculate the norm
    Tensor<double, 2> norm2d = (GorF.square().sum(array<int, 1>{2}).sqrt() + 1e-8);
    Tensor<double, 3> norm = norm2d.reshape(array<int, 3>{SIZE, SIZE, 1}).broadcast(array<int, 3>{1, 1, 2});
    
    // Calculate the dot product
    Tensor<double, 2> dot_product2d = (ForG * GorF).sum(array<int, 1>{2});
    Tensor<double, 3> dot_product = dot_product2d.reshape(array<int, 3>{SIZE, SIZE, 1}).broadcast(array<int, 3>{1, 1, 2});

    // Calculate the updates
    Tensor<double, 3> V_broadcast = V.reshape(array<int, 3>{SIZE, SIZE, 1}).broadcast(array<int, 3>{1, 1, 2});
    Tensor<double, 3> update = ForG - GorF * (V_broadcast + dot_product) / (norm.square());
    
    // Update ForG
    ForG = (1 - weight_ForG) * ForG + lr * weight_ForG * update;
}

int main() {
    Tensor<double, 3> ForG(SIZE, SIZE, 2);
    Tensor<double, 2> V(SIZE, SIZE);
    Tensor<double, 3> GorF(SIZE, SIZE, 2);

    ForG.setRandom();
    V.setRandom();
    GorF.setRandom();

    double lr = 0.01;
    double weight_ForG = 0.1;
    auto start = high_resolution_clock::now();

    for (int i = 0; i < 10000; ++i) {
        update_ForG(ForG, V, GorF, lr, weight_ForG);
    }

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start).count();

    std::cout << "Time taken for 10,000 updates: " << duration/10000 << " microseconds" << std::endl;
    return 0;
}


