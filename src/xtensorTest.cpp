//
// Created by Daniel Pommer on 03.12.25.
//
#include <iostream>
#include <xtensor.hpp>

int main() {
    xt::xarray<double> arr1
    {{1.0, 2.0, 3.0},
    {2.0, 5.0, 7.0},
    {2.0, 5.0, 7.0}};

    xt::xarray<double> arr2{5.0, 6.0, 7.0};

    xt::xarray<double> res = xt::view(arr1, 1) + arr2;

    std::cout << res;
}

