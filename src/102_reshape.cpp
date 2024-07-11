#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xadapt.hpp>

int main(int argc, char* argv[])
{
    xt::xarray<int> arr
      {1, 2, 3, 4, 5, 6, 7, 8, 9};

    arr.reshape({3, 3});

    std::cout << arr << std::endl;
    std::cout << xt::adapt(arr.shape()) << std::endl; // with: #include <xtensor/xadapt.hpp>
    return 0;
}