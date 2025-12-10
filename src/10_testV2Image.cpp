//
// Created by Daniel Pommer on 09.12.25.
//
#include <conversions.h>
#include <imaging.h>

int main() {
    std::random_device myRandomDevice;
    unsigned seed = myRandomDevice();
    std::default_random_engine rng(seed);
    auto engine = xt::random::get_default_random_engine();
    xt::xtensor<float, 2> tensor2 = xt::random::randn<float>({100, 100, 2}, 0, 20, engine);
    cv::Mat mat2 = xtensorToCvMat(tensor2);
    cv::Mat V = V2image(tensor2, 0.1);
    std::cout << tensor2 << std::endl;
    std::cout << mat2 << std::endl;
}