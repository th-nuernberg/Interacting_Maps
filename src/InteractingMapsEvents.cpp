#include <interacting_maps.h>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <sstream>
#include <cassert>
#include <numeric>
#include "Instrumentor.h"
#include <cmath>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define PI 3.14159265
#define EXECUTE_TEST 0
#define EXECUTE_FRAMEBASED 0
#define SMALL_MATRIX_METHOD 1

// Define DEBUG_LOG macro that logs with function name in debug mode
#ifdef DEBUG
#define DEBUG_LOG(message) \
        std::cout << "DEBUG (" << __func__ << "): " << message << std::endl << \
        "###########################################" << std::endl;
#else
#define DEBUG_LOG(message) // No-op in release mode
#endif

#define PROFILING 1
#if PROFILING
#define PROFILE_SCOPE(name) InstrumentationTimer timer##__LINE__(name)
#define PROFILE_FUNCTION() PROFILE_SCOPE(__FUNCTION__)
// #define PROFILE_FUNCTION() PROFILE_SCOPE(__FUNCSIG__) (Includes call attributes, whole signature of function)
#else
#define PROFILE_SCOPE(name)
#define PROFILE_FUNCTION()
#endif

std::ostream& operator << (std::ostream &os, const Event &e) {
    return (os << "Time: " << e.time << " Coords: " << e.coordinates[0] << " " << e.coordinates[1] << " Polarity: " << e.polarity);
}

std::string Event::toString() const {
    std::stringstream ss;
    ss << (*this);
    return ss.str();
}

std::vector<std::string> split_string(std::stringstream sstream, char delimiter){
    std::string segment;
    std::vector<std::string> seglist;

    while(std::getline(sstream, segment, delimiter))
    {
        seglist.push_back(segment);
    }

    return seglist;
}

std::vector<float> update_FG(std::vector<float> F, std::vector<float> G, float V, const float lr, const float weight_FG, float eps=1e-8, float gamma=255.0){
    std::vector<float> update_F(2);
    float norm = std::abs((G[0] * G[0]  + G[1]  * G[1]));
    update_F[0] = F[0] - ((G[0] / norm) * (V + (F[0]  * G[0]  + F[1]  * G[1])));
    update_F[1] = F[1] - ((G[1] / norm) * (V + (F[0]  * G[0]  + F[1]  * G[1])));
    F[0] = (1 - weight_FG) * F[0] + lr * weight_FG * update_F[0];
    F[1] = (1 - weight_FG) * F[1] + lr * weight_FG * update_F[1];
    if (F[0] > gamma){
        F[0] = gamma;
    }
    if (F[1] > gamma){
        F[1] = gamma;
    }
    if (F[0] < -gamma){
        F[0] = -gamma;
    }
    if (F[1] < -gamma){
        F[1] = -gamma;
    }
    if (std::abs(F[0]) < eps){
        F[0] = 0;
    }
    if (std::abs(F[1]) < eps){
        F[1] = 0;
    }
    return F;
}

std::vector<float> update_GF(std::vector<float> G, std::vector<float> F, float V, const float lr, const float weight_GF, float eps=1e-8, float gamma=255.0){
    std::vector<float> update_G(2);
    float norm = std::abs((F[0] * F[0]  + F[1]  * F[1]));
    update_G[0] = G[0] - ((F[0] / norm) * (V + (G[0]  * F[0]  + G[1]  * F[1])));
    update_G[1] = G[1] - ((F[1] / norm) * (V + (G[0]  * F[0]  + G[1]  * F[1])));
    G[0] = (1 - weight_GF) * G[0] + lr * weight_GF * update_G[0];
    G[1] = (1 - weight_GF) * G[1] + lr * weight_GF * update_G[1];
    if (G[0] > gamma){
        G[0] = gamma;
    }
    if (G[1] > gamma){
        G[1] = gamma;
    }
    if (G[0] < -gamma){
        G[0] = -gamma;
    }
    if (G[1] < -gamma){
        G[1] = -gamma;
    }
    if (std::abs(G[0]) < eps){
        G[0] = 0;
    }
    if (std::abs(G[1]) < eps){
        G[1] = 0;
    }
    return G;
}

