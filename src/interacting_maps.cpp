#include "interacting_maps.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <sstream>
#include <cassert>
#include <numeric>

namespace fs = std::filesystem;

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

std::vector<int> digitize(const std::vector<double>& input, const std::vector<double>& bins) {
    std::vector<int> indices;
    for (const double& value : input) {
        // Find the index where 'value' should be inserted to keep 'bins' sorted
        auto it = std::upper_bound(bins.begin(), bins.end(), value);
        // Calculate the index
        int index = it - bins.begin();
        indices.push_back(index);
    }
    return indices;
}

std::string create_folder_and_update_gitignore(const std::string& folder_name) {
    // Get the absolute path of the current working directory
    fs::path current_directory = fs::current_path();
    
    // Create the output folder if it does not exist
    fs::path output_folder_path = current_directory / "output";
    // Create the directory if it does not exist
    if (!fs::exists(output_folder_path)) {
        fs::create_directory(output_folder_path);
    }

    // Same for the actual folder
    fs::path folder_path = output_folder_path / folder_name;
    if (!fs::exists(folder_path)) {
        fs::create_directory(folder_path);
    }
    
    // Path to the .gitignore file
    fs::path gitignore_path = current_directory/ ".gitignore";
    
    // Check if the folder is already in .gitignore
    bool folder_in_gitignore = false;
    if (fs::exists(gitignore_path)) {
        std::ifstream gitignore_file(gitignore_path);
        std::string line;
        while (std::getline(gitignore_file, line)) {
            if (line == folder_name || line == "/" + folder_name) {
                folder_in_gitignore = true;
                break;
            }
        }
    }
    
    // Add the folder to .gitignore if it's not already there
    if (!folder_in_gitignore) {
        std::ofstream gitignore_file(gitignore_path, std::ios_base::app);
        gitignore_file << "\n" << folder_name << "\n";
    }
    
    // Return the absolute path of the new folder
    return folder_path.string();
}

void read_calib(const std::string file_path, std::vector<float>& calibration_data){
    fs::path current_directory = fs::current_path();
    std::string path = current_directory / file_path;
    if (fs::exists(path)) {
        std::ifstream calibration_file(path);
        std::string::size_type size;
        for (std::string line; std::getline(calibration_file, line, ' ');) {
            calibration_data.push_back(std::stof(line, &size));
        }
    }
}

void read_events(const std::string file_path, std::vector<Event>& events, float start_time, float end_time, int max_events = INT32_MAX){
    fs::path current_directory = fs::current_path();
    std::string path = current_directory / file_path;
    if (fs::exists(path)) {
        std::ifstream event_file(path);
        int counter;
        float time;
        int width, height, polarity;
        while (event_file >> time >> width >> height >> polarity){
            if (time < start_time) continue;
            if (time > end_time) break;
            if (counter > max_events) break;
            Event event;
            event.time = time;
            std::vector<int> coords = {width, height};
            event.coordinates = coords;
            event.polarity = polarity;
            events.push_back(event);
            counter++;
        }
    }
}

std::vector<std::vector<Event>> bin_events(std::vector<Event>& events, float bucketsize = 0.05){
    float min_time = events.front().time;
    float max_time = events.back().time;
    int n_buckets = int((max_time-min_time)/bucketsize);

    Eigen::VectorXf bins = Eigen::VectorXf::LinSpaced(n_buckets, min_time, max_time);
    std::vector<std::vector<Event>> buckets(n_buckets, std::vector<Event>());


    // Find the indices where each event belongs according to its time stamp
    // std::vector<int> indices;
    int index;
    for (const Event& event : events) {
        // Find the index where 'value' should be inserted to keep 'bins' sorted
        auto it = std::lower_bound(bins.begin(), bins.end(), event.time);
        // Calculate the index
        if (it == bins.begin()) {
            index = 0;
        }

        // If the iterator points to the end, timeValue is greater than all bins
        if (it == bins.end()) {
            index =  bins.size(); // timeValue is beyond the last bin
        }

        index = std::distance(bins.begin(), it);

        // std::cout << index << " ";
        // indices.push_back(index);
        buckets[*it].push_back(event);
    }
    return buckets;
}

void create_frames(std::vector<std::vector<Event>>& bucketed_events, std::vector<Tensor<float,2>>& frames, int camera_height, int camera_width){
    for (std::vector<Event> event_vector : bucketed_events){
        Tensor<float,2> frame(camera_height, camera_width);
        Tensor<float,2> cum_frame(camera_height, camera_width);
        for (Event event : event_vector){
            frame(event.coordinates.at(0), event.coordinates.at(1)) = event.polarity;
            cum_frame(event.coordinates.at(0), event.coordinates.at(1)) += event.polarity;
        }
        frames.push_back(frame);
    }
}

void create_sparse_matrix(int N, Tensor<float,2>& V, SpMat& result){

    // Step 1: Create the repeated identity matrix part
    std::vector<Triplet<float>> tripletList;
    tripletList.reserve(N*3*2);
    for (int i = 0; i<N*3; i++){
        tripletList.push_back(Triplet<float>(i,i%3,1.0));
    }

    // Step 2: Create the diagonal and off-diagonal values from V
    int j = 3;
    array<int,1> one_dim{{V.size()}};
    Tensor<float,1> k = V.reshape(one_dim);
    for (int i = 0; i<N*3; i++){
        // std::cout << " i: " << i << " j: " << j << ", " << std::endl;
        tripletList.push_back(Triplet<float>(i,j,k(i)));
        if (i%3 == 2){
            j++;
        }
    }

    result.setFromTriplets(tripletList.begin(), tripletList.end());
}

void norm_tensor_along_dim3(Tensor<float,3>& T, Tensor<float,2>& norm){
    array<int,1> dims({2});
    norm = T.square().sum(dims).sqrt();
}

void find_C(int N_x, int N_y, float view_angle_x, float view_angle_y, float rs, Tensor<float,3>& C, Tensor<float,3>& dxdC, Tensor<float,3>& dydC){
    float height = tan(view_angle_x/2);
    float width = tan(view_angle_y/2);

    // Constant 1 Matrix for calculations
    Tensor<float, 2> constant_one(N_x, N_y);
    constant_one.setConstant(1.0f);

    // Create X Index Matrix
    Tensor<float, 2> x_index(N_x, N_y);
    x_index.setConstant(1.0f);
    x_index = x_index.cumsum(0)-x_index;
    x_index = height * constant_one * (-constant_one + (2 * x_index)/((N_x-1)*constant_one));

    // Create X Index Matrix
    Tensor<float, 2> y_index(N_x, N_y);
    y_index.setConstant(1.0f);
    y_index = y_index.cumsum(1)-y_index;
    y_index = width * constant_one * (constant_one - (2 * y_index)/((N_y-1)*constant_one));

    C.chip(0,2) = x_index;
    C.chip(1,2) = y_index;
    C.chip(2,2) = rs*constant_one;

    Tensor<float, 2> norm(N_x, N_y);
    array<int,1> dims({2});
    norm = C.square().sum(dims).sqrt();

    C.chip(0,2) = C.chip(0,2)/norm;
    C.chip(1,2) = C.chip(1,2)/norm;
    C.chip(2,2) = C.chip(2,2)/norm;

    // Calculate derivative dxdC and dydC
    dxdC.chip(0,2) = height * constant_one/((N_x-1)*constant_one);
    dydC.chip(1,2) = - width * constant_one/((N_y-1)*constant_one);

    std::cout << C << std::endl;
    std::cout << dxdC << std::endl;
    std::cout << dydC << std::endl;
}

void update_F_from_G(Tensor<float,3>& F, const Tensor<float,2>& V, const Tensor<float,3>& G, const float lr, const float weight_FG){
    
    array<IndexPair<int>, 1>dot_prod_dims = {IndexPair<int>(2,2)};
    Eigen::Tensor<float, 2> dot_prod = F.contract(G, dot_prod_dims);
    Eigen::Tensor<float, 2> norm_G = G.contract(G, dot_prod_dims);

    int dim1 = 180;
    int dim2 = 240;
    int dim3 = 3;

    for (int i = 0; i<dim1)










    const auto& dimensions = F.dimensions();
    Tensor<float, 2> G_norm(dimensions[0], dimensions[1]);
    norm_tensor_along_dim3(G, G_norm);
    Tensor<float,3> update_F(dimensions[0], dimensions[1], dimensions[2]);
    

    array<int,3> bcast({1,1,3});
    array<int,3> three_dims{{dimensions[0], dimensions[1], 1}};
    Tensor<float,3> G_norm_tensor = G_norm.reshape(three_dims);
    Tensor<float,3> G_norm_tensor_3D = G_norm_tensor.broadcast(bcast);

    array<IndexPair<int>, 1>matmul_dims = {IndexPair<int>(2,2)};
    array<int, 1> sum_dims({2});
    Tensor<float,2> tmp0 = F.contract(G, matmul_dims).eval().sum(sum_dims);
    Tensor<float,2> tmp = V + tmp0;
    Tensor<float,3> tmp_3D =  tmp.reshape(three_dims);
    Tensor<float,3> tmp_3D_bc =  tmp_3D.broadcast(bcast);
    update_F = F - G/G_norm_tensor_3D * tmp_3D_bc;
    F = (1-weight_FG)*F + lr * weight_FG * update_F;
}

int main() {
    // Create results_folder
    std::string folder_name = "results";
    std::string folder_path = create_folder_and_update_gitignore(folder_name);
    std::cout << "Folder created at: " << folder_path << std::endl;

    // Read calibration file
    std::string calib_path = "../res/shapes_rotation/calib.txt";
    std::vector<float> calibration_data;
    read_calib(calib_path, calibration_data);
    // for (float data : calibration_data){
    //     std::cout << data << std::endl;
    // }

    // Read events file
    std::string event_path = "../res/shapes_rotation/eventsshort.txt";
    std::vector<Event> event_data;
    read_events(event_path, event_data, 0.0, 1.0);
    // for (Event event : event_data){
    //     std::cout << event << std::endl;
    // }

    // Bin events
    std::vector<std::vector<Event>> bucketed_events;
    bucketed_events = bin_events(event_data, 0.05);

    // Create frames
    std::vector<Tensor<float,2>> frames;
    create_frames(bucketed_events, frames, 180, 240);
    
    // Test Sparsematrix creation
    int M = 100;
    int N = M*M;
    SpMat m(N*3,N+3);
    Tensor<float,2> V(N,3);
    V.setRandom();
    create_sparse_matrix(N, V, m);

    int x = 4;
    int y = 6;
    int z = 3;

    Tensor<float,3> C(x,y,z);
    C.setZero();
    Tensor<float,3> dxdC(x,y,z);
    dxdC.setZero();
    Tensor<float,3> dydC(x,y,z);
    dydC.setZero();
    find_C(x, y, 90, 90, 1, C, dxdC, dydC);

    Tensor<float,3> F(x,y,3);
    F.setConstant(1.0);
    Tensor<float,2> V2(x,y);
    V2.setConstant(2.0);
    Tensor<float,3> G(x,y,3);
    G.setConstant(3.0);
    update_F_from_G(F, V2, G, 1.0, 0.5);

    return 0;
}