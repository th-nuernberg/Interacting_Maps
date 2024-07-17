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

    // std::cout << C << std::endl;
    // std::cout << dxdC << std::endl;
    // std::cout << dydC << std::endl;
}

void crossProductTensors(const Eigen::Tensor<float, 3>& A, const Eigen::Tensor<float, 3>& B, Eigen::Tensor<float, 3>& C) {
    assert(A.dimensions() == B.dimensions() && "Tensors A and B must have the same shape");

    // Use Eigen's tensor operations to compute the cross product
    C.chip(0, 2) = A.chip(1, 2) * B.chip(2, 2) - A.chip(2, 2) * B.chip(1, 2);
    C.chip(1, 2) = A.chip(2, 2) * B.chip(0, 2) - A.chip(0, 2) * B.chip(2, 2);
    C.chip(2, 2) = A.chip(0, 2) * B.chip(1, 2) - A.chip(1, 2) * B.chip(0, 2);
}

// Function to compute the cross product of two 3-tensors
void crossProductTensors_loop(const Eigen::Tensor<float, 3>& A, const Eigen::Tensor<float, 3>& B, Eigen::Tensor<float, 3>& C) {
    assert(A.dimensions() == B.dimensions() && "Tensors A and B must have the same shape");
    for (int i = 0; i < A.dimension(0); ++i) {
        for (int j = 0; j < A.dimension(1); ++j) {
            C(i, j, 0) = A(i, j, 1) * B(i, j, 2) - A(i, j, 2) * B(i, j, 1);
            C(i, j, 1) = A(i, j, 2) * B(i, j, 0) - A(i, j, 0) * B(i, j, 2);
            C(i, j, 2) = A(i, j, 0) * B(i, j, 1) - A(i, j, 1) * B(i, j, 0);
        }
    }
}

// def m32(V, C_x, C_y):
//     x_comp = jnp.sign((jnp.einsum("ijk,ijk->ij", V, (jnp.cross(C_y, jnp.cross(C_x, C_y)))))) * vector_distance(V, C_y)/vector_distance(C_x, C_y)
//     y_comp = jnp.sign((jnp.einsum("ijk,ijk->ij", V, (jnp.cross(C_x, jnp.cross(C_y, C_x)))))) * vector_distance(V, C_x)/vector_distance(C_y, C_x)
//     return jnp.array([x_comp, y_comp]).transpose((1,2,0))

// def vector_distance(vec1, vec2):
//     return jnp.linalg.norm(jnp.cross(vec1, vec2), ord=2, axis=2)/jnp.linalg.norm(vec2, ord=2,axis=2)

void vector_distance(Tensor<float,3> &vec1, Tensor<float,3> &vec2, Tensor<float,2> distance){
    const auto& dimensions = vec1.dimensions();
    Tensor<float,3> cross_product(dimensions);
    Tensor<float,2> norm(dimensions[0], dimensions[1]);
    Tensor<float,2> norm2(dimensions[0], dimensions[1]);
    // std::cout << vec1.dimensions() << vec2.dimensions() << std::endl;
    crossProductTensors(vec1, vec2, cross_product);
    norm_tensor_along_dim3(cross_product, norm);
    norm_tensor_along_dim3(vec2, norm2);
    distance = norm/norm2;
}

float sign_func(float x)
{
    // Apply via a.unaryExpr(std::ptr_fun(sign_func))
    if (x > 0)
        return +1.0;
    else if (x == 0)
        return 0.0;
    else
        return -1.0;
}

// Function using nested loops to compute the dot product
void computeDotProductWithLoops(const Eigen::Tensor<float, 3>& A, const Eigen::Tensor<float, 3>& B, Eigen::Tensor<float, 2>& D) {
    const int m = A.dimension(0);
    const int n = A.dimension(1);
    const int d = A.dimension(2);

    // #pragma omp parallel for collapse(2)
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float dotProduct = 0.0f; // Initialize the dot product for position (i, j)
            for (int k = 0; k < d; ++k) {
                dotProduct += A(i, j, k) * B(i, j, k);
            }
            D(i, j) = dotProduct; // Store the result in tensor D
        }
    }
}

void m32(Tensor<float,3> &In, Tensor<float,3> &C_x, Tensor<float,3> &C_y, Tensor<float,3> &Out){
    const auto& dimensions = In.dimensions();
    Tensor<float,3> C1(dimensions);
    Tensor<float,3> C2(dimensions);
    Tensor<float,2> dot(dimensions[0], dimensions[1]);
    Tensor<float,2> sign(dimensions[0], dimensions[1]);
    Tensor<float,2> distance1(dimensions[0], dimensions[1]);
    Tensor<float,2> distance2(dimensions[0], dimensions[1]);
    
    crossProductTensors(C_x,C_y,C1);
    crossProductTensors(C_y,C1,C2);
    computeDotProductWithLoops(In,C2,dot);
    sign = dot.unaryExpr(std::ptr_fun(sign_func));
    vector_distance(In, C_y, distance1);
    vector_distance(C_x, C_y, distance2);
    Out.chip(0,2) = sign * distance1/distance2;

    crossProductTensors(C_x,-C1,C2);
    computeDotProductWithLoops(In,C2,dot);
    sign = dot.unaryExpr(std::ptr_fun(sign_func));
    vector_distance(In, C_x, distance1);
    vector_distance(C_y, C_x, distance2);
    // Tensor<float,2>res1 = sign * distance1/distance2;
    Out.chip(1,2) = sign * distance1/distance2;
}

// // Function using .chip() to compute the dot product
// void computeDotProductWithChip(const Eigen::Tensor<float, 3>& A, const Eigen::Tensor<float, 3>& B, Eigen::Tensor<float, 2>& D) {
//     const int m = A.dimension(0);
//     const int n = A.dimension(1);
//     Tensor<float, 1> a_slice;
//     Tensor<float, 1> b_slice;
//     Tensor<float, 0> res;
//     for (int i = 0; i < m; ++i) {
//         for (int j = 0; j < n; ++j) {
//             // Extract the slices and compute the dot product
//             a_slice = A.chip(i, 0).chip(j, 0);
//             b_slice = B.chip(i, 0).chip(j, 0);
//             res = (a_slice * b_slice).sum().sqrt();
//             D(i, j) = res(); // Compute the dot product
//         }
//     }
// }



// Function to time the performance of a given dot product computation function
template<typename Func>
void timeDotProductComputation(Func func, const Eigen::Tensor<float, 3>& A, const Eigen::Tensor<float, 3>& B, Eigen::Tensor<float, 2>& D, int iterations) {
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        func(A, B, D);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time elapsed: " << elapsed.count() << " seconds" << std::endl;
}


void update_F_from_G(Tensor<float,3>& F, Tensor<float,2>& V, Tensor<float,3>& G, float lr, float weight_FG){

    const auto& dimensions = F.dimensions();
    Tensor<float,3> update_F(dimensions);
    for (int i = 0; i<dimensions[0]; i++){
        for (int j = 0; j<dimensions[1]; j++){
            update_F(i,j,0) = F(i,j,0) - ((G(i,j,0)/(G(i,j,0) * G(i,j,0) + G(i,j,1) * G(i,j,1))) * (V(i,j) + (F(i,j,0) * G(i,j,0) + F(i,j,1) * G(i,j,1))));
            update_F(i,j,1) = F(i,j,1) - ((G(i,j,1)/(G(i,j,0) * G(i,j,0) + G(i,j,1) * G(i,j,1))) * (V(i,j) + (F(i,j,0) * G(i,j,0) + F(i,j,1) * G(i,j,1))));
        }
    }
    F = (1-weight_FG)*F + lr * weight_FG * update_F;
    // const auto& dimensions = F.dimensions();
    // Tensor<float, 2> G_norm(dimensions[0], dimensions[1]);
    // norm_tensor_along_dim3(G, G_norm);
    // Tensor<float,3> update_F(dimensions[0], dimensions[1], dimensions[2]);
    

    // array<int,3> bcast({1,1,3});
    // array<int,3> three_dims{{dimensions[0], dimensions[1], 1}};
    // Tensor<float,3> G_norm_tensor = G_norm.reshape(three_dims);
    // Tensor<float,3> G_norm_tensor_3D = G_norm_tensor.broadcast(bcast);

    // array<IndexPair<int>, 1>matmul_dims = {IndexPair<int>(2,2)};
    // array<int, 1> sum_dims({2});
    // Tensor<float,2> tmp0 = F.contract(G, matmul_dims).eval().sum(sum_dims);
    // Tensor<float,2> tmp = V + tmp0;
    // Tensor<float,3> tmp_3D =  tmp.reshape(three_dims);
    // Tensor<float,3> tmp_3D_bc =  tmp_3D.broadcast(bcast);
    // update_F = F - G/G_norm_tensor_3D * tmp_3D_bc;
}

void update_G_from_I(Tensor<float,3> &G, Tensor<float,3> &I_gradient, int weight_GI){
    G = (1 - weight_GI) * G + weight_GI*I_gradient;
}

void update_I_from_V(Tensor<float,2> &I, Tensor<float,2> &cum_V, int weight_IV=0.5, int time_step=0.05){
    I = (1 - weight_IV)*I + weight_IV*cum_V;
    const auto& dimensions = I.dimensions();
    for (int i; i<dimensions[0]; i++){
        for (int j; j<dimensions[1]; j++){
            if (I(i,j)>0){
                if (I(i,j)>time_step){
                    I(i,j) -= time_step;
                }
                else{
                    I(i,j) = 0;
                }
            }
            if (I(i,j)<0){
                if (I(i,j)<time_step){
                    I(i,j) += time_step;
                }
                else{
                    I(i,j) = 0;
                }
            }
        }
    }
    // I = jnp.where(I > 0, jnp.maximum(I - time_step, 0), I) # decay to neutral value
    // I = jnp.where(I < 0, jnp.minimum(I + time_step, 0), I) # decay to neutral value

}

void update_I_from_G(Tensor<float,2> &I, Tensor<float,3> &I_gradient, Tensor<float,3> &G, int weight_IG=0.5){
    const auto& dimensions = I.dimensions();
    Tensor<float,3> temp_map = G - I_gradient;
    Tensor<float,2> x_update(dimensions);
    Tensor<float,2> y_update(dimensions);
    for (int i; i<dimensions[0]-1; i++){
        for (int j; j<dimensions[1]-1; j++){
            if (i==0){
                x_update(i,j) = temp_map(i,j,0);
            }
            else{
                x_update(i,j) = temp_map(i,j,0) - temp_map(i-1,j,0);
            }
            if (j==0){
                y_update(i,j) = temp_map(i,j,1);
            }
            else{
                x_update(i,j) = temp_map(i,j,1) - temp_map(i,j-1,1);
            }
        }
    }
    I = (1 - weight_IG)*I + weight_IG*(I - x_update - y_update);
    // # Gradient Implementation, the paper mentions an effect in x-direction which gets computed as a difference between matrix components in x-direction. 
    // # This is very similar to how gradients are implemented which is why they are used here. The temp map consists of an x and a y component since, the
    // # gradient of I consists of two components. One for the x and one for the y direction.
    // x_update = jnp.gradient(temp_map[:,:,0])
    // y_update = jnp.gradient(temp_map[:,:,1])
    // return (1 - weight_IG)*I + weight_IG*(I - x_update[0] - y_update[1])

    // Like Paper Implementation
    // x_update = np.zeros((temp_map.shape[:-1]))
    // y_update = np.zeros((temp_map.shape[:-1]))
    // x_update[0,:] = temp_map[0,:,0] # - 0 // Out of bound entries are set to 0
    // x_update[1:,:] = temp_map[1:,:,0] - temp_map[:-1,:,0]
    // y_update[:,0] = temp_map[:,0,1] # - 0 // Out of bound entries are set to 0
    // y_update[:,1:] = temp_map[:,1:,1] - temp_map[:,:-1,1]
    // return (1 - weight_IG)*I + weight_IG*(I - x_update - y_update)
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
    SpMat sparse_m(N*3,N+3);
    Tensor<float,2> V(N,3);
    V.setRandom();
    create_sparse_matrix(N, V, sparse_m);

    int n = 10;
    int m = 5;
    int p = 3;

    Tensor<float,3> C(n,m,p);
    C.setZero();
    Tensor<float,3> dxdC(n,m,p);
    dxdC.setZero();
    Tensor<float,3> dydC(n,m,p);
    dydC.setZero();
    find_C(n, m, 90, 90, 1, C, dxdC, dydC);
    std::cout << "C" << std::endl;
    std::cout << C << std::endl;
    std::cout << "dxdC" << std::endl;
    std::cout << dxdC << std::endl;
    std::cout << "dydC" << std::endl;
    std::cout << dydC << std::endl;

    Tensor<float,3> F(n,m,p);
    F.setConstant(1.0);
    Tensor<float,2> V2(n,m);
    V2.setRandom();
    Tensor<float,3> G(n,m,p);
    G.setConstant(3.0);
    // std::cout << "F before update" << std::endl;
    // std::cout << F << std::endl;
    update_F_from_G(F, V2, G, 1.0, 0.5);
    // std::cout << "F after update" << std::endl;
    // std::cout << F << std::endl;



    // // Tensor cross product
    // // Define the dimensions of the tensors
    Eigen::array<Eigen::Index, 3> dimensions = {n, m, p};

    // // Initialize tensors A and B with random values
    Eigen::Tensor<float, 3> A(dimensions);
    // Eigen::Tensor<float, 3> B(dimensions);
    // Eigen::Tensor<float, 3> D(dimensions);

    A.setRandom();
    // B.setRandom();
    // int iterations = 1000;

    // // Timing the chip-based implementation
    // auto start_chip = std::chrono::high_resolution_clock::now();
    // for (int i = 0; i < iterations; ++i) {
    //     crossProductTensors(A, B, D);
    // }
    // auto end_chip = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> duration_chip = end_chip - start_chip;
    // std::cout << "Time for chip-based implementation: " << duration_chip.count()/iterations << " seconds\n";

    // // Timing the loop-based implementation
    // auto start_loop = std::chrono::high_resolution_clock::now();
    // for (int i = 0; i < iterations; ++i) {
    //     crossProductTensors_loop(A, B, D);
    // }
    // auto end_loop = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> duration_loop = end_loop - start_loop;
    // std::cout << "Time for loop-based implementation: " << duration_loop.count()/iterations << " seconds\n";


    // // Initialize tensors with random values
    // A.setRandom();
    // B.setRandom();

    // // Define the resulting tensor D with shape (m, n)
    // Eigen::Tensor<float, 2> E(n, m);

    // // // Time the dot product computation using .chip() (CHATGPT USES CHIP JUST FOR FANCY INDEXING)
    // // std::cout << "Timing dot product computation with .chip():" << std::endl;
    // // timeDotProductComputation(computeDotProductWithChip, A, B, E, 1000);

    // // Time the dot product computation using nested loops
    // std::cout << "Timing dot product computation with loops:" << std::endl;
    // timeDotProductComputation(computeDotProductWithLoops, A, B, E, 1000);


    // M32 Test
    Tensor<float,3> A2(n,m,2);
    m32(A, dxdC, dydC, A2);

    std::cout << A << std::endl;
    std::cout << A2 << std::endl;

    return 0;
}