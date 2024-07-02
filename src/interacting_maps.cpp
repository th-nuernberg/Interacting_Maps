#include "interacting_maps.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <sstream>

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

        std::cout << index << " ";
        // indices.push_back(index);
        buckets[*it].push_back(event);
    }
    return buckets;
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
    for (float data : calibration_data){
        std::cout << data << std::endl;
    }

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

    return 0;
}