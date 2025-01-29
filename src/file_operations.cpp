//
// Created by daniel on 11/25/24.
//
#include "file_operations.h"

/**
 * Saves a 3Tensor as string to a file on disk
 * @param t 3Tensor to be saved
 * @param fileName path on disk
 */
void writeToFile(const Tensor3f &t, const std::string &fileName){
    std::ofstream file(fileName);
    if (file.is_open())
    {
        file << t;
    }
}

/**
 * Saves a 3-Tensor as string to a file on disk
 * @param t 3-Tensor to be saved
 * @param fileName path on disk
 * @param y vertical coordinate of entry
 * @param x horizontal coordinate of entry
 */
void writeToFile(const Tensor3f &t, int y, int x, const std::string &fileName){
    std::ofstream file(fileName, std::ios::app);
    if (file.is_open())
    {
        file << t(y, x, 0) << " " << t(y, x, 1) << std::endl;
    }
}

void writeToFile(const Tensor<float,1> &t, const std::string &fileName){
    std::ofstream file(fileName);
    if (file.is_open())
    {
        file << t;
    }
}

/**
 * Saves a 2Tensor as string to a file on disk
 * @param t 2Tensor to be saved
 * @param fileName path on disk
 */
void writeToFile(const Tensor2f &t, const std::string &fileName){
    std::ofstream file(fileName);
    if (file.is_open())
    {
        file << t;
    }
}

/**
 * Saves an entry of a 2-Tensor as string to a file on disk
 * @param t 2-Tensor to be saved
 * @param fileName path on disk
 * @param y vertical coordinate of entry
 * @param x horizontal coordinate of entry
 */
void writeToFile(const Tensor2f &t, int y, int x, const std::string &fileName){
    std::ofstream file(fileName, std::ios::app);
    if (file.is_open())
    {
        file << t(x,y) << std::endl;
    }
}

/**
 * Saves a Eigen matrix as string to a file on disk
 * @param t matrix to be saved
 * @param fileName path on disk
 */
void writeToFile(const MatrixXfRowMajor &t, const std::string &fileName){
    std::ofstream file(fileName);
    if (file.is_open())
    {
        file << t;
    }
}

/**
 * Saves a string s to a file;
 * @param s
 * @param fileName
 */
void writeToFile(const std::string &s, const std::string &fileName){
    std::ofstream file(fileName);
    if (file.is_open())
    {
        file << s;
    }
}

void writeToFile(float V, const std::string &fileName) {
    std::ofstream file(fileName, std::ios::app);
    if (file.is_open())
    {
        file << V << std::endl;
    }
}

/**
 * Creates a results at current directory/output/name and ads the path to .gitignore to contain git bloat
 * @param folder_name path to folder which is to be created
 * @return path to created folder
 */
fs::path create_folder_and_update_gitignore(const std::string &folder_name) {
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
    return folder_path;
}
/**
 * Reads out a single line text file consisting of a string of float
 * @param file_path path to the file
 * @param calibration_data std::vector of contained floats
 */
void read_single_line_txt(const std::string &file_path, std::vector<float> &calibration_data){
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
/**
 * Converts a vector of calibration data floats to an Calibration_Data struct for further use
 * @param raw_data std::vector of floats of calibration data
 * @param frame_height height of the image for which the calibration data is for
 * @param frame_width width of the image for which the calibration data is for
 * @return Combined Calibration_Data
 */
Calibration_Data get_calibration_data(const std::vector<float> &raw_data, int frame_height, int frame_width){
    Calibration_Data data;
    data.focal_point = std::vector<float>(raw_data.begin(), raw_data.begin()+2);
    data.camera_matrix = MatrixXf({{data.focal_point[0], 0, raw_data[2]},
                                   {0, data.focal_point[1], raw_data[3]},
                                   {0, 0, 1}});
    data.distortion_coefficients = std::vector<float>(raw_data.begin()+4, raw_data.end());
    data.view_angles = std::vector<float>({2*std::atan(frame_height/(2*data.focal_point[0])),
                                           2*std::atan(frame_width/(2*data.focal_point[1]))});
    return data;
}

/**
 * Reads out events from disk and converts it to a std::vector of Events. Expects a file with on event
 * per line in chronological order.
 * @param file_path Path to the event file
 * @param events Empty vector in which the events get written
 * @param start_time time stamp from which on out to consider frames
 * @param end_time time stamp after which events get ignored
 * @param event_factor allows scaling the event intensity with an factor
 * @param max_events upper limit on the amount of events to save if end_time is not reached before
 */
void read_events(const std::string &file_path, std::vector<Event> &events, float start_time, float end_time, int max_events = INT32_MAX){
    fs::path current_directory = fs::current_path();
    std::string path = current_directory / file_path;
    if (fs::exists(path)) {
        std::ifstream event_file(path);
        int counter = 0;
        float time;
        int width, height, polarity;
        while (event_file >> time >> width >> height >> polarity){
            if (time < start_time) continue;
            if (time > end_time) break;
            if (counter > max_events) break;
            Event event;
            event.time = time;
            std::vector<int> coords = {height, width};
            event.coordinates = coords;
            event.polarity = (polarity*2 - 1);
            events.push_back(event);
            counter++;
        }
    }
}


