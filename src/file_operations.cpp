//
// Created by daniel on 11/25/24.
//
#include "file_operations.h"
#include <iostream>
#include <memory>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iterator>

// Helper function to check if a string ends with a given suffix
bool endsWith(const std::string &str, const std::string &suffix) {
    if (str.length() < suffix.length()) return false;
    return str.compare(str.length() - suffix.length(), suffix.length(), suffix) == 0;
}


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

void writeToFile(const Tensor1f &t, const std::string &fileName){
    std::ofstream file(fileName);
    if (file.is_open())
    {
        file << t;
    }
}

void writeToFile(const float time, const float loss, const std::string &fileName){
    std::ofstream file(fileName);
    if (file.is_open())
    {
        file << time << loss << std::endl;
    }
}

void writeToFile(const float time, const Tensor1f &t, const std::string &fileName, bool append = true) {
    if (append) {
        std::ofstream file(fileName, std::ios::app);
        if (file.is_open())
        {
            file << time << " " << t[0] << " " << t[1] << " " << t[2] << std::endl;
        }
    }
    else {
        std::ofstream file(fileName);
        if (file.is_open())
        {
            file << time << " " << t[0] << " " << t[1] << " " << t[2] << std::endl;
        }
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
 * Creates the results folder at <current directory>/output/<folder_name>, including parents.
 * output/ is ignored as a whole in .gitignore, so no per-run entries are needed.
 * @param folder_name name of the results folder inside output/
 * @return path to created folder
 */
fs::path create_results_folder(const std::string &folder_name) {
    fs::path folder_path = fs::current_path() / "output" / folder_name;
    fs::create_directories(folder_path);
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
        // Whitespace-separated floats; stream extraction also copes with newlines and repeated spaces.
        for (float value; calibration_file >> value;) {
            calibration_data.push_back(value);
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
    // {vertical, horizontal}: the vertical angle uses fy, the horizontal one fx
    data.view_angles = std::vector<float>({2*std::atan(frame_height/(2*data.focal_point[1])),
                                           2*std::atan(frame_width/(2*data.focal_point[0]))});
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
void read_events(const std::string &file_path, std::vector<std::shared_ptr<Event>> &events, float start_time, float end_time, int max_events = INT32_MAX, std::string timeFormat="muS"){
    fs::path current_directory = fs::current_path();
    std::string path = current_directory / file_path;
    if (fs::exists(path)) {
        if (fs::exists(path) && endsWith(file_path, ".hdf5")) {
            try {
                // Open the HDF5 file
                H5::H5File file(path, H5F_ACC_RDONLY);

                // Open the "events" group
                H5::Group eventsGroup = file.openGroup("events");

                // Open the "time" dataset
                H5::DataSet timeDataset = eventsGroup.openDataSet("time");
                H5::DataSpace timeSpace = timeDataset.getSpace();
                hsize_t timeDims[1];
                timeSpace.getSimpleExtentDims(timeDims);
                size_t numEvents = std::min(static_cast<size_t>(timeDims[0]), static_cast<size_t>(max_events));

                // Read the "time" data
                std::vector<float> times(numEvents);
                timeDataset.read(times.data(), H5::PredType::NATIVE_FLOAT);

                // Open and read the "y", "x", and "polarity" datasets
                H5::DataSet yDataset = eventsGroup.openDataSet("y");
                H5::DataSet xDataset = eventsGroup.openDataSet("x");
                H5::DataSet polarityDataset = eventsGroup.openDataSet("polarity");

                std::vector<int> y(numEvents), x(numEvents), polarity(numEvents);
                yDataset.read(y.data(), H5::PredType::NATIVE_INT);
                xDataset.read(x.data(), H5::PredType::NATIVE_INT);
                polarityDataset.read(polarity.data(), H5::PredType::NATIVE_INT);

                // Find the first and last indices for the time range
                float firstTime = times[0];
                size_t firstIdx = searchSorted(times, firstTime + start_time);
                size_t lastIdx = searchSorted(times, firstTime + end_time);

                // Add events to the output vector
                for (size_t i = firstIdx; i < lastIdx; ++i) {
                    std::vector<int> coords = {y[i], x[i]};
                    events.emplace_back(std::make_shared<CameraEvent>(times[i], coords, polarity[i] * 2 - 1));
                }
            } catch (const H5::Exception &e) {
                std::cerr << "HDF5 Error: " << e.getDetailMsg() << std::endl;
            }
    }
        std::ifstream event_file(path);
        int counter = 0;
        float time;
        int width, height, polarity;
        while (event_file >> time >> width >> height >> polarity){
            if (time < start_time) continue;
            if (time > end_time) break;
            if (counter > max_events) break;
            std::vector coords = {height, width};
            events.emplace_back(std::make_shared<CameraEvent>(time, coords, polarity*2-1));
            counter++;
        }
    }
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
bool load_events_txt(const std::string &file_path, std::vector<RawCameraEvent> &events){
    std::ifstream file(fs::current_path() / file_path, std::ios::binary);
    if (!file) return false;
    std::string buffer((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    const char *p = buffer.c_str();
    const char *end = p + buffer.size();
    // Skip a header line such as "time x y p"
    while (p < end && std::isspace(static_cast<unsigned char>(*p))) ++p;
    if (p < end && !std::isdigit(static_cast<unsigned char>(*p)) && *p != '-' && *p != '.') {
        while (p < end && *p != '\n') ++p;
    }
    events.reserve(buffer.size() / 24);
    while (p < end) {
        char *next;
        const float time = std::strtof(p, &next);
        if (next == p) break;
        const long x = std::strtol(next, &next, 10);
        const long y = std::strtol(next, &next, 10);
        const long polarity = std::strtol(next, &next, 10);
        events.push_back({time, static_cast<int16_t>(y), static_cast<int16_t>(x), static_cast<int8_t>(polarity * 2 - 1)});
        p = next;
    }
    return true;
}

void slice_events(const std::vector<RawCameraEvent> &all, float start_time, float end_time, std::vector<std::shared_ptr<Event>> &events){
    auto first = std::lower_bound(all.begin(), all.end(), start_time,
                                  [](const RawCameraEvent &e, float t) { return e.time < t; });
    auto last = std::upper_bound(first, all.end(), end_time,
                                 [](float t, const RawCameraEvent &e) { return t < e.time; });
    events.reserve(events.size() + static_cast<size_t>(last - first));
    for (auto it = first; it != last; ++it) {
        std::vector<int> coords = {it->y, it->x};
        events.emplace_back(std::make_shared<CameraEvent>(it->time, coords, it->polarity));
    }
}

void read_imu(const std::string &file_path, std::vector<std::shared_ptr<Event>> &events, float start_time, float end_time, int max_events = INT32_MAX){
    fs::path current_directory = fs::current_path();
    std::string path = current_directory / file_path;
    if (fs::exists(path)) {
        std::ifstream event_file(path);
        int counter = 0;
        float time,a1,a2,a3,g1,g2,g3;
        std::vector<float> accel;
        std::vector<float> angVel;
        while (event_file >> time >> a1 >> a2 >> a3 >> g1 >> g2 >> g3){
            if (time < start_time) continue;
            if (time > end_time) break;
            if (counter > max_events) break;
            accel = {a2, a1, a3};
            angVel = {g2, g1, g3};
            events.emplace_back(std::make_shared<IMUEvent>(time, accel, angVel));
            counter++;
        }
    }
}

void readImage(const std::string &file_path, std::vector<std::shared_ptr<Event>> &events, float start_time, float end_time, int max_events = INT32_MAX){
    fs::path current_directory = fs::current_path();
    std::string path = current_directory / file_path;
    if (fs::exists(path)) {
        std::ifstream event_file(path);
        int counter = 0;
        float time;
        std::string imagePath;
        while (event_file >> time >> imagePath) {
            if (time < start_time) continue;
            if (time > end_time) break;
            if (counter > max_events) break;
            cv::Mat image;
            image = cv::imread(current_directory / "res" / "boxes_rotation" / imagePath, cv::IMREAD_GRAYSCALE);
            std::cout << "Loaded image: " << imagePath << " at time " << time << std::endl;
            // Convert to float32
            cv::Mat img32f;
            image.convertTo(img32f, CV_32F, 1/255.0);
            // Check if the image was loaded successfully
            if (image.empty()) {
                std::cerr << "Error: Could not load image at " << imagePath << std::endl;
            } else {
                events.emplace_back(std::make_shared<ImageEvent>(time, img32f));
                counter++;
            }
        }
    }
}

void mergeTimeCollections(std::vector<std::shared_ptr<Event>>& collection1, std::vector<std::shared_ptr<Event>>& collection2, std::vector<std::shared_ptr<Event>> &mergedCollection) {
    size_t pointer1 = 0;
    size_t pointer2 = 0;

    // Traverse both collections and merge them
    while (pointer1 < collection1.size() && pointer2 < collection2.size()) {
        if (collection1[pointer1]->time < collection2[pointer2]->time) {
            mergedCollection.push_back(collection1[pointer1]);
            pointer1++;
        } else {
            mergedCollection.push_back(collection2[pointer2]);
            pointer2++;
        }
    }

    // Append the remaining elements of collection1, if any
    while (pointer1 < collection1.size()) {
        mergedCollection.push_back(collection1[pointer1]);
        pointer1++;
    }

    // Append the remaining elements of collection2, if any
    while (pointer2 < collection2.size()) {
        mergedCollection.push_back(collection2[pointer2]);
        pointer2++;
    }
}

// Function to find the first index where time >= target
size_t searchSorted(const std::vector<float>& times, float target) {
    auto it = std::lower_bound(times.begin(), times.end(), target);
    return std::distance(times.begin(), it);
}
