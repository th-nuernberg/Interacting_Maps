//
// Created by daniel on 11/25/24.
//

#ifndef INTERACTINGMAPS_FILE_OPERATIONS_H
#define INTERACTINGMAPS_FILE_OPERATIONS_H

#include <filesystem>
#include <cstdint>
#include <fstream>
#include <H5Cpp.h>
#include <string>
#include "datatypes.h"

namespace fs = std::filesystem;

void writeToFile(const Tensor3f &t, const std::string &fileName);

void writeToFile(const Tensor3f &t, int y, int x, const std::string &fileName);

void writeToFile(const Tensor1f &t, const std::string &fileName);

void writeToFile(const float time, const float loss, const std::string &fileName);

void writeToFile(const float time, const Tensor1f &t, const std::string &fileName, bool append) ;

void writeToFile(const Tensor2f &t, const std::string &fileName);

void writeToFile(const Tensor2f &t, int y, int x, const std::string &fileName);

void writeToFile(const MatrixXfRowMajor &t, const std::string &fileName);

void writeToFile(const std::string &s, const std::string &fileName);

void writeToFile(float V, const std::string &fileName);

fs::path create_results_folder(const std::string &folder_name);

void read_single_line_txt(const std::string &file_path, std::vector<float> &calibration_data);

Calibration_Data get_calibration_data(const std::vector<float> &raw_data, int frame_height, int frame_width);

void read_events(const std::string &file_path, std::vector<std::shared_ptr<Event>> &events, float start_time, float end_time, int max_events, std::string timeFormat);

// Compact camera event for holding a whole recording in memory (12 bytes instead of a heap-allocated CameraEvent)
struct RawCameraEvent {
    float   time;
    int16_t y;
    int16_t x;
    int8_t  polarity;   // -1 or +1
};

/**
 * Reads a whole events.txt (one "time x y polarity[0,1]" per line, chronological) into memory in one pass.
 * A non-numeric first line (header) is skipped.
 * @param file_path path to the event file, relative to the working directory
 * @param events receives all events
 * @return false if the file could not be opened
 */
bool load_events_txt(const std::string &file_path, std::vector<RawCameraEvent> &events);

/**
 * Appends the events with start_time <= time <= end_time as CameraEvents (same bounds as read_events)
 * @param all events loaded by load_events_txt
 * @param start_time first time to include
 * @param end_time last time to include
 * @param events receives the slice
 */
void slice_events(const std::vector<RawCameraEvent> &all, float start_time, float end_time, std::vector<std::shared_ptr<Event>> &events);

void read_imu(const std::string &file_path, std::vector<std::shared_ptr<Event>> &events, float start_time, float end_time, int max_events);

void readImage(const std::string &file_path, std::vector<std::shared_ptr<Event>> &events, float start_time, float end_time, int max_events);

void mergeTimeCollections(std::vector<std::shared_ptr<Event>>& collection1, std::vector<std::shared_ptr<Event>>& collection2, std::vector<std::shared_ptr<Event>> &mergedCollection);

size_t searchSorted(const std::vector<float>& times, float target);

std::vector<std::vector<int>> readHDF5Events(
    const std::string& filePath,
    int maxEvents,
    double startTime,
    double endTime
);
#endif //INTERACTINGMAPS_FILE_OPERATIONS_H
