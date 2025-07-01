//
// Created by daniel on 11/25/24.
//

#ifndef INTERACTINGMAPS_FILE_OPERATIONS_H
#define INTERACTINGMAPS_FILE_OPERATIONS_H

#include <string>
#include <filesystem>
#include <fstream>
#include "datatypes.h"

namespace fs = std::filesystem;


void writeToFile(const Tensor3f &t, const std::string &fileName);

void writeToFile(const Tensor3f &t, int y, int x, const std::string &fileName);

void writeToFile(const Tensor<float,1> &t, const std::string &fileName);

void writeToFile(const float time, const Tensor<float,1> &t, const std::string &fileName, bool append) ;

void writeToFile(const Tensor2f &t, const std::string &fileName);

void writeToFile(const Tensor2f &t, int y, int x, const std::string &fileName);

void writeToFile(const MatrixXfRowMajor &t, const std::string &fileName);

void writeToFile(const std::string &s, const std::string &fileName);

void writeToFile(float V, const std::string &fileName);

fs::path create_folder_and_update_gitignore(const std::string &folder_name);

void read_single_line_txt(const std::string &file_path, std::vector<float> &calibration_data);

Calibration_Data get_calibration_data(const std::vector<float> &raw_data, int frame_height, int frame_width);

void read_events(const std::string &file_path, std::vector<std::shared_ptr<Event>> &events, float start_time, float end_time, int max_events);

void read_imu(const std::string &file_path, std::vector<std::shared_ptr<Event>> &events, float start_time, float end_time, int max_events);

void readImage(const std::string &file_path, std::vector<std::shared_ptr<Event>> &events, float start_time, float end_time, int max_events);

void mergeTimeCollections(std::vector<std::shared_ptr<Event>>& collection1, std::vector<std::shared_ptr<Event>>& collection2, std::vector<std::shared_ptr<Event>> &mergedCollection);
#endif //INTERACTINGMAPS_FILE_OPERATIONS_H
