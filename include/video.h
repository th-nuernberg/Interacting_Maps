//
// Created by daniel on 2/13/25.
//

#ifndef INTERACTINGMAPS_VIDEO_H
#define INTERACTINGMAPS_VIDEO_H
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

namespace VideoCreator {

/**
 * @brief Get a sorted list of image filenames from the specified folder.
 *
 * @param folderPath Path to the folder containing images.
 * @return A vector of sorted image file paths.
 */
    std::vector<std::string> getSortedImageFiles(const std::string& folderPath);

/**
 * @brief Create an MP4 video from a folder of images.
 *
 * @param folderPath Path to the folder containing images.
 * @param outputFile Path to the output MP4 file.
 * @param fps Frames per second for the video.
 */
    void createMP4Video(const std::string& folderPath, const std::string& outputFile, int fps = 30);

} // namespace VideoCreator

#endif //INTERACTINGMAPS_VIDEO_H
