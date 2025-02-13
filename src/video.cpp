//
// Created by daniel on 2/13/25.
//
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <iostream>

namespace fs = std::filesystem;

namespace VideoCreator {

    std::vector<std::string> getSortedImageFiles(const std::string &folderPath) {
        std::vector<std::string> imageFiles;

        for (const auto &entry: fs::directory_iterator(folderPath)) {
            if (entry.path().extension() == ".png" && entry.path().filename().string().find("VIGF_") == 0) {
                imageFiles.push_back(entry.path().string());
            }
        }

        std::sort(imageFiles.begin(), imageFiles.end()); // Sort lexicographically (should work if zero-padded)
        return imageFiles;
    }

    void createMP4Video(const std::string &folderPath, const std::string &outputFile, int fps = 30) {
        auto imageFiles = getSortedImageFiles(folderPath);

        if (imageFiles.empty()) {
            std::cerr << "No images found in the folder!" << std::endl;
            return;
        }

        cv::Mat frame = cv::imread(imageFiles[0]);
        if (frame.empty()) {
            std::cerr << "Error reading the first image." << std::endl;
            return;
        }

        // Define the codec and create a VideoWriter object
        int fourcc = cv::VideoWriter::fourcc('H', '2', '6', '4'); // MP4 encoding
        cv::Size frameSize(frame.cols, frame.rows);
        cv::VideoWriter videoWriter(outputFile, fourcc, fps, frameSize, true);

        if (!videoWriter.isOpened()) {
            std::cerr << "Could not open the video file for writing." << std::endl;
            return;
        }

        for (const auto &file: imageFiles) {
            frame = cv::imread(file);
            if (frame.empty()) {
                std::cerr << "Skipping invalid image: " << file << std::endl;
                continue;
            }
            videoWriter.write(frame);
        }

        std::cout << "Video saved as " << outputFile << std::endl;
    }

}