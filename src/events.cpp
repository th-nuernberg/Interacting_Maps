//
// Created by root on 7/29/25.
//

#include <events.h>

std::vector<std::vector<Event>> bin_events(std::vector<Event> &events, float bin_size = 0.05){
    std::vector<std::vector<Event>> bins;
    if (events.empty()) {
        return bins;  // Return empty if the input vector is empty
    }

    Event minVal = events.front();  // The lowest number in the sorted vector
    float currentBinStart = minVal.time;

    std::vector<Event> currentBin;

    for (const Event &event : events) {
        if (event.time >= currentBinStart && event.time < currentBinStart + bin_size) {
            currentBin.push_back(event);
        } else {
            // Push the current bin into bins and start a new bin
            bins.push_back(currentBin);
            currentBin.clear();
            currentBinStart += bin_size;
            // Keep adjusting the currentBinStart if the number falls outside the current bin
            while (event.time >= currentBinStart + bin_size) {
                currentBinStart += bin_size;
                bins.emplace_back(); // Add an empty bin for skipped bins
            }
            currentBin.push_back(event);
        }
    }
    // Push the last bin
    bins.push_back(currentBin);
    return bins;
}

void create_frames(const std::vector<std::vector<CameraEvent>> &bucketed_events, std::vector<Tensor2f> &frames, const int camera_height, const int camera_width, float eventContribution){
    int i = 0;
    Tensor2f frame(camera_height, camera_width);
    Tensor2f cum_frame(camera_height, camera_width);
    for (const std::vector<CameraEvent> &event_vector : bucketed_events){
        frame.setZero();
        cum_frame.setZero();
        for (CameraEvent event : event_vector){

            frame(event.coordinates.at(0), event.coordinates.at(1)) = (float) event.polarity * eventContribution;
        }
        frames[i] = frame;
        i++;
    }
}