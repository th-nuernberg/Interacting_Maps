//
// Created by root on 7/29/25.
//

#ifndef EVENTS_H
#define EVENTS_H
#include <vector>
#include <datatypes.h>

/**
 * Take a vector of events and sort them into a collection of bins. Sorting happens according to the time dimension of
 * the events. Size of the bins is given in seconds.
 * @param events std::vector of Events
 * @param bin_size float
 * @return vector of vectors of binned events. First bin contains first timespan, last bin the last timespan.
 *
 */
std::vector<std::vector<Event>> bin_events(std::vector<Event> &events, float bin_size);

/**
 * Convert a collection of binned events to a collection of frames (i.e. matrices representing an image)
 * @param bucketed_events collection of events sorted into bins based on time
 * @param frames returning collection of frames
 * @param camera_height height of the frame
 * @param camera_width width of the frame
 */
void create_frames(std::vector<std::vector<Event>> &bucketed_events, std::vector<Tensor2f> &frames, int camera_height, int camera_width);



#endif //EVENTS_H
