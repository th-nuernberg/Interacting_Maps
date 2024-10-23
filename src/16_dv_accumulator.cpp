//
// Created by arbeit on 10/23/24.
//
#include <dv-processing/core/frame.hpp>

#include <dv-processing/io/camera_capture.hpp>

#include <dv-processing/data/generate.hpp>


#include <opencv2/highgui.hpp>


int main() {

    using namespace std::chrono_literals;

    // Generate 1000 events within 2 second interval. These will be sliced correctly by the slicer.

    const dv::EventStore store = dv::data::generate::uniformEventsWithinTimeRange(0, 2s, cv::Size(100, 100), 1000);

    // Initialize an accumulator with some resolution

    dv::Accumulator accumulator(cv::Size(100, 100));

    // Apply configuration, these values can be modified to taste

    accumulator.setMinPotential(0.f);

    accumulator.setMaxPotential(1.f);

    accumulator.setNeutralPotential(0.5f);

    accumulator.setEventContribution(0.15f);

    accumulator.setDecayFunction(dv::Accumulator::Decay::EXPONENTIAL);

    accumulator.setDecayParam(1e+6);

    accumulator.setIgnorePolarity(false);

    accumulator.setSynchronousDecay(false);


    // Initialize a preview window

    cv::namedWindow("Preview", cv::WINDOW_NORMAL);


    // Initialize a slicer

    dv::EventStreamSlicer slicer;


    // Register a callback every 33 milliseconds

    slicer.doEveryTimeInterval(33ms, [&accumulator](const dv::EventStore &events) {
        // Pass events into the accumulator and generate a preview frame

        accumulator.accept(events);

        dv::Frame frame = accumulator.generateFrame();
        // Show the accumulated image

        cv::imwrite("Preview.png" , frame.image);

//        cv::waitKey(0);

    });


    // Run the event processing while the camera is connected

    slicer.accept(store);


    return 0;

}