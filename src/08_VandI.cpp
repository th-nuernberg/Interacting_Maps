//
// Created by Daniel Pommer on 08.12.25.
//
#include <dv-processing/io/camera/discovery.hpp>
#include <dv-processing/visualization/event_visualizer.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <dv-processing/noise/k_noise_filter.hpp>

#include <csignal>
static std::atomic<bool> globalShutdown(false);


static void handleShutdown(int) {
    globalShutdown.store(true);
}

int main() {
    using namespace std::chrono_literals;
    static constexpr int ESC_KEYCODE = 27;
    // Install signal handlers for a clean shutdown
    std::signal(SIGINT, handleShutdown);
    std::signal(SIGTERM, handleShutdown);
    // Open the specified camera
    auto capture = dv::io::camera::DAVIS("00001088");

    std::cout << "Camera [" << capture.getCameraName() << "] has been opened!" << std::endl;
    std::cout << "Resolution [" << capture.getEventResolution()->width << "x" << capture.getEventResolution()->height
              << "]." << std::endl;
    // Check whether the IMU stream is available

    if (capture.isImuStreamAvailable()) {
        // Print the imu data stream capability
        std::cout << "* IMU measurements" << std::endl;
    }

    int ROI_width = 100;
    int ROI_height = 100;
    cv::Size ROI_size = cv::Size(ROI_width, ROI_height);

    // Set ROI for events
    //capture.setCropAreaEvents({0, 0, ROI_width, ROI_height});
    // Set ROI for frames
    //capture.setCropAreaFrames({0, 0, ROI_width, ROI_height});
    // Fails with anything other than 0 for x and y.

    // Setting camera readout to events and frames (default).
    capture.setEventsRunning(true);
    capture.setFramesRunning(true);

    // FRAME OPTIONS
    // Configure frame output mode to color (default), only on COLOR cameras. Other mode available: GRAYSCALE
    capture.setColorMode(dv::io::camera::parser::DAVIS::ColorMode::DEFAULT);
    // Enable frame auto-exposure (default behavior)
    capture.setAutoExposure(true);
    // Disable auto-exposure, set frame exposure (here 10ms)
    capture.setAutoExposure(false);
    capture.setExposureDuration(10ms);
    // Read current frame exposure duration value
    std::chrono::microseconds duration = capture.getExposureDuration();
    // Set frame interval duration (here 33ms for ~30FPS)
    capture.setFrameInterval(33ms);
    // Read current frame interval duration value
    std::chrono::microseconds interval = capture.getFrameInterval();

    // ACCUM AND VIS
    // Initialize an accumulator with some resolution
    //dv::visualization::EventVisualizer visualizer(*capture.getEventResolution());
    //dv::Accumulator accumulator(*capture.getEventResolution());
    dv::visualization::EventVisualizer visualizer(ROI_size);
    dv::Accumulator accumulator(ROI_size);

    // Apply event color scheme configuration, these values can be modified to taste
    visualizer.setBackgroundColor(dv::visualization::colors::white);
    visualizer.setPositiveColor(dv::visualization::colors::green);
    visualizer.setNegativeColor(dv::visualization::colors::red);

    // Apply accumulator configuration, these values can be modified to taste
    accumulator.setMinPotential(0.f);
    accumulator.setMaxPotential(1.f);
    accumulator.setNeutralPotential(0.5f);
    accumulator.setEventContribution(0.1f);
    accumulator.setDecayFunction(dv::Accumulator::Decay::EXPONENTIAL);
    accumulator.setDecayParam(1e+7);
    accumulator.setIgnorePolarity(false);
    accumulator.setSynchronousDecay(false);

    // Initialize region filter using hardcoded coordinates
    dv::EventRegionFilter regionFilter(cv::Rect(0, 0, ROI_width, ROI_height));
    dv::noise::KNoiseFilter kNoiseFilter(ROI_size);

    // Initialize a preview window
    cv::namedWindow("Events", cv::WINDOW_NORMAL);
    cv::namedWindow("Images", cv::WINDOW_NORMAL);
    cv::namedWindow("RealFrame", cv::WINDOW_NORMAL);

    // Initialize a slicer
    dv::EventStreamSlicer slicer;

    // Define options for text in preview images
    const cv::Point2i textPosition(20, 20);
    const cv::Point2i textShift(0, 20);
    const double fontScale      = 0.7;
    const cv::Scalar fontColor  = dv::visualization::colors::red;
    const int32_t fontThickness = 2;

    // Register a callback every 33 milliseconds
    slicer.doEveryTimeInterval(100ms, [&visualizer, &accumulator, &kNoiseFilter, &regionFilter, &textPosition, &fontColor, &fontScale, &textShift](const dv::EventStore &events) {
        regionFilter.accept(events);
        dv::EventStore regionFilteredEvents = regionFilter.generateEvents();
        kNoiseFilter.accept(regionFilteredEvents);
        dv::EventStore kNoiseFiltered = kNoiseFilter.generateEvents();
        // cv::Mat kNoiseFilterPreview   = visualizer.generateImage(kNoiseFiltered);
        // cv::putText(kNoiseFilterPreview, "K-Noise filter", textPosition, cv::FONT_HERSHEY_SIMPLEX, fontScale, fontColor,
        //     fontThickness);
        // cv::putText(kNoiseFilterPreview, fmt::format("Reduction factor: {:.2f}", kNoiseFilter.getReductionFactor()),
        //     textPosition + textShift, cv::FONT_HERSHEY_SIMPLEX, fontScale, fontColor, fontThickness);

        cv::Mat image = visualizer.generateImage(regionFilteredEvents);
        // Pass events into the accumulator and generate a preview frame
        accumulator.accept(regionFilteredEvents);
        dv::Frame frame = accumulator.generateFrame();

        // Show the event image and the accumulated image
        cv::imshow("Events", image);
        cv::imshow("Image", frame.image);
        cv::waitKey(10);
    });
    // Run the event processing while the camera is connected
    while (!globalShutdown && capture.isRunning()) {
        // Receive events, check if anything was received
        if (const auto events = capture.getNextEventBatch()) {
            // If so, pass the events into the slicer to handle them
            slicer.accept(*events);
        }
        // Read a frame, check whether it is correct.
        // The method does not wait for frame arrive, it returns immediately with
        // the latest available frame or if no data is available, returns a `std::nullopt`.
        if (const auto frame = capture.getNextFrame(); frame.has_value()) {
            std::cout << *frame << std::endl;
            // Show a preview of the image
            cv::imshow("RealFrame", frame->image);
        }
        /*if (const auto imuBatch = capture.getNextImuBatch(); imuBatch.has_value() && !imuBatch->empty()) {
            std::cout << "Received " << imuBatch->size() << " IMU measurements" << std::endl;
        }
        else {
            // No data has arrived yet, short sleep to reduce CPU load.
            std::this_thread::sleep_for(1ms);
        }*/
    }
    return 0;
}