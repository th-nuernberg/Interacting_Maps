#include <interacting_maps.h>
#include <iostream>
#include <filesystem>
#include <string>
#include <sstream>
#include <cassert>
#include <memory>
#include "Instrumentor.h"
#include <cmath>
#include <update.h>
#include <cost.h>
#include "file_operations.h"
#include "imaging.h"
#include <xtensor.hpp>
#include <dv-processing/io/camera/discovery.hpp>
#include <dv-processing/visualization/event_visualizer.hpp>
#include <dv-processing/noise/k_noise_filter.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <csignal>

static std::atomic<bool> globalShutdown(false);

static void handleShutdown(int) {
    globalShutdown.store(true);
}

namespace po = boost::program_options;

#ifdef PROFILING
#define PROFILE_SCOPE(name) InstrumentationTimer timer##__LINE__(name)
#define PROFILE_FUNCTION() PROFILE_SCOPE(__FUNCTION__)
// #define PROFILE_FUNCTION() PROFILE_SCOPE(__FUNCSIG__) (Includes call attributes, whole signature of function)
#define PROFILE_MAIN(name)
#else
#define PROFILE_SCOPE(name)
#define PROFILE_FUNCTION()
#define PROFILE_MAIN(name) InstrumentationTimer timer##__LINE__(name)
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  INTERACTING MAPS MAIN FUNCTION  ////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void event_step(const float V, Tensor2f &MI, Tensor3f &delta_I, Tensor3f &GIDiff, Tensor3f &GIDiffGradient, Tensor3f &F, Tensor3f &G, Tensor1f &R, const Tensor3f &CCM, const Tensor3f &dCdx, const Tensor3f &dCdy, const Matrix3f &A, Vector3f &B, const std::vector<std::vector<Matrix3f>> &Identity_minus_outerProducts, std::vector<std::vector<Vector3f>> &old_points, std::unordered_map<std::string,float> &parameters, std::vector<int> &permutation, int y, int x){
    PROFILE_FUNCTION();
    array<Index, 2> dimensions = MI.dimensions();
    update_IV(MI, V, y, x, parameters["minPotential"], parameters["maxPotential"], parameters["weight_IV"]);
    // Image (MI) got changed through update by V. we need to update all surrounding gradient values. Because of the change at this pixel
    {
        PROFILE_SCOPE("GRADIENTS");
        if (y>0){
            computeGradient(MI, delta_I, y-1, x);
            update_GI(G, delta_I, y-1, x, parameters["weight_GI"], parameters["eps"], parameters["gamma"]);
        }
        if (x>0){
            computeGradient(MI, delta_I, y, x-1);
            update_GI(G, delta_I, y, x-1, parameters["weight_GI"], parameters["eps"], parameters["gamma"]);
        }
        if (y<dimensions[0]-1){
            computeGradient(MI, delta_I, y+1, x);
            update_GI(G, delta_I, y+1, x, parameters["weight_GI"], parameters["eps"], parameters["gamma"]);
        }
        if (x<dimensions[1]-1){
            computeGradient(MI, delta_I, y, x+1);
            update_GI(G, delta_I, y, x+1, parameters["weight_GI"], parameters["eps"], parameters["gamma"]);
        }
    }

    //computeGradient(MI, delta_I, y, x);
    update_GI(G, delta_I, y, x, parameters["weight_GI"], parameters["eps"], parameters["gamma"]);
    updateGIDiffGradient(G, delta_I, GIDiff, GIDiffGradient, y, x);
    update_IG(MI, GIDiffGradient, y, x, parameters["weight_IG"]);
    //computeGradient(MI, delta_I, y, x);

    for (const auto& element : permutation){
        switch( element ){
            default:
                std::cout << "Unknown number in permutation" << std::endl;
            case 0:
                update_FG(F, V, G, y, x, parameters["lr"], parameters["weight_FG"], parameters["eps"], parameters["gamma"]);
                break;
            case 1:
                // Gets called separately because we do not want to do an update of F based on R with every event since this update is global
                update_FR(F, CCM, dCdx, dCdy, R, parameters["weight_FR"], parameters["eps"], parameters["gamma"]);
                break;
            case 2:
                update_GF(G, V, F, y, x, parameters["lr"], parameters["weight_GF"], parameters["eps"], parameters["gamma"]);
                break;
            case 3:
                update_RF(R, F, CCM, dCdx, dCdy, A, B, Identity_minus_outerProducts, old_points, parameters["weight_RF"], y, x);
                break;
        }
    }
}

void event_step(const xt::xtensor<float, 2> &V, xt::xtensor<float, 2> &I, xt::xtensor<float, 3> &delta_I, xt::xtensor<float, 3> &GIDiff, xt::xtensor<float, 3> &GIDiffGradient, xt::xtensor<float, 3> &F, xt::xtensor<float, 3> &G, xt::xtensor<float, 1> &R, const xt::xtensor<float, 3> &CCM, const xt::xtensor<float, 3> &dCdx, const xt::xtensor<float, 3> &dCdy, const xt::xtensor<float, 2> &A, xt::xtensor<float, 1> &B, const xt::xtensor<float, 4> &Identity_minus_outerProducts, xt::xtensor<float, 3> &old_points, xt::xtensor<float, 3> &C1, xt::xtensor<float, 3> &C2, xt::xtensor<float, 2> &dot, xt::xtensor<float, 2> &distance, std::unordered_map<std::string,float> &parameters, std::vector<int> &permutation, std::default_random_engine &rng){
    PROFILE_FUNCTION();
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (const auto& element : permutation){
        switch( element ){
            default:
                std::cout << "Unknown number in permutation" << std::endl;
            case 0:
                update_FG(F, V, G, parameters["lr"], parameters["weight_FG"], parameters["eps"], parameters["gamma"]);
                break;
            case 1:
                // if (dis(rng) < 1.0) {
                update_FR(F, CCM, dCdx, dCdy, R, C1, C2, dot, distance, parameters["weight_FR"], parameters["eps"], parameters["gamma"]);
                // }
                break;
            case 2:
                update_FG(G, V, F, parameters["lr"], parameters["weight_GF"], parameters["eps"], parameters["gamma"]);
                break;
            case 3:
                update_RF(R, F, CCM, dCdx, dCdy, A, B, Identity_minus_outerProducts, old_points, parameters["weight_RF"]);
                break;
            case 4:
                update_GI(G, delta_I, parameters["weight_GI"], parameters["eps"], parameters["gamma"]);
                updateGIDiffGradient(G, delta_I, GIDiff, GIDiffGradient);
                break;
            case 5:
                update_IG(I, GIDiffGradient, parameters["weight_IG"]);
                computeGradient(I, delta_I);
                updateGIDiffGradient(G, delta_I, GIDiff, GIDiffGradient);
                break;
        }
    }
}

void randomInit(Tensor3f &T, const float lower, const float upper) {
    const auto &dimensions = T.dimensions();
    Tensor3f T1(dimensions[0], dimensions[1], dimensions[2]);
    Tensor3f T2(dimensions[0], dimensions[1], dimensions[2]);
    T.setRandom();
    T1.setConstant(lower);
    T2.setConstant(upper - lower);
    T = T*T2 + T1;
}

void softReset(Tensor3f &T, const float lower, const float upper) {
    const auto &dimensions = T.dimensions();
    Tensor3f T1(dimensions[0], dimensions[1], dimensions[2]);
    Tensor3f T2(dimensions[0], dimensions[1], dimensions[2]);
    Tensor3f T3(dimensions[0], dimensions[1], dimensions[2]);
    T1.setRandom();
    T2.setConstant(lower);
    T3.setConstant(upper - lower);
    T = T + T1*T3 + T2;
}

void randomInit(Tensor1f &T, const float lower, const float upper) {
    const auto &dimensions = T.dimensions();
    Tensor1f T1(dimensions[0]);
    Tensor1f T2(dimensions[0]);
    T.setRandom();
    T1.setConstant(lower);
    T2.setConstant(upper - lower);
    T = T*T2 + T1;
}

int main(int argc, char* argv[]) {

    // Define the command-line options
    po::options_description desc("Allowed options");
    desc.add_options()
            ("help,h", "Produce help message")
            ("startTime,f", po::value<float>()->default_value(10), "Where to start with event consideration")
            ("endTime,f", po::value<float>()->default_value(15), "Where to end with event consideration")
            ("timeFormat,s", po::value<std::string>()->default_value("s"), "What format are the times: seconds, milliseconds, microseconds (s,ms,mus)")
            ("timeStep,f", po::value<float>()->default_value(0.001), "Size of the event frames")
            ("resourceDirectory,s", po::value<std::string>()->default_value("live"), "Which dataset to use, searches in res directory")
            ("resultsDirectory,s", po::value<std::string>()->default_value("live"), "Where to store the results, located in output directory")
            ("addTime,b", po::value<bool>()->default_value(false), "Add time to output folder?")
            ("startIndex,i", po::value<int>()->default_value(0), "With what index to start for the images")
            ("fuseR,b", po::value<bool>()->default_value(false), "Fuse with imu.txt?")
            ("fuseI,b", po::value<bool>()->default_value(false), "Fuse with images?");
    // Parse command-line arguments
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    // Display help message if requested
    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 0;
    }
    // Retrieve values (using defaults if not provided)
    float startTime = vm["startTime"].as<float>();
    float endTime = vm["endTime"].as<float>();
    float timeStep = vm["timeStep"].as<float>();
    bool addTime = vm["addTime"].as<bool>();
    int startIndex = vm["startIndex"].as<int>();
    std::string timeFormat = vm["timeFormat"].as<std::string>();
    std::string resourceDirectory = vm["resourceDirectory"].as<std::string>();
    std::string resultsDirectory = vm["resultsDirectory"].as<std::string>();

    std::vector permutation {0,1,2,3,4}; // Which update steps to take; 1 is not needed
    // FG, FR, GF, RF, GI, IG
    bool eventBased = false;

    std::unordered_map<std::string,float> parameters;
    parameters["startTime"] = startTime;                                    // in seconds
    parameters["endTime"] = endTime;                                        // in seconds
    parameters["time_step"] = timeStep;                                     // in seconds
    parameters["weight_FG"] = 0.2;                                          // [0-1]
    parameters["weight_FR"] = 0.8;                                          // [0-1]
    parameters["weight_GF"] = 0.2;                                          // [0-1]
    parameters["weight_GI"] = 0.8;                                          // [0-1]
    parameters["weight_IG"] = 0.2;                                          // [0-1]
    parameters["weight_IV"] = 1.0;                                          // [0-1]
    parameters["weight_RF"] = 0.8;                                          // [0-1]
    parameters["weight_RIMU"] = 0.0;                                        // [0-1]
    parameters["weight_Ifusion"] = 0.0;                                     // [0-1]
    parameters["lr"] = 1.0;                                                 // [0-1]
    parameters["eventContribution"] = 20.0f;                                // mainly important for the visibility of the intensity image
    parameters["eps"] = 0.00001;                                            // lowest value allowed for F, G,...
    parameters["gamma"] = 255;                                              // highest value allowed for F, G,...
    parameters["decayParam"] = 1e-1;                                        // 1e-1 for exponential decay
    parameters["minPotential"] = 0.0;                                       // minimum Value for Image
    parameters["maxPotential"] = 255.0;                                     // maximal Value for Image
    parameters["neutralPotential"] = 128;                                   // base value where image decays back to
    parameters["convergenceSteps"] = 2;
    //parameters["fps"] = 1.0f/parameters["time_step"];                       // how often shown images are update
    //parameters["FR_updates_per_second"] = 1.0f/parameters["time_step"];     // how often the FR update is performed; It is not done after every event
    //parameters["updateIterationsFR"] = 4;                                   // more iterations -> F captures general movement of scene/camera better but significantly more computation time

    // Split time interval into sub intervals to allow loading of larger files.
    int nIntervals = 1;
    float maxIntervalLength = 0.05;
    std::vector intervals = {startTime, endTime};
    if (endTime - startTime > maxIntervalLength) {
        float currentTime = startTime;
        nIntervals = 0;
        intervals = {startTime};
        while (currentTime + maxIntervalLength < endTime) {
            currentTime += maxIntervalLength;
            intervals.push_back(currentTime);
            nIntervals++;
        }
        intervals.push_back(endTime);
        nIntervals++;
    }

    std::cout << "Parsed startTime: " << startTime << "\n";
    std::cout << "Parsed endTime: " << endTime << "\n";
    std::cout << "Parsed timeStep: " << timeStep << "\n";
    std::cout << "Parsed resourceDirectory: " << resourceDirectory << "\n";
    std::cout << "Parsed resultsDirectory: " << resultsDirectory << "\n";

    // Create results_folder
    std::string folder_name;
    if (addTime) {
        auto clock_time = std::chrono::system_clock::now();
        std::time_t time = std::chrono::system_clock::to_time_t(clock_time);
        folder_name = resultsDirectory + " " + std::ctime(&time);
    }
    else{
        folder_name = resultsDirectory;
    }
    fs::path folder_path = create_folder_and_update_gitignore(folder_name);
    std::cout << "Created Folder " << folder_name << std::endl;

    std::string profiler_name = "Profiler.json";
    fs::path profiler_path = folder_path / profiler_name;
    Instrumentor::Get().BeginSession("Interacting Maps", profiler_path);
    std::cout << "Setup Profiler" << std::endl;

    std::string calibrationPath = "res/" + resourceDirectory + "/calib.txt";
    std::string eventPath = "res/" + resourceDirectory + "/events.txt";
    std::string imuPath = "res/" + resourceDirectory + "/imu.txt";
    std::string imagesPath = "res/" + resourceDirectory + "/images.txt";
    std::string settingsPath = "res/" + resourceDirectory + "/settings.txt";
    fs::path R_path = folder_path / ("R.txt");
    if (fs::exists(R_path)) {
        try {
            fs::remove(R_path);
        } catch (const fs::filesystem_error& e) {
            std::cerr << "Error deleting file: " << e.what() << '\n';
        }
    }
    fs::path VLossPath = folder_path / ("VLoss.txt");
    std::cout << "Parsed calibrationPath: " << calibrationPath << "\n";

    // Read resolution from file
    std::vector<float> settings;
    read_single_line_txt(settingsPath, settings);
    std::cout << "Parsed Settings: " << settingsPath << "\n";

    // Set sizes according to read settings
    int height = int(settings[0]); // in pixels
    int rows = int(settings[0]); // in pixels
    int width = int(settings[1]); // in pixels
    int cols = int(settings[1]); // in pixels

    std::random_device myRandomDevice;
    unsigned seed = myRandomDevice();
    std::default_random_engine rng(seed);

    //##################################################################################################################
    // Optic flow F, temporal derivative V, spatial derivative G, intensity I, rotation vector R
    xt::xtensor<float, 2> V = xt::zeros<float>({height, width});

    // Initialize optical flow
    auto engine = xt::random::get_default_random_engine();
    xt::xtensor<float, 3> F = xt::random::randn<float>({height, width, 2}, 0, 1, engine);

    // Initialize spatial gradient G
    xt::xtensor<float, 3> G = xt::random::randn<float>({height, width, 2}, 0, 1, engine);

    // Initialize intensity image I
    xt::xtensor<float, 2> I = xt::ones<float>({height, width})*128.0;
    xt::xtensor<float, 3> delta_I = xt::zeros_like(F);
    Tensor2f decayTimeSurface(height, width);
    decayTimeSurface.setConstant(parameters["startTime"]);

    // For the "I from G" update rule we need helper values.
    xt::xtensor<float, 3> GIDiff = xt::random::randn<float>({height, width, 2}, 0, 1, engine);
    xt::xtensor<float, 3> GIDiffGradient = xt::random::randn<float>({height, width, 2}, 0, 1, engine);

    // Initialize rotational velocity to a random vector with values between -10 and 10
    xt::xtensor<float, 1> R = xt::random::randn<float>({3}, 0, 1, engine);

    //##################################################################################################################
    // Memory Image for I to remember previous image
    xt::xtensor<float, 2> MI = xt::ones<float>({rows, cols})*parameters["neutralPotential"];
    xt::xtensor<float, 2> decayBase = xt::ones<float>({rows, cols})*parameters["neutralPotential"];
    xt::xtensor<float, 2> expDecay = xt::ones<float>({rows, cols});

    // Tensors for Image decay
    xt::xtensor<float, 2> np = xt::ones<float>({rows, cols})*parameters["neutralPotential"];
    xt::xtensor<float, 2> t = xt::zeros<float>({rows, cols});
    xt::xtensor<float, 2> dP = xt::ones<float>({rows, cols})*parameters["decayParameter"];

    // Read calibration file
    std::vector<float> raw_calibration_data;
    read_single_line_txt(calibrationPath, raw_calibration_data);
    Calibration_Data calibration_data = get_calibration_data(raw_calibration_data, height, width);
    std::cout << "Readout calibration file at " << calibrationPath << std::endl;
    // Camera calibration matrix (C/CCM) and dCdx/dCdy
    xt::xtensor<float, 3> CCM = xt::zeros<float>({height, width, 3});
    xt::xtensor<float, 3> dCdx = xt::zeros<float>({height, width, 3});
    xt::xtensor<float, 3> dCdy = xt::zeros<float>({height, width, 3});
    find_C(static_cast<size_t>(width), static_cast<size_t>(height), calibration_data.view_angles[1], calibration_data.view_angles[0], 1.0f, CCM, dCdx, dCdy);
    std::cout << "Calculated Camera Matrix" << std::endl;

    // A matrix and outerProducts for update_R
    xt::xtensor<float, 2> A = xt::zeros<float>({3, 3});
    xt::xtensor<float, 1> B = xt::zeros<float>({3});
    // Create a 2D vector with uninitialized elements
    xt::xtensor<float, 4> Identity_minus_outerProducts = xt::zeros<float>({rows, cols, 3, 3});
    xt::xtensor<float, 3> old_points = xt::zeros<float>({rows, cols, 3});
    setup_R_update(CCM, A, B, Identity_minus_outerProducts, old_points);

    // SETUP EVENT CAMERA
    using namespace std::chrono_literals;
    static constexpr int ESC_KEYCODE = 27;
    // Install signal handlers for a clean shutdown
    std::signal(SIGINT, handleShutdown);
    std::signal(SIGTERM, handleShutdown);
    // Open the specified camera
    auto capture = dv::io::camera::DAVIS("00001088");
    std::cout << "Camera [" << capture.getCameraName() << "] has been opened!" << std::endl;
    std::cout << "Resolution [" << capture.getEventResolution()->width << "x" << capture.getEventResolution()->height << "]." << std::endl;
    if (capture.isImuStreamAvailable()) {
        // Print the imu data stream capability
        std::cout << "* IMU measurements" << std::endl;
    }
    cv::Size ROI_size = cv::Size(width, height);
    // Setting camera readout to events and frames (default).
    capture.setEventsRunning(true);
    capture.setFramesRunning(true);
    // Configure frame output mode to color (default), only on COLOR cameras. Other mode available: GRAYSCALE
    capture.setColorMode(dv::io::camera::parser::DAVIS::ColorMode::DEFAULT);
    // Enable frame auto-exposure (default behavior)
    capture.setAutoExposure(true);
    // Disable auto-exposure, set frame exposure (here 10ms)
    capture.setAutoExposure(false);
    capture.setExposureDuration(50ms);
    // Read current frame exposure duration value
    std::chrono::microseconds duration = capture.getExposureDuration();
    // Set frame interval duration (here 33ms for ~30FPS)
    capture.setFrameInterval(50ms);
    // Read current frame interval duration value
    std::chrono::microseconds interval = capture.getFrameInterval();
    // Initialize an accumulator with some resolution
    //dv::visualization::EventVisualizer visualizer(*capture.getEventResolution());
    //dv::Accumulator accumulator(*capture.getEventResolution());
    dv::visualization::EventVisualizer visualizer(ROI_size);
    dv::Accumulator accumulator(ROI_size);
    // Apply event color scheme configuration, these values can be modified to taste
    visualizer.setBackgroundColor(dv::visualization::colors::black);
    visualizer.setPositiveColor(dv::visualization::colors::green);
    visualizer.setNegativeColor(dv::visualization::colors::red);
    // Apply accumulator configuration, these values can be modified to taste
    accumulator.setMinPotential(0.f);
    accumulator.setMaxPotential(1.f);
    accumulator.setNeutralPotential(0.5f);
    accumulator.setEventContribution(parameters["eventContribution"]/255.0f);
    accumulator.setDecayFunction(dv::Accumulator::Decay::EXPONENTIAL);
    accumulator.setDecayParam(1e5);
    accumulator.setIgnorePolarity(false);
    accumulator.setSynchronousDecay(false);
    dv::EventRegionFilter regionFilter(cv::Rect(0, 0, width, height));
    dv::noise::KNoiseFilter kNoiseFilter(ROI_size);
    // Initialize a slicer
    dv::EventStreamSlicer slicer;
    // Register a callback every 33 milliseconds

    std::cout << "Setup complete; Streaming start!" << std::endl;

    // Initialize a preview window
    cv::namedWindow("Events", cv::WINDOW_NORMAL);
    cv::namedWindow("Images", cv::WINDOW_NORMAL);
    cv::namedWindow("RealFrame", cv::WINDOW_NORMAL);
    cv::namedWindow("VIGF", cv::WINDOW_NORMAL);

    const shape_type3 &shape = CCM.shape();
    xt::xtensor<float, 3> C1(shape);
    xt::xtensor<float, 3> C2(shape);
    xt::xtensor<float, 2> dot({shape[0], shape[1]});
    xt::xtensor<float, 2> distance({shape[0], shape[1]});

    slicer.doEveryTimeInterval(50ms, [&visualizer, &accumulator, &kNoiseFilter, &regionFilter, &V, &I, &delta_I, &GIDiff, &GIDiffGradient, &F, &G, &R, &CCM, &dCdx, &dCdy, &A, &B,
                               &Identity_minus_outerProducts, &old_points, &parameters, &permutation, &rng, &C1, &C2, &dot, &distance](const dv::EventStore &events) {
        regionFilter.accept(events);
        dv::EventStore regionFilteredEvents = regionFilter.generateEvents();
        kNoiseFilter.accept(regionFilteredEvents);
        dv::EventStore kNoiseFiltered = kNoiseFilter.generateEvents();
        //cv::Mat kNoiseFilterPreview   = visualizer.generateImage(kNoiseFiltered);
        // cv::putText(kNoiseFilterPreview, "K-Noise filter", textPosition, cv::FONT_HERSHEY_SIMPLEX, fontScale, fontColor,
        //     fontThickness);
        // cv::putText(kNoiseFilterPreview, fmt::format("Reduction factor: {:.2f}", kNoiseFilter.getReductionFactor()),
        //     textPosition + textShift, cv::FONT_HERSHEY_SIMPLEX, fontScale, fontColor, fontThickness);

        // Pass events into the accumulator and generate a preview frame
        accumulator.accept(kNoiseFiltered);
        dv::Frame frame = accumulator.generateFrame();
        cv::Mat eventImage = visualizer.generateImage(kNoiseFiltered);
        //std::cout << "Accumulated" << std::endl;

        V = cvMatToV(eventImage, 0, parameters["eventContribution"]);
        cv::Mat imageV = V2image(V, 1.0);

        I = cvMatToI(frame.image);
        //I = xt::transpose(I);

        //std::cout << "Transformed" << std::endl;

        for (int convergenceStep = 0; convergenceStep<static_cast<int>(parameters["convergenceSteps"]); ++convergenceStep) {
            std::shuffle(std::begin(permutation), std::end(permutation), rng);
            event_step(V, I, delta_I, GIDiff, GIDiffGradient, F, G, R, CCM, dCdx, dCdy, A, B,
                               Identity_minus_outerProducts, old_points, C1, C2, dot, distance, parameters, permutation, rng);
        }
        //std::cout << "Converged" << std::endl;
        cv::Mat VIGF = create_VIGF(V, I, G, F, "VIGF.png", false, 0.1, false);
        //std::cout << "VIGFed" << std::endl;
        // Show the event image and the accumulated image
        cv::imshow("VIGF", VIGF);
        cv::waitKey(1);

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
        // if (const auto frame = capture.getNextFrame(); frame.has_value()) {
        //     std::cout << *frame << std::endl;
        //     // Show a preview of the image
        //     cv::imshow("RealFrame", frame->image);
        // }
        if (const std::optional<std::vector<dv::IMU>> imuBatch = capture.getNextImuBatch(); imuBatch.has_value() && !imuBatch->empty()) {
            //std::cout << "Received " << imuBatch->size() << " IMU measurements" << std::endl;
            dv::IMU imu_ = imuBatch->at(0);
            auto R_ = imu_.getAngularVelocities();
            R[1] = R_[0];
            R[0] = R_[1];
            R[2] = R_[2];

        }
        else {
            // No data has arrived yet, short sleep to reduce CPU load.
            std::this_thread::sleep_for(1ms);
        }
    }
    Instrumentor::Get().EndSession();
    return 0;
}