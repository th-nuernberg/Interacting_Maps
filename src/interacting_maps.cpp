#include <interacting_maps.h>
#include <iostream>
#include <filesystem>
#include <string>
#include <sstream>
#include <cassert>
#include <memory>
#include "Instrumentor.h"
#include <cmath>
#include <boost/program_options.hpp>

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
                // update_FR(F, CCM, dCdx, dCdy, R, parameters["weight_FR"], parameters["eps"], parameters["gamma"]);
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

void randomInit(Tensor3f &T, const float lower, const float upper) {
    const auto &dimensions = T.dimensions();
    Tensor3f T1(dimensions[0], dimensions[1], dimensions[2]);
    Tensor3f T2(dimensions[0], dimensions[1], dimensions[2]);
    T.setRandom();
    T1.setConstant(lower);
    T2.setConstant(upper - lower);
    T = T*T2 + T1;
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
            ("startTime,f", po::value<float>()->default_value(0), "Where to start with event consideration")
            ("endTime,f", po::value<float>()->default_value(10), "Where to end with event consideration")
            ("timeStep,f", po::value<float>()->default_value(0.0460299576597383), "Size of the event frames")
            ("resourceDirectory,s", po::value<std::string>()->default_value("boxes_rotation"), "Which dataset to use, searches in res directory")
            ("resultsDirectory,s", po::value<std::string>()->default_value("boxes_rotation"), "Where to store the results, located in output directory")
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
    // Split time interval into sub intervals to allow loading of larger files.
    int nIntervals = 1;
    float maxIntervalLength = 2.5;
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

    float timeStep = vm["timeStep"].as<float>();
    bool addTime = vm["addTime"].as<bool>();
    int startIndex = vm["startIndex"].as<int>();
    std::string resourceDirectory = vm["resourceDirectory"].as<std::string>();
    std::string resultsDirectory = vm["resultsDirectory"].as<std::string>();

    std::cout << "Parsed startTime: " << startTime << "\n";
    std::cout << "Parsed endTime: " << endTime << "\n";
    std::cout << "Parsed timeStep: " << timeStep << "\n";
    std::cout << "Parsed resourceDirectory: " << resourceDirectory << "\n";
    std::cout << "Parsed resultsDirectory: " << resultsDirectory << "\n";

    auto start = std::chrono::high_resolution_clock::now();

    //##################################################################################################################
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

    std::unordered_map<std::string,float> parameters;
    parameters["startTime"] = startTime;                                   // in seconds
    parameters["endTime"] = endTime;                                       // in seconds
    parameters["time_step"] = timeStep;                                     // in seconds
    parameters["weight_FG"] = 0.2;                                          // [0-1]
    parameters["weight_FR"] = 0.8;                                          // [0-1]
    parameters["weight_GF"] = 0.2;                                          // [0-1]
    parameters["weight_GI"] = 0.2;                                          // [0-1]
    parameters["weight_IG"] = 0.2;                                          // [0-1]
    parameters["weight_IV"] = 1.0;                                          // [0-1]
    parameters["weight_RF"] = 0.2;                                          // [0-1]
    parameters["weight_RIMU"] = 0.0;                                     // [0-1]
    parameters["weight_Ifusion"] = 0.0;                                     // [0-1]
    parameters["lr"] = 1.0;                                                 // [0-1]
    parameters["eventContribution"] = 10.0f;                                   // mainly important for the visibility of the intensity image
    parameters["eps"] = 0.00001;                                            // lowest value allowed for F, G,...
    parameters["gamma"] = 255;                                              // highest value allowed for F, G,...
    parameters["decayParam"] = 1e-1;                                        // 1e-1 for exponential decay
    parameters["minPotential"] = 0.0;                                       // minimum Value for Image
    parameters["maxPotential"] = 255.0;                                       // maximal Value for Image
    parameters["neutralPotential"] = 128;                                   // base value where image decays back to
    parameters["fps"] = 1.0f/parameters["time_step"];                       // how often shown images are update
    parameters["FR_updates_per_second"] = 1.0f/parameters["time_step"];     // how often the FR update is performed; It is not done after every event
    parameters["updateIterationsFR"] = 2;                                  // more iterations -> F captures general movement of scene/camera better but significantly more computation time

    // Read resolution from file
    std::vector<float> settings;
    read_single_line_txt(settingsPath, settings);
    std::cout << "Parsed Settings: " << settingsPath << "\n";

    // Set sizes according to read settings
    int height = int(settings[0]); // in pixels
    int rows = int(settings[0]); // in pixels
    int width = int(settings[1]); // in pixels
    int cols = int(settings[1]); // in pixels

    // iterations are done after event calculations for a frame are done
    std::vector permutation {0,2,3}; // Which update steps to take; 1 is not needed
    std::random_device myRandomDevice;
    unsigned seed = myRandomDevice();
    std::default_random_engine rng(seed);

    //##################################################################################################################
    // Optic flow F, temporal derivative V, spatial derivative G, intensity I, rotation vector R
    Tensor2f V_Vis(height, width);
    V_Vis.setZero();
    float V;
    cv::Mat VIGF;



    // Initialize optical flow
    Tensor3f F(height, width, 2);
    randomInit(F, -1, 1);

    // Initialize spatial gradient G
    Tensor3f G(height, width, 2);
    G.setZero();
    Tensor3f delta_I(height, width,2);
    delta_I.setZero();

    // Initialize intensity image I
    Tensor2f I(height, width);
    I.setConstant(128.0);

    // For the image we want to decay the image intensity. We save for each pixel how old the
    // information is.
    Tensor2f decayTimeSurface(height, width);
    decayTimeSurface.setConstant(parameters["startTime"]);

    // For the "I from G" update rule we need helper values.
    Tensor3f GIDiff(height, width,2);
    randomInit(GIDiff, -1, 1);
    Tensor3f GIDiffGradient(height, width,2);
    randomInit(GIDiffGradient, -1, 1);

    // Initialize rotational velocity to a random vector with values between -1 and 1
    Tensor1f R(3);
    randomInit(R, -10, 10);

    //##################################################################################################################
    // Read calibration file
    std::vector<float> raw_calibration_data;
    read_single_line_txt(calibrationPath, raw_calibration_data);
    Calibration_Data calibration_data = get_calibration_data(raw_calibration_data, height, width);
    std::cout << "Readout calibration file at " << calibrationPath << std::endl;

    //##################################################################################################################
    // Bin events
    //std::vector<std::vector<Event>> binned_events;
    //binned_events = bin_events(event_data, parameters["time_step"]);
    //std::cout << "Binned events" << std::endl;

    //##################################################################################################################
    // Create frames
    //size_t frame_count = binned_events.size();
    //std::vector<Tensor2f> frames(frame_count);
    //create_frames(binned_events, frames, height, width, parameters["eventContribution"]);
    //std::cout << "Created frames " << frame_count << " out of " << event_data.size() << " events" << std::endl;

    //##################################################################################################################
    // Camera calibration matrix (C/CCM) and dCdx/dCdy
    Tensor3f CCM(height, width,3);
    CCM.setZero();
    Tensor3f dCdx(height, width,3);
    dCdx.setZero();
    Tensor3f dCdy(height, width,3);
    dCdy.setZero();
    find_C(width, height, calibration_data.view_angles[1], calibration_data.view_angles[0], 1.0f, CCM, dCdx, dCdy);
    std::cout << "Calculated Camera Matrix" << std::endl;

    //##################################################################################################################
    // A matrix and outerProducts for update_R
    Matrix3f A = Matrix3f::Zero();
    Vector3f B = Vector3f::Zero();
    // Create a 2D vector with uninitialized elements
    std::vector<std::vector<Matrix3f>> Identity_minus_outerProducts;
    std::vector<std::vector<Vector3f>> old_points;
    Identity_minus_outerProducts.resize(rows);  // Resize to have the number of rows
    old_points.resize(rows);  // Resize to have the number of rows

    for (int i = 0; i < rows; ++i) {
        Identity_minus_outerProducts[i].resize(cols);  // Resize each row but do not initialize
        old_points[i].resize(cols);  // Resize each row but do not initialize
    }
    setup_R_update(CCM, A, B, Identity_minus_outerProducts, old_points);

    //##################################################################################################################
    // Memory Image for I to remember previous image
    Tensor2f MI(height, width);
    MI.setConstant(parameters["neutralPotential"]);

    Tensor2f decayBase(height, width);
    decayBase.setConstant(parameters["neutralPotential"]);

    Tensor2f expDecay(height, width);
    expDecay.setConstant(1.0);

    // For keeping track of the current Event
    int y;
    int x;
    std::vector<float> ang_velocity = {0,0,0};
    std::vector<float> acceleration = {0,0,0};
    bool cEventFlag = true;
    //std::vector<Event> frameEvents;

    // Tensors for Image decay
    Tensor2f nP(I.dimensions());    // neutralPotential
    Tensor2f t(I.dimensions());     // time
    Tensor2f dP(I.dimensions());    // decayParameter

    auto start_realtime = std::chrono::high_resolution_clock::now();

    int vis_counter = -1;
    int FR_update_counter = 0;

    nP.setConstant(parameters["neutralPotential"]);
    dP.setConstant(parameters["decayParam"]);

    for (int currentInterval = 0; currentInterval<nIntervals; ++currentInterval) {
        //##################################################################################################################
        // Read events file

        std::vector<std::shared_ptr<Event>> cameraEventData;
        read_events(eventPath, cameraEventData, intervals[currentInterval], intervals[currentInterval+1], INT32_MAX);
        std::cout << "Readout events at " << eventPath << " for time " << intervals[currentInterval] << " to " << intervals[currentInterval + 1] << std::endl;

        std::vector<std::shared_ptr<Event>> event_data;

        if (vm["fuseR"].as<bool>()) {
            std::vector<std::shared_ptr<Event>> imuEventData;
            read_imu(imuPath, imuEventData, intervals[currentInterval], intervals[currentInterval+1], INT32_MAX);
            std::cout << "Readout IMU data at " << imuPath << " for time " << intervals[currentInterval] << " to " << intervals[currentInterval + 1] << std::endl;
            mergeTimeCollections(cameraEventData, imuEventData, event_data);
        }
        else if (vm["fuseI"].as<bool>()) {
            std::vector<std::shared_ptr<Event>> imageEventData;
            readImage(imagesPath, imageEventData, intervals[currentInterval], intervals[currentInterval+1], INT32_MAX);
            std::cout << "Readout Image data at " << imuPath << " for time " << intervals[currentInterval] << " to " << intervals[currentInterval + 1] << std::endl;
            mergeTimeCollections(cameraEventData, imageEventData, event_data);
        }
        else {
            event_data = cameraEventData;
        }

        for (const auto& event : event_data) {
            // Shuffle the order of operations for the interacting maps operations
            std::shuffle(std::begin(permutation), std::end(permutation), rng);

            if (auto* cEvent = dynamic_cast<CameraEvent*>(event.get())) {
                PROFILE_SCOPE("CAMERA_EVENT");
                y = cEvent->coordinates[0];
                x = cEvent->coordinates[1];
                V = static_cast<float>(cEvent->polarity) * parameters["eventContribution"];
                //decayTimeSurface(y,x) = event->time;

                // For Showing the events as an image increase the intensity
                V_Vis(y, x) = V;

                // Perform an update step for the current event for I G R and F
                exponentialDecay(MI, decayTimeSurface, y, x, event->time, parameters["neutralPotential"], parameters["decayParam"]);
                for (int i = 0; i < 1; ++i) {
                    event_step(V, MI, delta_I, GIDiff, GIDiffGradient, F, G, R, CCM, dCdx, dCdy, A, B,
                               Identity_minus_outerProducts, old_points, parameters, permutation, y, x);
                }

                if (parameters["startTime"] + static_cast<float>(FR_update_counter) * static_cast<float>(1 / parameters["FR_updates_per_second"]) < event->time) {
                    t.setConstant(event->time);
                    for (int i = 0; i < static_cast<int>(parameters["updateIterationsFR"]); ++i) {
                        update_FR(F, CCM, dCdx, dCdy, R, parameters["weight_FR"], parameters["eps"], parameters["gamma"]);
                    }

                }

            } else if (auto* imuEvent = dynamic_cast<IMUEvent*>(event.get()))  {
                update_RIMU(R, imuEvent->ang_velocities, parameters["weight_RIMU"]);
            }

            else if (auto* imageEvent = dynamic_cast<ImageEvent*>(event.get()))  {
                std::cout << "Fused image at time " << imageEvent->time << std::endl;
                update_Ifusion(MI, imageEvent->image, parameters["weight_Ifusion"]);
                decayTimeSurface.setConstant(imageEvent->time);
            }

            // Starting from the start time we count up. If the current time (event->time)
            // reaches the time of the next "frame" we want to save to disk
            if (parameters["startTime"] + static_cast<float>(vis_counter) * (1 / parameters["fps"]) < event->time) {
                vis_counter++;
                std::cout << "Frame " << startIndex+vis_counter << "/"
                          << static_cast<int>((parameters["endTime"] - parameters["startTime"]) * parameters["fps"]) << std::endl;
                {
                    PROFILE_SCOPE("BETWEEN FRAMES");
                    //writeToFile(CCM, folder_path / ("C" + std::to_string(counter) + ".txt"));
                    //writeToFile(V_Vis, folder_path / ("V" + std::to_string(counter) + ".txt"));
                    //writeToFile(MI, folder_path / ("MI" + std::to_string(counter)  + ".txt"));
                    //writeToFile(I, folder_path / ("I" + std::to_string(counter)  + ".txt"));
                    //writeToFile(delta_I, folder_path / ("I_gradient" + std::to_string(counter)  + ".txt"));
                    //writeToFile(F, folder_path / ("F" + std::to_string(counter)  + ".txt"));
                    //writeToFile(G, folder_path / ("G" + std::to_string(counter)  + ".txt"));
                    writeToFile(event->time, R, R_path, true);
                }

    #ifdef IMAGES
                    float loss = VFG_check(V_Vis, F, G);
                    //std::cout << "VFG Check: " << loss << std::endl;
                    writeToFile(event->time, loss, VLossPath);

                    std::stringstream filename;
                    filename.fill('0');
                    filename.width(8);
                    filename<<std::to_string(static_cast<int>((startIndex + vis_counter)));

                    std::string image_name = "VIGF_" + filename.str() + ".png";

                    fs::path image_path = folder_path / image_name;
                    //create_VIGF(Tensor2Matrix(V_Vis), Tensor2Matrix(MI), G, F, image_path, true, cutoff);
                    //image_name = "VvsFG" + std::to_string(int(counter)) + ".png";
                    //image_path = folder_path / image_name;
                    //plot_VvsFG(Tensor2Matrix(V_Vis), F, G, image_path, true);

                    create_VIGF(Tensor2Matrix(V_Vis), Tensor2Matrix(MI), G, F, image_path, true, cutoff);
                    saveImage(Tensor2Matrix(MI), folder_path / ("frame_" + filename.str() + ".png"), true);
                    // cv::imshow("VIGF", VIGF);
                    // // Press 'q' to exit
                    // if (cv::waitKey(1) == 'q') {
                    //     break;
                    // }
                    V_Vis.setZero();
                    F.setRandom();
                    F = F * F2 - F1;
                    F = F * F3;
                    G.setZero();

    #endif
                //globalDecay(MI, decayTimeSurface, nP, t, dP);
            }

            if (parameters["startTime"] + static_cast<float>(FR_update_counter) * (1 / parameters["FR_updates_per_second"]) <event->time) {
                FR_update_counter++;
            }
        }
    }


    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> elapsed = end - start;
    std::chrono::duration<float> elapsed_realtime = end - start_realtime;
    std::stringstream ss;
    std::stringstream ssrt;
    ss << "Time elapsed: " << elapsed.count() << " seconds" << std::endl;
    ssrt << "Time elapsed: " << elapsed_realtime.count() << " seconds" << std::endl;
    writeToFile(ss.str(), folder_path / "time_complete.txt");
    writeToFile(ssrt.str(), folder_path / "time_realtime.txt");
    std::cout << "Algorithm took: " << elapsed_realtime.count() << "seconds/ Real elapsed time: " << parameters["endTime"] - parameters["startTime"] << std::endl;


    std::string outputFile = "output.mp4";

#ifdef IMAGES
    VideoCreator::createMP4Video(folder_path, folder_path / outputFile, static_cast<int>((parameters["fps"])));
#endif

    Instrumentor::Get().EndSession();
}