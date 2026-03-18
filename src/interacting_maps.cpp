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

struct Resolution
{
    int height;
    int width;
};

Resolution readResolution(const fs::path& settingsPath)
{
    std::vector<float> settings;
    read_single_line_txt(settingsPath, settings);
    std::cout << "Parsed Settings: " << settingsPath << "\n";
    return { int(settings[0]), int(settings[1]) };
}

Calibration_Data readCalibration(const fs::path& calibrationPath, int height, int width)
{
    std::vector<float> raw_calibration_data;
    read_single_line_txt(calibrationPath, raw_calibration_data);
    std::cout << "Readout calibration file at " << calibrationPath << "\n";
    return get_calibration_data(raw_calibration_data, height, width);
}

struct ProgramArgs
{
    float startTime;
    float endTime;
    float timeStep;
    float maxIntervalLength;
    int startIndex;
    std::string timeFormat;
    std::string resourceDirectory;
    std::string resultsDirectory;
    bool fuseR;
    bool fuseI;
};

std::optional<ProgramArgs> parseArgs(int argc, char* argv[], const po::options_description& desc)
{
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return std::nullopt;
    }

    return ProgramArgs {
        .startTime         = vm["startTime"].as<float>(),
        .endTime           = vm["endTime"].as<float>(),
        .timeStep          = vm["timeStep"].as<float>(),
        .maxIntervalLength = vm["maxIntervalLength"].as<float>(),
        .startIndex        = vm["startIndex"].as<int>(),
        .timeFormat        = vm["timeFormat"].as<std::string>(),
        .resourceDirectory = vm["resourceDirectory"].as<std::string>(),
        .resultsDirectory  = vm["resultsDirectory"].as<std::string>(),
        .fuseR             = vm["fuseR"].as<bool>(),
        .fuseI             = vm["fuseI"].as<bool>()
    };
}
struct Parameters
{
    // Timing
    float startTime;                    // in seconds
    float endTime;                      // in seconds
    float time_step;                    // in seconds
    float fps;
    float FR_updates_per_second;
    int   updateIterationsFR = 4;
    int   event_steps = 1;

    // Weights [0-1]
    float weight_FG    = 0.2f;
    float weight_FR    = 0.8f;
    float weight_GF    = 0.2f;
    float weight_GI    = 0.2f;
    float weight_IG    = 0.2f;
    float weight_IV    = 1.0f;
    float weight_RF    = 0.8f;
    float weight_RIMU = 0.0f;
    float weight_Ifusion = 0.0f;
    float lr           = 1.0f;
    int   FR_update_counter = 100;

    // Image
    float eventContribution = 10.0f;
    float minPotential      = 0.0f;
    float maxPotential      = 255.0f;
    float neutralPotential  = 128.0f;
    int   vis_counter = -1;

    // Bounds
    float eps   = 0.00001f;
    float gamma = 255.0f;

    // Decay
    float decayParam = 1e-1f;
};

Parameters initParameters(float startTime, float endTime, float timeStep)
{
    Parameters p;

    p.startTime             = startTime;
    p.endTime               = endTime;
    p.time_step             = timeStep;
    p.fps                   = 1.0f / timeStep;
    p.FR_updates_per_second = 1.0f / timeStep;

    return p;
}

struct OpticalFlowState
{
    Tensor2f V_Vis;
    float V = 0.0f;
    cv::Mat VIGF;

    Tensor3f F;               // Optical flow
    Tensor3f G;               // Spatial gradient
    Tensor3f delta_I;

    Tensor2f I;               // Intensity image
    Tensor2f decayTimeSurface;

    Tensor3f GIDiff;          // Helper values for "I from G" update rule
    Tensor3f GIDiffGradient;

    Tensor1f R;               // Rotational velocity

    // Camera calibration
    Tensor3f CCM;
    Tensor3f dCdx;
    Tensor3f dCdy;

    // R update helpers
    Matrix3f A;
    Vector3f B;
    std::vector<std::vector<Matrix3f>> Identity_minus_outerProducts;
    std::vector<std::vector<Vector3f>> old_points;

    // Memory image
    Tensor2f MI;
    Tensor2f decayBase;
    Tensor2f expDecay;
    Tensor2f neutralPotential;
    Tensor2f time;
    Tensor2f decayParam;

    // Tensors for Image decay

};

OpticalFlowState initOpticalFlowState(int height, int width, const Parameters& parameters,
                                      const Calibration_Data& calibration_data)
{
    OpticalFlowState s;

    s.V_Vis = Tensor2f(height, width);
    s.V_Vis.setZero();

    s.F = Tensor3f(height, width, 2);
    randomInit(s.F, -1, 1);

    s.G = Tensor3f(height, width, 2);
    s.G.setZero();

    s.delta_I = Tensor3f(height, width, 2);
    s.delta_I.setZero();

    s.I = Tensor2f(height, width);
    s.I.setConstant(parameters.neutralPotential);

    s.decayTimeSurface = Tensor2f(height, width);
    s.decayTimeSurface.setConstant(parameters.startTime);

    s.GIDiff = Tensor3f(height, width, 2);
    randomInit(s.GIDiff, -1, 1);

    s.GIDiffGradient = Tensor3f(height, width, 2);
    randomInit(s.GIDiffGradient, -1, 1);

    s.R = Tensor1f(3);
    randomInit(s.R, -10, 10);

    // Camera calibration
    s.CCM  = Tensor3f(height, width, 3);  s.CCM.setZero();
    s.dCdx = Tensor3f(height, width, 3);  s.dCdx.setZero();
    s.dCdy = Tensor3f(height, width, 3);  s.dCdy.setZero();
    find_C(width, height, calibration_data.view_angles[1], calibration_data.view_angles[0],
           1.0f, s.CCM, s.dCdx, s.dCdy);

    // R update helpers
    s.A = Matrix3f::Zero();
    s.B = Vector3f::Zero();
    s.Identity_minus_outerProducts.resize(height, std::vector<Matrix3f>(width));
    s.old_points.resize(height, std::vector<Vector3f>(width));
    setup_R_update(s.CCM, s.A, s.B, s.Identity_minus_outerProducts, s.old_points);

    // Memory image
    s.MI = Tensor2f(height, width);
    s.MI.setConstant(parameters.neutralPotential);

    s.decayBase = Tensor2f(height, width);
    s.decayBase.setConstant(parameters.neutralPotential);

    s.expDecay = Tensor2f(height, width);
    s.expDecay.setConstant(1.0f);

    s.neutralPotential(s.I.dimensions());    // neutralPotential
    s.time(s.I.dimensions());     // time
    s.decayParam(s.I.dimensions());    // decayParameter
    s.neutralPotential.setConstant(parameters.neutralPotential);
    s.decayParam.setConstant(parameters.decayParam);

    return s;
}

void event_step(const float V, Tensor2f &MI, Tensor3f &delta_I, Tensor3f &GIDiff, Tensor3f &GIDiffGradient, Tensor3f &F, Tensor3f &G, Tensor1f &R, const Tensor3f &CCM, const Tensor3f &dCdx, const Tensor3f &dCdy, const Matrix3f &A, Vector3f &B, const std::vector<std::vector<Matrix3f>> &Identity_minus_outerProducts, std::vector<std::vector<Vector3f>> &old_points, Parameters &parameters, std::vector<int> &permutation, int y, int x){
    PROFILE_FUNCTION();
    array<Index, 2> dimensions = MI.dimensions();
    update_IV(MI, V, y, x, parameters.minPotential, parameters.maxPotential, parameters.weight_IV);
    // Image (MI) got changed through update by V. we need to update all surrounding gradient values. Because of the change at this pixel
    {
        PROFILE_SCOPE("GRADIENTS");
        if (y>0){
            computeGradient(MI, delta_I, y-1, x);
            update_GI(G, delta_I, y-1, x, parameters.weight_GI, parameters.eps, parameters.gamma);
        }
        if (x>0){
            computeGradient(MI, delta_I, y, x-1);
            update_GI(G, delta_I, y, x-1, parameters.weight_GI, parameters.eps, parameters.gamma);
        }
        if (y<dimensions[0]-1){
            computeGradient(MI, delta_I, y+1, x);
            update_GI(G, delta_I, y+1, x, parameters.weight_GI, parameters.eps,parameters.gamma);
        }
        if (x<dimensions[1]-1){
            computeGradient(MI, delta_I, y, x+1);
            update_GI(G, delta_I, y, x+1, parameters.weight_GI, parameters.eps, parameters.gamma);
        }
    }

    //computeGradient(MI, delta_I, y, x);
    update_GI(G, delta_I, y, x, parameters.weight_GI, parameters.eps, parameters.gamma);
    updateGIDiffGradient(G, delta_I, GIDiff, GIDiffGradient, y, x);
    update_IG(MI, GIDiffGradient, y, x, parameters.weight_IG);
    //computeGradient(MI, delta_I, y, x);

    for (const auto& element : permutation){
        switch( element ){
            default:
                std::cout << "Unknown number in permutation" << std::endl;
            case 0:
                update_FG(F, V, G, y, x, parameters.lr, parameters.weight_FG, parameters.eps, parameters.gamma);
                break;
            case 1:
                // Gets called separately because we do not want to do an update of F based on R with every event since this update is global
                update_FR(F, CCM, dCdx, dCdy, R, parameters.weight_FR, parameters.eps, parameters.gamma);
                break;
            case 2:
                update_GF(G, V, F, y, x, parameters.lr, parameters.weight_GF, parameters.eps, parameters.gamma);
                break;
            case 3:
                update_RF(R, F, CCM, dCdx, dCdy, A, B, Identity_minus_outerProducts, old_points, parameters.weight_RF, y, x);
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

std::vector<float> splitTimeInterval(float startTime, float endTime, float maxIntervalLength = 0.01f)
{
    std::vector<float> intervals = {startTime, endTime};

    if (endTime - startTime > maxIntervalLength) {
        intervals = {startTime};
        float currentTime = startTime;

        while (currentTime + maxIntervalLength < endTime) {
            currentTime += maxIntervalLength;
            intervals.push_back(currentTime);
        }
        intervals.push_back(endTime);
    }

    return intervals;
}

fs::path setupFilePath(const fs::path& folder_path, const std::string& filename)
{
    fs::path file_path = folder_path / filename;

    if (fs::exists(file_path)) {
        try {
            fs::remove(file_path);
        } catch (const fs::filesystem_error& e) {
            std::cerr << "Error deleting file: " << e.what() << '\n';
        }
    }

    return file_path;
}

int main(int argc, char* argv[]) {

    // Define the command-line options
    po::options_description desc("Allowed options");
    desc.add_options()
            ("help,h", "Produce help message")
            ("startTime,f", po::value<float>()->default_value(0), "Where to start with event consideration")
            ("endTime,f", po::value<float>()->default_value(5), "Where to end with event consideration")
            ("timeFormat,s", po::value<std::string>()->default_value("uS"), "What format are the times: seconds, milliseconds, microseconds (s,ms,mus)")
            ("timeStep,f", po::value<float>()->default_value(0.01), "Size of the event frames")
            ("maxIntervalLength,f", po::value<float>()->default_value(1.0), "Load events from time slice of this size at a time")
            ("resourceDirectory,s", po::value<std::string>()->default_value("shapes_rotation"), "Which dataset to use, searches in res directory")
            ("resultsDirectory,s", po::value<std::string>()->default_value("shapes_rotation"), "Where to store the results, located in output directory")
            ("startIndex,i", po::value<int>()->default_value(0), "With what index to start for the images")
            ("fuseR,b", po::value<bool>()->default_value(false), "Fuse with imu.txt?")
            ("fuseI,b", po::value<bool>()->default_value(false), "Fuse with images folder?");

    auto args = parseArgs(argc, argv, desc);
    if (!args) return 0;

    // Split time interval into sub intervals to allow loading of larger files.
    std::vector intervals = {args->startTime, args->endTime};
    intervals = splitTimeInterval(args->startTime, args->endTime, args->maxIntervalLength);

    // Inform user
    std::cout << "Parsed startTime: " << args->startTime << "\n";
    std::cout << "Parsed endTime: " << args->endTime << "\n";
    std::cout << "Parsed timeStep: " << args->timeStep << "\n";
    std::cout << "Parsed resourceDirectory: " << args->resourceDirectory << "\n";
    std::cout << "Parsed resultsDirectory: " << args->resultsDirectory << "\n";

    // Create results_folder
    fs::path folder_path = create_folder_and_update_gitignore(args->resultsDirectory);
    std::cout << "Created Folder " << args->resultsDirectory << std::endl;

    // Setup Profiling
    std::string profiler_name = "Profiler.json";
    fs::path profiler_path = folder_path / profiler_name;
    Instrumentor::Get().BeginSession("Interacting Maps", profiler_path);
    std::cout << "Setup Profiler" << std::endl;

    // Setup needed files
    fs::path basePath        = fs::path("res") / args->resourceDirectory;
    fs::path calibrationPath = basePath / "calib.txt";
    fs::path eventPath       = basePath / "events.txt";
    fs::path imuPath         = basePath / "imu.txt";
    fs::path imagesPath      = basePath / "images.txt";
    fs::path settingsPath    = basePath / "settings.txt";
    fs::path R_path = setupFilePath(folder_path, "R.txt");
    fs::path VLossPath = setupFilePath(folder_path, "VLoss.txt");



    // Read resolution and calibration from file
    Resolution res                    = readResolution(settingsPath);
    Calibration_Data calibration_data = readCalibration(calibrationPath, res.height, res.width);
    Parameters parameters = initParameters(args->startTime, args->endTime, args->timeStep);
    OpticalFlowState state = initOpticalFlowState(res.height, res.width, parameters, calibration_data);

    // iterations are done after event calculations for a frame are done
    std::vector permutation {0,2,3}; // Which update steps to take; 1 is not needed
    std::random_device myRandomDevice;
    unsigned seed = myRandomDevice();
    std::default_random_engine rng(seed);

    // For keeping track of the current Event
    // std::vector<int> currentCoordinates = {0,0};
    // CameraEvent currentCameraEvent = CameraEvent(args->startTime, currentCoordinates, 0);
    // std::vector<float> ang_velocity = {0,0,0};
    // std::vector<float> lin_acceleration = {0,0,0};
    // IMUEvent currentImuEvent = IMUEvent(args->startTime, lin_acceleration, ang_velocity);

    auto start_realtime = std::chrono::high_resolution_clock::now();

    for (int currentInterval = 0; currentInterval<intervals.size(); ++currentInterval) {
        //##################################################################################################################
        // Read events file

        std::vector<std::shared_ptr<Event>> cameraEventData;
        read_events(eventPath, cameraEventData, intervals[currentInterval], intervals[currentInterval+1], INT32_MAX, args->timeFormat);
        std::cout << "Readout events at " << eventPath << " for time " << intervals[currentInterval] << " to " << intervals[currentInterval + 1] << std::endl;
        std::cout << "Read " << cameraEventData.size() << " events." << std::endl;
        std::vector<std::shared_ptr<Event>> event_data;

        if (args->fuseR) {
            std::vector<std::shared_ptr<Event>> imuEventData;
            read_imu(imuPath, imuEventData, intervals[currentInterval], intervals[currentInterval+1], INT32_MAX);
            std::cout << "Readout IMU data at " << imuPath << " for time " << intervals[currentInterval] << " to " << intervals[currentInterval + 1] << std::endl;
            mergeTimeCollections(cameraEventData, imuEventData, event_data);
        }
        else if (args->fuseI) {
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
                state.V = static_cast<float>(cEvent->polarity) * parameters.eventContribution;

                // For Showing the events as an image increase the intensity
                state.V_Vis(cEvent->coordinates[0], cEvent->coordinates[1]) = state.V;

                // Perform an update step for the current event for I G R and F
                exponentialDecay(state.MI, state.decayTimeSurface, cEvent->coordinates[0], cEvent->coordinates[1], event->time, parameters.neutralPotential, parameters.decayParam);
                for (int i = 0; i < parameters.event_steps; ++i) {
                    event_step(state.V, state.MI, state.delta_I, state.GIDiff, state.GIDiffGradient, state.F, state.G, state.R, state.CCM, state.dCdx, state.dCdy, state.A, state.B,
                               state.Identity_minus_outerProducts, state.old_points, parameters, permutation, cEvent->coordinates[0], cEvent->coordinates[1]);
                }

                if (parameters.startTime + static_cast<float>(parameters.FR_update_counter) * static_cast<float>(1 / parameters.FR_updates_per_second) < event->time) {
                    state.time.setConstant(event->time);
                    for (int i = 0; i < static_cast<int>(parameters.updateIterationsFR); ++i) {
                        update_FR(state.F, state.CCM, state.dCdx, state.dCdy, state.R, parameters.weight_FR, parameters.eps, parameters.gamma);
                    }

                }

            } else if (auto* imuEvent = dynamic_cast<IMUEvent*>(event.get()))  {
                update_RIMU(state.R, imuEvent->ang_velocity, parameters.weight_RIMU);
            }

            else if (auto* imageEvent = dynamic_cast<ImageEvent*>(event.get()))  {
                std::cout << "Fused image at time " << imageEvent->time << std::endl;
                update_Ifusion(state.MI, imageEvent->image, parameters.weight_Ifusion);
                state.decayTimeSurface.setConstant(imageEvent->time);
            }

            // Starting from the start time we count up. If the current time (event->time)
            // reaches the time of the next "frame" we want to save to disk
            if (parameters.startTime + static_cast<float>(parameters.vis_counter) * (1 / parameters.fps) < event->time) {
                parameters.vis_counter++;
                std::cout << "Frame " << args->startIndex+parameters.vis_counter << "/"
                          << static_cast<int>((parameters.endTime - parameters.startTime) * parameters.fps) << std::endl;
                {
                    PROFILE_SCOPE("BETWEEN FRAMES");
                    //writeToFile(CCM, folder_path / ("C" + std::to_string(counter) + ".txt"));
                    //writeToFile(V_Vis, folder_path / ("V" + std::to_string(counter) + ".txt"));
                    //writeToFile(MI, folder_path / ("MI" + std::to_string(counter)  + ".txt"));
                    //writeToFile(I, folder_path / ("I" + std::to_string(counter)  + ".txt"));
                    //writeToFile(delta_I, folder_path / ("I_gradient" + std::to_string(counter)  + ".txt"));
                    //writeToFile(F, folder_path / ("F" + std::to_string(counter)  + ".txt"));
                    //writeToFile(G, folder_path / ("G" + std::to_string(counter)  + ".txt"));
                    writeToFile(event->time, state.R, R_path, true);
                }

    #ifdef IMAGES
                    float loss = VFG_check(state.V_Vis, state.F, state.G);
                    //std::cout << "VFG Check: " << loss << std::endl;
                    writeToFile(event->time, loss, VLossPath);

                    std::stringstream filename;
                    filename.fill('0');
                    filename.width(8);
                    filename<<std::to_string(static_cast<int>((args->startIndex + parameters.vis_counter)));

                    std::string image_name = "VIGF_" + filename.str() + ".png";
                    fs::path image_path = folder_path / image_name;
                    create_VIGF(Tensor2Matrix(state.V_Vis), Tensor2Matrix(state.MI), state.G, state.F, image_path, true, 0.1);
                    float cost1, cost2, cost3;
                    cost1 = costFR(state.F, state.CCM, state.dCdx, state.dCdy, state.R);
                    cost2 = costFG(state.F, state.V_Vis, state.G);
                    cost3 = costGI(state.G, state.delta_I);
                    std::cout << "Costs: " << cost1 << " " << cost2 << " " << cost3 << std::endl;
                    saveImage(Tensor2Matrix(state.MI), folder_path / ("frame_" + filename.str() + ".png"), true);
                    state.V_Vis.setZero();
                    //randomInit(F, -1, 1);
                    state.G.setZero();

    #endif
                //globalDecay(MI, decayTimeSurface, nP, t, dP);
            }

            if (parameters.startTime+ static_cast<float>(parameters.FR_update_counter) * (1 / parameters.FR_updates_per_second) <event->time) {
                parameters.FR_update_counter++;
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> elapsed_realtime = end - start_realtime;
    std::stringstream ssTime;
    ssTime << "Time elapsed: " << elapsed_realtime.count() << " seconds" << std::endl;
    writeToFile(ssTime.str(), folder_path / "time_realtime.txt");
    std::cout << "Algorithm took: " << elapsed_realtime.count() << "seconds/ Real elapsed time: " << parameters.endTime - parameters.startTime << std::endl;

    std::string outputFile = "output.mp4";

//#ifdef IMAGES
//    VideoCreator::createMP4Video(folder_path, folder_path / outputFile, static_cast<int>((parameters["fps"])));
//#endif

    Instrumentor::Get().EndSession();
}