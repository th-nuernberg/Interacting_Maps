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

#include <boost/program_options.hpp>
#include "file_operations.h"
#include "imaging.h"
#include "timing.h"
#include "rotation_plot.h"
#include <opencv2/highgui.hpp>
#include <atomic>
#include <mutex>
#include <thread>
#include <fstream>

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
    unsigned seed;
    bool decay;
    bool clamp;
    bool resetG;
    bool silentG;
    bool asyncFR;
    float fps;
    bool show;
    bool pace;
    bool writeFrames;
    bool rotationFromTiming;
    std::string onsetRule;
    double onsetTime;
    double evidenceLifetime;
    bool evidenceAccumulate;
    bool plotRotation;
    int dumpEvents;
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
        .fuseI             = vm["fuseI"].as<bool>(),
        .seed              = vm["seed"].as<unsigned>(),
        .decay             = vm["decay"].as<bool>(),
        .clamp             = vm["clamp"].as<bool>(),
        .resetG            = vm["resetG"].as<bool>(),
        .silentG           = vm["silentG"].as<bool>(),
        .asyncFR           = vm["asyncFR"].as<bool>(),
        .fps               = vm["fps"].as<float>(),
        .show              = vm["show"].as<bool>(),
        .pace              = vm["pace"].as<bool>(),
        .writeFrames       = vm["writeFrames"].as<bool>(),
        .rotationFromTiming = vm["rotationFromTiming"].as<bool>(),
        .onsetRule         = vm["onsetRule"].as<std::string>(),
        .onsetTime         = vm["onsetTime"].as<double>(),
        .evidenceLifetime  = vm["evidenceLifetime"].as<double>(),
        .evidenceAccumulate = vm["evidenceAccumulate"].as<bool>(),
        .plotRotation      = vm["plotRotation"].as<bool>(),
        .dumpEvents        = vm["dumpEvents"].as<int>()
    };
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
            ("startTime", po::value<float>()->default_value(0), "Where to start with event consideration")
            ("endTime", po::value<float>()->default_value(5), "Where to end with event consideration")
            ("timeFormat", po::value<std::string>()->default_value("uS"), "What format are the times: seconds, milliseconds, microseconds (s,ms,mus)")
            ("timeStep", po::value<float>()->default_value(0.01), "Size of the event frames")
            ("maxIntervalLength", po::value<float>()->default_value(1.0), "Load events from time slice of this size at a time")
            ("resourceDirectory", po::value<std::string>()->default_value("shapes_rotation"), "Which dataset to use, searches in res directory")
            ("resultsDirectory", po::value<std::string>()->default_value("shapes_rotation"), "Where to store the results, located in output directory")
            ("startIndex,i", po::value<int>()->default_value(0), "With what index to start for the images")
            ("fuseR", po::value<bool>()->default_value(false), "Fuse with imu.txt?")
            ("fuseI", po::value<bool>()->default_value(false), "Fuse with images folder?")
            ("seed", po::value<unsigned>()->default_value(0), "Seed for random initialisation and update order")
            ("decay", po::value<bool>()->default_value(true), "Decay I towards the neutral potential between events (not in the paper)")
            ("clamp", po::value<bool>()->default_value(false), "Clip F and G to +-gamma and zero tiny values (old display-range bound, not in the paper)")
            ("resetG", po::value<bool>()->default_value(false), "IMAGES builds: zero G after each output frame (old hack, not in the paper)")
            ("silentG", po::value<bool>()->default_value(true), "Treat time without events at a pixel as V = 0 for the G <- F relation")
            ("asyncFR", po::value<bool>()->default_value(true), "Event-driven F <- R (closed-form relaxation) instead of global passes every timeStep")
            ("fps", po::value<float>()->default_value(60), "Rate of output frames, live display and R.txt samples")
            ("show", po::value<bool>()->default_value(false), "IMAGES builds: stream the V/I/G/F frames to a live window")
            ("pace", po::value<bool>()->default_value(true), "With --show: play back in real time instead of as fast as possible")
            ("writeFrames", po::value<bool>()->default_value(true), "IMAGES builds: write the V/I/G/F frames as PNG files")
            ("rotationFromTiming", po::value<bool>()->default_value(false), "F and R (rad/s) from the arrival times of edges instead of F <- G, V and the F <-> R loop")
            ("onsetRule", po::value<std::string>()->default_value("depression"), "With rotationFromTiming: arrival time per pixel, latest | gap | depression")
            ("onsetTime", po::value<double>()->default_value(0.02), "With rotationFromTiming: depression recovery time or burst gap (s)")
            ("evidenceLifetime", po::value<double>()->default_value(0.01), "With rotationFromTiming: lifetime of flow measurements (s)")
            ("evidenceAccumulate", po::value<bool>()->default_value(false), "With rotationFromTiming: add measurements at a pixel instead of replacing them")
            ("plotRotation", po::value<bool>()->default_value(true), "IMAGES builds with rotationFromTiming: plot R against the gyro (display only) below the maps")
            ("dumpEvents", po::value<int>()->default_value(0), "With rotationFromTiming: write the maps at every n-th event to event_diagnostics.bin (0 = off)");

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
    fs::path folder_path = create_results_folder(args->resultsDirectory);
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
    parameters.seed = args->seed;
    parameters.useDecay = args->decay;
    parameters.clampMaps = args->clamp;
    parameters.resetGEachFrame = args->resetG;
    parameters.silentRelaxation = args->silentG;
    parameters.asyncFR = args->asyncFR;
    parameters.fps = args->fps;
    parameters.rotationFromTiming = args->rotationFromTiming;
    parameters.onsetRule = args->onsetRule;
    parameters.onsetTime = args->onsetTime;
    parameters.evidenceLifetime = args->evidenceLifetime;
    parameters.evidenceAccumulate = args->evidenceAccumulate;
    OpticalFlowState state = initOpticalFlowState(res.height, res.width, parameters, calibration_data);

    // Rotation from event timing: maps O, F evidence and R's sums; relations in include/timing.h
    timing::OnsetMap onsetMap(res.height, res.width);
    const auto onsetRule = timing::makeOnsetRule(parameters.onsetRule, parameters.onsetTime, parameters.onsetUse);
    const timing::NormalFlowSettings normalFlowSettings{parameters.normalFlowRadius, parameters.normalFlowHorizon,
                                                       parameters.normalFlowTimeFloor, parameters.normalFlowMaxSpeed};
    const timing::MotionModel motionModel = parameters.rotationFromTiming
        ? timing::MotionModel::rotation(calibration_data, res.height, res.width) : timing::MotionModel{};
    timing::FlowEvidence flowEvidence(res.height, res.width, parameters.evidenceLifetime, parameters.evidenceAccumulate);
    timing::RotationFromFlow rotationFromFlow(parameters.evidenceLifetime);
    const double flowPriorKappa = 1.0 / (parameters.flowPriorSigma * parameters.flowPriorSigma);
    std::vector<int> noFlowUpdates;   // with rotationFromTiming the event step only updates I and G

    // Plot of R against the gyro, for the display only: the gyro never reaches the network unless --fuseR.
    const bool plotRotation = args->plotRotation && parameters.rotationFromTiming && fs::exists(imuPath);
    const bool dumpEvents = args->dumpEvents > 0 && parameters.rotationFromTiming && fs::exists(imuPath);
    const GyroReference gyroReference = (plotRotation || dumpEvents) ? GyroReference::load(imuPath, parameters.R_cam_imu) : GyroReference{};
    RotationPlot rotationPlot(parameters.startTime, parameters.endTime);
    const double imuLag = 0.004;   // s, IMU behind the events (experiments/v_units.py)
    const int plotHeight = 400;

    // Diagnostics for experiments/polarity_check.py: the maps at an event pixel just before the event's I/G update
    struct EventDiagnostics {
        double t;
        float y, x, p;
        float G[2], F[2], MR[2], MRgyro[2];     // (y, x); MR = flow predicted by R, MRgyro = by the gyro
        float Nvalid, N[2], Nspeed;             // normal flow of this event, unit direction (y, x) and px/s
        float I, Imin, Imax;                    // I at the pixel and over its 3 x 3 stencil
        float dtPrevious;                       // s since this pixel's previous event (any polarity)
    };
    std::ofstream diagnosticsFile;
    std::vector<double> lastEventAt;
    long long cameraEventCount = 0;
    if (dumpEvents) {
        diagnosticsFile.open(setupFilePath(folder_path, "event_diagnostics.bin"), std::ios::binary);
        lastEventAt.assign(static_cast<size_t>(res.height) * res.width, -1.0);
        std::cout << "Writing every " << args->dumpEvents << "th event to event_diagnostics.bin ("
                  << sizeof(EventDiagnostics) << " bytes each)" << std::endl;
    }
    if (parameters.rotationFromTiming) {
        state.R.setZero();
        state.F.setZero();
        std::cout << "Rotation from event timing: onset rule " << onsetRule->name() << " (" << parameters.onsetTime
                  << " s), evidence lifetime " << parameters.evidenceLifetime << " s" << std::endl;
    }

    // iterations are done after event calculations for a frame are done
    std::vector permutation {0,2,3}; // Which update steps to take; 1 is not needed
    std::seed_seq order_seed{parameters.seed, 2u};
    std::mt19937 rng(order_seed);

    // For keeping track of the current Event
    // std::vector<int> currentCoordinates = {0,0};
    // CameraEvent currentCameraEvent = CameraEvent(args->startTime, currentCoordinates, 0);
    // std::vector<float> ang_velocity = {0,0,0};
    // std::vector<float> lin_acceleration = {0,0,0};
    // IMUEvent currentImuEvent = IMUEvent(args->startTime, lin_acceleration, ang_velocity);

    // Load the whole event file once; the interval loop below only slices it. Not counted in the RTF.
    auto start_loading = std::chrono::high_resolution_clock::now();
    std::vector<RawCameraEvent> allCameraEvents;
    if (!load_events_txt(eventPath, allCameraEvents)) {
        std::cerr << "Could not open " << eventPath << std::endl;
        return 1;
    }
    std::chrono::duration<float> elapsed_loading = std::chrono::high_resolution_clock::now() - start_loading;
    std::cout << "Loaded " << allCameraEvents.size() << " events from " << eventPath << " in " << elapsed_loading.count() << " s" << std::endl;

#ifdef IMAGES
    if (args->show) {
        cv::namedWindow("Interacting Maps", cv::WINDOW_NORMAL);
        // 2x the panel; with the rotation plot below, 1.2x so that it fits on a laptop screen
        if (plotRotation) cv::resizeWindow("Interacting Maps", 756, static_cast<int>(1.2 * (380 + plotHeight)));
        else cv::resizeWindow("Interacting Maps", 1260, 760);
    }
#else
    if (args->show || !args->writeFrames) {
        std::cout << "--show and --writeFrames only apply to IMAGES builds" << std::endl;
    }
#endif

    auto start_realtime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> maxPaceLag{0.0f};   // how far the paced display fell behind the data clock

    // With --show the algorithm runs on a worker thread: on macOS the window must be serviced from the main
    // thread, and each event pump (waitKey / pollKey) takes ~12 ms, which would otherwise stall the algorithm.
    std::mutex frameMutex;
    cv::Mat latestFrame;
    std::atomic<bool> finished{false};
    auto runAlgorithm = [&]() {
        // intervals holds n+1 boundaries for n slices
        for (size_t currentInterval = 0; currentInterval + 1 < intervals.size(); ++currentInterval) {
            //##################################################################################################################
            // Read events file

            const auto start_interval = std::chrono::high_resolution_clock::now();
            std::vector<std::shared_ptr<Event>> cameraEventData;
            slice_events(allCameraEvents, intervals[currentInterval], intervals[currentInterval+1], cameraEventData);
            std::cout << "Events for time " << intervals[currentInterval] << " to " << intervals[currentInterval + 1] << std::endl;
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
                    if (parameters.useDecay) {
                        decayNeighbourhood(state.MI, state.decayTimeSurface, cEvent->coordinates[0], cEvent->coordinates[1], event->time, parameters.neutralPotential, parameters.decayParam);
                    }
                    timing::NormalFlow normalFlow;
                    if (parameters.rotationFromTiming) {
                        // events -> O -> N -> F evidence -> R (rad/s); stored F where this event reads it
                        const int ey = cEvent->coordinates[0], ex = cEvent->coordinates[1];
                        onsetRule->update(onsetMap, ey, ex, cEvent->polarity, event->time);
                        normalFlow = timing::fitNormalFlow(onsetMap, ey, ex, cEvent->polarity,
                                                                                    event->time, normalFlowSettings);
                        rotationFromFlow.fadeTo(event->time);
                        rotationFromFlow.apply(motionModel.at(ey, ex), flowEvidence.add(ey, ex, normalFlow, event->time));
                        const Eigen::Vector3f &R = rotationFromFlow.rotation();
                        for (int k = 0; k < 3; ++k) state.R(k) = R(k);
                        timing::readFlowNeighbourhood(flowEvidence, motionModel, R, flowPriorKappa, ey, ex, event->time, state.F);
                    } else if (parameters.asyncFR) {
                        // F <- R since the last event: B in closed form, F where this event reads it.
                        relaxB(state.B, state.BTime, state.A, state.R, event->time, parameters.rate_FR);
                        relaxFRNeighbourhood(state.F, state.CCM, state.dCdx, state.dCdy, state.R, state.FTimeSurface, state.old_points,
                                             cEvent->coordinates[0], cEvent->coordinates[1], event->time, parameters.rate_FR);
                    }
                    if (parameters.silentRelaxation) {
                        relaxSilentNeighbourhood(state.G, state.F, state.MI, state.delta_I, state.GTimeSurface, cEvent->coordinates[0], cEvent->coordinates[1], event->time, parameters.rate_GI_silent, parameters.rate_GF_silent);
                    }
                    if (dumpEvents) {
                        const int ey = cEvent->coordinates[0], ex = cEvent->coordinates[1];
                        double &last = lastEventAt[static_cast<size_t>(ey) * res.width + ex];
                        if (cameraEventCount++ % args->dumpEvents == 0) {
                            const Eigen::Vector3f w = gyroReference.at(event->time + imuLag);
                            const Eigen::Vector3d Rgyro(-w.y(), -w.x(), -w.z());
                            const Eigen::Vector2d MR = motionModel.at(ey, ex) * Eigen::Vector3d(state.R(0), state.R(1), state.R(2));
                            const Eigen::Vector2d MRgyro = motionModel.at(ey, ex) * Rgyro;
                            float Imin = state.MI(ey, ex), Imax = Imin;
                            for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
                                const int yy = std::clamp(ey + dy, 0, res.height - 1), xx = std::clamp(ex + dx, 0, res.width - 1);
                                Imin = std::min(Imin, state.MI(yy, xx));
                                Imax = std::max(Imax, state.MI(yy, xx));
                            }
                            const EventDiagnostics d{event->time, static_cast<float>(ey), static_cast<float>(ex),
                                static_cast<float>(cEvent->polarity),
                                {state.G(ey, ex, 0), state.G(ey, ex, 1)}, {state.F(ey, ex, 0), state.F(ey, ex, 1)},
                                {static_cast<float>(MR(0)), static_cast<float>(MR(1))},
                                {static_cast<float>(MRgyro(0)), static_cast<float>(MRgyro(1))},
                                normalFlow.valid ? 1.0f : 0.0f, {normalFlow.n(0), normalFlow.n(1)}, normalFlow.speed,
                                state.MI(ey, ex), Imin, Imax,
                                last < 0 ? std::numeric_limits<float>::infinity() : static_cast<float>(event->time - last)};
                            diagnosticsFile.write(reinterpret_cast<const char *>(&d), sizeof(d));
                        }
                        last = event->time;
                    }
                    for (int i = 0; i < parameters.event_steps; ++i) {
                        event_step(state.V, state.MI, state.delta_I, state.GIDiff, state.GIDiffGradient, state.F, state.G, state.R, state.CCM, state.dCdx, state.dCdy, state.A, state.B,
                                   state.Identity_minus_outerProducts, state.old_points, parameters,
                                   parameters.rotationFromTiming ? noFlowUpdates : permutation, cEvent->coordinates[0], cEvent->coordinates[1]);
                    }
                    if (!parameters.rotationFromTiming && parameters.asyncFR) {
                        // F <- G and G <- F may have run after F -> R in this step; B must see the final F.
                        register_R_point(state.F, state.CCM, state.dCdx, state.dCdy, state.B, state.Identity_minus_outerProducts, state.old_points,
                                         cEvent->coordinates[0], cEvent->coordinates[1]);
                    }

                    if (!parameters.rotationFromTiming && !parameters.asyncFR && parameters.startTime + static_cast<float>(parameters.FR_update_counter) * static_cast<float>(1 / parameters.FR_updates_per_second) < event->time) {
                        state.time.setConstant(event->time);
                        for (int i = 0; i < static_cast<int>(parameters.updateIterationsFR); ++i) {
                            update_FR(state.F, state.CCM, state.dCdx, state.dCdy, state.R, parameters.weight_FR, parameters.zeroThreshold(), parameters.clampBound());
                        }
                        // F changed at every pixel; the F -> R running sums must follow.
                        rebuild_R_sums(state.F, state.CCM, state.dCdx, state.dCdy, state.B,
                                       state.Identity_minus_outerProducts, state.old_points);

                    }

                } else if (auto* imuEvent = dynamic_cast<IMUEvent*>(event.get()))  {
                    if (parameters.rotationFromTiming) {
                        // The gyro as one more measurement of R (rad/s), weighted by its precision: a weak input
                        const auto &g = imuEvent->ang_velocity;
                        const Vector3f omega_cam = parameters.R_cam_imu * Vector3f(g[0], g[1], g[2]);
                        rotationFromFlow.fadeTo(event->time);
                        rotationFromFlow.addMeasurement(-Eigen::Vector3f(omega_cam.y(), omega_cam.x(), omega_cam.z()),
                                                        1.0 / (parameters.gyroSigma * parameters.gyroSigma));
                        const Eigen::Vector3f &R = rotationFromFlow.rotation();
                        for (int k = 0; k < 3; ++k) state.R(k) = R(k);
                    } else {
                        update_RIMU(state.R, imuEvent->ang_velocity, parameters.weight_RIMU, parameters.R_cam_imu, parameters.imuScale);
                    }
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
                        // Decay is lazy: show every pixel at the frame time, without touching the state.
                        const Tensor2f MI_now = parameters.useDecay
                            ? decayedCopy(state.MI, state.decayTimeSurface, event->time, parameters.neutralPotential, parameters.decayParam)
                            : state.MI;
                        if (parameters.rotationFromTiming) {
                            // bring the stored F up to the frame time everywhere
                            timing::readFlowAll(flowEvidence, motionModel, rotationFromFlow.rotation(), flowPriorKappa,
                                                event->time, state.F);
                        }
                        const Tensor3f F_now = !parameters.rotationFromTiming && parameters.asyncFR
                            ? relaxedFRCopy(state.F, state.CCM, state.dCdx, state.dCdy, state.R, state.FTimeSurface, event->time, parameters.rate_FR)
                            : state.F;
                        const Tensor3f G_now = parameters.silentRelaxation
                            ? relaxedSilentCopy(state.G, F_now, MI_now, state.GTimeSurface, event->time, parameters.rate_GI_silent, parameters.rate_GF_silent)
                            : state.G;
                        float loss = VFG_check(state.V_Vis, F_now, G_now);
                        //std::cout << "VFG Check: " << loss << std::endl;
                        writeToFile(event->time, loss, VLossPath);

                        std::stringstream filename;
                        filename.fill('0');
                        filename.width(8);
                        filename<<std::to_string(static_cast<int>((args->startIndex + parameters.vis_counter)));

                        std::string image_name = "VIGF_" + filename.str() + ".png";
                        fs::path image_path = folder_path / image_name;
                        // I is in event steps around an unobservable offset; show 0 as mid grey, 10 grey levels per step
                        const float displayGreyPerEvent = 10.0f;
                        const Tensor2f I_display = MI_now * displayGreyPerEvent + 128.0f;
                        cv::Mat vigf = create_VIGF(Tensor2Matrix(state.V_Vis), Tensor2Matrix(I_display), G_now, F_now, image_path, false, 0.1);
                        if (plotRotation) {
                            // omega = -R, from (y, x, z) storage to (x, y, z)
                            const Eigen::Vector3f omega(-state.R(1), -state.R(0), -state.R(2));
                            rotationPlot.add(event->time, omega, gyroReference.at(event->time + imuLag));
                            cv::vconcat(vigf, rotationPlot.render(vigf.cols, plotHeight), vigf);
                        }
                        if (args->writeFrames) {
                            cv::imwrite(image_path.string(), vigf);
                        }
                        if (args->show) {
                            if (args->pace) {
                                const auto due = start_realtime + std::chrono::duration_cast<std::chrono::high_resolution_clock::duration>(
                                    std::chrono::duration<float>(event->time - parameters.startTime));
                                const auto now = std::chrono::high_resolution_clock::now();
                                if (now < due) {
                                    std::this_thread::sleep_until(due);
                                } else {
                                    maxPaceLag = std::max(maxPaceLag, std::chrono::duration<float>(now - due));
                                }
                            }
                            // The main thread shows it; servicing the window takes ~12 ms per call on macOS.
                            std::lock_guard<std::mutex> lock(frameMutex);
                            latestFrame = vigf.clone();
                        }
                        float cost1, cost2, cost3;
                        cost1 = costFR(F_now, state.CCM, state.dCdx, state.dCdy, state.R);
                        cost2 = costFG(F_now, state.V_Vis, G_now);
                        cost3 = costGI(G_now, state.delta_I);
                        std::cout << "Costs: " << cost1 << " " << cost2 << " " << cost3 << std::endl;
                        if (args->writeFrames) {
                            saveImage(Tensor2Matrix(MI_now), folder_path / ("frame_" + filename.str() + ".png"), true);
                        }
                        state.V_Vis.setZero();
                        //randomInit(F, -1, 1);
                        if (parameters.resetGEachFrame) {
                            state.G.setZero();
                        }

        #endif
                    //globalDecay(MI, decayTimeSurface, nP, t, dP);
                }

                if (parameters.startTime+ static_cast<float>(parameters.FR_update_counter) * (1 / parameters.FR_updates_per_second) <event->time) {
                    parameters.FR_update_counter++;
                }
            }
            const std::chrono::duration<float> elapsed_interval = std::chrono::high_resolution_clock::now() - start_interval;
            const float intervalSpan = intervals[currentInterval + 1] - intervals[currentInterval];
            std::cout << "Interval " << intervals[currentInterval] << "-" << intervals[currentInterval + 1] << " s: "
                      << cameraEventData.size() << " events, " << elapsed_interval.count() << " s, RTF = "
                      << elapsed_interval.count() / intervalSpan << std::endl;
        }
        finished = true;
    };

#ifdef IMAGES
    if (args->show) {
        std::thread worker(runAlgorithm);
        while (!finished) {
            cv::Mat frame;
            {
                std::lock_guard<std::mutex> lock(frameMutex);
                std::swap(frame, latestFrame);
            }
            if (!frame.empty()) {
                cv::imshow("Interacting Maps", frame);
            }
            cv::pollKey();
        }
        worker.join();
    } else {
        runAlgorithm();
    }
#else
    runAlgorithm();
#endif

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> elapsed_realtime = end - start_realtime;
    std::stringstream ssTime;
    // Real-time factor: processing time per second of recording (< 1 is faster than real time). The span ends at the
    // last event if the recording is shorter than endTime; loading the event file is excluded.
    const float lastEventTime = allCameraEvents.empty() ? parameters.startTime : allCameraEvents.back().time;
    const float dataSpan = std::min(parameters.endTime, lastEventTime) - parameters.startTime;
    const float rtf = elapsed_realtime.count() / dataSpan;
    ssTime << "Time elapsed: " << elapsed_realtime.count() << " seconds" << std::endl;
    ssTime << "Data span: " << dataSpan << " seconds" << std::endl;
    ssTime << "RTF: " << rtf << std::endl;
    writeToFile(ssTime.str(), folder_path / "time_realtime.txt");
    if (args->show && args->pace) {
        std::cout << "Paced to real time; the RTF below includes the waiting. Largest lag behind real time: "
                  << maxPaceLag.count() * 1000.0f << " ms" << std::endl;
    }
    std::cout << "Algorithm took: " << elapsed_realtime.count() << " s for " << dataSpan << " s of data, RTF = " << rtf << " (event loading: " << elapsed_loading.count() << " s, excluded)" << std::endl;

    std::string outputFile = "output.mp4";

//#ifdef IMAGES
//    VideoCreator::createMP4Video(folder_path, folder_path / outputFile, static_cast<int>((parameters["fps"])));
//#endif

    Instrumentor::Get().EndSession();
}