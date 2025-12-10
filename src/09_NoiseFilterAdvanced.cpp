//
// Created by Daniel Pommer on 08.12.25.
//
#include <dv-processing/core/frame.hpp>
#include <dv-processing/io/mono_camera_recording.hpp>
#include <dv-processing/noise/background_activity_noise_filter.hpp>
#include <dv-processing/noise/fast_decay_noise_filter.hpp>
#include <dv-processing/noise/frequency_filters.hpp>
#include <dv-processing/noise/k_noise_filter.hpp>
#include <dv-processing/visualization/event_visualizer.hpp>

#include <CLI/CLI.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <thread>

int main(int ac, char **av) {
	using namespace std::chrono_literals;

	std::string aedat4Path;

	CLI::App app{"Command-line aedat4 preview player of recorded frames"};

	app.add_option("-i,--input", aedat4Path, "Path to an input aedat4 file to be played.")->required();
	try {
		app.parse(ac, av);
	}
	catch (const CLI::ParseError &e) {
		return app.exit(e);
	}

	// Construct the reader
	dv::io::MonoCameraRecording reader(aedat4Path);

	// Test whether the frame stream available
	if (!reader.isEventStreamAvailable()) {
		throw dv::exceptions::InvalidArgument<std::string>(
			fmt::format("Noise filtering sample requires an aedat4 with an event stream, the supplied file [{}] does "
						"not contain event stream",
				aedat4Path));
	}

	// Create a display window for the frames
	cv::namedWindow("Noise Filtering", cv::WINDOW_AUTOSIZE);

	dv::EventStreamSlicer slicer;

	const cv::Size resolution = *reader.getEventResolution();

	// Define used filters
	dv::noise::BackgroundActivityNoiseFilter<> BGAFilter(resolution);
	dv::noise::KNoiseFilter kNoiseFilter(resolution);
	dv::noise::FastDecayNoiseFilter<> fastFilter(resolution);
	dv::noise::LowPassFilter lowPassFilter(resolution, 500.0f);
	dv::noise::HighPassFilter highPassFilter(resolution, 500.0f);

	dv::visualization::EventVisualizer visualizer(resolution);

	// Define options for text in preview images
	const cv::Point2i textPosition(20, 20);
	const cv::Point2i textShift(0, 20);
	const double fontScale      = 0.7;
	const cv::Scalar fontColor  = dv::visualization::colors::red;
	const int32_t fontThickness = 2;

	slicer.doEveryTimeInterval(33ms, [&](const dv::EventStore &events) {
		// No filter
		cv::Mat noFilter = visualizer.generateImage(events);
		cv::putText(noFilter, "No filter", textPosition, cv::FONT_HERSHEY_SIMPLEX, fontScale, fontColor, fontThickness);
		cv::putText(noFilter, "Reduction factor: 0.00", textPosition + textShift, cv::FONT_HERSHEY_SIMPLEX, fontScale,
			fontColor, fontThickness);

		// Background activity noise filter
		BGAFilter.accept(events);
		dv::EventStore BGAFiltered = BGAFilter.generateEvents();
		cv::Mat BGAFilterPreview   = visualizer.generateImage(BGAFiltered);
		cv::putText(BGAFilterPreview, "Background activity filter", textPosition, cv::FONT_HERSHEY_SIMPLEX, fontScale,
			fontColor, fontThickness);
		cv::putText(BGAFilterPreview, fmt::format("Reduction factor: {:.2f}", BGAFilter.getReductionFactor()),
			textPosition + textShift, cv::FONT_HERSHEY_SIMPLEX, fontScale, fontColor, fontThickness);

		// K-Noise filter
		kNoiseFilter.accept(events);
		dv::EventStore kNoiseFiltered = kNoiseFilter.generateEvents();
		cv::Mat kNoiseFilterPreview   = visualizer.generateImage(kNoiseFiltered);
		cv::putText(kNoiseFilterPreview, "K-Noise filter", textPosition, cv::FONT_HERSHEY_SIMPLEX, fontScale, fontColor,
			fontThickness);
		cv::putText(kNoiseFilterPreview, fmt::format("Reduction factor: {:.2f}", kNoiseFilter.getReductionFactor()),
			textPosition + textShift, cv::FONT_HERSHEY_SIMPLEX, fontScale, fontColor, fontThickness);

		// Fast decay noise filter
		fastFilter.accept(events);
		dv::EventStore fastFiltered = fastFilter.generateEvents();
		cv::Mat fastFilterPreview   = visualizer.generateImage(fastFiltered);
		cv::putText(fastFilterPreview, "Fast filter", textPosition, cv::FONT_HERSHEY_SIMPLEX, fontScale, fontColor,
			fontThickness);
		cv::putText(fastFilterPreview, fmt::format("Reduction factor: {:.2f}", fastFilter.getReductionFactor()),
			textPosition + textShift, cv::FONT_HERSHEY_SIMPLEX, fontScale, fontColor, fontThickness);

		// Low-pass frequency filter
		lowPassFilter.accept(events);
		dv::EventStore lowPassFiltered = lowPassFilter.generateEvents();
		cv::Mat lowPassFilterPreview   = visualizer.generateImage(lowPassFiltered);
		cv::putText(lowPassFilterPreview, "Low-pass filter", textPosition, cv::FONT_HERSHEY_SIMPLEX, fontScale,
			fontColor, fontThickness);
		cv::putText(lowPassFilterPreview, fmt::format("Reduction factor: {:.2f}", lowPassFilter.getReductionFactor()),
			textPosition + textShift, cv::FONT_HERSHEY_SIMPLEX, fontScale, fontColor, fontThickness);

		// High-pass frequency filter
		highPassFilter.accept(events);
		dv::EventStore highPassFiltered = highPassFilter.generateEvents();
		cv::Mat highPassFilterPreview   = visualizer.generateImage(highPassFiltered);
		cv::putText(highPassFilterPreview, "High-pass filter", textPosition, cv::FONT_HERSHEY_SIMPLEX, fontScale,
			fontColor, fontThickness);
		cv::putText(highPassFilterPreview, fmt::format("Reduction factor: {:.2f}", highPassFilter.getReductionFactor()),
			textPosition + textShift, cv::FONT_HERSHEY_SIMPLEX, fontScale, fontColor, fontThickness);

		// Combine images
		cv::Mat preview;
		cv::Mat firstCol, secondCol, thirdCol;
		cv::vconcat(std::vector<cv::Mat>({noFilter, BGAFilterPreview}), firstCol);
		cv::vconcat(std::vector<cv::Mat>({kNoiseFilterPreview, fastFilterPreview}), secondCol);
		cv::vconcat(std::vector<cv::Mat>({lowPassFilterPreview, highPassFilterPreview}), thirdCol);

		cv::hconcat(std::vector<cv::Mat>({firstCol, secondCol, thirdCol}), preview);

		cv::imshow("Noise Filtering", preview);
		cv::waitKey(1);
	});

	while (const auto events = reader.getNextEventBatch()) {
		slicer.accept(*events);
	}

	return EXIT_SUCCESS;
}
