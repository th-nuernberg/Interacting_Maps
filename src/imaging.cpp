//
// Created by daniel on 11/25/24.
//
#include "imaging.h"
#include "datatypes.h"
#include "conversions.h"

#define PI 3.14159265

/**
 * Takes a frame and applies the camera_matrix and distortion parameters to undistort the image
 * @param frame Image in form of an Eigen matrix
 * @param camera_matrix contains camera parameters like focal length; in form of an opencv matrix
 * @param distortion_parameters contains distortion parameters for radial distortion, etc. opencv matrix
 * @return Image in form of an Eigen matrix
 */
MatrixXfRowMajor undistort_frame(const MatrixXfRowMajor &frame, const cv::Mat &camera_matrix, const cv::Mat &distortion_parameters) {
    cv::Mat image = eigenToCvMat(frame);
    return cvMatToEigen(undistort_image(image, camera_matrix, distortion_parameters));
}
/**
 * Takes a frame and applies the camera_matrix and distortion parameters to undistort the image
 * @param frame Image in form of an Eigen matrix
 * @param camera_matrix contains camera parameters like focal length; in form of an opencv matrix
 * @param distortion_parameters contains distortion parameters for radial distortion, etc. opencv matrix
 * @return Image in form of an Eigen matrix
 */
cv::Mat undistort_image(const cv::Mat &image, const cv::Mat &camera_matrix, const cv::Mat &distortion_parameters) {
    // cv::Mat new_camera_matrix;
    // cv::Rect roi;
    // new_camera_matrix = cv::getOptimalNewCameraMatrix(camera_matrix, distortion_parameters, cv::Size(width, height), 1, cv::Size(width, height), &roi);
    cv::Mat undistorted_image;
    cv::undistort(image, undistorted_image, camera_matrix, distortion_parameters, camera_matrix);
    return undistorted_image;
}
/**
 * Takes a vector of images to undisort
 * @param images std::vector of distorted images in form of opencv matrices
 * @param camera_matrix contains camera parameters like focal length; in form of an opencv matrix
 * @param distortion_parameters contains distortion parameters for radial distortion, etc. opencv matrix
 * @return std::vector of undistorted images
 */
std::vector<cv::Mat> undistort_images(const std::vector<cv::Mat> &images, const MatrixXfRowMajor &camera_matrix, const MatrixXfRowMajor &distortion_coefficients) {
    std::vector<cv::Mat> undistorted_images;

    // Convert Eigen matrices to cv::Mat
    cv::Mat camera_matrix_cv = eigenToCvMat(camera_matrix);
    cv::Mat distortion_coefficients_cv = eigenToCvMat(distortion_coefficients);

    for (const auto& image : images) {
        undistorted_images.push_back(undistort_image(image, camera_matrix_cv, distortion_coefficients_cv));
    }
    return undistorted_images;
}

/**
 * receives a frame and converts it to an greyscale image
 * @param frame in form of an Eigen matrix
 * @return opencv matrix of the greyscale image
 */
cv::Mat frame2grayscale(const MatrixXfRowMajor &frame) {
    // Convert MatrixXfRowMajor to cv::Mat
    cv::Mat frame_cv = eigenToCvMat(frame);
    cv::Mat output;

    // Find min and max polarity
    double min_polarity, max_polarity;
    cv::minMaxLoc(frame_cv, &min_polarity, &max_polarity);

    // Normalize the frame
    frame_cv.convertTo(frame_cv, CV_32FC3, 1.0 / (max_polarity - min_polarity), -min_polarity / (max_polarity - min_polarity));

    // Scale to 0-255 and convert to CV_8U
    frame_cv.convertTo(output, CV_8UC1, 255.0);

    return output;
}

/**
 * Converts the an Event frame to an image. positive polarity results in green coloring, negative in red.
 * @param V agglomerated Events in a Eigen matrix
 * @param cutoff Events with intensity less than cutoff are not visualised
 * @return opencv matrix of the colorcoded event frame
 */
cv::Mat V2image(const MatrixXfRowMajor &V, const float cutoff=0.1) {
    // Determine the shape of the image
    long rows = V.rows();
    long cols = V.cols();

    // Create an empty image with 3 channels (BGR)
    cv::Mat image = cv::Mat::zeros(rows, cols, CV_8UC3);

    // Process on_events (V > 0)
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (V(i, j) > cutoff) {
                image.at<cv::Vec3b>(i, j)[1] = 255; // Set green channel
            }
            if (V(i, j) < -cutoff) {
                image.at<cv::Vec3b>(i, j)[2] = 255; // Set red channel
            }
        }
    }

    return image;
}

/**
 * Converts a vector field to an opencv image. Vectors are color coded according to their direction
 * @param vector_field 3Tensor containing a 2 dimensional vector for each pixel of the image
 * @return returns a bgr_image as an opencv matrix
 */
cv::Mat vector_field2image(const Tensor3f &vector_field) {
    const int rows = vector_field.dimension(0);
    const int cols = vector_field.dimension(1);

    // Calculate angles and saturations
    MatrixXfRowMajor angles(rows, cols);
    MatrixXfRowMajor saturations(rows, cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float y = vector_field(i, j, 0);
            float x = vector_field(i, j, 1);
            float angle = std::atan2(y, x);
//            if (angle < 0){
//                angle += 2*PI;
//            }
            angles(i,j) = angle;
            saturations(i, j) = std::sqrt(x*x + y*y);
        }
    }
    //std::cout << vector_field(0,0,0) << " " << vector_field(0,0,1) << std::endl;
    //std::cout << angles(0,0) << std::endl;
    //std::cout << saturations(0,0) << std::endl;

    // Normalize angles to [0, 179]
    cv::Mat hue(rows, cols, CV_8UC1);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            hue.at<uint8_t>(i, j) = static_cast<uint8_t>((angles(i, j) + M_PI)/(2 * M_PI) * 179);
        }
    }

    //std::cout << static_cast<uint16_t>(hue.at<uint8_t>(0,0)) << std::endl;

    // Value channel (full brightness)
    cv::Mat value(rows, cols, CV_8UC1, cv::Scalar(255));


    // Normalize saturations to [0, 255]
//    float max_saturation = 10;
    float max_saturation = saturations.maxCoeff();
    float min_saturation = saturations.minCoeff();
    //std::cout << max_saturation << std::endl;
    cv::Mat saturation(rows, cols, CV_8UC1);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (saturations(i,j) == 0){
                value.at<uint8_t>(i, j) = 0;
            }
            else{
                saturation.at<uint8_t>(i, j) = static_cast<uint8_t>(std::max(std::min(255.0 * (saturations(i, j)) / (max_saturation), 255.0),100.0));
            }
            //saturation.at<uint8_t>(i, j) = static_cast<uint8_t>(std::min(255.0 * (saturations(i, j) - min_saturation) / (max_saturation - min_saturation), 255.0));
        }
    }
    //std::cout << static_cast<uint16_t>(saturation.at<uint8_t>(0,0)) << std::endl;
    //double minVal;
    //double maxVal;
    //cv::Point minLoc;
    //cv::Point maxLoc;
    //minMaxLoc( saturation, &minVal, &maxVal, &minLoc, &maxLoc );
    //std::cout << "min val: " << minVal << std::endl;
    //std::cout << "max val: " << maxVal << std::endl;

    // Merge HSV channels
    std::vector<cv::Mat> hsv_channels = {hue, saturation, value};
    cv::Mat hsv_image;
    cv::merge(hsv_channels, hsv_image);

    // Convert HSV image to BGR format
    cv::Mat bgr_image;
    cv::cvtColor(hsv_image, bgr_image, cv::COLOR_HSV2BGR);

    return bgr_image;
}

/**
 * Creates a 3 Tensor of a square 2D grid with 2D vectors which point from the center radially outward
 * @param grid_size square side length of the grid
 * @return 3 Tensor of 2D vectors
 */
Tensor3f create_outward_vector_field(int grid_size) {
    // Create a 2D grid of points using linspace equivalent
    VectorXf x = VectorXf::LinSpaced(grid_size, -1.0, 1.0);
    VectorXf y = VectorXf::LinSpaced(grid_size, 1.0, -1.0);

    // Initialize matrices for meshgrid-like behavior (xv, yv)
    MatrixXfRowMajor xv(grid_size, grid_size);
    MatrixXfRowMajor yv(grid_size, grid_size);

    // Create meshgrid by repeating x and y values across rows and columns
    for (int i = 0; i < grid_size; ++i) {
        xv.row(i) = x.transpose();
        yv.col(i) = y;
    }

    // Create a 3D Tensor to store the 2D vector field (x, y components at each point)
    Tensor3f vectors(grid_size, grid_size, 2);

    // Calculate distances from origin and store the normalized vectors
    for (int i = 0; i < grid_size; ++i) {
        for (int j = 0; j < grid_size; ++j) {
            float x_val = xv(i, j);
            float y_val = yv(i, j);
            float distance = std::sqrt(x_val * x_val + y_val * y_val);

            // Prevent division by zero at the origin
            if (distance == 0) distance = 1.0;

            // Normalize the vector and assign it
            vectors(i, j, 1) = x_val / distance;
            vectors(i, j, 0) = y_val / distance;
        }
    }

    return vectors;
}

/**
 * Creates a image mask for a square image which only shows a circular ring around the center of the image
 * @param image_size
 * @param inner_radius
 * @param outer_radius
 * @return
 */
cv::Mat create_circular_band_mask(const cv::Size &image_size, float inner_radius, float outer_radius) {
    cv::Mat mask(image_size, CV_8UC1, cv::Scalar(0));

    int center_x = image_size.width / 2;
    int center_y = image_size.height / 2;

    for (int i = 0; i < image_size.height; ++i) {
        for (int j = 0; j < image_size.width; ++j) {
            float distance = std::sqrt((i - center_y) * (i - center_y) + (j - center_x) * (j - center_x));
            if (distance >= inner_radius && distance <= outer_radius) {
                mask.at<uchar>(i, j) = 255;  // Inside the band
            }
        }
    }

    return mask;
}

/**
 * Creates a square image colored ring to visualise the direction vectors are pointing in a vectorfield
 * @param grid_size Size if the image of the colored ring
 * @return bgr image of the ring in form of a opencv matrix
 */
cv::Mat create_colorwheel(int grid_size) {
    Tensor3f vector_field = create_outward_vector_field(grid_size);

    cv::Mat image;
    image = vector_field2image(vector_field);

    // Define inner and outer radius
    float inner_radius = grid_size / 4.0f;
    float outer_radius = grid_size / 2.0f;

    // Create the circular band mask
    cv::Mat mask = create_circular_band_mask(image.size(), inner_radius, outer_radius);

    // Apply mask to the image
    cv::Mat colourwheel = image.clone();
    colourwheel.setTo(cv::Scalar(255, 255, 255), ~mask);  // Outside the mask: set to white
    return colourwheel;
}

/**
 * Visualise the Event information V, image I, spatial gradient field G, and optical flow F in a single image
 * @param V Eigen matrix of Event frame
 * @param I Eigen matrix of light intensities
 * @param G 3 Tensor containing the spatial gradients of the image
 * @param F 3 Tensor containing the optical flow of the image
 * @param path where to save the image on the disk if desired
 * @param save if a save to disk is desired
 * @param cutoff Events with intensity less than cutoff are not visualised
 * @return bgr image of the joined visualisation as opencv matrix
 */
cv::Mat create_VIGF(const MatrixXfRowMajor &V, const MatrixXfRowMajor &I, const Tensor3f &G, const Tensor3f &F, const std::string &path = "VIGF", const bool save = false, const float cutoff=0.1) {
    cv::Mat V_img = V2image(V, cutoff);
    cv::Mat I_img = frame2grayscale(I);
    cv::Mat G_img = vector_field2image(G);
    cv::Mat F_img = vector_field2image(F);

    bool masking = false;

    cv::Mat masked_F;

    if (masking){
        cv::Mat mask;
        cv::transform(V_img, mask, cv::Matx13f(1,1,1));
        mask = mask/255;

        //    std::cout<<mask.size() << F_img.size() <<std::endl;
//        std::cout<<mask.channels() << std::endl;
//    //    std::cout<< F_img.channels() <<std::endl;
//        std::cout<<mask.type() << std::endl;
        //    std::cout << F_img.type() <<std::endl;


        cv::cvtColor(mask, mask, cv::COLOR_GRAY2BGR);

//    std::cout<<mask.size() << F_img.size() <<std::endl;
//    std::cout<<mask.channels() << std::endl;
//    std::cout<< F_img.channels() <<std::endl;
//    std::cout<<mask.type() << std::endl;
//    std::cout << F_img.type() <<std::endl;

        cv::multiply(mask, F_img, masked_F);
    }else{
         masked_F = F_img;
    }



    long rows = V.rows();
    long cols = V.cols();
    long I_rows = I.rows();
    long I_cols = I.cols();

    // Calculate the size of the color wheel
    int colourwheel_size = cols / 2;

    // Calculate the size of the final output image (double size + padding and color wheel)
    int y_size = rows * 2 + 20;
    int x_size = cols * 2 + 30 + colourwheel_size;

    cv::Mat image(y_size, x_size, CV_8UC3, cv::Scalar(0, 0, 0));
//    std::cout << image.size() << std::endl;

    // Place V image
    V_img.copyTo(image(cv::Rect(5, 5, cols, rows)));

    // Place I image (convert to BGR first)
    cvtColor(I_img, I_img, cv::COLOR_GRAY2BGR);
    I_img.copyTo(image(cv::Rect(cols + 15 + colourwheel_size, 5, I_cols, I_rows)));

    // Place G image
    G_img.copyTo(image(cv::Rect(5, rows + 10, cols, rows)));

    // Place F image
    masked_F.copyTo(image(cv::Rect(cols + 15 + colourwheel_size, rows + 10, cols, rows)));

    // Create and place the color wheel
    cv::Mat colourwheel = create_colorwheel(colourwheel_size);
    colourwheel.copyTo(image(cv::Rect(cols + 10, 2 * rows + 10 - colourwheel_size, colourwheel_size, colourwheel_size)));

    // Save the image if required
    if (save && !path.empty()) {
        cv::imwrite(path, image);
    }

    return image;
}

void saveImage(const MatrixXfRowMajor &Image, const std::string &path = "Image.png", const bool Imode = true) {
    cv::Mat grayImage;
    if (Imode){
        grayImage = frame2grayscale(Image);
        cvtColor(grayImage, grayImage, cv::COLOR_GRAY2BGR);
    }
    else{
        grayImage = V2image(Image, 0.1);
    }

    cv::imwrite(path, grayImage);
}

void saveImage(const Tensor3f &Image, const std::string &path = "Image.png") {
    cv::Mat grayImage = vector_field2image(Image);
    cv::imwrite(path, grayImage);
}


/**
 * Creates a colorbar of given height and width representing given max and min values with a specific colormap
 * @param globalMin minimum value of the colorbar
 * @param globalMax maximum value of the colorbar
 * @param height height of the colorbar image
 * @param width width of the colorbar image
 * @param colormapType which colormap to use, default is VIRIDIS
 * @return returns an image of the colormap to be included in plots
 */
cv::Mat createColorbar(double globalMin, double globalMax, int height, int width, int colormapType = cv::COLORMAP_VIRIDIS) {
    // Create a vertical gradient (values from globalMin to globalMax)
    cv::Mat colorbar(height, width, CV_32F);

    // Fill the colorbar with values from globalMin to globalMax
    for (int i = 0; i < height; ++i) {
        float value = globalMin + i * (globalMax - globalMin) / height;
        colorbar.row(i).setTo(value);
    }

    // Normalize the colorbar to the range [0, 255]
    cv::Mat colorbarNormalized;
    colorbar.convertTo(colorbarNormalized, CV_8U, 255.0 / (globalMax - globalMin), -255.0 * globalMin / (globalMax - globalMin));

    // Apply the same colormap to the colorbar
    cv::Mat colorbarColored;
    cv::applyColorMap(colorbarNormalized, colorbarColored, colormapType);

    // Add labels to the colorbar (min, max, and optionally intermediate ticks)
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 0.4;
    int thickness = 1;
    int baseline = 0;

    // Put the min value at the bottom
    std::string minLabel = std::to_string(globalMin);
    cv::Size minTextSize = cv::getTextSize(minLabel, fontFace, fontScale, thickness, &baseline);
    cv::putText(colorbarColored, minLabel, cv::Point(5, height - 5), fontFace, fontScale, cv::Scalar(255, 255, 255), thickness);

    // Put the max value at the top
    std::string maxLabel = std::to_string(globalMax);
    cv::Size maxTextSize = cv::getTextSize(maxLabel, fontFace, fontScale, thickness, &baseline);
    cv::putText(colorbarColored, maxLabel, cv::Point(5, maxTextSize.height), fontFace, fontScale, cv::Scalar(255, 255, 255), thickness);

    // Optionally, you can add intermediate ticks and labels here as needed
    return colorbarColored;
}



/**
 * Plots the Event information (as stand in for the temporal gradient) against the dot product of
 * the spatial gradient G and the optical flow F
 * @param V Event information as Eigen matrix
 * @param F Optical flow as 3Tensor
 * @param G Spatial gradient as 3Tensor
 * @param path where to save the image on the disk if desired
 * @param save if a save to disk is desired
 * @return bgr image of the plot as opencv matrix
 */
cv::Mat plot_VvsFG(const MatrixXfRowMajor &V, const Tensor3f &F, const Tensor3f &G, const std::string &path = "VvsFG", bool save = false){
    long rows = V.rows();
    long cols = V.cols();
    cv::Mat image(rows + 50, cols * 3 + 50, CV_8UC3, cv::Scalar(0, 0, 0));

    // Step 1: Prepare Matrices
    // Convert V to cvMat
    cv::Mat V_img;
    V_img = eigenToCvMat(V);

    // Calculate the dot product of F and G, convert to cvMat
    MatrixXfRowMajor FdotG(rows, cols);
    cv::Mat FdotG_img;
    FdotG = Tensor2Matrix(-(F*G).sum(array<int,1>({2})));
    FdotG_img = eigenToCvMat(FdotG); // Multiply FG elementwise then sum over 3rd dim > dot product

    // Calculate the difference between V and the dot product
    MatrixXfRowMajor diff(rows, cols);
    cv::Mat diff_img;
    diff = V - FdotG;
    diff_img = eigenToCvMat(diff);

    // Step 2: Find the global min and max values across all three matrices
    double minVal1, maxVal1, minVal2, maxVal2, minVal3, maxVal3;
    cv::minMaxLoc(V_img, &minVal1, &maxVal1);
    cv::minMaxLoc(FdotG_img, &minVal2, &maxVal2);
    cv::minMaxLoc(diff_img, &minVal3, &maxVal3);

    double globalMin = std::min({minVal1, minVal2, minVal3});
    double globalMax = std::max({maxVal1, maxVal2, maxVal3});

    // Step 3: Normalize all matrices to the global range [0, 255]
    cv::Mat normalizedMatrix1, normalizedMatrix2, normalizedMatrix3;
    // Normalize each matrix to the range [0, 255] using the global min and max
    V_img.convertTo(normalizedMatrix1, CV_8U, 255.0 / (globalMax - globalMin), -255.0 * globalMin / (globalMax - globalMin));
    FdotG_img.convertTo(normalizedMatrix2, CV_8U, 255.0 / (globalMax - globalMin), -255.0 * globalMin / (globalMax - globalMin));
    diff_img.convertTo(normalizedMatrix3, CV_8U, 255.0 / (globalMax - globalMin), -255.0 * globalMin / (globalMax - globalMin));

    // Step 4: Apply the same colormap to each normalized matrix
    cv::Mat coloredMatrix1, coloredMatrix2, coloredMatrix3;
    cv::applyColorMap(normalizedMatrix1, coloredMatrix1, cv::COLORMAP_VIRIDIS);
    cv::applyColorMap(normalizedMatrix2, coloredMatrix2, cv::COLORMAP_VIRIDIS);
    cv::applyColorMap(normalizedMatrix3, coloredMatrix3, cv::COLORMAP_VIRIDIS);

    // Step 5: Concatenate the three matrices horizontally or vertically for display
    cv::Mat combinedMatrix;
    cv::hconcat(std::vector<cv::Mat>{coloredMatrix1, coloredMatrix2, coloredMatrix3}, combinedMatrix);

    // Step 6: Create a colorbar (we'll use height of the matrix and width of 50 pixels for the colorbar)
    int colorbarWidth = 50;
    cv::Mat colorbar = createColorbar(globalMin, globalMax, combinedMatrix.rows, colorbarWidth);

    // Step 7: Concatenate the colorbar with the combined image
    cv::Mat coloredImage;
    cv::hconcat(combinedMatrix, colorbar, coloredImage);

    // Step 8: Add a black region at the top of the image for the title
    int titleHeight = 50;  // Height of the title area
    cv::Mat titleImage = cv::Mat::zeros(titleHeight, coloredImage.cols, CV_8UC3);

    // Step 9: Add the title text to the top region
    std::string title = "V | - F dot G | Difference";
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 0.5;
    int thickness = 2;
    cv::Scalar color(255, 255, 255); // White text
    int baseline = 0;

    // Get the text size to center the title
    cv::Size textSize = cv::getTextSize(title, fontFace, fontScale, thickness, &baseline);
    cv::Point textOrg((titleImage.cols - textSize.width) / 2, (titleHeight + textSize.height) / 2);

    // Put the title on the titleImage
    cv::putText(titleImage, title, textOrg, fontFace, fontScale, color, thickness);

    // Step 10: Concatenate the title image with the matrix image
    cv::Mat finalImage;
    cv::vconcat(titleImage, coloredImage, finalImage);

    // Step 11: Display the final image with the title
    if (save && !path.empty()) {
        imwrite(path, finalImage);
    }
    else{
        cv::imshow("V vs F dot G", finalImage);
        cv::waitKey(0); // Wait for a key press before closing the window
    }
    return finalImage;
}
