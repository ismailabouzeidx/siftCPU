#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>

// Function to calculate kernel size based on sigma
int calculate_kernel_size(double sigma) {
    int size = std::ceil(6 * sigma);
    return (size % 2 == 0) ? size + 1 : size; // Ensure kernel size is odd
}

// Function to apply Gaussian blur to an image
cv::Mat apply_gaussian_blur(const cv::Mat& image, double sigma) {
    int kernel_size = calculate_kernel_size(sigma);
    cv::Mat blurred_image;
    cv::GaussianBlur(image, blurred_image, cv::Size(kernel_size, kernel_size), sigma);
    return blurred_image;
}

// Function to calculate the sigma for each level
double calculate_sigma(double initial_sigma, int levels_per_octave, int level) {
    double k = std::pow(2, 1.0 / levels_per_octave); // Scale factor
    return initial_sigma * std::pow(k, level);
}

std::vector<cv::KeyPoint> detect_keypoints(const std::vector<std::vector<cv::Mat>>& DoG, int border_width = 1) {
    std::vector<cv::KeyPoint> keypoints;

    for (int octave = 0; octave < DoG.size(); octave++) {
        for (int level = 1; level < DoG[octave].size() - 1; level++) { // Avoid the first and last levels
            const cv::Mat& prev = DoG[octave][level - 1];
            const cv::Mat& current = DoG[octave][level];
            const cv::Mat& next = DoG[octave][level + 1];

            for (int y = border_width; y < current.rows - border_width; y++) {
                for (int x = border_width; x < current.cols - border_width; x++) {
                    float value = current.at<float>(y, x);

                    // Check if the pixel is a local extremum
                    bool is_extremum = true;

                    // Compare with neighbors in the previous, current, and next levels
                    for (int dy = -1; dy <= 1; dy++) {
                        for (int dx = -1; dx <= 1; dx++) {
                            if (dx == 0 && dy == 0) continue; // Skip the center pixel
                            if (value < prev.at<float>(y + dy, x + dx) || value > prev.at<float>(y + dy, x + dx) ||
                                value < current.at<float>(y + dy, x + dx) || value > current.at<float>(y + dy, x + dx) ||
                                value < next.at<float>(y + dy, x + dx) || value > next.at<float>(y + dy, x + dx)) {
                                is_extremum = false;
                                break;
                            }
                        }
                        if (!is_extremum) break;
                    }

                    if (is_extremum) {
                        // Create a keypoint and add to the list
                        keypoints.emplace_back(cv::KeyPoint(
                            x, y, 1.0f, -1, value, octave * DoG[octave].size() + level
                        ));
                    }
                }
            }
        }
    }

    return keypoints;
}

int main() {
    // Load the image
    cv::Mat image = cv::imread("C:/Users/User/Desktop/box.png", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error loading image!" << std::endl;
        return -1;
    }

    // OpenCV SIFT Parameters
    double initial_sigma = 1.6;     // Initial sigma for SIFT
    int levels_per_octave = 4;     // Number of levels per octave
    int total_levels = levels_per_octave + 3; // Total levels (3 + 3 extra)
    int num_octaves = 4;           // Total number of octaves

    // Pre-blur the base image
    cv::Mat base = apply_gaussian_blur(image, std::sqrt(initial_sigma * initial_sigma - 0.5 * 0.5));

    // Gaussian Pyramid
    std::vector<std::vector<cv::Mat>> octaves;

    for (int octave = 0; octave < num_octaves; octave++) {
        std::vector<cv::Mat> levels;
        cv::Mat current_image = base.clone(); // Start with the base image for this octave
        double prev_sigma = 0;

        for (int level = 0; level < total_levels; level++) {
            double current_sigma = calculate_sigma(initial_sigma, levels_per_octave, level);
            double sigma_diff = std::sqrt(current_sigma * current_sigma - prev_sigma * prev_sigma);

            // Apply Gaussian blur incrementally
            cv::Mat blurred_image = apply_gaussian_blur(current_image, sigma_diff);
            levels.push_back(blurred_image);

            // Use the current blurred image as the input for the next level
            current_image = blurred_image;
            prev_sigma = current_sigma;

            // Debug: Show each level
            std::cout << "Octave: " << octave << ", Level: " << level
                << ", Sigma: " << current_sigma << std::endl;
        }

        octaves.push_back(levels); // Store the levels for this octave

        // Downsample the last blurred image for the next octave
        cv::resize(current_image, base, cv::Size(current_image.cols / 2, current_image.rows / 2), 0, 0, cv::INTER_LINEAR);
    }


    // DoG formulation
    std::vector<std::vector<cv::Mat>> DoG(octaves.size()); // Initialize DoG vector with the same number of octaves as the Gaussian Pyramid
    for (int octave = 0; octave < octaves.size(); octave++) {
        for (int level = 0; level < octaves[octave].size() - 1; level++) {
            cv::Mat DoG_diff;
            cv::subtract(octaves[octave][level + 1], octaves[octave][level], DoG_diff); // Compute DoG: G_{i+1} - G_{i}
            DoG[octave].push_back(DoG_diff);
            cv::imshow("window_name", DoG_diff);
            cv::waitKey(0);
        }
    }

    
    return 0;
}
