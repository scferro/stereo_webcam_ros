#include "stereo_depth_webcam/DepthProcessor.h"

namespace DepthVision {
    DepthProcessor::DepthProcessor(const ApplicationConfig& config) : config_(config) {
        // Create stereo matcher with detailed configuration
        stereo_matcher_ = cv::StereoSGBM::create(
            0,                    // Minimum disparity 
            64,                   // Number of disparities (must be divisible by 16)
            5,                    // Block size for matching
            2 * 5 * 11 * 11,      // P1 parameter (controlling smooth regions)
            8 * 3 * 11 * 11,      // P2 parameter (controlling less smooth regions)
            1,                    // Maximum allowed difference in disparity 
            63,                   // Prefilter cap for normalization
            10,                   // Uniqueness ratio for matched pixels
            200,                  // Speckle window size to filter out noise
            8                     // Speckle range for noise filtering
        );
        
        right_matcher_ = cv::ximgproc::createRightMatcher(stereo_matcher_);
        wls_filter_ = cv::ximgproc::createDisparityWLSFilter(stereo_matcher_);
    }
    
    std::tuple<cv::Mat, cv::Mat, cv::Mat> DepthProcessor::computeDepth(
        const cv::Mat& left_frame, 
        const cv::Mat& right_frame,
        const cv::Mat& Q_matrix
    ) {
        // Compute advanced depth using Q matrix
        cv::Mat depth_map = computeAdvancedDepth(left_frame, right_frame, Q_matrix);
        
        // Apply additional filtering
        depth_map = filterDepthMap(depth_map);
        
        // Create visualizations
        cv::Mat depth_vis = createDepthVisualization(depth_map);
        cv::Mat invalid_mask = createInvalidMask(depth_map);
        
        return {depth_map, depth_vis, invalid_mask};
    }
    
    void DepthProcessor::updateParameters(float min_depth, float max_depth) {
        config_.min_depth = min_depth;
        config_.max_depth = max_depth;
    }
    
    cv::Mat DepthProcessor::computeAdvancedDepth(
        const cv::Mat& left_frame, 
        const cv::Mat& right_frame,
        const cv::Mat& Q_matrix
    ) {
        // Validate input matrices
        if (left_frame.empty() || right_frame.empty() || Q_matrix.empty()) {
            Logger::log(Logger::Level::ERROR, "Invalid input matrices for depth computation");
            return cv::Mat();
        }

        // Convert to grayscale
        cv::Mat left_gray, right_gray;
        cv::cvtColor(left_frame, left_gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(right_frame, right_gray, cv::COLOR_BGR2GRAY);

        // Compute disparity
        cv::Mat disparity;
        try {
            stereo_matcher_->compute(left_gray, right_gray, disparity);
        } catch (const cv::Exception& e) {
            Logger::log(Logger::Level::ERROR, "Disparity computation failed: " + std::string(e.what()));
            return cv::Mat();
        }

        // Convert disparity to float
        cv::Mat float_disparity;
        disparity.convertTo(float_disparity, CV_32F, 1.0 / 16.0);

        // Reproject to 3D
        cv::Mat points4D;
        try {
            cv::reprojectImageTo3D(float_disparity, points4D, Q_matrix);
        } catch (const cv::Exception& e) {
            Logger::log(Logger::Level::ERROR, "3D reprojection failed: " + std::string(e.what()));
            return cv::Mat();
        }

        // Extract depth (Z coordinate)
        cv::Mat depth_channels[3];
        cv::split(points4D, depth_channels);
        cv::Mat depth_map = depth_channels[2];  // Z coordinate

        // Filter out extreme depths
        cv::Mat mask;
        cv::inRange(depth_map, config_.min_depth, config_.max_depth, mask);
        
        cv::Mat filtered_depth_map;
        depth_map.copyTo(filtered_depth_map, mask);

        return filtered_depth_map;
    }
    
    cv::Mat DepthProcessor::filterDepthMap(const cv::Mat& depth_map) {
        // Ensure the depth map is in 32-bit floating-point format
        cv::Mat filtered_depth;
        if (depth_map.type() != CV_32F) {
            depth_map.convertTo(filtered_depth, CV_32F);
        } else {
            filtered_depth = depth_map.clone();
        }
        
        // Median filtering to remove noise
        cv::Mat median_filtered;
        cv::medianBlur(filtered_depth, median_filtered, 3);
        
        // Bilateral filtering with explicit checks
        try {
            cv::Mat bilateral_filtered;
            cv::bilateralFilter(median_filtered, bilateral_filtered, 7, 75, 75);
            return bilateral_filtered;
        } catch (const cv::Exception& e) {
            // If bilateral filtering fails, return median filtered image
            Logger::log(Logger::Level::WARNING, 
                "Bilateral filtering failed. Returning median filtered depth map.");
            return median_filtered;
        }
    }
    
    cv::Mat DepthProcessor::createDepthVisualization(const cv::Mat& depth_map) {
        // Normalize depth map to 0-255 range for visualization
        cv::Mat normalized_depth;
        cv::normalize(depth_map, normalized_depth, 0, 255, cv::NORM_MINMAX, CV_8U);
        
        // Apply color map for intuitive depth representation
        cv::Mat depth_color;
        cv::applyColorMap(normalized_depth, depth_color, cv::COLORMAP_HOT);
        
        return depth_color;
    }
    
    cv::Mat DepthProcessor::createInvalidMask(const cv::Mat& depth_map) {
        // Create a mask for depth values outside valid range
        cv::Mat mask;
        cv::inRange(depth_map, 
            config_.min_depth, 
            config_.max_depth, 
            mask
        );
        return mask;
    }
}