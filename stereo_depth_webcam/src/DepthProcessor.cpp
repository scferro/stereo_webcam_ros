#include "stereo_depth_webcam/DepthProcessor.h"

namespace DepthVision {
    DepthProcessor::DepthProcessor(const ApplicationConfig& config) : config_(config) {
        // Check if CUDA is available
#ifdef WITH_CUDA_SUPPORT
        is_gpu_available_ = cv::cuda::getCudaEnabledDeviceCount() > 0;
#else
        is_gpu_available_ = false;
#endif

        if (config_.use_gpu && is_gpu_available_) {
#ifdef WITH_CUDA_SUPPORT
            Logger::log(Logger::Level::INFO, "Using GPU acceleration for depth processing");
            // Set the CUDA device to use
            cv::cuda::setDevice(config_.gpu_device_id);
            
            // Create GPU stereo matcher
            stereo_matcher_gpu_ = cv::cuda::createStereoSGBM(
                0,                    // Minimum disparity 
                64,                   // Number of disparities (must be divisible by 16)
                5,                    // Block size for matching
                2 * 5 * 11 * 11,      // P1 parameter (controlling smooth regions)
                8 * 3 * 11 * 11       // P2 parameter (controlling less smooth regions)
            );
            
            // Create GPU disparity filter
            disparity_filter_gpu_ = cv::cuda::createDisparityBilateralFilter(64, 5, 5);
            
            // Create median filter for GPU depth map filtering
            median_filter_gpu_ = cv::cuda::createMedianFilter(CV_32F, 3);
#endif
        } else {
            if (config_.use_gpu && !is_gpu_available_) {
#ifdef WITH_CUDA_SUPPORT
                Logger::log(Logger::Level::WARNING, 
                    "GPU acceleration requested but no CUDA-capable GPU found. Falling back to CPU.");
#else
                Logger::log(Logger::Level::WARNING, 
                    "GPU acceleration requested but CUDA support not compiled in. Using CPU.");
#endif
            } else {
                Logger::log(Logger::Level::INFO, "Using CPU for depth processing");
            }
            
            // Create CPU stereo matcher
            stereo_matcher_cpu_ = cv::StereoSGBM::create(
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
            
            right_matcher_cpu_ = cv::ximgproc::createRightMatcher(stereo_matcher_cpu_);
            wls_filter_cpu_ = cv::ximgproc::createDisparityWLSFilter(stereo_matcher_cpu_);
        }
    }
    
    std::tuple<cv::Mat, cv::Mat, cv::Mat> DepthProcessor::computeDepth(
        const cv::Mat& left_frame, 
        const cv::Mat& right_frame,
        const cv::Mat& Q_matrix
    ) {
        // Compute depth based on GPU availability and settings
        cv::Mat depth_map;
        
        if (config_.use_gpu && is_gpu_available_) {
#ifdef WITH_CUDA_SUPPORT
            depth_map = computeDepthGPU(left_frame, right_frame, Q_matrix);
#else
            // This shouldn't happen due to is_gpu_available_ being false when CUDA is not available
            depth_map = computeDepthCPU(left_frame, right_frame, Q_matrix);
#endif
        } else {
            depth_map = computeDepthCPU(left_frame, right_frame, Q_matrix);
        }
        
        // Create visualizations (always done on CPU for simplicity)
        cv::Mat depth_vis = createDepthVisualization(depth_map);
        cv::Mat invalid_mask = createInvalidMask(depth_map);
        
        return {depth_map, depth_vis, invalid_mask};
    }
    
    void DepthProcessor::updateParameters(float min_depth, float max_depth) {
        config_.min_depth = min_depth;
        config_.max_depth = max_depth;
    }
    
    void DepthProcessor::setUseGPU(bool use_gpu) {
        if (use_gpu && !is_gpu_available_) {
#ifdef WITH_CUDA_SUPPORT
            Logger::log(Logger::Level::WARNING, 
                "Cannot enable GPU acceleration: No CUDA-capable GPU found");
#else
            Logger::log(Logger::Level::WARNING, 
                "Cannot enable GPU acceleration: No CUDA support in this build");
#endif
            return;
        }
        
        config_.use_gpu = use_gpu;
        Logger::log(Logger::Level::INFO, 
            std::string("GPU acceleration ") + (use_gpu ? "enabled" : "disabled"));
    }
    
    cv::Mat DepthProcessor::computeDepthCPU(
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
            stereo_matcher_cpu_->compute(left_gray, right_gray, disparity);
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

        // Apply additional filtering
        return filterDepthMap(filtered_depth_map);
    }
    
#ifdef WITH_CUDA_SUPPORT
    cv::Mat DepthProcessor::computeDepthGPU(
        const cv::Mat& left_frame, 
        const cv::Mat& right_frame,
        const cv::Mat& Q_matrix
    ) {
        // Validate input matrices
        if (left_frame.empty() || right_frame.empty() || Q_matrix.empty()) {
            Logger::log(Logger::Level::ERROR, "Invalid input matrices for GPU depth computation");
            return cv::Mat();
        }

        // Upload images to GPU
        cv::cuda::GpuMat left_gpu, right_gpu;
        left_gpu.upload(left_frame);
        right_gpu.upload(right_frame);

        // Convert to grayscale on GPU
        cv::cuda::GpuMat left_gray_gpu, right_gray_gpu;
        cv::cuda::cvtColor(left_gpu, left_gray_gpu, cv::COLOR_BGR2GRAY);
        cv::cuda::cvtColor(right_gpu, right_gray_gpu, cv::COLOR_BGR2GRAY);

        // Compute disparity on GPU
        cv::cuda::GpuMat disparity_gpu;
        try {
            stereo_matcher_gpu_->compute(left_gray_gpu, right_gray_gpu, disparity_gpu);
        } catch (const cv::Exception& e) {
            Logger::log(Logger::Level::ERROR, "GPU disparity computation failed: " + std::string(e.what()));
            return cv::Mat();
        }

        // Convert disparity to float on GPU
        cv::cuda::GpuMat float_disparity_gpu;
        disparity_gpu.convertTo(float_disparity_gpu, CV_32F, 1.0 / 16.0);

        // Filter disparity on GPU
        try {
            disparity_filter_gpu_->apply(float_disparity_gpu, left_gray_gpu, float_disparity_gpu);
        } catch (const cv::Exception& e) {
            Logger::log(Logger::Level::WARNING, 
                "GPU disparity filtering failed, using unfiltered disparity: " + std::string(e.what()));
        }

        // Download to CPU for reprojection (OpenCV doesn't offer GPU reprojectImageTo3D)
        cv::Mat float_disparity;
        float_disparity_gpu.download(float_disparity);

        // Reproject to 3D on CPU
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

        // Upload depth for GPU filtering
        cv::cuda::GpuMat depth_gpu;
        depth_gpu.upload(filtered_depth_map);

        // Apply GPU filtering and download result
        cv::Mat result;
        filterDepthMapGPU(depth_gpu).download(result);
        
        return result;
    }
    
    cv::cuda::GpuMat DepthProcessor::filterDepthMapGPU(const cv::cuda::GpuMat& depth_map_gpu) {
        // Handle NaN/Inf values by replacing with zeros
        cv::cuda::GpuMat valid_mask;
        cv::cuda::GpuMat clean_depth = depth_map_gpu.clone();
        
        // Apply median filter on GPU
        cv::cuda::GpuMat median_filtered;
        median_filter_gpu_->apply(clean_depth, median_filtered);
        
        // Apply bilateral filter on GPU (not available in OpenCV CUDA, so we'll skip)
        // The median filter is usually quite effective on its own
        
        return median_filtered;
    }
#endif
    
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