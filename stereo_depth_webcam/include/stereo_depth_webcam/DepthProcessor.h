#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <stereo_depth_webcam/config.h>

namespace DepthVision {
    class DepthProcessor {
    public:
        DepthProcessor(const ApplicationConfig& config);
        
        // Updated method signature to include Q matrix
        std::tuple<cv::Mat, cv::Mat, cv::Mat> computeDepth(
            const cv::Mat& left_frame, 
            const cv::Mat& right_frame,
            const cv::Mat& Q_matrix
        );
        
        // Method to update parameters at runtime
        void updateParameters(float min_depth, float max_depth);
        
    private:
        // Advanced depth computation methods
        cv::Mat computeAdvancedDepth(
            const cv::Mat& left_frame, 
            const cv::Mat& right_frame,
            const cv::Mat& Q_matrix
        );
        
        cv::Mat createDepthVisualization(const cv::Mat& depth_map);
        cv::Mat createInvalidMask(const cv::Mat& depth_map);
        cv::Mat addDepthScaleBar(const cv::Mat& depth_image);
        cv::Mat filterDepthMap(const cv::Mat& depth_map);
        
        // Stereo matching parameters
        cv::Ptr<cv::StereoSGBM> stereo_matcher_;
        cv::Ptr<cv::StereoMatcher> right_matcher_;
        cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter_;
        ApplicationConfig config_;
    };
}