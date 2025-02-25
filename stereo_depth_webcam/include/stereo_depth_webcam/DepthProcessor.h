#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>

// Conditionally include CUDA headers based on compilation flag
#ifdef WITH_CUDA_SUPPORT
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#endif

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
        
        // Method to enable/disable GPU processing
        void setUseGPU(bool use_gpu);
        
    private:
        // CPU-based depth computation
        cv::Mat computeDepthCPU(
            const cv::Mat& left_frame, 
            const cv::Mat& right_frame,
            const cv::Mat& Q_matrix
        );
        
#ifdef WITH_CUDA_SUPPORT
        // GPU-based depth computation - only available with CUDA
        cv::Mat computeDepthGPU(
            const cv::Mat& left_frame, 
            const cv::Mat& right_frame,
            const cv::Mat& Q_matrix
        );
        
        // GPU-based filter methods
        cv::cuda::GpuMat filterDepthMapGPU(const cv::cuda::GpuMat& depth_map_gpu);
#endif
        
        // Helper methods available on all platforms
        cv::Mat createDepthVisualization(const cv::Mat& depth_map);
        cv::Mat createInvalidMask(const cv::Mat& depth_map);
        cv::Mat filterDepthMap(const cv::Mat& depth_map);
        
        // CPU-based components
        cv::Ptr<cv::StereoSGBM> stereo_matcher_cpu_;
        cv::Ptr<cv::StereoMatcher> right_matcher_cpu_;
        cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter_cpu_;
        
#ifdef WITH_CUDA_SUPPORT
        // GPU-based components - only declared when CUDA is available
        cv::Ptr<cv::cuda::StereoSGBM> stereo_matcher_gpu_;
        cv::Ptr<cv::cuda::DisparityBilateralFilter> disparity_filter_gpu_;
        cv::Ptr<cv::cuda::Filter> median_filter_gpu_;
#endif
        
        // Configuration
        ApplicationConfig config_;
        bool is_gpu_available_;
    };
}