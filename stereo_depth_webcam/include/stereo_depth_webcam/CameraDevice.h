#pragma once

#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <stereo_depth_webcam/config.h>
#include <sensor_msgs/msg/camera_info.hpp>

namespace DepthVision {
    class CameraDevice {
    public:
        // Constructor with configuration
        explicit CameraDevice(const ApplicationConfig& config);
        
        // Capture stereo frames
        std::pair<cv::Mat, cv::Mat> captureFrames();
        
        // Apply camera calibration/undistortion
        std::pair<cv::Mat, cv::Mat> undistortFrames(
            const cv::Mat& left_frame, 
            const cv::Mat& right_frame
        );
        
        // Getters for key calibration matrices
        cv::Mat getQMatrix() const { return Q_matrix_; }
        
        // Check if camera is operational
        bool isOpen() const;
        
        // Release camera resources
        void release();
        
        // Generate camera info messages for ROS
        sensor_msgs::msg::CameraInfo getLeftCameraInfo() const;
        sensor_msgs::msg::CameraInfo getRightCameraInfo() const;

    private:
        // Convert JSON matrix to OpenCV matrix
        cv::Mat jsonToMat(const nlohmann::json& json_matrix);
        
        // Convert OpenCV matrix to ROS message
        void matToRosMatrix(const cv::Mat& mat, 
                           double* ros_matrix, 
                           const int rows, 
                           const int cols) const;

        cv::VideoCapture camera_;
        cv::Mat camera_matrix_left_;
        cv::Mat dist_coeffs_left_;
        cv::Mat camera_matrix_right_;
        cv::Mat dist_coeffs_right_;
        cv::Mat Q_matrix_;
        cv::Mat R_matrix_;
        cv::Mat T_matrix_;
        cv::Mat P_left_;
        cv::Mat P_right_;
        
        ApplicationConfig config_;
    };
}