#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>
#include <stereo_depth_webcam/CameraDevice.h>
#include <stereo_depth_webcam/DepthProcessor.h>
#include <stereo_depth_webcam/config.h>
#include <filesystem>

class StereoDepthNode : public rclcpp::Node {
public:
    StereoDepthNode() : Node("stereo_depth_node") {
        // Declare parameters
        this->declare_parameter("config_file", "");
        
        // Get config file path
        std::string config_file = this->get_parameter("config_file").as_string();
        if (config_file.empty()) {
            RCLCPP_ERROR(this->get_logger(), "No configuration file specified. Use --ros-args -p config_file:=path/to/config.yaml");
            throw std::runtime_error("Configuration file required");
        }
        
        // Make sure config file exists and is readable
        if (!std::filesystem::exists(config_file)) {
            RCLCPP_ERROR(this->get_logger(), "Configuration file not found: %s", config_file.c_str());
            throw std::runtime_error("Configuration file not found");
        }
        
        RCLCPP_INFO(this->get_logger(), "Loading configuration from: %s", config_file.c_str());
        
        // Load configuration
        config_ = std::make_shared<DepthVision::ApplicationConfig>();
        if (!config_->loadConfig(config_file)) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load configuration");
            throw std::runtime_error("Configuration loading failed");
        }
        
        // Set up ROS logging for DepthVision logger
        auto logger = this->get_logger();
        auto log_callback = [logger](DepthVision::Logger::Level level, const std::string& message) {
            switch (level) {
                case DepthVision::Logger::Level::INFO:
                    RCLCPP_INFO(logger, "%s", message.c_str());
                    break;
                case DepthVision::Logger::Level::WARNING:
                    RCLCPP_WARN(logger, "%s", message.c_str());
                    break;
                case DepthVision::Logger::Level::ERROR:
                    RCLCPP_ERROR(logger, "%s", message.c_str());
                    break;
            }
        };
        
        // Initialize the camera device
        try {
            camera_ = std::make_unique<DepthVision::CameraDevice>(*config_);
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Camera initialization failed: %s", e.what());
            throw;
        }
        
        // Initialize the depth processor
        processor_ = std::make_unique<DepthVision::DepthProcessor>(*config_);
        
        // Create publishers
        rgb_pub_ = image_transport::create_publisher(this, config_->camera_name + "/rgb/image_raw");
        depth_pub_ = image_transport::create_publisher(this, config_->camera_name + "/depth/image_raw");
        depth_visual_pub_ = image_transport::create_publisher(this, config_->camera_name + "/depth/image_visual");
        
        // Camera info publishers
        left_info_pub_ = this->create_publisher<sensor_msgs::msg::CameraInfo>(
            config_->camera_name + "/rgb/camera_info", 10);
        depth_info_pub_ = this->create_publisher<sensor_msgs::msg::CameraInfo>(
            config_->camera_name + "/depth/camera_info", 10);
        
        // Create timer for frame capture
        double frame_period = 1.0 / config_->frame_rate;
        timer_ = this->create_wall_timer(
            std::chrono::duration<double>(frame_period),
            std::bind(&StereoDepthNode::captureCallback, this));
        
        RCLCPP_INFO(this->get_logger(), "Stereo depth node initialized");
        RCLCPP_INFO(this->get_logger(), "Publishing at %.1f FPS", config_->frame_rate);
    }
    
    ~StereoDepthNode() {
        if (camera_) {
            camera_->release();
        }
    }
    
private:
    void captureCallback() {
        // Check if camera is still open
        if (!camera_ || !camera_->isOpen()) {
            RCLCPP_ERROR(this->get_logger(), "Camera not available");
            return;
        }
        
        // Capture stereo frames
        auto [left_frame, right_frame] = camera_->captureFrames();
        
        // Validate frame capture
        if (left_frame.empty() || right_frame.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to capture frames");
            return;
        }
        
        // Undistort frames if calibration is available
        std::tie(left_frame, right_frame) = camera_->undistortFrames(left_frame, right_frame);
        
        // Compute depth using Q matrix from camera calibration
        auto [depth_map, depth_vis, invalid_mask] = 
            processor_->computeDepth(left_frame, right_frame, camera_->getQMatrix());
        
        // Create and publish messages
        auto current_time = this->now();
        
        // Prepare camera info messages
        auto left_info = camera_->getLeftCameraInfo();
        auto right_info = camera_->getRightCameraInfo();
        
        // Update timestamps
        left_info.header.stamp = current_time;
        right_info.header.stamp = current_time;
        
        // Publish RGB image
        sensor_msgs::msg::Image::SharedPtr rgb_msg = 
            cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", left_frame).toImageMsg();
        rgb_msg->header.stamp = current_time;
        rgb_msg->header.frame_id = config_->rgb_frame_id;
        rgb_pub_.publish(rgb_msg);
        
        // Publish left camera info
        left_info_pub_->publish(left_info);
        
        // Publish depth image if valid
        if (!depth_map.empty()) {
            // Convert to ROS format (meters, 32FC1)
            sensor_msgs::msg::Image::SharedPtr depth_msg = 
                cv_bridge::CvImage(std_msgs::msg::Header(), "32FC1", depth_map).toImageMsg();
            depth_msg->header.stamp = current_time;
            depth_msg->header.frame_id = config_->depth_frame_id;
            depth_pub_.publish(depth_msg);
            
            // Publish depth camera info (same as left camera info but with depth frame_id)
            right_info.header.frame_id = config_->depth_frame_id;
            depth_info_pub_->publish(right_info);
            
            // Publish visualization
            if (!depth_vis.empty()) {
                sensor_msgs::msg::Image::SharedPtr depth_vis_msg = 
                    cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", depth_vis).toImageMsg();
                depth_vis_msg->header.stamp = current_time;
                depth_vis_msg->header.frame_id = config_->depth_frame_id;
                depth_visual_pub_.publish(depth_vis_msg);
            }
        }
    }
    
    // ROS publishers
    image_transport::Publisher rgb_pub_;
    image_transport::Publisher depth_pub_;
    image_transport::Publisher depth_visual_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr left_info_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr depth_info_pub_;
    
    // Timer for regular frame capture
    rclcpp::TimerBase::SharedPtr timer_;
    
    // Configuration and components
    std::shared_ptr<DepthVision::ApplicationConfig> config_;
    std::unique_ptr<DepthVision::CameraDevice> camera_;
    std::unique_ptr<DepthVision::DepthProcessor> processor_;
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    
    try {
        auto node = std::make_shared<StereoDepthNode>();
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("stereo_depth_node"), "Exception: %s", e.what());
        return 1;
    }
    
    rclcpp::shutdown();
    return 0;
}