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
        this->declare_parameter("camera_index", -1);
        this->declare_parameter("width", -1);
        this->declare_parameter("height", -1);
        this->declare_parameter("frame_rate", -1.0);
        this->declare_parameter("publish_rgb", true);
        this->declare_parameter("publish_depth", true);
        this->declare_parameter("publish_depth_visual", true);
        this->declare_parameter("use_gpu", true);
        this->declare_parameter("gpu_device_id", 0);
        
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
        
        // Override with command-line parameters if provided
        int camera_index = this->get_parameter("camera_index").as_int();
        int width = this->get_parameter("width").as_int();
        int height = this->get_parameter("height").as_int();
        double frame_rate = this->get_parameter("frame_rate").as_double();
        
        if (camera_index >= 0) {
            RCLCPP_INFO(this->get_logger(), "Overriding camera index: %d", camera_index);
            config_->camera_index = camera_index;
        }
        
        if (width > 0 && height > 0) {
            RCLCPP_INFO(this->get_logger(), "Overriding resolution: %dx%d", width, height);
            config_->single_size = cv::Size(width, height);
            config_->combined_size = cv::Size(width * 2, height);
        }
        
        if (frame_rate > 0) {
            RCLCPP_INFO(this->get_logger(), "Overriding frame rate: %.1f FPS", frame_rate);
            config_->frame_rate = frame_rate;
        }
        
        // Override publish control flags
        config_->publish_rgb = this->get_parameter("publish_rgb").as_bool();
        config_->publish_depth = this->get_parameter("publish_depth").as_bool();
        config_->publish_depth_visual = this->get_parameter("publish_depth_visual").as_bool();
        
        // Override GPU settings
        config_->use_gpu = this->get_parameter("use_gpu").as_bool();
        config_->gpu_device_id = this->get_parameter("gpu_device_id").as_int();
        
        RCLCPP_INFO(this->get_logger(), "Publishing RGB: %s", config_->publish_rgb ? "true" : "false");
        RCLCPP_INFO(this->get_logger(), "Publishing Depth: %s", config_->publish_depth ? "true" : "false");
        RCLCPP_INFO(this->get_logger(), "Publishing Depth Visual: %s", config_->publish_depth_visual ? "true" : "false");
        RCLCPP_INFO(this->get_logger(), "GPU Acceleration: %s", config_->use_gpu ? "enabled" : "disabled");
        
        // Check CUDA availability
#ifdef WITH_CUDA_SUPPORT
        int cuda_devices = cv::cuda::getCudaEnabledDeviceCount();
        if (config_->use_gpu) {
            if (cuda_devices > 0) {
                RCLCPP_INFO(this->get_logger(), "Found %d CUDA-capable device(s)", cuda_devices);
                
                // Print CUDA device info
                if (config_->gpu_device_id < cuda_devices) {
                    cv::cuda::DeviceInfo device_info(config_->gpu_device_id);
                    RCLCPP_INFO(this->get_logger(), "Using GPU device %d: %s", 
                        config_->gpu_device_id, device_info.name().c_str());
                    RCLCPP_INFO(this->get_logger(), "  Compute capability: %d.%d", 
                        device_info.majorVersion(), device_info.minorVersion());
                    RCLCPP_INFO(this->get_logger(), "  Total memory: %.2f GB", 
                        device_info.totalMemory() / (1024.0 * 1024.0 * 1024.0));
                } else {
                    RCLCPP_WARN(this->get_logger(), "Invalid GPU device ID %d. Using device 0 instead.",
                        config_->gpu_device_id);
                    config_->gpu_device_id = 0;
                }
            } else {
                RCLCPP_WARN(this->get_logger(), "No CUDA-capable GPU found. Falling back to CPU processing.");
                config_->use_gpu = false;
            }
        }
#else
        if (config_->use_gpu) {
            RCLCPP_WARN(this->get_logger(), "CUDA support not compiled in. GPU acceleration is unavailable.");
            config_->use_gpu = false;
        }
#endif
        
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
        if (config_->publish_rgb) {
            rgb_pub_ = image_transport::create_publisher(this, config_->camera_name + "/rgb/image_raw");
            left_info_pub_ = this->create_publisher<sensor_msgs::msg::CameraInfo>(
                config_->camera_name + "/rgb/camera_info", 10);
        }
        
        if (config_->publish_depth) {
            depth_pub_ = image_transport::create_publisher(this, config_->camera_name + "/depth/image_raw");
            depth_info_pub_ = this->create_publisher<sensor_msgs::msg::CameraInfo>(
                config_->camera_name + "/depth/camera_info", 10);
        }
        
        if (config_->publish_depth_visual) {
            depth_visual_pub_ = image_transport::create_publisher(this, config_->camera_name + "/depth/image_visual");
        }
        
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
        if (config_->publish_rgb) {
            sensor_msgs::msg::Image::SharedPtr rgb_msg = 
                cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", left_frame).toImageMsg();
            rgb_msg->header.stamp = current_time;
            rgb_msg->header.frame_id = config_->rgb_frame_id;
            rgb_pub_.publish(rgb_msg);
            
            // Publish left camera info
            left_info_pub_->publish(left_info);
        }
        
        // Publish depth image if valid
        if (!depth_map.empty() && (config_->publish_depth || config_->publish_depth_visual)) {
            // Set depth camera info frame ID
            right_info.header.frame_id = config_->depth_frame_id;
            
            if (config_->publish_depth) {
                // Convert to ROS format (meters, 32FC1)
                sensor_msgs::msg::Image::SharedPtr depth_msg = 
                    cv_bridge::CvImage(std_msgs::msg::Header(), "32FC1", depth_map).toImageMsg();
                depth_msg->header.stamp = current_time;
                depth_msg->header.frame_id = config_->depth_frame_id;
                depth_pub_.publish(depth_msg);
                
                // Publish depth camera info
                depth_info_pub_->publish(right_info);
            }
            
            // Publish visualization
            if (config_->publish_depth_visual && !depth_vis.empty()) {
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