#pragma once

#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <yaml-cpp/yaml.h>

namespace DepthVision {
    // Forward declaration of logger
    class Logger {
    public:
        enum class Level { INFO, WARNING, ERROR };
        
        static void log(Level level, const std::string& message);
    };

    // Application configuration structure
    struct ApplicationConfig {
        // Configuration parameters with default values
        int camera_index = 0;
        cv::Size combined_size{1280, 480};
        cv::Size single_size{640, 480};
        
        // Depth estimation parameters
        float min_depth = 0.1f;
        float max_depth = 4.0f;
        
        // Visualization flags
        bool show_depth = true;
        bool show_raw = true;
        
        // Frame rate control
        double frame_rate = 30.0;
        
        // ROS parameters
        std::string camera_name = "stereo_camera";
        std::string rgb_frame_id = "camera_rgb_optical_frame";
        std::string depth_frame_id = "camera_depth_optical_frame";
        
        // Calibration file path
        std::string calibration_path = "";
        
        // Path to the loaded config file (for reference to find calibration file)
        std::string config_file_path = "";
        
        // JSON storage for calibration data
        nlohmann::json calibration_data;

        // Method to load calibration
        bool loadCalibration() {
            try {
                // If calibration_path doesn't contain a full path, assume it's in the same directory as the config file
                std::string full_calibration_path = calibration_path;
                if (calibration_path.find('/') == std::string::npos) {
                    // Extract directory from the loaded config file path
                    std::filesystem::path config_dir = std::filesystem::path(config_file_path).parent_path();
                    full_calibration_path = (config_dir / calibration_path).string();
                    Logger::log(Logger::Level::INFO, 
                        "Looking for calibration file at: " + full_calibration_path);
                }
                
                // Open the calibration file
                std::ifstream file(full_calibration_path);
                
                // Check if file is open
                if (!file.is_open()) {
                    Logger::log(Logger::Level::ERROR, 
                        "Failed to open calibration file: " + full_calibration_path);
                    return false;
                }

                // Parse JSON file
                calibration_data = nlohmann::json::parse(file);
                
                // Validate key sections
                if (!calibration_data.contains("left_camera") || 
                    !calibration_data.contains("right_camera") ||
                    !calibration_data.contains("stereo")) {
                    Logger::log(Logger::Level::ERROR, 
                        "Invalid calibration file structure");
                    return false;
                }

                // Update single size from calibration if available
                if (calibration_data.contains("resolution") && 
                    calibration_data["resolution"].contains("single")) {
                    auto& res = calibration_data["resolution"]["single"];
                    single_size = cv::Size(res[0], res[1]);
                    combined_size = cv::Size(single_size.width * 2, single_size.height);
                }

                return true;
            } catch (const std::exception& e) {
                Logger::log(Logger::Level::ERROR, 
                    "Calibration loading error: " + std::string(e.what()));
                return false;
            }
        }
        
        // Load configuration from YAML file
        bool loadConfig(const std::string& config_path) {
            try {
                // Store the config file path for reference
                config_file_path = config_path;
                
                // Load YAML file
                YAML::Node config = YAML::LoadFile(config_path);
                
                // Camera parameters
                if (config["camera"]) {
                    if (config["camera"]["index"])
                        camera_index = config["camera"]["index"].as<int>();
                    
                    if (config["camera"]["width"] && config["camera"]["height"]) {
                        int width = config["camera"]["width"].as<int>();
                        int height = config["camera"]["height"].as<int>();
                        single_size = cv::Size(width, height);
                        combined_size = cv::Size(width * 2, height);
                    }
                    
                    if (config["camera"]["frame_rate"])
                        frame_rate = config["camera"]["frame_rate"].as<double>();
                    
                    if (config["camera"]["calibration_file"])
                        calibration_path = config["camera"]["calibration_file"].as<std::string>();
                }
                
                // Depth parameters
                if (config["depth"]) {
                    if (config["depth"]["min_depth"])
                        min_depth = config["depth"]["min_depth"].as<float>();
                    
                    if (config["depth"]["max_depth"])
                        max_depth = config["depth"]["max_depth"].as<float>();
                }
                
                // ROS parameters
                if (config["ros"]) {
                    if (config["ros"]["camera_name"])
                        camera_name = config["ros"]["camera_name"].as<std::string>();
                    
                    if (config["ros"]["rgb_frame_id"])
                        rgb_frame_id = config["ros"]["rgb_frame_id"].as<std::string>();
                    
                    if (config["ros"]["depth_frame_id"])
                        depth_frame_id = config["ros"]["depth_frame_id"].as<std::string>();
                }
                
                // Display parameters
                if (config["display"]) {
                    if (config["display"]["show_depth"])
                        show_depth = config["display"]["show_depth"].as<bool>();
                    
                    if (config["display"]["show_raw"])
                        show_raw = config["display"]["show_raw"].as<bool>();
                }
                
                Logger::log(Logger::Level::INFO, "Configuration loaded successfully");
                return true;
                
            } catch (const std::exception& e) {
                Logger::log(Logger::Level::ERROR, 
                    "Configuration loading error: " + std::string(e.what()));
                return false;
            }
        }
    };

    // Implement Logger method outside the class declaration
    inline void Logger::log(Level level, const std::string& message) {
        switch (level) {
            case Level::INFO:
                std::cout << "[INFO] " << message << std::endl;
                break;
            case Level::WARNING:
                std::cerr << "[WARNING] " << message << std::endl;
                break;
            case Level::ERROR:
                std::cerr << "[ERROR] " << message << std::endl;
                break;
        }
    }
}