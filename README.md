# Stereo Depth Webcam with GPU Acceleration

A ROS 2 package for stereo depth sensing using a standard USB webcam with side-by-side stereo images, with optional GPU acceleration using CUDA.

## Overview

This package provides tools for:
- Stereo camera calibration
- Real-time depth map generation (CPU or GPU)
- Publishing RGB, depth, and depth visualization images to ROS topics

## Requirements

- ROS 2 (tested with Humble)
- OpenCV 4.x with CUDA support (for GPU acceleration)
- CUDA Toolkit (for GPU acceleration)
- NVIDIA GPU (for GPU acceleration)
- nlohmann_json
- yaml-cpp

## Installation

Clone this repository into your ROS 2 workspace and build it:

```bash
cd ~/ros2_ws/src
git clone https://github.com/your_username/stereo_depth_webcam.git
cd ..
colcon build --packages-select stereo_depth_webcam
source install/setup.bash
```

### GPU Support

If you want to use GPU acceleration:

1. Make sure you have the CUDA toolkit installed and configured
2. Ensure OpenCV is built with CUDA support
3. Enable GPU acceleration in the configuration (enabled by default)

## Camera Calibration

Before using the stereo camera for depth sensing, you need to calibrate it:

1. Print a checkerboard pattern and measure the size of each square.
2. Run the calibration tool:

```bash
# Using provided script directly
ros2 run stereo_depth_webcam calibration.py --device 0 --width 1280 --height 480 --output ~/ros2_ws/src/stereo_depth_webcam/config/stereo_calibration.json

# Or using the launch file
ros2 launch stereo_depth_webcam calibration.launch.py
```

3. Follow the on-screen instructions, moving the checkerboard to different positions in the camera view.
4. The calibration results will be saved to the specified output file.

## Configuration

Edit the `config/stereo_config.yaml` file to match your camera setup:

```yaml
camera:
  index: 0                    # Camera device index
  width: 640                  # Width of single camera frame
  height: 480                 # Height of camera frame
  frame_rate: 30.0            # Camera frame rate
  calibration_file: "stereo_calibration.json"

depth:
  min_depth: 0.1              # Minimum depth (meters)
  max_depth: 4.0              # Maximum depth (meters)

gpu:
  enabled: true               # Use GPU acceleration if available
  device_id: 0                # CUDA device ID (if multiple GPUs)

ros:
  camera_name: "stereo_camera"  # Base name for camera topics
  rgb_frame_id: "camera_rgb_optical_frame"  # TF frame for RGB image
  depth_frame_id: "camera_depth_optical_frame"  # TF frame for depth image
  publish_rgb: true           # Whether to publish RGB images
  publish_depth: true         # Whether to publish raw depth images
  publish_depth_visual: true  # Whether to publish visualization depth images
```

## Running the Stereo Depth Node

To start the stereo depth processing:

```bash
# Run with default settings
ros2 launch stereo_depth_webcam stereo_depth.launch.py

# Run with specific parameters
ros2 launch stereo_depth_webcam stereo_depth.launch.py use_gpu:=false frame_rate:=15.0
```

## Topics

The node publishes the following topics:

- `/stereo_camera/rgb/image_raw` - RGB image from the left camera
- `/stereo_camera/rgb/camera_info` - Camera information for the RGB image
- `/stereo_camera/depth/image_raw` - Depth map (32-bit float, in meters)
- `/stereo_camera/depth/image_visual` - Colored visualization of the depth map
- `/stereo_camera/depth/camera_info` - Camera information for the depth image

## Viewing the Results

You can use ROS tools to visualize the results:

```bash
# View RGB image
ros2 run rqt_image_view rqt_image_view /stereo_camera/rgb/image_raw

# View depth visualization
ros2 run rqt_image_view rqt_image_view /stereo_camera/depth/image_visual
```

## Parameters

The following parameters can be set via the ROS parameter system:

- `config_file` - Path to the configuration YAML file
- `camera_index` - Camera device index (overrides config)
- `width` - Width of single camera frame (overrides config)
- `height` - Height of camera frame (overrides config)
- `frame_rate` - Frame rate to publish at (overrides config)
- `publish_rgb` - Whether to publish RGB images
- `publish_depth` - Whether to publish raw depth images
- `publish_depth_visual` - Whether to publish visualization depth images
- `use_gpu` - Use GPU acceleration if available
- `gpu_device_id` - CUDA device ID to use

## Performance Considerations

### CPU vs GPU

This package supports both CPU and GPU processing for stereo depth computation:

- **CPU Processing**: Works on any system but may be slower for high resolutions
- **GPU Processing**: Significantly faster but requires CUDA-compatible NVIDIA GPU

The performance difference is especially noticeable at higher resolutions or frame rates. On a modern GPU, you can expect 2-10x faster processing compared to CPU.

### Resolution vs Accuracy

Higher resolutions provide more detailed depth maps but require more processing power. Consider these tradeoffs:

- Low resolution (320x240): Fast but less detailed depth maps
- Medium resolution (640x480): Good balance for most applications
- High resolution (1280x720): Detailed but requires GPU for real-time performance

## License

This package is released under the Apache License 2.0.