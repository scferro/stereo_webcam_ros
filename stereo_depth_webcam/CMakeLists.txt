cmake_minimum_required(VERSION 3.8)
project(stereo_depth_webcam)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(camera_info_manager REQUIRED)
find_package(stereo_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenCV REQUIRED)

# Include directories
include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIR}
)

# C++ node executable
add_executable(stereo_depth_node 
  src/stereo_depth_node.cpp
  src/CameraDevice.cpp
  src/DepthProcessor.cpp
)

ament_target_dependencies(stereo_depth_node
  rclcpp
  sensor_msgs
  std_msgs
  cv_bridge
  image_transport
  camera_info_manager
  stereo_msgs
  message_filters
  geometry_msgs
  OpenCV
)

target_link_libraries(stereo_depth_node
  ${OpenCV_LIBRARIES}
  yaml-cpp
)

# Install C++ executables
install(TARGETS
  stereo_depth_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install Python scripts directly (not as a Python package)
install(PROGRAMS
  scripts/calibration.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

# Install config files
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
)

# Install include files
install(DIRECTORY
  include/
  DESTINATION include/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()