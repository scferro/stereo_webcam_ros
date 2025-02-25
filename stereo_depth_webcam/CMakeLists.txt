cmake_minimum_required(VERSION 3.8)
project(stereo_depth_webcam CXX)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Try to find CUDA - optional
find_package(CUDA QUIET)
set(HAS_CUDA FALSE)
if(CUDA_FOUND)
  set(HAS_CUDA TRUE)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++14)
  message(STATUS "CUDA found. Enabling GPU support.")
else()
  message(STATUS "CUDA not found. GPU support will be disabled.")
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
find_package(nlohmann_json REQUIRED)
find_package(CUDA 12.0 REQUIRED)

# Find OpenCV - first try with CUDA components, then fall back to standard
if(HAS_CUDA)
  find_package(OpenCV QUIET COMPONENTS core imgproc highgui features2d cudawarping cudafilters cudastereo cudaimgproc cudaarithm)
  if(NOT OpenCV_FOUND OR NOT OpenCV_CUDA_VERSION)
    set(HAS_CUDA FALSE)
    message(STATUS "OpenCV CUDA modules not found. GPU support will be disabled.")
    find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui features2d)
  else()
    message(STATUS "OpenCV CUDA modules found. GPU support enabled.")
  endif()
else()
  find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui features2d)
endif()

# Define compilation options for GPU support
if(HAS_CUDA)
  add_definitions(-DWITH_CUDA_SUPPORT)
  message(STATUS "Building with CUDA support.")
else()
  message(STATUS "Building without CUDA support.")
endif()

# Include directories
include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIR}
)

if(HAS_CUDA)
  include_directories(${CUDA_INCLUDE_DIRS})
endif()

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
  nlohmann_json::nlohmann_json
)

if(HAS_CUDA)
  target_link_libraries(stereo_depth_node ${CUDA_LIBRARIES})
endif()

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