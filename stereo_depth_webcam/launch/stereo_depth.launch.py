from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare arguments
    declared_arguments = [
        DeclareLaunchArgument(
            'config_file',
            default_value=PathJoinSubstitution([
                FindPackageShare('stereo_depth_webcam'),
                'config',
                'stereo_config.yaml'
            ]),
            description='Path to config file'
        ),
        DeclareLaunchArgument(
            'camera_index',
            default_value='-1',
            description='Camera device index (overrides config file if >= 0)'
        ),
        DeclareLaunchArgument(
            'width',
            default_value='-1',
            description='Image width (overrides config file if > 0)'
        ),
        DeclareLaunchArgument(
            'height',
            default_value='-1',
            description='Image height (overrides config file if > 0)'
        ),
        DeclareLaunchArgument(
            'frame_rate',
            default_value='-1.0',
            description='Frame rate (overrides config file if > 0)'
        ),
        DeclareLaunchArgument(
            'publish_rgb',
            default_value='true',
            description='Whether to publish RGB images'
        ),
        DeclareLaunchArgument(
            'publish_depth',
            default_value='true',
            description='Whether to publish raw depth images'
        ),
        DeclareLaunchArgument(
            'publish_depth_visual',
            default_value='true',
            description='Whether to publish visualization depth images'
        ),
        DeclareLaunchArgument(
            'use_gpu',
            default_value='true',
            description='Use GPU acceleration if available'
        ),
        DeclareLaunchArgument(
            'gpu_device_id',
            default_value='0',
            description='CUDA device ID to use'
        ),
    ]

    # Get the values of the arguments
    config_file = LaunchConfiguration('config_file')
    camera_index = LaunchConfiguration('camera_index')
    width = LaunchConfiguration('width')
    height = LaunchConfiguration('height')
    frame_rate = LaunchConfiguration('frame_rate')
    publish_rgb = LaunchConfiguration('publish_rgb')
    publish_depth = LaunchConfiguration('publish_depth')
    publish_depth_visual = LaunchConfiguration('publish_depth_visual')
    use_gpu = LaunchConfiguration('use_gpu')
    gpu_device_id = LaunchConfiguration('gpu_device_id')

    # Define our nodes
    stereo_depth_node = Node(
        package='stereo_depth_webcam',
        executable='stereo_depth_node',
        name='stereo_depth_node',
        parameters=[{
            'config_file': config_file,
            'camera_index': camera_index,
            'width': width,
            'height': height,
            'frame_rate': frame_rate,
            'publish_rgb': publish_rgb,
            'publish_depth': publish_depth,
            'publish_depth_visual': publish_depth_visual,
            'use_gpu': use_gpu,
            'gpu_device_id': gpu_device_id
        }],
        output='screen'
    )

    # Create the launch description and populate
    ld = LaunchDescription(declared_arguments)
    
    # Add the nodes to the launch description
    ld.add_action(stereo_depth_node)
    
    return ld