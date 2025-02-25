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
    ]

    # Get the values of the arguments
    config_file = LaunchConfiguration('config_file')

    # Define our nodes
    stereo_depth_node = Node(
        package='stereo_depth_webcam',
        executable='stereo_depth_node',
        name='stereo_depth_node',
        parameters=[
            {'config_file': config_file}
        ],
        output='screen'
    )

    # Create the launch description and populate
    ld = LaunchDescription(declared_arguments)
    
    # Add the nodes to the launch description
    ld.add_action(stereo_depth_node)
    
    return ld