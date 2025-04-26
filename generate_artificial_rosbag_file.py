import os
import shutil
import math
import random
import numpy as np
import rclpy
import rosbag2_py
from std_msgs.msg import Header, Float64
from geometry_msgs.msg import Pose, Twist, PoseStamped
from nav_msgs.msg import Odometry
import rclpy.serialization
from rclpy.node import Node

def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles (roll, pitch, yaw) in radians to quaternion (qx, qy, qz, qw).

    Returns:
        qx, qy, qz, qw
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return qx, qy, qz, qw

def generate_trajectory_and_control(num_points=5, dt=0.1):
    """
    Simulate a dynamic bicycle model with slip angle (beta) 
    and add small noise to simulate real-world imperfections.
    """

    # Vehicle parameters
    mass = 1500.0  # kg
    L = 2.5  # Wheelbase [m]
    Lr = L / 2.0  # Rear axle to COG [m]
    drag_coeff = 0.3  # aerodynamic drag coefficient
    frontal_area = 2.2  # m^2
    air_density = 1.225  # kg/m^3
    rolling_resistance_coeff = 0.015
    max_engine_force = 4000.0  # N
    g = 9.81  # gravity

    max_steer = 0.4  # ~23 degrees

    # Initialize states
    x = 0.0
    y = 0.0
    yaw = 0.0
    v = 0.0
    slip_angle = 0.0

    odometries = []
    throttles = []
    steering_angles = []

    # Noise parameters
    pos_noise_std = 0.01  
    yaw_noise_std = math.radians(0.01) 
    vel_noise_std = 0.02  
    angular_vel_noise_std = math.radians(0.05)  
    throttle_noise_std = 0.05  
    steering_noise_std = math.radians(0.05)  

    for i in range(num_points):
        # -------- Generate control inputs --------
        # Random throttle inputs with slow variations
        throttle_val = np.clip(0.5 + 0.5 * np.random.randn() * 0.1, 0.0, 1.0)
        steering_val = np.clip(0.2 * math.sin(0.01 * i) + 0.1 * np.random.randn(), -max_steer, max_steer)

        # -------- Vehicle longitudinal dynamics --------
        engine_force = throttle_val * max_engine_force
        drag_force = 0.5 * air_density * drag_coeff * frontal_area * v**2
        rolling_force = rolling_resistance_coeff * mass * g

        total_force = engine_force - drag_force - rolling_force
        acceleration = total_force / mass

        v += acceleration * dt
        v = max(v, 0.0)  # No negative speeds

        # -------- Vehicle lateral/slip dynamics --------
        beta = math.atan2(Lr * math.tan(steering_val), L)  # assuming 2.5 m wheelbase

        # -------- Update position --------
        x += v * math.cos(yaw + beta) * dt
        y += v * math.sin(yaw + beta) * dt
        yaw += (v / 2.5) * math.tan(steering_val) * dt  # Bicycle model
        yaw = (yaw + math.pi) % (2 * math.pi) - math.pi  # normalize between [-pi, pi]

        # Add noise to the state
        noisy_x = x + random.gauss(0, pos_noise_std)
        noisy_y = y + random.gauss(0, pos_noise_std)
        noisy_yaw = yaw + random.gauss(0, yaw_noise_std)
        noisy_vx = v * math.cos(beta) + random.gauss(0, vel_noise_std)
        noisy_vy = v * math.sin(beta) + random.gauss(0, vel_noise_std)
        noisy_angular_z = (v / L) * math.tan(steering_val) + random.gauss(0, angular_vel_noise_std)

        # Create Odometry message
        odom = Odometry()
        odom.header = Header()
        odom.header.stamp.sec = int(i * dt)
        odom.header.stamp.nanosec = int((i * dt - int(i * dt)) * 1e9)
        odom.header.frame_id = 'map'
        odom.child_frame_id = 'base_link'

        # Fill pose
        qx, qy, qz, qw = euler_to_quaternion(0.0, 0.0, noisy_yaw)

        odom.pose.pose = Pose()
        odom.pose.pose.position.x = noisy_x
        odom.pose.pose.position.y = noisy_y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw

        # Fill twist (velocity)
        odom.twist.twist = Twist()
        odom.twist.twist.linear.x = noisy_vx
        odom.twist.twist.linear.y = noisy_vy
        odom.twist.twist.linear.z = 0.0
        odom.twist.twist.angular.z = noisy_angular_z

        odometries.append(odom)

        # Control input messages with noise
        throttle_msg = Float64()
        throttle_msg.data = throttle_val + random.gauss(0, throttle_noise_std)

        steering_msg = Float64()
        steering_msg.data = steering_val + random.gauss(0, steering_noise_std)

        throttles.append(throttle_msg)
        steering_angles.append(steering_msg)

    return odometries, throttles, steering_angles


def create_bag_file():
    bag_dir = 'rosbag/kinematic_trajectory_ros2.bag'

    if os.path.exists(bag_dir):
        shutil.rmtree(bag_dir)
        print(f"Deleted existing directory: {bag_dir}")

    # Open bag
    storage_options = rosbag2_py._storage.StorageOptions(uri=bag_dir, storage_id='sqlite3')
    converter_options = rosbag2_py._storage.ConverterOptions('')
    writer = rosbag2_py.SequentialWriter()
    writer.open(storage_options, converter_options)

    # Create topic metadata for Odometry
    odometry_topic = '/vehicle/odometry'
    odometry_msg_type = 'nav_msgs/Odometry'
    serialization_format = 'cdr'

    topic_metadata_odom = rosbag2_py._storage.TopicMetadata(
        name=odometry_topic,
        type=odometry_msg_type,
        serialization_format=serialization_format
    )

    # Create topic metadata for Throttle (Float64)
    throttle_topic = '/vehicle/throttle'
    throttle_msg_type = 'std_msgs/Float64'

    topic_metadata_throttle = rosbag2_py._storage.TopicMetadata(
        name=throttle_topic,
        type=throttle_msg_type,
        serialization_format=serialization_format
    )

    # Create topic metadata for Steering (Float64)
    steering_topic = '/vehicle/steering'
    steering_msg_type = 'std_msgs/Float64'

    topic_metadata_steering = rosbag2_py._storage.TopicMetadata(
        name=steering_topic,
        type=steering_msg_type,
        serialization_format=serialization_format
    )

    # Register all topics
    writer.create_topic(topic_metadata_odom)
    writer.create_topic(topic_metadata_throttle)
    writer.create_topic(topic_metadata_steering)

    # Generate odometries, throttles, and steering angles
    odometries, throttles, steering_angles = generate_trajectory_and_control(3000)

    rclpy.init()

    for i in range(len(odometries)):
        # Serialize each odometry message
        serialized_odom = rclpy.serialization.serialize_message(odometries[i])
        timestamp_odom = odometries[i].header.stamp.sec * 10**9 + odometries[i].header.stamp.nanosec
        writer.write(odometry_topic, serialized_odom, timestamp_odom)

        # Serialize the throttle message
        serialized_throttle = rclpy.serialization.serialize_message(throttles[i])
        timestamp_throttle = odometries[i].header.stamp.sec * 10**9 + odometries[i].header.stamp.nanosec
        writer.write(throttle_topic, serialized_throttle, timestamp_throttle)

        # Serialize the steering message
        serialized_steering = rclpy.serialization.serialize_message(steering_angles[i])
        timestamp_steering = odometries[i].header.stamp.sec * 10**9 + odometries[i].header.stamp.nanosec
        writer.write(steering_topic, serialized_steering, timestamp_steering)

    rclpy.shutdown()
    writer.close()

    print("Bag file created with odometry, throttle, and steering messages")

# Create the bag file
create_bag_file()