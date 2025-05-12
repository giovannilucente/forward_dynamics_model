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

def pacejka_force(alpha, B, C, D, E):
    """
    Calculate the lateral force using Pacejka's Magic Formula
    :param alpha: Slip angle in radians
    :param B, C, D, E: Pacejka parameters for the tire
    :return: Lateral force
    """
    return D * math.sin(C * math.atan(B * alpha - E * (B * alpha - math.atan(B * alpha))))


def generate_trajectory_and_control(num_points=500, dt=0.1):
    # Vehicle parameters
    mass = 1500.0  # kg
    L = 2.5
    Lr = L / 2.0
    Lf = L - Lr
    drag_coeff = 0.3
    frontal_area = 2.2
    air_density = 1.225
    rolling_resistance_coeff = 0.015
    max_engine_force = 4000.0
    max_brake_force = 6000.0
    g = 9.81
    v_max = 20
    max_steer = 0.4
    tau = 0.5
    Iz = 2250.0

    Cf = 8000.0
    Cr = 10000.0

    # Example Pacejka parameters (for illustrative purposes)
    B_f = 10  # Stiffness factor for front tire
    C_f = 1.9  # Shape factor for front tire
    D_f = 1500  # Peak lateral force for front tire
    E_f = 0.97  # Curvature factor for front tire

    B_r = 12  # Stiffness factor for rear tire
    C_r = 2.0  # Shape factor for rear tire
    D_r = 1600  # Peak lateral force for rear tire
    E_r = 0.98  # Curvature factor for rear tire

    x = 0.0
    y = 0.0
    yaw = 0.0
    v = 0.0
    yaw_rate = 0.0
    beta = 0.0

    odometries = []
    throttles = []
    steering_angles = []
    brakes = []

    # Noise parameters
    pos_noise_std = 0.01  
    yaw_noise_std = math.radians(3) 
    vel_noise_std = 0.02 
    angular_vel_noise_std = math.radians(3)

    for i in range(num_points):
        # -------- Generate control inputs --------
        # Smooth throttle (adjust to simulate urban driving speed)
        throttle_val = np.clip(0.2 + 0.2 * np.random.randn(), 0.0, 1.0)  # Keep throttle lower for city driving

        # Increase likelihood of braking (city driving with frequent stops)
        if random.random() < 0.2:  # 20% chance to brake (increased frequency)
            brake_val = np.clip(abs(np.random.randn()) * 0.5, 0.0, 1.0)  # More frequent and stronger braking
            throttle_val = 0.0  # If braking, throttle to 0
        else:
            brake_val = 0.0

        # Smooth steering (simulate more frequent and sharp turns)
        steering_val = np.clip(0.15 * math.sin(0.03 * i) + 0.1 * np.random.randn(), -max_steer, max_steer)
        
        # -------- Vehicle longitudinal dynamics --------
        engine_force = throttle_val * max_engine_force
        brake_force = brake_val * max_brake_force
        drag_force = 0.5 * air_density * drag_coeff * frontal_area * v**2
        rolling_force = rolling_resistance_coeff * mass * g

        total_force = engine_force - brake_force - drag_force - rolling_force
        acceleration = total_force / mass

        v += acceleration * dt
        v = max(v, 0.0)
        v = min(v, v_max)  # Cap the speed to 40 km/h

        # -------- Vehicle lateral/slip dynamics --------
        if v > 1.0:
            alpha_f = steering_val - math.atan2((yaw_rate * Lf + v * math.sin(beta)), (v * math.cos(beta) + 1e-6))
            alpha_r = -math.atan2((yaw_rate * Lr - v * math.sin(beta)), (v * math.cos(beta) + 1e-6))

            Fyf = - pacejka_force(alpha_f, B_f, C_f, D_f, E_f)
            Fyr = - pacejka_force(alpha_r, B_r, C_r, D_r, E_r)

            yaw_acc = (Lf * Fyf - Lr * Fyr) / Iz
            yaw_rate += yaw_acc * dt

            lat_acceleration = (Fyf + Fyr) / mass
            v_lat = lat_acceleration * dt

            beta = math.atan2(v_lat, v + 1e-6)
            
        else:
            yaw_acc = 0.0
            yaw_rate *= 0.95  # Damp yaw_rate slowly
            beta = steering_val * Lr / L
        

        # -------- Update position --------
        x += v * math.cos(yaw + beta) * dt
        y += v * math.sin(yaw + beta) * dt
        yaw += yaw_rate * dt
        yaw = (yaw + np.pi) % (2 * np.pi) - np.pi  # Normalize yaw to [-pi, pi]

        # Add noise
        noisy_x = x + random.gauss(0, pos_noise_std)
        noisy_y = y + random.gauss(0, pos_noise_std)
        noisy_yaw = yaw + random.gauss(0, yaw_noise_std)
        noisy_vx = v * math.cos(beta) + random.gauss(0, vel_noise_std)
        noisy_vy = v * math.sin(beta) + random.gauss(0, vel_noise_std)
        noisy_angular_z = yaw_rate + random.gauss(0, angular_vel_noise_std)

        # Create Odometry
        odom = Odometry()
        odom.header = Header()
        odom.header.stamp.sec = int(i * dt)
        odom.header.stamp.nanosec = int((i * dt - int(i * dt)) * 1e9)
        odom.header.frame_id = 'map'
        odom.child_frame_id = 'base_link'

        qx, qy, qz, qw = euler_to_quaternion(0.0, 0.0, noisy_yaw)

        odom.pose.pose = Pose()
        odom.pose.pose.position.x = noisy_x
        odom.pose.pose.position.y = noisy_y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw

        odom.twist.twist = Twist()
        odom.twist.twist.linear.x = noisy_vx
        odom.twist.twist.linear.y = noisy_vy
        odom.twist.twist.linear.z = 0.0
        odom.twist.twist.angular.z = noisy_angular_z

        odometries.append(odom)

        # Control inputs
        throttle_msg = Float64()
        throttle_msg.data = throttle_val

        steering_msg = Float64()
        steering_msg.data = steering_val

        brake_msg = Float64()
        brake_msg.data = brake_val

        throttles.append(throttle_msg)
        steering_angles.append(steering_msg)
        brakes.append(brake_msg)

    return odometries, throttles, steering_angles, brakes


def create_bag_file():
    bag_dir = 'rosbag/kinematic_trajectory_ros2.bag'

    if os.path.exists(bag_dir):
        shutil.rmtree(bag_dir)
        print(f"Deleted existing directory: {bag_dir}")

    storage_options = rosbag2_py._storage.StorageOptions(uri=bag_dir, storage_id='sqlite3')
    converter_options = rosbag2_py._storage.ConverterOptions('')
    writer = rosbag2_py.SequentialWriter()
    writer.open(storage_options, converter_options)

    odometry_topic = '/vehicle/odometry'
    odometry_msg_type = 'nav_msgs/Odometry'
    serialization_format = 'cdr'

    throttle_topic = '/vehicle/throttle'
    throttle_msg_type = 'std_msgs/Float64'

    steering_topic = '/vehicle/steering'
    steering_msg_type = 'std_msgs/Float64'

    braking_topic = '/vehicle/braking'  
    brake_msg_type = 'std_msgs/Float64'

    # Register all topics
    writer.create_topic(rosbag2_py._storage.TopicMetadata(name=odometry_topic, type=odometry_msg_type, serialization_format=serialization_format))
    writer.create_topic(rosbag2_py._storage.TopicMetadata(name=throttle_topic, type=throttle_msg_type, serialization_format=serialization_format))
    writer.create_topic(rosbag2_py._storage.TopicMetadata(name=steering_topic, type=steering_msg_type, serialization_format=serialization_format))
    writer.create_topic(rosbag2_py._storage.TopicMetadata(name=braking_topic, type=brake_msg_type, serialization_format=serialization_format))

    odometries, throttles, steering_angles, brakes = generate_trajectory_and_control(30000)

    rclpy.init()

    for i in range(len(odometries)):
        timestamp = odometries[i].header.stamp.sec * 10**9 + odometries[i].header.stamp.nanosec

        writer.write(odometry_topic, rclpy.serialization.serialize_message(odometries[i]), timestamp)
        writer.write(throttle_topic, rclpy.serialization.serialize_message(throttles[i]), timestamp)
        writer.write(steering_topic, rclpy.serialization.serialize_message(steering_angles[i]), timestamp)
        writer.write(braking_topic, rclpy.serialization.serialize_message(brakes[i]), timestamp)

    rclpy.shutdown()
    writer.close()

    print("Bag file created with odometry, throttle, braking and steering messages")


# Run
create_bag_file()