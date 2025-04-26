import os
import csv
import rclpy
import math
import numpy as np 
import rosbag2_py
import matplotlib.pyplot as plt  # <<<<<<<<<<<<<<<<
from rosidl_runtime_py.utilities import get_message
import rclpy.serialization

# Define the bag path
bag_path = 'rosbag/kinematic_trajectory_ros2.bag'

# Output CSV file
output_csv = 'trajectory_csv/trajectory_data.csv'

# Topics
odometry_topic = '/vehicle/odometry'
throttle_topic = '/vehicle/throttle'
brake_topic = '/vehicle/braking'
steering_topic = '/vehicle/steering'

# Message types
Odometry = get_message('nav_msgs/msg/Odometry')
Float64 = get_message('std_msgs/msg/Float64')

def quaternion_to_euler(qx, qy, qz, qw):
    """Convert quaternion (qx, qy, qz, qw) to Euler angles (roll, pitch, yaw)."""
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

def read_rosbag_and_write_csv(bag_path, output_csv):
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions()
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_dict = {topic.name: topic.type for topic in topic_types}

    # Data storage
    data = {}

    # For plotting
    xs = []
    ys = []
    yaws = []

    while reader.has_next():
        topic, serialized_msg, timestamp = reader.read_next()

        # Deserialize message
        msg_type_str = type_dict[topic]
        if msg_type_str == 'nav_msgs/Odometry':
            msg_type = Odometry
        elif msg_type_str == 'std_msgs/Float64':
            msg_type = Float64
        else:
            continue

        msg = rclpy.serialization.deserialize_message(serialized_msg, msg_type)

        # Timestamp in seconds
        time_sec = timestamp / 1e9

        if time_sec not in data:
            data[time_sec] = {}

        if topic == odometry_topic:
            pos = msg.pose.pose.position
            ori = msg.pose.pose.orientation
            twist = msg.twist.twist

            roll, pitch, yaw = quaternion_to_euler(ori.x, ori.y, ori.z, ori.w)

            data[time_sec]['pos_x'] = pos.x
            data[time_sec]['pos_y'] = pos.y
            data[time_sec]['pos_z'] = pos.z
            data[time_sec]['yaw'] = yaw
            data[time_sec]['vel_x'] = twist.linear.x
            data[time_sec]['vel_y'] = twist.linear.y
            data[time_sec]['yaw_rate'] = twist.angular.z

            # Save for plotting
            xs.append(pos.x)
            ys.append(pos.y)
            yaws.append(yaw)

        elif topic == throttle_topic:
            data[time_sec]['throttle'] = msg.data

        elif topic == steering_topic:
            data[time_sec]['steering'] = msg.data
        
        elif topic == brake_topic:
            data[time_sec]['braking'] = msg.data

    # Sort timestamps
    sorted_times = sorted(data.keys())

    # Write to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, mode='w', newline='') as csv_file:
        fieldnames = ['timestamp', 'pos_x', 'pos_y', 'pos_z', 'yaw', 'vel_x', 'vel_y', 'yaw_rate', 'throttle', 'braking', 'steering']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for time in sorted_times:
            row = {'timestamp': time}
            row.update(data[time])
            writer.writerow(row)

    print(f"CSV file written to {output_csv}")

    # ------ Plotting ------
    plt.figure(figsize=(8, 6))
    plt.title("Vehicle Trajectory with Yaw Direction")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.axis('equal')

    for x, y, yaw in zip(xs, ys, yaws):
        dx = math.cos(yaw) * 0.5  # Arrow size
        dy = math.sin(yaw) * 0.5
        plt.arrow(x, y, dx, dy, head_width=0.1, head_length=0.1, fc='blue', ec='blue')

    plt.grid(True)
    plt.show()
    # ----------------------

    # --- Compute statistics ---

    # Extract data into numpy arrays
    velocities = []
    yaw_rates = []

    for time in sorted_times:
        entry = data[time]
        if 'vel_x' in entry and 'vel_y' in entry:
            speed = math.sqrt(entry['vel_x']**2 + entry['vel_y']**2)
            velocities.append(speed)

        if 'yaw_rate' in entry:
            yaw_rates.append(entry['yaw_rate'])

    velocities = np.array(velocities)
    yaw_rates = np.array(yaw_rates)

    # Compute statistics
    print("\n=== Trajectory Statistics ===")
    print(f"Speed (m/s): min={velocities.min():.2f}, max={velocities.max():.2f}, mean={velocities.mean():.2f}")
    print(f"Yaw Rate (rad/s): min={yaw_rates.min():.4f}, max={yaw_rates.max():.4f}, mean={yaw_rates.mean():.4f}")
    print("================================\n")

# Initialize rclpy
rclpy.init()

# Execute
read_rosbag_and_write_csv(bag_path, output_csv)

# Shutdown rclpy
rclpy.shutdown()
