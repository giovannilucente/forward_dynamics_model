import os
import csv
import rclpy
import rosbag2_py
from rosidl_runtime_py.utilities import get_message
import rclpy.serialization

# Define the bag path
bag_path = 'rosbag/kinematic_trajectory_ros2.bag'

# Output CSV file
output_csv = 'rosbag/trajectory_data.csv'

# Topics
odometry_topic = '/vehicle/odometry'
throttle_topic = '/vehicle/throttle'
steering_topic = '/vehicle/steering'

# Message types
Odometry = get_message('nav_msgs/msg/Odometry')
Float64 = get_message('std_msgs/msg/Float64')

def read_rosbag_and_write_csv(bag_path, output_csv):
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions()
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_dict = {topic.name: topic.type for topic in topic_types}

    # Data storage
    data = {}

    while reader.has_next():
        topic, serialized_msg, timestamp = reader.read_next()

        # Deserialize message
        msg_type_str = type_dict[topic]
        if msg_type_str == 'nav_msgs/Odometry':
            msg_type = Odometry
        elif msg_type_str == 'std_msgs/Float64':
            msg_type = Float64
        else:
            continue  # Ignore unknown topics

        msg = rclpy.serialization.deserialize_message(serialized_msg, msg_type)

        # Timestamp in seconds
        time_sec = timestamp / 1e9

        if time_sec not in data:
            data[time_sec] = {}

        if topic == odometry_topic:
            data[time_sec]['pos_x'] = msg.pose.pose.position.x
            data[time_sec]['pos_y'] = msg.pose.pose.position.y
            data[time_sec]['pos_z'] = msg.pose.pose.position.z
            data[time_sec]['orient_w'] = msg.pose.pose.orientation.w
            data[time_sec]['vel_x'] = msg.twist.twist.linear.x
            data[time_sec]['vel_y'] = msg.twist.twist.linear.y
            data[time_sec]['ang_vel_z'] = msg.twist.twist.angular.z

        elif topic == throttle_topic:
            data[time_sec]['throttle'] = msg.data

        elif topic == steering_topic:
            data[time_sec]['steering'] = msg.data

    # Sort timestamps
    sorted_times = sorted(data.keys())

    # Write to CSV
    with open(output_csv, mode='w', newline='') as csv_file:
        fieldnames = ['timestamp', 'pos_x', 'pos_y', 'pos_z', 'orient_w', 'vel_x', 'vel_y', 'ang_vel_z', 'throttle', 'steering']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for time in sorted_times:
            row = {'timestamp': time}
            row.update(data[time])
            writer.writerow(row)

    print(f"CSV file written to {output_csv}")

# Initialize rclpy
rclpy.init()

# Execute
read_rosbag_and_write_csv(bag_path, output_csv)

# Shutdown rclpy
rclpy.shutdown()
