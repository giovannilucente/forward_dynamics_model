# Forward dynamics model learned from ROS2 rosbag file
This repository contains scripts for training a forward dynamics model using ROS2 rosbag driving datasets. 
In case a recorded rosbag file is unavailable, this repository also provides a script to generate an artificial rosbag file of a vehicle trajectory.

## Installation and Setup 
To install and run this project locally, follow these steps:

### 1. Clone the repository
First, clone the repository to your local machine:
```bash
git clone https://github.com/giovannilucente/forward_dynamics_model.git
cd forward_dynamics_model
```
### 2. Install the requirements
Install all the required dependencies listed in the requirements.txt:
```bash
pip install -r requirements.txt
```

### 3. Generate an artificial ROS2 rosbag
Place the ROS2 rosbag file in the /rosbag folder. 
If a ROS2 rosbag file is not available, you can generate one by running:
```bash
python3 generate_artificial_rosbag_file.py
```
The rosbag characteristics are:
```bash
Files:             kinematic_trajectory_ros2.bag_0.db3
Bag size:          27.9 MiB
Storage id:        sqlite3
Duration:          2999.900000000s
Start:             Jan  1 1970 01:00:00.000000000 (0.000000000)
End:               Jan  1 1970 01:49:59.900000000 (2999.900000000)
Messages:          120000
Topic information: Topic: /vehicle/braking | Type: std_msgs/Float64 | Count: 30000 | Serialization Format: cdr
                   Topic: /vehicle/steering | Type: std_msgs/Float64 | Count: 30000 | Serialization Format: cdr
                   Topic: /vehicle/throttle | Type: std_msgs/Float64 | Count: 30000 | Serialization Format: cdr
                   Topic: /vehicle/odometry | Type: nav_msgs/Odometry | Count: 30000 | Serialization Format: cdr
```
The repository works if these types of messages are present inside the rosbag file. 
The generated trajectory is based on a realistic dynamic bicycle model, incorporating front and rear slip angles, non-linear lateral forces, and longitudinal dynamics that also account for drag and rolling forces.

### 4. Extract a CSV from the ROS2 rosbag 
To generate a corresponding CSV file run:
```bash
python3 extract_data.py  
```
This will generate a CSV file with the following fieldnames:
```bash
['timestamp', 'pos_x', 'pos_y', 'pos_z', 'yaw', 'vel_x', 'vel_y', 'yaw_rate', 'throttle', 'braking', 'steering']
```

### 5. Train the model
To train the model run:
```bash
python3 train.py  
```
The model inputs are: 
```bash
['vel_x', 'vel_y', 'yaw', 'yaw_rate', 'steering', 'throttle', 'braking', 'dt']
```
The model predicts the following outputs:
```bash
['d_pos_x', 'd_pos_y', 'd_yaw', 'next_vel_x', 'next_vel_y', 'next_yaw_rate']
```
Where d_pos_x, d_pos_y, and d_yaw represent the x, y, and yaw differences between two consecutive timesteps, respectively.
The model is a three-layer perceptron with a hidden layer dimension of 128 and a ReLU activation function.
The performance on the test dataset is as follows:
```bash
Test RMSE per output:
d_pos_x: 0.0265 m
d_pos_y: 0.0283 m
d_yaw:   0.0737 rad
vel_x:   0.1090 m/s
vel_y:   0.0488 m/s
yaw_rate:0.0688 rad/s
```
