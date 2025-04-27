# Forward dynamics model learned from ROS2 rosbag file
This repository contains the scripts to train a forward dynamics model from ROS2 rosbag driving datasets.
In case you do not have a recorded rosbag file, this repository provides also a script to generate an artificial rosbag file of a vehicle trajectory.

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

### 3. Generate a ROS2 rosbag
Place the ROS2 rosbag file in the folder /rosbag.
As an example, is possible to generate a ROS2 rosbag file by running:
```bash
python3 generate_artificial_rosbag_file.py
```

### 4. Extract a CSV from the ROS2 rosbag 
To generate a corresponding CSV file run:
```bash
python3 extract_data.py  
```

### 5. Train the model
To train the model run:
```bash
python3 train.py  
```
