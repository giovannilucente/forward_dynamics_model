import torch
import torch.nn as nn

class DynamicsModel(nn.Module):
    def __init__(self, input_dim=8, output_dim=6, hidden_dim=128):
        """
        input_dim: 8 (vel_x, vel_y, yaw, yaw_rate, steering, throttle, braking, dt)
        output_dim: 6 (d_pos_x, d_pos_y, d_yaw, next_vel_x, next_vel_y, next_yaw_rate)
        hidden_dim: Number of neurons in the hidden layers
        """
        super(DynamicsModel, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        x is a tensor of shape (batch_size, 7), where:
        - x[:, 0] = vel_x
        - x[:, 1] = vel_y
        - x[:, 2] = yaw
        - x[:, 3] = yaw_rate
        - x[:, 4] = steering
        - x[:, 5] = throttle
        - x[:, 6] = braking
        - x[:, 7] = dt
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
