import torch
import torch.nn as nn

class DynamicsModel(nn.Module):
    def __init__(self, input_dim=8, output_dim=6, hidden_dim=128):
        """
        input_dim: 8 (pos_x, pos_y, yaw, vel_x, vel_y, yaw_rate, steering, throttle)
        output_dim: 6 (next_pos_x, next_pos_y, next_yaw, next_vel_x, next_vel_y, next_yaw_rate)
        """
        super(DynamicsModel, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x