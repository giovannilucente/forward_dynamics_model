import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from model import DynamicsModel

# Load dataset
csv_file = 'trajectory_csv/trajectory_data.csv'
df = pd.read_csv(csv_file)

df = df.dropna()

# Prepare input/output
states = df[['pos_x', 'pos_y', 'yaw', 'vel_x', 'vel_y', 'yaw_rate']].values
controls = df[['steering', 'throttle']].values

inputs = torch.tensor(
    np.hstack([states[:-1], controls[:-1]]),  # Stack state + control
    dtype=torch.float32
)

targets = torch.tensor(
    states[1:],  # Next state (shifted)
    dtype=torch.float32
)

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(inputs, targets, test_size=0.2, random_state=42)

# Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DynamicsModel().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
num_epochs = 2000

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train.to(device))
    loss = criterion(outputs, y_train.to(device))
    loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val.to(device))
        val_loss = criterion(val_outputs, y_val.to(device))

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}")

# Save model
torch.save(model.state_dict(), 'trained_dynamics_model.pth')
print("Model saved successfully.")