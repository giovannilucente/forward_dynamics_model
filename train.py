import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from model import DynamicsModel

# Load dataset
csv_file = 'trajectory_csv/trajectory_data.csv'
df = pd.read_csv(csv_file)
df = df.dropna()

# Calculate dt: Time step between consecutive samples
timestamps = df['timestamp'].values
dt = np.diff(timestamps)  # Time difference between consecutive timestamps

# Ensure dt has the same shape as inputs
dt = np.concatenate(([0], dt))  # Add a zero at the beginning to match the shape

# Prepare Inputs (NOW INCLUDING braking)
inputs = df[['vel_x', 'vel_y', 'yaw', 'yaw_rate', 'steering', 'throttle', 'braking']].values[:-1]
inputs = np.hstack([inputs, dt[:-1].reshape(-1, 1)])  # Add dt as the last column

# Prepare Outputs
d_pos_x = df['pos_x'].values[1:] - df['pos_x'].values[:-1]
d_pos_y = df['pos_y'].values[1:] - df['pos_y'].values[:-1]
d_yaw = df['yaw'].values[1:] - df['yaw'].values[:-1]
d_yaw = (d_yaw + np.pi) % (2 * np.pi) - np.pi  # Wrap angle between [-π, π]

next_vel_x = df['vel_x'].values[1:]
next_vel_y = df['vel_y'].values[1:]
next_yaw_rate = df['yaw_rate'].values[1:]

targets = np.stack([d_pos_x, d_pos_y, d_yaw, next_vel_x, next_vel_y, next_yaw_rate], axis=1)

# Train/Val/Test split (70/15/15)
X_temp, X_test, y_temp, y_test = train_test_split(inputs, targets, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42)  
# 0.1765 * 0.85 ≈ 0.15 → so val ≈ 15%

# Convert to torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DynamicsModel().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
num_epochs = 2000
train_losses = []
val_losses = []

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

    train_losses.append(loss.item())
    val_losses.append(val_loss.item())

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}")

# Save model
torch.save(model.state_dict(), 'trained_dynamics_model.pth')
print("Model saved successfully.")

# Plot losses
plt.figure(figsize=(8,5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)
plt.title('Training and Validation Loss')
plt.show()

# --- Test Evaluation ---
model.eval()
with torch.no_grad():
    test_outputs = model(X_test.to(device)).cpu().numpy()
    test_targets = y_test.cpu().numpy()

# Compute RMSE
rmse = np.sqrt(np.mean((test_outputs - test_targets)**2, axis=0))
print("\nTest RMSE per output:")
print(f"d_pos_x: {rmse[0]:.4f} m")
print(f"d_pos_y: {rmse[1]:.4f} m")
print(f"d_yaw:   {rmse[2]:.4f} rad")
print(f"vel_x:   {rmse[3]:.4f} m/s")
print(f"vel_y:   {rmse[4]:.4f} m/s")
print(f"yaw_rate:{rmse[5]:.4f} rad/s")

mean_rmse = np.mean(rmse)
print(f"\nMean RMSE across all outputs: {mean_rmse:.4f}")

# --- Bicycle Model Evaluation (Baseline) ---
print("\n--- Bicycle Model Baseline Evaluation ---")

L = 2.5  # Wheelbase in meters

# Use test set from original dataframe to get inputs
test_inputs_np = X_test.numpy()
test_targets_np = y_test.numpy()

# Bicycle model predictions
pred_d_pos_x = []
pred_d_pos_y = []
pred_d_yaw = []
pred_next_vel_x = []
pred_next_vel_y = []
pred_next_yaw_rate = []

for i in range(test_inputs_np.shape[0]):
    vel_x, vel_y, yaw, yaw_rate, steering, throttle, braking, dt = test_inputs_np[i]
    v = np.sqrt(vel_x**2 + vel_y**2)
    a = throttle - braking
    delta = steering

    next_v = v + a * dt
    next_yaw = yaw + (v / L) * delta * dt
    next_vx = next_v * np.cos(next_yaw)
    next_vy = next_v * np.sin(next_yaw)

    dx = v * np.cos(yaw) * dt
    dy = v * np.sin(yaw) * dt
    dyaw = (v / L) * delta * dt
    next_yawrate = (v / L) * delta

    pred_d_pos_x.append(dx)
    pred_d_pos_y.append(dy)
    pred_d_yaw.append(((dyaw + np.pi) % (2 * np.pi)) - np.pi)  # Wrap
    pred_next_vel_x.append(next_vx)
    pred_next_vel_y.append(next_vy)
    pred_next_yaw_rate.append(next_yawrate)

# Stack predictions
baseline_preds = np.stack([
    pred_d_pos_x,
    pred_d_pos_y,
    pred_d_yaw,
    pred_next_vel_x,
    pred_next_vel_y,
    pred_next_yaw_rate
], axis=1)

# Compute RMSE
rmse_baseline = np.sqrt(np.mean((baseline_preds - test_targets_np) ** 2, axis=0))

# Print results
print("\nBicycle Model RMSE per output:")
print(f"d_pos_x: {rmse_baseline[0]:.4f} m")
print(f"d_pos_y: {rmse_baseline[1]:.4f} m")
print(f"d_yaw:   {rmse_baseline[2]:.4f} rad")
print(f"vel_x:   {rmse_baseline[3]:.4f} m/s")
print(f"vel_y:   {rmse_baseline[4]:.4f} m/s")
print(f"yaw_rate:{rmse_baseline[5]:.4f} rad/s")

mean_rmse_baseline = np.mean(rmse_baseline)
print(f"\nMean RMSE across all outputs (Bicycle Model): {mean_rmse_baseline:.4f}")

# Load trained weights
model = DynamicsModel()
model.load_state_dict(torch.load("trained_dynamics_model.pth"))
model.eval()

# Convert to TorchScript
example_input = torch.randn(1, 8)  # (vel_x, vel_y, yaw, yaw_rate, steering, throttle, braking, dt)
traced_script_module = torch.jit.trace(model, example_input)

# Save it
traced_script_module.save("trained_dynamics_model_script.pt")