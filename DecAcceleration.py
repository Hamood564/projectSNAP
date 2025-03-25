import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Load waveguide profile
def load_profile(filename):
    """Loads waveguide edge profile coordinates from a CSV file"""
    data = pd.read_csv(filename)
    return data['X'].values, data['Y'].values


# Compute Center of Mass (COM)
def compute_center_of_mass(x, y):
    centroid_x = np.mean(x)
    centroid_y = np.mean(y)
    return centroid_x, centroid_y


# Adjust the Waveguide Positioning
def adjust_waveguide_position(x, y, new_center_x, new_center_y):
    x_adjusted = x - new_center_x
    y_adjusted = y - new_center_y
    return x_adjusted, y_adjusted


# Compute Motion Path and Kinematic Profiles
def compute_motion_profile(x, y, time_step=0.05):

    # Compute Tangent and Normal Angles
    dx = np.gradient(x)
    dy = np.gradient(y)
    tangent_angles = np.arctan2(dy, dx)
    normal_angles = tangent_angles + np.pi / 2  # Compute normal angles

    # Compute Velocity Profiles (Vx, Vy)
    velocity_x = dx / time_step  # Vx
    velocity_y = dy / time_step  # Vy
    velocity = np.sqrt(velocity_x ** 2 + velocity_y ** 2)  # Speed magnitude

    # Compute Acceleration Profiles (Ax, Ay)
    acceleration_x = np.gradient(velocity_x) / time_step  # Ax
    acceleration_y = np.gradient(velocity_y) / time_step  # Ay
    acceleration = np.sqrt(acceleration_x ** 2 + acceleration_y ** 2)  # Acceleration magnitude

    # Compute Rotation Speed
    rotation_speed = np.gradient(np.degrees(normal_angles)) / time_step  # Angular velocity

    return velocity_x, velocity_y, acceleration_x, acceleration_y, normal_angles, rotation_speed


# Load Data
filename = "motion_data.csv"
x_vals, y_vals = load_profile(filename)

# Compute Center of Mass
com_x, com_y = compute_center_of_mass(x_vals, y_vals)

# Align Waveguide to Reduce Acceleration
x_adjusted, y_adjusted = adjust_waveguide_position(x_vals, y_vals, com_x, com_y)

# Compute Motion Path with Kinematic Profiles
velocity_x, velocity_y, acceleration_x, acceleration_y, heading_angles, rotation_speed = compute_motion_profile(
    x_adjusted, y_adjusted)

# Generate Time Axis
time_axis = np.linspace(0, len(velocity_x) * 0.05, len(velocity_x))  # Assuming 50ms time steps

# Plot Vx and Vy Separately
plt.figure(figsize=(12, 6))
plt.plot(time_axis, velocity_x, 'b-', label="Vx (X-axis Velocity)")
plt.plot(time_axis, velocity_y, 'r-', label="Vy (Y-axis Velocity)")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (mm/s)")
plt.title("Velocity Profile (Vx & Vy)")
plt.legend()
plt.grid(True)
plt.show()

# Plot Ax and Ay Separately
plt.figure(figsize=(12, 6))
plt.plot(time_axis, acceleration_x, 'g-', label="Ax (X-axis Acceleration)")
plt.plot(time_axis, acceleration_y, 'm-', label="Ay (Y-axis Acceleration)")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (mm/s²)")
plt.title("Acceleration Profile (Ax & Ay)")
plt.legend()
plt.grid(True)
plt.show()
