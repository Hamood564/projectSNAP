import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def load_coordinates(filename):
    """Load X, Y coordinates from CSV file."""
    data = pd.read_csv(filename)
    x = data.iloc[:, 0].values  # Assuming first column is X
    y = data.iloc[:, 1].values  # Assuming second column is Y
    return x, y

def smooth_data(data, window_size=15, poly_order=3):
    """Smooth the data using Savitzky-Golay filter."""
    return savgol_filter(data, window_size, poly_order)

def compute_derivatives(x, y, dt=0.1):
    """Compute velocity and acceleration using smoothed derivatives."""
    # Smooth position data
    x_smooth = smooth_data(x)
    y_smooth = smooth_data(y)

    # Compute velocity (first derivative)
    vx = np.gradient(x_smooth, dt)
    vy = np.gradient(y_smooth, dt)

    # Compute acceleration (second derivative)
    ax = np.gradient(vx, dt)
    ay = np.gradient(vy, dt)

    return x_smooth, y_smooth, vx, vy, ax, ay

def compute_heading(x, y):
    """Compute heading angle (theta) and angular velocity."""
    theta = np.arctan2(np.gradient(y), np.gradient(x))  # Compute heading angle
    omega = np.gradient(theta)  # Compute angular velocity

    # Smooth the heading and angular velocity
    theta_smooth = smooth_data(theta)
    omega_smooth = smooth_data(omega)

    return theta_smooth, omega_smooth

def plot_motion_profiles(x, y, vx, vy, ax, ay, angles, omega):
    """Plot trajectory, velocity, acceleration, and angular velocity profiles."""
    time = np.linspace(0, len(x) * 0.1, len(x))

    fig, axs = plt.subplots(3, 2, figsize=(12, 10))

    # Plot trajectory
    axs[0, 0].plot(x, y, label="Path")
    axs[0, 0].set_title("Trajectory")
    axs[0, 0].set_xlabel("X Position")
    axs[0, 0].set_ylabel("Y Position")
    axs[0, 0].legend()

    # Plot velocity
    axs[0, 1].plot(time, vx, label="Vx")
    axs[0, 1].plot(time, vy, label="Vy")
    axs[0, 1].set_title("Velocity Profile")
    axs[0, 1].set_xlabel("Time (s)")
    axs[0, 1].set_ylabel("Velocity")
    axs[0, 1].legend()

    # Plot acceleration
    axs[1, 0].plot(time, ax, label="Ax")
    axs[1, 0].plot(time, ay, label="Ay")
    axs[1, 0].set_title("Acceleration Profile")
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].set_ylabel("Acceleration")
    axs[1, 0].legend()

    # Plot heading angle
    axs[1, 1].plot(time, angles, label="Heading Angle")
    axs[1, 1].set_title("Heading Angle Profile")
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].set_ylabel("Angle (radians)")
    axs[1, 1].legend()

    # Plot rotation speed
    axs[2, 0].plot(time, omega, label="Angular Velocity")
    axs[2, 0].set_title("Rotation Speed")
    axs[2, 0].set_xlabel("Time (s)")
    axs[2, 0].set_ylabel("Angular Velocity (rad/s)")
    axs[2, 0].legend()

    plt.tight_layout()
    plt.show()

def generate_motion_profile(filename):
    """Main function to generate and plot motion profiles."""
    x, y = load_coordinates(filename)
    x, y, vx, vy, ax, ay = compute_derivatives(x, y)
    angles, omega = compute_heading(x, y)

    plot_motion_profiles(x, y, vx, vy, ax, ay, angles, omega)

    return x, y, vx, vy, ax, ay, angles, omega

# Run motion profile generation
filename = "motion_data.csv"  # Ensure this file is in the same directory
x, y, vx, vy, ax, ay, angles, omega = generate_motion_profile(filename)
