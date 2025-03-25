import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def load_coordinates(filename):
    """Load X, Y coordinates from CSV file."""
    data = pd.read_csv(filename)
    x = data.iloc[:, 0].values  # First column is X
    y = data.iloc[:, 1].values  # Second column is Y
    return x, y

def smooth_data(data, window_size=11, poly_order=3):
    """Smooth data using Savitzky-Golay filter."""
    return savgol_filter(data, window_size, poly_order)

def compute_motion_profile(x, y, dt=0.1):
    """Compute velocity, acceleration, and theta for motion profile."""
    # Smooth position data
    x_smooth = smooth_data(x)
    y_smooth = smooth_data(y)

    # Compute velocity
    vx = np.gradient(x_smooth, dt)
    vy = np.gradient(y_smooth, dt)

    # Compute acceleration
    ax = np.gradient(vx, dt)
    ay = np.gradient(vy, dt)

    # Compute heading angle (theta) in radians
    theta = np.arctan2(vy, vx)  # Robot orientation
    omega = np.gradient(theta, dt)  # Angular velocity

    # Convert theta from radians to degrees
    theta_deg = np.degrees(theta)

    return x_smooth, y_smooth, vx, vy, ax, ay, theta_deg, omega

def save_motion_profile(filename, x, y, theta):
    """Save motion profile (X, Y, Theta) to CSV."""
    df = pd.DataFrame({"X Position (mm)": x, "Y Position (mm)": y, "Rotation Angle (deg)": theta})
    df.to_csv(filename, index=False)
    print(f"Motion profile saved to {filename}")

def plot_motion_profiles(x, y, theta):
    """Plot X-Y motion and rotation angle over time."""
    time = np.linspace(0, len(x) * 0.1, len(x))  # Generate time axis

    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Plot X-Y motion path
    axs[0].plot(x, y, label="XY Motion Path", color="b")
    axs[0].set_title("Motion Path of XY Stage")
    axs[0].set_xlabel("X Position (mm)")
    axs[0].set_ylabel("Y Position (mm)")
    axs[0].legend()
    axs[0].grid(True)

    # Plot rotation angle (theta) vs time
    axs[1].plot(time, theta, label="Rotation Stage", color="r")
    axs[1].set_title("Rotation Stage (Theta) Over Time")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Rotation Angle (degrees)")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

def generate_motion_profile(filename):
    """Main function to compute, save, and plot motion profile."""
    x, y = load_coordinates(filename)
    x, y, vx, vy, ax, ay, theta, omega = compute_motion_profile(x, y)

    # Save motion profile
    save_motion_profile("motion_profile14.csv", x, y, theta)

    # Plot motion profiles
    plot_motion_profiles(x, y, theta)

    return x, y, vx, vy, ax, ay, theta, omega

# Run motion profile generation
filename = "motion_data.csv"  # Ensure this file is in the same directory
x, y, vx, vy, ax, ay, theta, omega = generate_motion_profile(filename)
