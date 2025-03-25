import numpy as np
import matplotlib.pyplot as plt


def load_coordinates(filename):
    """Load x, y coordinates from a CSV file."""
    data = np.loadtxt(filename, delimiter=',', skiprows=1)  # Skip header row
    x, y = data[:, 0], data[:, 1]  # Extract X and Y
    return x, y


def compute_curvature(x, y):
    """Calculate local curvature of the path."""
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    curvature = np.abs(dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** (3 / 2)
    curvature = np.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)
    return curvature


def compute_velocity_profile(x, y, mode="constant", V_const=100, V_max=120, V_min=50):
    """Generate motion profile with either constant or variable velocity."""
    distances = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)  # Compute segment distances

    if mode == "constant":
        times = np.cumsum(distances / V_const)  # Use fixed speed
    elif mode == "variable":
        curvature = compute_curvature(x, y)
        velocity = V_max - (V_max - V_min) * (curvature / np.max(curvature + 1e-5))  # Adjust velocity
        times = np.cumsum(distances / velocity[:-1])  # Use variable speed
    else:
        raise ValueError("Invalid mode. Choose 'constant' or 'variable'.")

    times = np.insert(times, 0, 0)  # Start time at zero
    return times


def generate_motion_profile(filename, mode="constant"):
    """Main function to generate motion profile."""
    x, y = load_coordinates(filename)
    times = compute_velocity_profile(x, y, mode=mode)

    # Compute heading angle (theta)
    dx = np.gradient(x)
    dy = np.gradient(y)
    angles = np.arctan2(dy, dx)

    return x, y, times, angles


def plot_motion_profile(x, y, times, angles):
    """Plot motion trajectory and key motion profiles."""
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(x, y, label="Path")
    plt.title("Trajectory")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(times, x, label="X Motion")
    plt.plot(times, y, label="Y Motion")
    plt.title("Motion Path (X-Y)")
    plt.xlabel("Time (s)")
    plt.ylabel("Position")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(times, angles, label="Rotation Angle")
    plt.title("Rotation Motion")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (radians)")
    plt.legend()

    plt.tight_layout()
    plt.show()


# Run the motion profile generation
filename = "motion_data.csv"  # Replace with actual filename
mode = "constant"  # Choose 'constant' or 'variable'
x, y, times, angles = generate_motion_profile(filename, mode)
plot_motion_profile(x, y, times, angles)
