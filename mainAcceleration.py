import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

# Load the XY profile data
def load_coordinates(filename):
    """Load X, Y coordinates from a CSV file."""
    data = np.loadtxt(filename, delimiter=',', skiprows=1)  # Skipping header row
    return data[:, 0], data[:, 1]

# Moving Average Filter
def moving_average_filter(data, window_size=5):
    """Apply a moving average filter to smooth the data."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

# Compute Curvature for Adaptive Velocity
def compute_curvature(x, y):
    """Compute curvature to adjust velocity based on edge complexity."""
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
    return curvature

# Generate B-Spline Smoothed Path
def smooth_path(x, y, smoothing_factor=0.1):
    """Generate a B-Spline smooth path from raw coordinates."""
    tck, _ = splprep([x, y], s=smoothing_factor)
    u_new = np.linspace(0, 1, len(x) * 2)  # Generate denser path points
    x_smooth, y_smooth = splev(u_new, tck)
    return np.array(x_smooth), np.array(y_smooth)

# Compute Motion Profile with Optimized Velocities
def generate_motion_profile(filename, probe_tolerance=1.0):
    """Generate motion path for X, Y stages and rotation theta with optimizations."""
    x, y = load_coordinates(filename)

    # Apply path smoothing
    x, y = smooth_path(x, y)

    # Apply Moving Average filter
    x = moving_average_filter(x)
    y = moving_average_filter(y)

    # Compute tangent angles for rotation stage
    dx = np.gradient(x)
    dy = np.gradient(y)
    angles = np.arctan2(dy, dx)  # Rotation angle at each point (radians)

    # Compute curvature-based velocity adjustments
    curvature = compute_curvature(x, y)
    base_velocity = 50  # mm/sec default speed
    velocity = base_velocity / (1 + 10 * curvature)  # Reduce speed in high-curvature areas
    velocity = np.clip(velocity, 10, 100)  # Limit velocity range

    # Adjust angle tolerance (allow small variations in probe rotation)
    angles = np.degrees(angles)  # Convert radians to degrees
    angles = moving_average_filter(angles, window_size=int(probe_tolerance))  # Smooth angle transitions

    return x, y, angles, velocity

# Run and visualize
filename = "motion_data.csv"  # Replace with actual CSV filename
x, y, angles, velocity = generate_motion_profile(filename)

# Plot the optimized motion path
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(x, y, 'b-', label="Smoothed Path")
plt.quiver(x, y, np.cos(np.radians(angles)), np.sin(np.radians(angles)), scale=20, color='r')
plt.xlabel("X Position (mm)")
plt.ylabel("Y Position (mm)")
plt.title("Optimized Waveguide Inspection Path")
plt.legend()
plt.grid()

# Plot the optimized velocity profile
plt.subplot(2, 1, 2)
plt.plot(np.arange(len(velocity)), velocity, 'g-', label="Velocity Profile (mm/sec)")
plt.xlabel("Path Position Index")
plt.ylabel("Velocity (mm/sec)")
plt.title("Adaptive Velocity Based on Curvature")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
