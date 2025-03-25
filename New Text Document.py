import numpy as np
import matplotlib.pyplot as plt

# Load provided x, y coordinates from a file
def load_coordinates(filename):
    data = np.loadtxt(filename, delimiter=',', skiprows=1)  # Assuming CSV format: x,y
    x_coords, y_coords = data[:, 0], data[:, 1]
    return x_coords, y_coords

# Smooth data to remove noise (optional)
def smooth_data(coords, window_size=5):
    return np.convolve(coords, np.ones(window_size)/window_size, mode='same')

# Compute velocity profile (using finite differences)
def compute_velocity(x, y, dt=0.1):
    velocity_x = np.gradient(x, dt)  # dx/dt
    velocity_y = np.gradient(y, dt)  # dy/dt
    return velocity_x, velocity_y

# Compute acceleration profile
def compute_acceleration(velocity_x, velocity_y, dt=0.1):
    acceleration_x = np.gradient(velocity_x, dt)  # dVx/dt
    acceleration_y = np.gradient(velocity_y, dt)  # dVy/dt
    return acceleration_x, acceleration_y

# Compute rotational motion (heading angle & angular velocity)
def compute_rotation(x, y, dt=0.1):
    angles = np.arctan2(np.gradient(y), np.gradient(x))  # Compute heading angle
    angular_velocity = np.gradient(angles, dt)  # d(theta)/dt
    return angles, angular_velocity

# Generate motion profile
def generate_motion_profile(filename):
    x, y = load_coordinates(filename)  # Load data from file

    # Optional: Smooth data
    x, y = smooth_data(x), smooth_data(y)

    # Compute velocity, acceleration, and rotation profiles
    velocity_x, velocity_y = compute_velocity(x, y)
    acceleration_x, acceleration_y = compute_acceleration(velocity_x, velocity_y)
    angles, angular_velocity = compute_rotation(x, y)

    # Return computed profiles
    return x, y, velocity_x, velocity_y, acceleration_x, acceleration_y, angles, angular_velocity

# Plot results for visualization
def plot_motion_profile(x, y, velocity_x, velocity_y, acceleration_x, acceleration_y, angles, angular_velocity):
    time = np.linspace(0, len(x) * 0.1, len(x))  # Time axis based on dt=0.1s

    plt.figure(figsize=(10, 6))

    # Position Plot
    plt.subplot(3, 2, 1)
    plt.plot(x, y, label="Path")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Trajectory")
    plt.legend()

    # Velocity Plot
    plt.subplot(3, 2, 2)
    plt.plot(time, velocity_x, label="Vx")
    plt.plot(time, velocity_y, label="Vy")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity")
    plt.title("Velocity Profile")
    plt.legend()

    # Acceleration Plot
    plt.subplot(3, 2, 3)
    plt.plot(time, acceleration_x, label="Ax")
    plt.plot(time, acceleration_y, label="Ay")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration")
    plt.title("Acceleration Profile")
    plt.legend()

    # Rotation Profile
    plt.subplot(3, 2, 4)
    plt.plot(time, angles, label="Heading Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (radians)")
    plt.title("Heading Angle Profile")
    plt.legend()

    # Angular Velocity
    plt.subplot(3, 2, 5)
    plt.plot(time, angular_velocity, label="Angular Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Angular Velocity (rad/s)")
    plt.title("Rotation Speed")
    plt.legend()

    plt.tight_layout()
    plt.show()

# Example usage
filename = "motion_data.csv"  # Replace with actual file name
x, y, vx, vy, ax, ay, angles, omega = generate_motion_profile(filename)
plot_motion_profile(x, y, vx, vy, ax, ay, angles, omega)
