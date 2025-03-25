import numpy as np
import matplotlib.pyplot as plt


# Load the edge profile data
def load_coordinates(filename):
    data = np.loadtxt(filename, delimiter=',', skiprows=1)  # Assuming CSV format: x,y
    return data[:, 0], data[:, 1]


# Compute tangents and normals for probe orientation
def compute_tangents_and_normals(x, y):
    dx = np.gradient(x)
    dy = np.gradient(y)

    # Compute tangent angles
    angles = np.arctan2(dy, dx)

    # Compute normal angles (perpendicular to tangents)
    normal_angles = angles + np.pi / 2

    return angles, normal_angles


# Compute SCARA inverse kinematics (joint angles for arm positioning)
def compute_scara_ik(x, y, l1=100, l2=100):
    """ Computes SCARA joint angles given the (x, y) end-effector path """
    theta1 = []
    theta2 = []

    for i in range(len(x)):
        r = np.sqrt(x[i] ** 2 + y[i] ** 2)
        cos_theta2 = (r ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
        theta2_i = np.arccos(np.clip(cos_theta2, -1, 1))  # Clip to avoid numerical issues
        theta1_i = np.arctan2(y[i], x[i]) - np.arctan2(l2 * np.sin(theta2_i), l1 + l2 * np.cos(theta2_i))

        theta1.append(np.degrees(theta1_i))
        theta2.append(np.degrees(theta2_i))

    return theta1, theta2


# Compute Hexapod DOF corrections
def compute_hexapod_corrections(x, y, normal_angles):
    """ Computes position corrections for a 6-DOF hexapod """
    z_offset = 0.01  # Fixed offset for waveguide height control
    roll = np.degrees(normal_angles)  # Align probe perpendicular to waveguide
    pitch = np.zeros_like(x)  # Assume no pitch correction needed
    yaw = np.zeros_like(x)  # Assume no yaw correction needed

    return x, y, z_offset, roll, pitch, yaw


# Compute Rotary-Linear Stage displacements
def compute_rotary_linear_motion(x, y, center_x=0, center_y=0):
    """ Computes rotary (θ) and linear (R) motion for a rotary-linear stage system """
    r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)  # Compute radial displacement
    theta = np.arctan2(y - center_y, x - center_x)  # Compute angular displacement

    return r, np.degrees(theta)


# Main function to generate motion profiles
def generate_motion_profiles(filename):
    # Load XY profile
    x, y = load_coordinates(filename)

    # Compute motion profiles
    angles, normal_angles = compute_tangents_and_normals(x, y)

    # Compute motion for different stage configurations
    theta1_scara, theta2_scara = compute_scara_ik(x, y)
    hexapod_x, hexapod_y, hexapod_z, roll, pitch, yaw = compute_hexapod_corrections(x, y, normal_angles)
    r_stage, theta_stage = compute_rotary_linear_motion(x, y)

    return x, y, angles, normal_angles, theta1_scara, theta2_scara, hexapod_x, hexapod_y, hexapod_z, roll, pitch, yaw, r_stage, theta_stage


# Plot motion profiles
def plot_motion_profiles(x, y, theta1_scara, theta2_scara, r_stage, theta_stage, roll):
    fig, axs = plt.subplots(3, 2, figsize=(12, 10))

    # Path Trajectory
    axs[0, 0].plot(x, y, label="Waveguide Path")
    axs[0, 0].set_title("XY Motion Profile")
    axs[0, 0].set_xlabel("X Position (mm)")
    axs[0, 0].set_ylabel("Y Position (mm)")
    axs[0, 0].legend()

    # SCARA Joint Angles
    axs[0, 1].plot(theta1_scara, label="SCARA Joint 1")
    axs[0, 1].plot(theta2_scara, label="SCARA Joint 2")
    axs[0, 1].set_title("SCARA Robot Joint Angles")
    axs[0, 1].legend()

    # Hexapod Roll Adjustments
    axs[1, 0].plot(roll, label="Hexapod Roll Correction")
    axs[1, 0].set_title("Hexapod Roll Compensation")
    axs[1, 0].legend()

    # Rotary-Linear Motion
    axs[1, 1].plot(r_stage, label="Radial Displacement (mm)")
    axs[1, 1].set_title("Rotary-Linear Motion: Radial Displacement")
    axs[1, 1].legend()

    axs[2, 0].plot(theta_stage, label="Rotary Angle (degrees)")
    axs[2, 0].set_title("Rotary-Linear Motion: Angular Displacement")
    axs[2, 0].legend()

    plt.tight_layout()
    plt.show()


# Run program
filename = "motion_data.csv"  # Replace with actual filename
x, y, angles, normal_angles, theta1_scara, theta2_scara, hexapod_x, hexapod_y, hexapod_z, roll, pitch, yaw, r_stage, theta_stage = generate_motion_profiles(
    filename)

# Display results
plot_motion_profiles(x, y, theta1_scara, theta2_scara, r_stage, theta_stage, roll)
