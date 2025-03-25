import numpy as np
import pandas as pd
from math import atan2, degrees


# Load waveguide edge profile (CSV format)
def load_profile(filename):
    data = pd.read_csv(filename)
    return data['X'].values, data['Y'].values


# Compute normals and motion path
def compute_motion_path(x, y):
    tangents = np.arctan2(np.gradient(y), np.gradient(x))  # Compute tangent angles
    normals = tangents + np.pi / 2  # Normal is perpendicular to tangent

    return x, y, np.degrees(normals)  # Convert angles to degrees


# Load data
x_vals, y_vals = load_profile("motion_data.csv")

# Compute motion path
x_motion, y_motion, theta_motion = compute_motion_path(x_vals, y_vals)

# Save motion path
motion_path = pd.DataFrame({'X': x_motion, 'Y': y_motion, 'Theta': theta_motion})
motion_path.to_csv("motion_profile2.csv", index=False)
