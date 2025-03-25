import numpy as np
import matplotlib.pyplot as plt


def compute_normals(x, y):

    dx = np.gradient(x)
    dy = np.gradient(y)

    # Compute normal vectors (perpendicular to tangent)
    normal_x = -dy
    normal_y = dx

    # Normalize vectors
    magnitude = np.sqrt(normal_x ** 2 + normal_y ** 2)
    normal_x /= magnitude
    normal_y /= magnitude

    # Compute rotation angles (degrees)
    angles = np.degrees(np.arctan2(normal_y, normal_x))

    return normal_x, normal_y, angles


# Example X, Y profile
x = np.linspace(0, 100, 50)
y = np.sin(x / 10) * 10  # Example waveguide shape

# Compute normal vectors
normal_x, normal_y, angles = compute_normals(x, y)

# Plot normal vectors
plt.figure(figsize=(8, 6))
plt.plot(x, y, 'b-', label="Waveguide Edge")
plt.quiver(x, y, normal_x, normal_y, angles, cmap="coolwarm", scale=15)
plt.xlabel("X Position (mm)")
plt.ylabel("Y Position (mm)")
plt.title("Probe Orientation Control")
plt.legend()
plt.grid()
plt.show()
