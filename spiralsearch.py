import numpy as np
import matplotlib.pyplot as plt

def spiral_search(radius, step_size=1):

    t = np.arange(0, 10*np.pi, step_size/radius)  # Spiral step function
    x = radius * (t / max(t)) * np.cos(t)  # Spiral X-coordinates
    y = radius * (t / max(t)) * np.sin(t)  # Spiral Y-coordinates
    return x, y

# Define search parameters
bounding_radius = 50  # Known bounding circle radius
x, y = spiral_search(bounding_radius)

# Plot search pattern
plt.figure(figsize=(6,6))
plt.plot(x, y, 'r-', label="Spiral Search Path")
plt.scatter(0, 0, c='blue', label="Start Position (Center)")
plt.xlabel("X Position (mm)")
plt.ylabel("Y Position (mm)")
plt.legend()
plt.grid(True)
plt.title("Spiral Search Path for Edge Detection")
plt.show()
