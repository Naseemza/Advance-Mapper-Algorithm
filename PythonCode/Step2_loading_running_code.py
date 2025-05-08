# Example usage
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gudhi as gd

# Define the torus parametric equations
def torus(u, v, R=3, r=1):
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)
    return np.column_stack((x, y, z))

# Parameters
n_points = 1000  # Number of points
R = 4  # Major radius
r = 2  # Minor radius

# Generate a regular grid of u and v values
u = np.linspace(0, 2 * np.pi, n_points)
v = np.linspace(0, 2 * np.pi, n_points)
u, v = np.meshgrid(u, v)
u = u.flatten()
v = v.flatten()

# Generate torus
xyz_points = torus(u, v, R, r)

# Randomly sample a subset of points
np.random.seed(42)  # For reproducibility
sample_indices = np.random.choice(xyz_points.shape[0], size=n_points, replace=False)
xyz_points_sampled = xyz_points[sample_indices]

# Normalize the data to [0, 1]
def normalize_data(data):
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    normalized_data = (data - min_vals) / (max_vals - min_vals)
    return normalized_data

xyz_points_normalized = normalize_data(xyz_points_sampled)

# Plot torus in 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xyz_points_normalized[:, 0], xyz_points_normalized[:, 1], xyz_points_normalized[:, 2], c='b', s=10)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Torus in 3D (Randomly Sampled)")
plt.show()






# Example of running the whole algorithm
xyz_points = xyz_points_normalized
k = 0.25  # Example value for k
cutoff_betti = 1 # Example cutoff Betti value
results_df_torus = process_structural_data(xyz_points, k, cutoff_betti)
