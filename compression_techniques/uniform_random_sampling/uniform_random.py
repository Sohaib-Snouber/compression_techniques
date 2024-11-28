import open3d as o3d
import numpy as np
import time

# Load the original point cloud
input_ply = "../../scene0.ply"
output_uniform_ply = "uniform_sampled_point_cloud.ply"
output_random_ply = "random_sampled_point_cloud.ply"

point_cloud = o3d.io.read_point_cloud(input_ply)
original_point_count = len(point_cloud.points)
print(f"Original Point Count: {original_point_count}")

# Remove invalid points (nan or inf)
points = np.asarray(point_cloud.points)
nan_points = np.isnan(points).any(axis=1)
inf_points = np.isinf(points).any(axis=1)
invalid_points = nan_points | inf_points
print(f"Number of Invalid Points (nan or inf): {invalid_points.sum()}")

if invalid_points.sum() > 0:
    points = points[~invalid_points]
    colors = np.asarray(point_cloud.colors) if point_cloud.has_colors() else None
    if colors is not None:
        colors = colors[~invalid_points]
    point_cloud.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
    print(f"Cleaned Point Count: {len(point_cloud.points)}")

# Parameters for sampling
uniform_ratio = 10  # Take every 10th point
random_ratio = 0.1  # Take 10% of the points randomly

# Prepare log file
log_file = "uniform_random_sampling_results.txt"
with open(log_file, "w") as log:
    # Write an overview of the method
    log.write("### Overview of Uniform/Random Sampling ###\n")
    log.write(
        "Sampling simplifies the point cloud by selecting a subset of points. "
        "Uniform sampling selects points at regular intervals, preserving structure, "
        "while random sampling selects points randomly, suitable for unstructured data.\n\n"
    )

    log.write("### Mathematical Concept ###\n")
    log.write("1. **Uniform Sampling**:\n")
    log.write(
        "   - Uniform sampling selects every `n`th point from the dataset, where `n` is the sampling ratio.\n"
        "   - Mathematically, the indices of selected points are:\n"
        "       I = {i | i mod n == 0, i ∈ {0, 1, ..., N-1}}\n"
        "     where N is the total number of points, and n is the uniform sampling ratio.\n"
    )
    log.write("2. **Random Sampling**:\n")
    log.write(
        "   - Random sampling selects a fixed percentage of points (`p`), chosen randomly.\n"
        "   - Mathematically, a subset of size k = p * N is selected uniformly at random:\n"
        "       S ⊆ {0, 1, ..., N-1}, |S| = k\n"
        "     where N is the total number of points, and p is the random sampling ratio.\n\n"
    )

    log.write("### Parameters ###\n")
    log.write(f"Uniform Ratio: {uniform_ratio}\n")
    log.write(f"Random Ratio: {random_ratio}\n\n")
    log.write(f"Original Point Count: {original_point_count}\n")
    log.write(f"Number of Invalid Points Removed: {invalid_points.sum()}\n")
    log.write(f"Cleaned Point Count: {len(point_cloud.points)}\n\n")

# Uniform Sampling
print("Performing Uniform Sampling...")
start_time = time.time()
indices_uniform = np.arange(0, len(points), uniform_ratio)
uniform_cloud = point_cloud.select_by_index(indices_uniform)
uniform_time = time.time() - start_time
print(f"Uniform Sampling completed in {uniform_time:.2f} seconds")
print(f"Uniform Sampled Point Count: {len(uniform_cloud.points)}")

# Save Uniform Sampled Point Cloud
o3d.io.write_point_cloud(output_uniform_ply, uniform_cloud)
print(f"Uniform sampled point cloud saved to {output_uniform_ply}")

# Log Uniform Sampling
with open(log_file, "a") as log:
    log.write("### Uniform Sampling Results ###\n")
    log.write(f"Uniform Sampled Point Count: {len(uniform_cloud.points)}\n")
    log.write(f"Uniform Sampling Time: {uniform_time:.2f} seconds\n")
    log.write(f"Uniform Sampled Point Cloud File: {output_uniform_ply}\n\n")

# Random Sampling
print("Performing Random Sampling...")
start_time = time.time()
total_points = len(points)
num_random_points = int(random_ratio * total_points)
indices_random = np.random.choice(total_points, num_random_points, replace=False)
random_cloud = point_cloud.select_by_index(indices_random)
random_time = time.time() - start_time
print(f"Random Sampling completed in {random_time:.2f} seconds")
print(f"Random Sampled Point Count: {len(random_cloud.points)}")

# Save Random Sampled Point Cloud
o3d.io.write_point_cloud(output_random_ply, random_cloud)
print(f"Random sampled point cloud saved to {output_random_ply}")

# Log Random Sampling
with open(log_file, "a") as log:
    log.write("### Random Sampling Results ###\n")
    log.write(f"Random Sampled Point Count: {len(random_cloud.points)}\n")
    log.write(f"Random Sampling Time: {random_time:.2f} seconds\n")
    log.write(f"Random Sampled Point Cloud File: {output_random_ply}\n\n")

print(f"Results saved to {log_file}")
