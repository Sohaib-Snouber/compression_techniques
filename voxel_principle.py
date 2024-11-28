import open3d as o3d
import numpy as np

# Load the point cloud
point_cloud = o3d.io.read_point_cloud("scene0.ply")
print(f"Original Point Count: {len(point_cloud.points)}")

# Convert points and colors to NumPy arrays
points = np.asarray(point_cloud.points)
colors = np.asarray(point_cloud.colors) if point_cloud.has_colors() else None

# Check for invalid points (nan or inf)
nan_points = np.isnan(points).any(axis=1)
inf_points = np.isinf(points).any(axis=1)
invalid_points = nan_points | inf_points
print(f"Number of Invalid Points (nan or inf): {invalid_points.sum()}")

# Remove invalid points
if invalid_points.sum() > 0:
    points = points[~invalid_points]
    if colors is not None:
        colors = colors[~invalid_points]
    print(f"Cleaned Point Count: {len(points)}")

# Voxel downsampling with color retention
voxel_size = 0.005  # Adjust voxel size as needed
voxel_indices = np.floor(points / voxel_size).astype(np.int32)
unique_voxels, inverse_indices = np.unique(voxel_indices, axis=0, return_inverse=True)

# Compute voxel centers (representative points)
voxel_centers = unique_voxels * voxel_size + voxel_size / 2

# Compute average colors for each voxel
downsampled_colors = None
if colors is not None:
    downsampled_colors = np.zeros((len(unique_voxels), 3))  # Initialize color array
    for i in range(len(unique_voxels)):
        voxel_mask = inverse_indices == i
        downsampled_colors[i] = np.mean(colors[voxel_mask], axis=0)  # Average color

# Create the downsampled point cloud
downsampled_point_cloud = o3d.geometry.PointCloud()
downsampled_point_cloud.points = o3d.utility.Vector3dVector(voxel_centers)
if downsampled_colors is not None:
    downsampled_point_cloud.colors = o3d.utility.Vector3dVector(downsampled_colors)

print(f"Downsampled Point Count: {len(downsampled_point_cloud.points)}")

# Visualize the downsampled point cloud
o3d.visualization.draw_geometries([downsampled_point_cloud])
