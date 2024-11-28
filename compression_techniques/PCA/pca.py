# import open3d as o3d
# import numpy as np
# import time
# from sklearn.decomposition import PCA

# # File paths
# input_ply = "../../scene0.ply"
# compressed_file = "pca_compressed_points.npz"
# output_ply_file = "pca_decompressed_point_cloud.ply"

# # Load the original point cloud
# point_cloud = o3d.io.read_point_cloud(input_ply)
# original_point_count = len(point_cloud.points)
# print(f"Original Point Count: {original_point_count}")

# # Remove invalid points
# points = np.asarray(point_cloud.points)
# nan_points = np.isnan(points).any(axis=1)
# inf_points = np.isinf(points).any(axis=1)
# invalid_points = nan_points | inf_points
# print(f"Number of Invalid Points (nan or inf): {invalid_points.sum()}")

# if invalid_points.sum() > 0:
#     points = points[~invalid_points]
#     colors = np.asarray(point_cloud.colors) if point_cloud.has_colors() else None
#     if colors is not None:
#         colors = colors[~invalid_points]
#     point_cloud.points = o3d.utility.Vector3dVector(points)
#     if colors is not None:
#         point_cloud.colors = o3d.utility.Vector3dVector(colors)
#     print(f"Cleaned Point Count: {len(point_cloud.points)}")

# # Perform PCA Compression
# print("Performing PCA Compression...")
# start_time = time.time()

# # Fit PCA and transform the points
# pca = PCA(n_components=3)  # Keep all 3 components (dimensionality doesn't change but data is compressed)
# transformed_points = pca.fit_transform(points)
# pca_time = time.time() - start_time
# print(f"PCA Compression completed in {pca_time:.2f} seconds")

# # Save compressed data
# np.savez(compressed_file, transformed_points=transformed_points, mean=pca.mean_, components=pca.components_)
# print(f"Compressed point cloud saved to {compressed_file}")

# # Perform PCA Decompression
# print("Reconstructing Point Cloud...")
# start_time = time.time()
# compressed_data = np.load(compressed_file)
# reconstructed_points = np.dot(compressed_data["transformed_points"], compressed_data["components"]) + compressed_data["mean"]
# reconstruction_time = time.time() - start_time
# print(f"Reconstruction completed in {reconstruction_time:.2f} seconds")

# # Create reconstructed point cloud
# reconstructed_cloud = o3d.geometry.PointCloud()
# reconstructed_cloud.points = o3d.utility.Vector3dVector(reconstructed_points)
# if colors is not None:
#     reconstructed_cloud.colors = o3d.utility.Vector3dVector(colors)

# # Save the reconstructed point cloud
# o3d.io.write_point_cloud(output_ply_file, reconstructed_cloud)
# print(f"Reconstructed point cloud saved to: {output_ply_file}")

# # Log results
# log_file = "pca_compression_results.txt"
# with open(log_file, "w") as log:
#     log.write("### Overview of PCA Compression ###\n")
#     log.write(
#         "Principal Component Analysis (PCA) reduces data dimensionality by projecting points onto the most significant axes of variation. "
#         "This retains the structure of the data while reducing redundancy and minimizing storage size.\n\n"
#     )
#     log.write("### Mathematical Concept ###\n")
#     log.write("1. **Compression**:\n")
#     log.write(
#         "   - PCA computes the covariance matrix of the data, extracting eigenvectors (principal axes) and eigenvalues (variance).\n"
#         "   - Data is projected onto these axes to create a transformed (compressed) representation.\n"
#     )
#     log.write("2. **Decompression**:\n")
#     log.write(
#         "   - Transformed data is projected back onto the original space using the inverse of the PCA transformation.\n\n"
#     )
#     log.write(f"### Results ###\n")
#     log.write(f"Original Point Count: {original_point_count}\n")
#     log.write(f"Cleaned Point Count: {len(point_cloud.points)}\n")
#     log.write(f"PCA Compression Time: {pca_time:.2f} seconds\n")
#     log.write(f"PCA Decompression Time: {reconstruction_time:.2f} seconds\n")
#     log.write(f"Compressed File Size: {compressed_file}\n")
#     log.write(f"Reconstructed Point Cloud File: {output_ply_file}\n")

# print(f"Results saved to {log_file}")


import open3d as o3d
import numpy as np
import time
from sklearn.decomposition import PCA

# File paths
input_ply = "../../scene0.ply"
compressed_file = "pca_compressed_points_with_colors.npz"
output_ply_file = "pca_decompressed_point_cloud_with_colors.ply"

# Load the original point cloud
point_cloud = o3d.io.read_point_cloud(input_ply)
original_point_count = len(point_cloud.points)
print(f"Original Point Count: {original_point_count}")

# Remove invalid points
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
else:
    colors = None

# Combine points and colors for PCA
if colors is not None:
    data = np.hstack((points, colors))  # Combine points and colors into a single array
else:
    data = points

# Perform PCA Compression
print("Performing PCA Compression...")
start_time = time.time()

# Fit PCA and transform the data
pca = PCA(n_components=data.shape[1])  # Keep all components (points + colors)
transformed_data = pca.fit_transform(data)
transformed_data = np.round(transformed_data, decimals=3).astype(np.float32)
pca_time = time.time() - start_time
print(f"PCA Compression completed in {pca_time:.2f} seconds")

# Save compressed data
np.savez_compressed(compressed_file, transformed_data=transformed_data, mean=pca.mean_, components=pca.components_)
print(f"Compressed point cloud saved to {compressed_file}")

# Perform PCA Decompression
print("Reconstructing Point Cloud...")
start_time = time.time()
compressed_data = np.load(compressed_file)
reconstructed_data = np.dot(compressed_data["transformed_data"], compressed_data["components"]) + compressed_data["mean"]
reconstruction_time = time.time() - start_time
print(f"Reconstruction completed in {reconstruction_time:.2f} seconds")

# Split reconstructed data into points and colors
if colors is not None:
    reconstructed_points = reconstructed_data[:, :3]  # First 3 columns are points
    reconstructed_colors = reconstructed_data[:, 3:]  # Last 3 columns are colors
else:
    reconstructed_points = reconstructed_data
    reconstructed_colors = None

# Create reconstructed point cloud
reconstructed_cloud = o3d.geometry.PointCloud()
reconstructed_cloud.points = o3d.utility.Vector3dVector(reconstructed_points)
if colors is not None:
    reconstructed_cloud.colors = o3d.utility.Vector3dVector(np.clip(reconstructed_colors, 0, 1))  # Clip colors to valid range

# Save the reconstructed point cloud
o3d.io.write_point_cloud(output_ply_file, reconstructed_cloud)
print(f"Reconstructed point cloud saved to: {output_ply_file}")

# Log results
log_file = "pca_compression_results_with_colors.txt"
with open(log_file, "w") as log:
    log.write("### Overview of PCA Compression ###\n")
    log.write(
        "Principal Component Analysis (PCA) reduces data dimensionality by projecting points and colors onto the most significant axes of variation. "
        "This retains the structure and appearance of the data while reducing redundancy and minimizing storage size.\n\n"
    )
    log.write("### Mathematical Concept ###\n")
    log.write("1. **Compression**:\n")
    log.write(
        "   - PCA computes the covariance matrix of the data (points + colors), extracting eigenvectors (principal axes) and eigenvalues (variance).\n"
        "   - Data is projected onto these axes to create a transformed (compressed) representation.\n"
    )
    log.write("2. **Decompression**:\n")
    log.write(
        "   - Transformed data is projected back onto the original space using the inverse of the PCA transformation.\n\n"
    )
    log.write(f"### Results ###\n")
    log.write(f"Original Point Count: {original_point_count}\n")
    log.write(f"Cleaned Point Count: {len(point_cloud.points)}\n")
    log.write(f"PCA Compression Time: {pca_time:.2f} seconds\n")
    log.write(f"PCA Decompression Time: {reconstruction_time:.2f} seconds\n")
    log.write(f"Compressed File Size: {compressed_file}\n")
    log.write(f"Reconstructed Point Cloud File: {output_ply_file}\n")

print(f"Results saved to {log_file}")
