import open3d as o3d
import numpy as np
import pywt  # PyWavelets library for wavelet transform
import json
import time

# File paths
input_ply = "../../scene0.ply"
compressed_file = "wavelet_compressed_points.json"
output_ply_file = "wavelet_decompressed_point_cloud.ply"

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

# Parameters for wavelet compression
wavelet = 'haar'  # Wavelet type
threshold = 0.01  # Coefficient threshold

# Compress point cloud with wavelet transform
print("Performing Wavelet Transform...")
start_time = time.time()

# Apply wavelet transform to each dimension
compressed_data = {}
for dim, axis_name in enumerate(['x', 'y', 'z']):
    coeffs = pywt.wavedec(points[:, dim], wavelet)
    coeffs_thresholded = [pywt.threshold(c, threshold * max(c)) for c in coeffs]
    compressed_data[axis_name] = [c.tolist() for c in coeffs_thresholded]  # Convert to list

if colors is not None:
    compressed_data["colors"] = (colors * 255).astype(np.uint8).tolist()  # Save colors

compression_time = time.time() - start_time
print(f"Wavelet Transform completed in {compression_time:.2f} seconds")

# Save compressed data
with open(compressed_file, "w") as f:
    json.dump(compressed_data, f)
print(f"Compressed point cloud saved to {compressed_file}")

# Decompress point cloud
print("Reconstructing Point Cloud...")
start_time = time.time()

# Reconstruct each dimension
reconstructed_points = []
for axis_name in ['x', 'y', 'z']:
    coeffs_list = compressed_data[axis_name]
    coeffs_arrays = [np.array(c) for c in coeffs_list]  # Convert lists back to numpy arrays
    reconstructed_points.append(pywt.waverec(coeffs_arrays, wavelet))
reconstructed_points = np.vstack(reconstructed_points).T

if "colors" in compressed_data:
    reconstructed_colors = np.array(compressed_data["colors"]) / 255.0  # Reconstruct colors
else:
    reconstructed_colors = None

reconstruction_time = time.time() - start_time
print(f"Reconstruction completed in {reconstruction_time:.2f} seconds")

# Create reconstructed point cloud
reconstructed_cloud = o3d.geometry.PointCloud()
reconstructed_cloud.points = o3d.utility.Vector3dVector(reconstructed_points)
if reconstructed_colors is not None:
    reconstructed_cloud.colors = o3d.utility.Vector3dVector(reconstructed_colors)

# Save the reconstructed point cloud
o3d.io.write_point_cloud(output_ply_file, reconstructed_cloud)
print(f"Reconstructed point cloud saved to: {output_ply_file}")

# Log results
log_file = "wavelet_compression_results.txt"
with open(log_file, "w") as log:
    log.write("### Overview of Wavelet Transform ###\n")
    log.write(
        "Wavelet Transform compresses point cloud data by decomposing it into approximation and detail coefficients. "
        "Significant coefficients are retained, and minor ones are discarded, achieving lossy compression with "
        "a focus on preserving major variations.\n\n"
    )
    log.write("### Parameters ###\n")
    log.write(f"Wavelet Type: {wavelet}\n")
    log.write(f"Threshold: {threshold}\n\n")
    log.write(f"Original Point Count: {original_point_count}\n")
    log.write(f"Cleaned Point Count: {len(point_cloud.points)}\n")
    log.write(f"Compression Time: {compression_time:.2f} seconds\n")
    log.write(f"Reconstruction Time: {reconstruction_time:.2f} seconds\n")
    log.write(f"Compressed File Size: {len(json.dumps(compressed_data))} bytes\n")
    log.write(f"Compressed File Saved As: {compressed_file}\n")
    log.write(f"Reconstructed Point Cloud File: {output_ply_file}\n")

print(f"Results saved to {log_file}")
