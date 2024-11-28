import open3d as o3d
import numpy as np
import time

# Load the original point cloud
point_cloud = o3d.io.read_point_cloud("../../scene0.ply")
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

# Parameters for quantization
resolutions = [0.01, 0.005, 0.001]  # Test multiple resolutions

# Prepare output file
log_file = "quantization_results.txt"
with open(log_file, "w") as log:
    # Write an overview of the method
    log.write("### Overview of Quantization ###\n")
    log.write(
        "Quantization reduces the precision of point cloud coordinates to compress data size. "
        "This is achieved by rounding coordinates to the nearest multiple of a given resolution.\n\n"
    )
    log.write("### Key Parameters ###\n")
    log.write("1. **Resolution**:\n")
    log.write("   - Specifies the granularity of the quantized values (e.g., 0.001 for millimeter precision).\n")
    log.write("   - Higher resolution retains more detail but compresses less.\n\n")
    log.write(f"Original Point Count: {original_point_count}\n")
    log.write(f"Number of Invalid Points Removed: {invalid_points.sum()}\n")
    log.write(f"Cleaned Point Count: {len(point_cloud.points)}\n\n")

    # Loop through resolutions
    for resolution in resolutions:
        print(f"\nTesting quantization with resolution: {resolution}")
        log.write(f"Testing Resolution: {resolution}\n")

        # Start timing
        start_time = time.time()

        # Apply quantization
        quantized_points = np.round(points / resolution) * resolution
        quantized_cloud = o3d.geometry.PointCloud()
        quantized_cloud.points = o3d.utility.Vector3dVector(quantized_points)

        # Preserve colors if present
        if colors is not None:
            quantized_cloud.colors = o3d.utility.Vector3dVector(colors)

        # Compute processing time
        processing_time = time.time() - start_time

        # Save the quantized point cloud
        output_file = f"quantized_resolution_{str(resolution).replace('.', '_')}.ply"
        o3d.io.write_point_cloud(output_file, quantized_cloud)

        # Log results
        log.write(f"  Quantized Point Count: {len(quantized_cloud.points)}\n")
        log.write(f"  Processing Time: {processing_time:.2f} seconds\n")
        log.write(f"  Saved as: {output_file}\n\n")
        print(f"  Quantized Point Count: {len(quantized_cloud.points)}")
        print(f"  Processing Time: {processing_time:.2f} seconds")
        print(f"  Saved as: {output_file}")

# Final log message
print(f"\nAll results saved to {log_file}")
