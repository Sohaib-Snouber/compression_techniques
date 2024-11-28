import open3d as o3d
import numpy as np
import time

# Load the original point cloud
original_cloud = o3d.io.read_point_cloud("../../scene0.ply")
original_point_count = len(original_cloud.points)
print(f"Original Point Count: {original_point_count}")

# Remove invalid points (nan or inf)
points = np.asarray(original_cloud.points)
nan_points = np.isnan(points).any(axis=1)
inf_points = np.isinf(points).any(axis=1)
invalid_points = nan_points | inf_points
print(f"Number of Invalid Points (nan or inf): {invalid_points.sum()}")

# Filter out invalid points
if invalid_points.sum() > 0:
    points = points[~invalid_points]
    colors = np.asarray(original_cloud.colors) if original_cloud.has_colors() else None
    if colors is not None:
        colors = colors[~invalid_points]
    original_cloud.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        original_cloud.colors = o3d.utility.Vector3dVector(colors)
    print(f"Cleaned Point Count: {len(original_cloud.points)}")

# Define voxel sizes to test
voxel_sizes = [0.001, 0.005, 0.01]

# Prepare output file
log_file = "voxel_downsampling_results.txt"
with open(log_file, "w") as log:
    # Write an overview of the method
    log.write("### Overview of Voxel Downsampling ###\n")
    log.write(
        "Voxel downsampling is a geometric simplification technique that groups points within fixed-size 3D cubes "
        "(voxels) and replaces them with a representative point, typically the centroid. This method effectively reduces "
        "the number of points while retaining the overall structure of the point cloud.\n\n"
    )
    log.write("### Key Features ###\n")
    log.write("1. **Simplification**: Reduces the number of points in dense areas.\n")
    log.write("2. **Efficiency**: Improves performance in tasks such as rendering, storage, and processing.\n")
    log.write("3. **Customizable Granularity**: The voxel size determines the level of detail retained.\n\n")
    log.write("### Mathematical Concept ###\n")
    log.write(
        "1. **Voxel Indexing**: Each point in the cloud is assigned to a voxel based on its coordinates:\n"
        "    Voxel_index = floor(Point_coordinate / Voxel_size)\n"
        "2. **Representative Point**: The points within each voxel are replaced with a single point, such as the centroid:\n"
        "    Centroid = mean(Points_in_voxel)\n"
        "3. **Color Retention**: The color of the representative point is computed as the average color of all points in the voxel.\n\n"
    )
    log.write(f"Original Point Count: {original_point_count}\n")
    log.write(f"Number of Invalid Points Removed: {invalid_points.sum()}\n")
    log.write(f"Cleaned Point Count: {len(original_cloud.points)}\n\n")


    # Function to compute Chamfer distance
    def chamfer_distance(cloud1, cloud2):
        dist1 = np.asarray(cloud1.compute_point_cloud_distance(cloud2))
        dist2 = np.asarray(cloud2.compute_point_cloud_distance(cloud1))
        return np.mean(dist1) + np.mean(dist2)

    # Loop through each voxel size
    for voxel_size in voxel_sizes:
        log.write(f"Testing Voxel Size: {voxel_size} meters\n")
        print(f"\nTesting Voxel Size: {voxel_size} meters")

        # Start timing
        start_time = time.time()

        # Apply voxel downsampling
        downsampled_cloud = original_cloud.voxel_down_sample(voxel_size=voxel_size)
        downsampled_point_count = len(downsampled_cloud.points)

        # Compute processing time
        processing_time = time.time() - start_time

        # Compute compression ratio
        compression_ratio = original_point_count / downsampled_point_count

        # Compute Chamfer distance
        fidelity_loss = chamfer_distance(original_cloud, downsampled_cloud)

        # Save the downsampled point cloud
        output_file = f"voxel_downsampled_{voxel_size}.ply"
        o3d.io.write_point_cloud(output_file, downsampled_cloud)

        # Log results
        log.write(f"  Downsampled Point Count: {downsampled_point_count}\n")
        log.write(f"  Compression Ratio: {compression_ratio:.2f}\n")
        log.write(f"  Processing Time: {processing_time:.2f} seconds\n")
        log.write(f"  Fidelity Loss (Chamfer Distance): {fidelity_loss:.6f}\n")
        log.write(f"  Saved as: {output_file}\n\n")

        # Print results
        print(f"  Downsampled Point Count: {downsampled_point_count}")
        print(f"  Compression Ratio: {compression_ratio:.2f}")
        print(f"  Processing Time: {processing_time:.2f} seconds")
        print(f"  Fidelity Loss (Chamfer Distance): {fidelity_loss:.6f}")
        print(f"  Saved as: {output_file}")

# Final log message
print(f"\nAll results saved to {log_file}")
