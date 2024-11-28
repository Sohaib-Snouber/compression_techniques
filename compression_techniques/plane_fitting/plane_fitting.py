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

# Parameters for RANSAC plane fitting
distance_thresholds = [0.01, 0.014]
num_iterations_list = [500, 1000, 10000]

# Prepare output file
log_file = "plane_fitting_results.txt"
with open(log_file, "w") as log:
    # Write an overview of the method
    log.write("### Overview of Plane Fitting (Region Growing) ###\n")
    log.write(
        "Plane fitting is a segmentation technique to extract planar surfaces from point clouds. Using RANSAC, the algorithm "
        "iteratively estimates the parameters of a plane by minimizing the distance of inliers to the plane. Region growing can "
        "expand the segmented plane by clustering neighboring points with similar normals.\n\n"
    )
    log.write("### Key Parameters ###\n")
    log.write("1. **Distance Threshold**:\n")
    log.write("   - Defines the maximum distance a point can have to the plane to be considered an inlier.\n")
    log.write("   - Smaller values: More precise plane fitting but fewer inliers.\n")
    log.write("   - Larger values: More inliers but risks over-segmentation.\n")
    log.write("2. **Number of Iterations**:\n")
    log.write("   - Higher values lead to better plane fitting but slower computation.\n")
    log.write("3. **Region Growing (Normal Similarity)**:\n")
    log.write("   - Clusters points based on proximity and similarity in normal vectors to expand planar regions.\n\n")
    log.write(f"Original Point Count: {original_point_count}\n")
    log.write(f"Number of Invalid Points Removed: {invalid_points.sum()}\n")
    log.write(f"Cleaned Point Count: {len(original_cloud.points)}\n\n")

    # Loop through parameter combinations
    for distance_threshold in distance_thresholds:
        for num_iterations in num_iterations_list:
            log.write(f"Testing distance_threshold: {distance_threshold}, num_iterations: {num_iterations}\n")
            print(f"\nTesting distance_threshold: {distance_threshold}, num_iterations: {num_iterations}")

            # Start timing
            start_time = time.time()

            # Apply RANSAC for plane segmentation
            plane_model, inliers = original_cloud.segment_plane(
                distance_threshold=distance_threshold,
                ransac_n=3,
                num_iterations=num_iterations,
            )
            inlier_cloud = original_cloud.select_by_index(inliers)
            outlier_cloud = original_cloud.select_by_index(inliers, invert=True)

            # Compute processing time
            processing_time = time.time() - start_time

            # Save the segmented plane and remaining points
            inlier_output = f"plane_inliers_dist_{distance_threshold}_iter_{num_iterations}.ply"
            outlier_output = f"plane_outliers_dist_{distance_threshold}_iter_{num_iterations}.ply"
            o3d.io.write_point_cloud(inlier_output, inlier_cloud)
            o3d.io.write_point_cloud(outlier_output, outlier_cloud)

            # Log results
            log.write(f"  Plane Model Coefficients: {plane_model}\n")
            log.write(f"  Number of Inliers: {len(inliers)}\n")
            log.write(f"  Processing Time: {processing_time:.2f} seconds\n")
            log.write(f"  Saved Plane Inliers as: {inlier_output}\n")
            log.write(f"  Saved Remaining Points as: {outlier_output}\n\n")

            # Print results
            print(f"  Plane Model Coefficients: {plane_model}")
            print(f"  Number of Inliers: {len(inliers)}")
            print(f"  Processing Time: {processing_time:.2f} seconds")
            print(f"  Saved Plane Inliers as: {inlier_output}")
            print(f"  Saved Remaining Points as: {outlier_output}")

# Final log message
print(f"\nAll results saved to {log_file}")
