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

# Downsample the point cloud
voxel_size = 0.005  # Adjust voxel size for desired downsampling
downsampled_cloud = original_cloud.voxel_down_sample(voxel_size=voxel_size)
downsampled_point_count = len(downsampled_cloud.points)
print(f"Downsampled Point Count: {downsampled_point_count}")

# Parameters for clustering
eps_list = [0.01, 0.03, 0.05]
min_points_list = [10, 20]

# Prepare output file
log_file = "clustering_simplification_results.txt"
with open(log_file, "w") as log:
    # Write an overview of the method
    log.write("### Overview of Clustering Simplification ###\n")
    log.write(
        "Clustering simplification is a geometric grouping method where points in the point cloud are grouped into "
        "clusters based on proximity. Each cluster can then be simplified, typically by representing it with its "
        "centroid or another reduced form.\n\n"
    )
    log.write("### Key Parameters ###\n")
    log.write("1. **eps (Cluster Radius)**:\n")
    log.write("   - Maximum distance between two points to be considered part of the same cluster.\n")
    log.write("   - Smaller values result in more, smaller clusters. Larger values result in fewer, larger clusters.\n")
    log.write("2. **min_points (Minimum Points per Cluster)**:\n")
    log.write("   - Minimum number of points required to form a cluster.\n")
    log.write("   - Smaller values allow tiny clusters, while larger values filter out noise and small clusters.\n\n")
    log.write("### Mathematical Concept ###\n")
    log.write(
        "Clusters are formed using DBSCAN, which groups points based on density. For each cluster, a centroid can "
        "be computed as the representative point:\n"
        "    C = (1/n) * sum(P_i), for i in 1 to n\n\n"
    )
    log.write(f"Original Point Count: {original_point_count}\n")
    log.write(f"Number of Invalid Points Removed: {invalid_points.sum()}\n")
    log.write(f"Cleaned Point Count: {len(original_cloud.points)}\n\n")
    log.write(f"Downsampled Point Count: {downsampled_point_count}\n\n")

    # Loop through parameter combinations
    for eps in eps_list:
        for min_points in min_points_list:
            log.write(f"Testing eps: {eps}, min_points: {min_points}\n")
            print(f"\nTesting eps: {eps}, min_points: {min_points}")

            # Start timing
            start_time = time.time()

            # Apply DBSCAN clustering
            labels = np.array(downsampled_cloud.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
            max_label = labels.max()

            # Filter only valid clusters (exclude noise points)
            cluster_indices = labels >= 0  # Points with a valid cluster label
            clustered_points = np.asarray(downsampled_cloud.points)[cluster_indices]
            clustered_colors = np.asarray(downsampled_cloud.colors)[cluster_indices]

            # Create a new point cloud for valid clusters
            clustered_cloud = o3d.geometry.PointCloud()
            clustered_cloud.points = o3d.utility.Vector3dVector(clustered_points)
            clustered_cloud.colors = o3d.utility.Vector3dVector(clustered_colors)

            # Create bounding boxes for clusters
            combined_geometry = [clustered_cloud]
            for cluster_id in range(max_label + 1):
                cluster_points = clustered_points[labels[cluster_indices] == cluster_id]
                if len(cluster_points) == 0:
                    continue

                # Create bounding box for the cluster
                cluster_cloud = o3d.geometry.PointCloud()
                cluster_cloud.points = o3d.utility.Vector3dVector(cluster_points)
                bounding_box = cluster_cloud.get_axis_aligned_bounding_box()
                bounding_box.color = (1, 0, 0)  # Red color for bounding box

                # Add bounding box to the combined geometry
                combined_geometry.append(bounding_box)

            # Compute processing time
            processing_time = time.time() - start_time

            # Display clusters with bounding boxes
            o3d.visualization.draw_geometries(combined_geometry, window_name=f"Clusters with Red Squares (eps={eps}, min_points={min_points})")

            # Save the clustered point cloud
            output_file = f"clustered_eps_{eps}_min_{min_points}.ply"
            o3d.io.write_point_cloud(output_file, downsampled_cloud)

            # Log results
            log.write(f"  Number of Clusters Detected: {max_label + 1}\n")
            log.write(f"  Processing Time: {processing_time:.2f} seconds\n")
            log.write(f"  Saved as: {output_file}\n\n")

            # Print results
            print(f"  Number of Clusters Detected: {max_label + 1}")
            print(f"  Processing Time: {processing_time:.2f} seconds")
            print(f"  Saved as: {output_file}")

# Final log message
print(f"\nAll results saved to {log_file}")
