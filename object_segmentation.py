import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# Create output directory for saving clustered point clouds
output_dir = "segmentation_results"
os.makedirs(output_dir, exist_ok=True)

# Start logging
log_file = os.path.join(output_dir, "segmentation_test_log.txt")

with open(log_file, "w") as log:
    total_start_time = time.time()

    # Load the point cloud
    point_cloud = o3d.io.read_point_cloud("scene0.ply")
    log.write(f"Original Point Count: {len(point_cloud.points)}\n")
    print(f"Original Point Count: {len(point_cloud.points)}")

    # Start the actual processing timer
    processing_start_time = time.time()

    # Downsample the point cloud
    voxel_size = 0.01
    point_cloud_downsampled = point_cloud.voxel_down_sample(voxel_size=voxel_size)
    downsampled_count = len(point_cloud_downsampled.points)
    print(f"Downsampled Point Count: {downsampled_count}")

    # Remove noise
    cl, ind = point_cloud_downsampled.remove_statistical_outlier(
        nb_neighbors=10, std_ratio=2.5)
    point_cloud_filtered = point_cloud_downsampled.select_by_index(ind)
    filtered_count = len(point_cloud_filtered.points)

    # Segment the largest plane using RANSAC
    plane_model, inliers = point_cloud_filtered.segment_plane(
        distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    inlier_cloud = point_cloud_filtered.select_by_index(inliers)
    outlier_cloud = point_cloud_filtered.select_by_index(inliers, invert=True)

    # Cluster the remaining points
    cluster_start_time = time.time()
    labels = np.array(outlier_cloud.cluster_dbscan(eps=0.05, min_points=10, print_progress=True))
    cluster_time = time.time() - cluster_start_time
    max_label = labels.max()

    # Find cluster centroids
    cluster_centroids = []
    for cluster_id in range(max_label + 1):
        cluster_points = np.asarray(outlier_cloud.points)[labels == cluster_id]
        if len(cluster_points) > 0:
            centroid = np.mean(cluster_points, axis=0)
            cluster_centroids.append(centroid)

    # Stop the processing timer
    processing_end_time = time.time()
    processing_time = processing_end_time - processing_start_time

    # Save plane, clusters, and centroids
    save_start_time = time.time()

    # Save the segmented plane
    inlier_cloud_path = os.path.join(output_dir, "plane_inliers.ply")
    o3d.io.write_point_cloud(inlier_cloud_path, inlier_cloud)

    # Save each cluster
    for cluster_id, centroid in enumerate(cluster_centroids):
        cluster_points = np.asarray(outlier_cloud.points)[labels == cluster_id]
        cluster_cloud = o3d.geometry.PointCloud()
        cluster_cloud.points = o3d.utility.Vector3dVector(cluster_points)
        cluster_path = os.path.join(output_dir, f"cluster_{cluster_id}.ply")
        o3d.io.write_point_cloud(cluster_path, cluster_cloud)

    # Save centroids
    centroid_cloud = o3d.geometry.PointCloud()
    centroid_cloud.points = o3d.utility.Vector3dVector(np.array(cluster_centroids))
    centroids_path = os.path.join(output_dir, "centroids.ply")
    o3d.io.write_point_cloud(centroids_path, centroid_cloud)

    # Stop the saving timer
    save_end_time = time.time()
    save_time = save_end_time - save_start_time

    # Log results
    log.write(f"Downsampled Point Count (voxel size {voxel_size}): {downsampled_count}\n")
    log.write(f"Filtered Point Count (after noise removal): {filtered_count}\n")
    log.write(f"Plane Model Coefficients: {plane_model}\n")
    log.write(f"Points in the plane (inliers): {len(inliers)}\n")
    log.write(f"Points after plane removal (outliers): {len(outlier_cloud.points)}\n")
    log.write(f"Number of Clusters Detected: {max_label + 1}\n")
    log.write(f"Clustering Time: {cluster_time:.2f} seconds\n")
    for cluster_id, centroid in enumerate(cluster_centroids):
        log.write(f"Cluster {cluster_id} Centroid: {centroid.tolist()}\n")
    log.write(f"Processing Time (Excluding Saving and Logging): {processing_time:.2f} seconds\n")
    log.write(f"File Saving Time: {save_time:.2f} seconds\n")

    # Log total runtime
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    log.write(f"Total Runtime (Including Saving and Logging): {total_time:.2f} seconds\n")

    print(f"Processing Time (Excluding Saving and Logging): {processing_time:.2f} seconds")
    print(f"File Saving Time: {save_time:.2f} seconds")
    print(f"Total Runtime (Including Saving and Logging): {total_time:.2f} seconds")
