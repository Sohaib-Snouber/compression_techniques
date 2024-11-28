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

# Parameters for statistical outlier removal
nb_neighbors_list = [10, 20, 30]
std_ratios = [1.0, 2.0, 3.0]

# Prepare output file
log_file = "statistical_filtering_results.txt"
with open(log_file, "w") as log:
    # Write an overview of the method
    log.write("### Overview of Statistical Filtering ###\n")
    log.write(
        "Statistical filtering is a noise reduction technique that removes outlier points based on statistical analysis "
        "of the local neighborhood. By examining the average distance to neighbors and removing points that deviate "
        "significantly, this method ensures a cleaner point cloud.\n\n"
    )
    log.write("### Key Features ###\n")
    log.write("1. **Noise Removal**: Eliminates sparse or isolated points.\n")
    log.write("2. **Improves Data Quality**: Suitable for preprocessing before further operations.\n")
    log.write("3. **Customizable Parameters**: Number of neighbors and standard deviation ratio can be adjusted.\n\n")
    log.write("### Parameter Explanations ###\n")
    log.write("1. **nb_neighbors (Number of Neighbors)**:\n")
    log.write("   - Defines the number of nearest neighbors to consider for each point.\n")
    log.write("   - Higher values:\n")
    log.write("     - Capture a larger local neighborhood.\n")
    log.write("     - Outlier removal becomes less sensitive, resulting in fewer points being removed.\n")
    log.write("   - Lower values:\n")
    log.write("     - Capture a smaller local neighborhood.\n")
    log.write("     - Outlier removal becomes more sensitive, potentially removing more points.\n")
    log.write("2. **std_ratio (Standard Deviation Multiplier)**:\n")
    log.write("   - Sets the threshold for outlier classification as a multiplier of the standard deviation.\n")
    log.write("   - Higher values:\n")
    log.write("     - Relax the threshold, tolerating more deviation and removing fewer points.\n")
    log.write("   - Lower values:\n")
    log.write("     - Tighten the threshold, allowing only points close to their neighbors to remain.\n\n")
    log.write("### Choosing Parameters ###\n")
    log.write("1. High Noise Data:\n")
    log.write("   - Use low `nb_neighbors` and low `std_ratio` for aggressive filtering.\n")
    log.write("2. Low Noise Data:\n")
    log.write("   - Use high `nb_neighbors` and high `std_ratio` for gentle filtering.\n\n")
    log.write(f"Original Point Count: {original_point_count}\n")
    log.write(f"Number of Invalid Points Removed: {invalid_points.sum()}\n")
    log.write(f"Cleaned Point Count: {len(original_cloud.points)}\n\n")

    # Loop through combinations of parameters
    for nb_neighbors in nb_neighbors_list:
        for std_ratio in std_ratios:
            log.write(f"Testing nb_neighbors: {nb_neighbors}, std_ratio: {std_ratio}\n")
            print(f"\nTesting nb_neighbors: {nb_neighbors}, std_ratio: {std_ratio}")

            # Start timing
            start_time = time.time()

            # Apply statistical filtering
            cl, ind = original_cloud.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
            filtered_cloud = original_cloud.select_by_index(ind)
            filtered_point_count = len(filtered_cloud.points)

            # Compute processing time
            processing_time = time.time() - start_time

            # Compute compression ratio
            compression_ratio = original_point_count / filtered_point_count

            # Save the filtered point cloud
            output_file = f"statistical_filtered_nb_{nb_neighbors}_std_{std_ratio}.ply"
            o3d.io.write_point_cloud(output_file, filtered_cloud)

            # Log results
            log.write(f"  Filtered Point Count: {filtered_point_count}\n")
            log.write(f"  Compression Ratio: {compression_ratio:.2f}\n")
            log.write(f"  Processing Time: {processing_time:.2f} seconds\n")
            log.write(f"  Saved as: {output_file}\n\n")

            # Print results
            print(f"  Filtered Point Count: {filtered_point_count}")
            print(f"  Compression Ratio: {compression_ratio:.2f}")
            print(f"  Processing Time: {processing_time:.2f} seconds")
            print(f"  Saved as: {output_file}")

# Final log message
print(f"\nAll results saved to {log_file}")
