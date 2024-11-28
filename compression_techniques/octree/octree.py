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

# Define octree depths to test
octree_depths = [4, 6, 8]

# Prepare output file
log_file = "octree_compression_results.txt"
with open(log_file, "w") as log:
    # Write an overview of the method
    log.write("### Overview of Octree Compression ###\n")
    log.write(
        "Octree compression is a hierarchical representation method that divides a 3D space into smaller subspaces "
        "called octants. Starting with a root node representing the entire space, the method recursively subdivides "
        "each octant into eight smaller octants, stopping at a predefined depth. The depth determines the level of detail.\n\n"
    )
    log.write("### Key Features ###\n")
    log.write("1. **Hierarchical Representation**: Efficiently represents 3D space in a tree structure.\n")
    log.write("2. **Compression**: Reduces point count by approximating multiple points within a voxel.\n")
    log.write("3. **Applications**: Commonly used in rendering, storage, and transmission of large 3D datasets.\n\n")
    log.write("### Mathematical Concept ###\n")
    log.write(
        "Each node in the octree represents a voxel, defined by its origin and size. The mathematical basis involves:\n"
        "  - **Subdivision**: Divide each voxel into eight smaller ones, represented by the formula:\n"
        "    Voxel_center[i] = Voxel_origin + (i * Voxel_size / 2), for i in {0, ..., 7}\n"
        "  - **Storage**: Store points in leaf nodes, which represent the smallest voxels.\n"
        "  - **Reconstruction**: Use the center of each voxel or average of contained points for reconstruction.\n\n"
    )
    log.write(f"Original Point Count: {original_point_count}\n")
    log.write(f"Number of Invalid Points Removed: {invalid_points.sum()}\n")
    log.write(f"Cleaned Point Count: {len(original_cloud.points)}\n\n")

    # Function to compute Chamfer distance
    def chamfer_distance(cloud1, cloud2):
        dist1 = np.asarray(cloud1.compute_point_cloud_distance(cloud2))
        dist2 = np.asarray(cloud2.compute_point_cloud_distance(cloud1))
        return np.mean(dist1) + np.mean(dist2)

    # Function to reconstruct point cloud from octree with colors
    def reconstruct_from_octree_with_colors(octree, original_cloud):
        points = []
        colors = []
        original_points = np.asarray(original_cloud.points)
        original_colors = np.asarray(original_cloud.colors)

        # Traversal to collect points and average colors from leaf nodes
        def collect_points_and_colors(node, node_info):
            if isinstance(node, o3d.geometry.OctreeLeafNode):  # Check if the node is a leaf
                voxel_center = node_info.origin
                points.append(voxel_center)

                # Collect indices for points in this voxel
                indices = node.indices
                voxel_colors = original_colors[indices]
                averaged_color = np.mean(voxel_colors, axis=0)  # Average the colors
                colors.append(averaged_color)

        octree.traverse(collect_points_and_colors)
        return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array(points))), o3d.utility.Vector3dVector(np.array(colors))

    # Loop through each octree depth
    for depth in octree_depths:
        log.write(f"Testing Octree Depth: {depth}\n")
        print(f"\nTesting Octree Depth: {depth}")

        # Start timing
        start_time = time.time()

        # Build octree
        octree = o3d.geometry.Octree(max_depth=depth)
        octree.convert_from_point_cloud(original_cloud, size_expand=0.01)

        # Reconstruct point cloud from octree
        reconstructed_cloud, reconstructed_colors = reconstruct_from_octree_with_colors(octree, original_cloud)
        reconstructed_cloud.colors = reconstructed_colors
        reconstructed_point_count = len(reconstructed_cloud.points)

        # Compute processing time
        processing_time = time.time() - start_time

        # Compute compression ratio
        compression_ratio = original_point_count / reconstructed_point_count

        # Compute Chamfer distance
        fidelity_loss = chamfer_distance(original_cloud, reconstructed_cloud)

        # Save the reconstructed point cloud
        output_file = f"octree_reconstructed_depth_{depth}.ply"
        o3d.io.write_point_cloud(output_file, reconstructed_cloud)

        # Log results
        log.write(f"  Reconstructed Point Count: {reconstructed_point_count}\n")
        log.write(f"  Compression Ratio: {compression_ratio:.2f}\n")
        log.write(f"  Processing Time: {processing_time:.2f} seconds\n")
        log.write(f"  Fidelity Loss (Chamfer Distance): {fidelity_loss:.6f}\n")
        log.write(f"  Saved as: {output_file}\n\n")

        # Print results
        print(f"  Reconstructed Point Count: {reconstructed_point_count}")
        print(f"  Compression Ratio: {compression_ratio:.2f}")
        print(f"  Processing Time: {processing_time:.2f} seconds")
        print(f"  Fidelity Loss (Chamfer Distance): {fidelity_loss:.6f}")
        print(f"  Saved as: {output_file}")

# Final log message
print(f"\nAll results saved to {log_file}")
