import open3d as o3d
import numpy as np

def clean_and_visualize(file_path, output_file):
    # Load the point cloud
    point_cloud = o3d.io.read_point_cloud(file_path)
    print(f"Original Point Count: {len(point_cloud.points)}")

    # Remove invalid points (NaN or Inf)
    points = np.asarray(point_cloud.points)
    nan_points = np.isnan(points).any(axis=1)
    inf_points = np.isinf(points).any(axis=1)
    invalid_points = nan_points | inf_points
    print(f"Number of Invalid Points (NaN or Inf): {invalid_points.sum()}")

    # Filter out invalid points
    if invalid_points.sum() > 0:
        points = points[~invalid_points]
        colors = np.asarray(point_cloud.colors) if point_cloud.has_colors() else None
        if colors is not None:
            colors = colors[~invalid_points]
        point_cloud.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            point_cloud.colors = o3d.utility.Vector3dVector(colors)
        print(f"Cleaned Point Count: {len(point_cloud.points)}")

    # Save the cleaned point cloud
    o3d.io.write_point_cloud(output_file, point_cloud)
    print(f"Cleaned point cloud saved to: {output_file}")

    # Visualize the cleaned point cloud
    o3d.visualization.draw_geometries(
        [point_cloud],
        window_name="Cleaned Point Cloud",
        width=800,
        height=600
    )

# File paths
input_file = "scene0.ply"  # Replace with your input .ply file
output_file = "cleaned_scene0.ply"  # Output file name

# Clean and visualize
clean_and_visualize(input_file, output_file)
