import open3d as o3d

# Function to load and visualize a .ply file
def visualize_ply(file_path):
    # Load the point cloud
    point_cloud = o3d.io.read_point_cloud(file_path)
    print(f"Loaded Point Cloud: {file_path}")
    print(f"Number of Points: {len(point_cloud.points)}")
    
    # Check if the point cloud has colors
    if point_cloud.has_colors():
        print("Point Cloud has colors.")
    else:
        print("Point Cloud does not have colors.")

    # Visualize the point cloud
    o3d.visualization.draw_geometries([point_cloud], 
                                      window_name="Point Cloud Visualization",
                                      width=800, 
                                      height=600)

# Path to your .ply file
file_path = "compression_techniques/autoencoders_ML/reconstructed_autoencoder.ply"  # Replace with your .ply file path

# Visualize the point cloud
visualize_ply(file_path)
