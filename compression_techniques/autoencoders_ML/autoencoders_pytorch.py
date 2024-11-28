import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import open3d as o3d
import time
import os

# Define Autoencoder Model
class PointCloudAutoencoder(nn.Module):
    def __init__(self, input_dim=3, latent_dim=128):
        super(PointCloudAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed

# Load Point Cloud
input_ply = "../../scene0.ply"
point_cloud = o3d.io.read_point_cloud(input_ply)
original_points = np.asarray(point_cloud.points)
print(f"Original Point Count: {len(original_points)}")

# Remove Invalid Points (NaN or Inf)
nan_mask = np.isnan(original_points).any(axis=1)
inf_mask = np.isinf(original_points).any(axis=1)
valid_points = original_points[~(nan_mask | inf_mask)]
print(f"Valid Point Count: {len(valid_points)}")

# Normalize Points
mean_point = np.mean(valid_points, axis=0)
normalized_points = valid_points - mean_point

# Convert to Torch Tensor
points_tensor = torch.tensor(normalized_points, dtype=torch.float32)

# Initialize Autoencoder
latent_dim = 128
autoencoder = PointCloudAutoencoder(input_dim=3, latent_dim=latent_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)

# Training Loop
epochs = 100
batch_size = 1024
start_time = time.time()
for epoch in range(epochs):
    optimizer.zero_grad()
    # Batch Processing
    indices = torch.randperm(points_tensor.size(0))[:batch_size]
    batch = points_tensor[indices]
    reconstructed_batch = autoencoder(batch)
    loss = criterion(reconstructed_batch, batch)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")

training_time = time.time() - start_time
print(f"Training completed in {training_time:.2f} seconds")

# Compress Point Cloud
print("Compressing Point Cloud...")
start_time = time.time()
compressed_points = autoencoder.encoder(points_tensor).detach().numpy()
compression_time = time.time() - start_time
print(f"Compression completed in {compression_time:.2f} seconds")

# Reconstruct Point Cloud
print("Reconstructing Point Cloud...")
start_time = time.time()
reconstructed_points = autoencoder.decoder(torch.tensor(compressed_points, dtype=torch.float32)).detach().numpy()
reconstructed_points += mean_point  # Reapply mean
reconstruction_time = time.time() - start_time
print(f"Reconstruction completed in {reconstruction_time:.2f} seconds")

# Save Reconstructed Point Cloud
reconstructed_cloud = o3d.geometry.PointCloud()
reconstructed_cloud.points = o3d.utility.Vector3dVector(reconstructed_points)
output_ply = "reconstructed_autoencoder.ply"
o3d.io.write_point_cloud(output_ply, reconstructed_cloud)
print(f"Reconstructed point cloud saved to {output_ply}")

# Log Results
log_file = "autoencoder_results.txt"
with open(log_file, "w") as log:
    log.write("### Autoencoder Compression Results ###\n")
    log.write("Autoencoders are neural networks designed to compress data into a latent space and reconstruct it.\n")
    log.write("\n")
    log.write("### Key Parameters ###\n")
    log.write(f"Latent Dimensionality: {latent_dim}\n")
    log.write(f"Number of Original Points: {len(original_points)}\n")
    log.write(f"Number of Valid Points: {len(valid_points)}\n")
    log.write(f"Batch Size: {batch_size}\n")
    log.write(f"Number of Epochs: {epochs}\n")
    log.write("\n")
    log.write("### Results ###\n")
    log.write(f"Training Time: {training_time:.2f} seconds\n")
    log.write(f"Compression Time: {compression_time:.2f} seconds\n")
    log.write(f"Reconstruction Time: {reconstruction_time:.2f} seconds\n")
    log.write(f"Reconstructed Point Cloud File: {output_ply}\n")
print(f"Results saved to {log_file}")
