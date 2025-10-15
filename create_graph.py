import torch
from torch_geometric.nn import radius_graph
from torch_geometric.data import Data
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

file_path = "/Users/Brennan/Downloads/CSE5835_FinalProj/position_tensor_train.pt"

pos_data = torch.load(file_path)
pos_data = pos_data[0]

file_path_2 = "/Users/Brennan/Downloads/CSE5835_FinalProj/velocity_tensor_train.pt"

vel_data = torch.load(file_path_2)
vel_data = vel_data[:, :, :2]

num_timesteps, num_particles, position_dim = pos_data.shape
velocity_dim = vel_data.shape[2]

input_tensor = torch.zeros(num_timesteps - 1, num_particles, position_dim + 5 * velocity_dim)

for t in range(1, num_timesteps):  # Skip the first timestep since velocities start from timestep 1
    # Current position at timestep t
    current_position = pos_data[t]  # Shape: [particles, position_dim]

    # Collect past 5 velocities (zero-padding for earlier timesteps)
    past_velocities = []
    for i in range(5):
        if t - i - 1 >= 0:  # Ensure index is valid
            past_velocities.append(vel_data[t - i - 1])  # Shape: [particles, velocity_dim]
        else:
            # Pad with zeros if not enough past timesteps
            past_velocities.append(torch.zeros(num_particles, velocity_dim))

    # Concatenate past velocities in reverse order (from t-4 to t)
    past_velocities = torch.cat(past_velocities[::-1], dim=1)  # Shape: [particles, 5 * velocity_dim]

    # Combine current position and past velocities into the input vector
    input_tensor[t - 1] = torch.cat([current_position, past_velocities], dim=1)

# Check final input tensor shape
print("Input Tensor Shape:", input_tensor.shape)

timesteps = [0, 49, 99, 149, 199, 249, 299]

# Create subplots (2 rows, 4 columns for a clean layout)
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()  # Flatten the axes array for easy indexing

for idx, i in enumerate(timesteps):
    # Extract data for the current timestep
    curr_timestep = input_tensor[i]  # Select the i-th timestep from input_tensor

    # Extract positions from the current timestep
    positions = curr_timestep[:, :2]  # First two columns are positions

    # Define the graph based on positions and connectivity radius
    connectivity_radius = 0.015
    edge_index = radius_graph(positions, r=connectivity_radius)

    # Compute edge attributes (relative positions and distances)
    source_nodes = edge_index[0]
    target_nodes = edge_index[1]
    relative_positions = positions[source_nodes] - positions[target_nodes]  # Shape: [num_edges, position_dim]
    distances = torch.norm(relative_positions, p=2, dim=-1, keepdim=True)  # Shape: [num_edges, 1]
    edge_attr = torch.cat([relative_positions, distances], dim=-1)  # Combine attributes

    # Create the PyG Data object
    graph = Data(x=curr_timestep, edge_index=edge_index, edge_attr=edge_attr)

    # Convert to NetworkX graph
    nx_graph = to_networkx(graph, node_attrs=None, edge_attrs=None)
    nx_graph = nx_graph.to_undirected()

    # Extract positions for nodes (for layout)
    pos = {j: positions[j].numpy() for j in range(positions.shape[0])}  # Node positions

    # Plot the graph on the current subplot
    ax = axes[idx]  # Select the correct subplot
    nx.draw_networkx_nodes(
        nx_graph,
        pos,
        ax=ax,
        node_size=20,            # Node size
        node_color="orange",     # Node fill color
        edgecolors="black",      # Border color
        linewidths=0.5           # Border width
    )
    nx.draw_networkx_edges(
        nx_graph,
        pos,
        ax=ax,
        edge_color="gray",       # Uniform edge color
        alpha=0.8                # Slight transparency for edges
    )

    # Add a title to the subplot
    ax.set_title(f"Timestep {i + 1}", fontsize=14)
    ax.axis("off")  # Hide axes for a cleaner look

# Adjust layout
fig.tight_layout()

# If there are unused subplots, hide them
for j in range(len(timesteps), len(axes)):
    axes[j].axis("off")

plt.show()