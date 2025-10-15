import torch
from torch_geometric.nn import radius_graph
from torch_geometric.data import Data

def build_graph(positions, velocities, left_bottom, top_right):
    # print(positions.shape, velocities.shape)
    connectivity_radius = 0.015
    edge_index = radius_graph(positions, r=connectivity_radius)

    source_nodes, target_nodes = edge_index
    relative_positions = positions[source_nodes] - positions[target_nodes]  # (num_edges, dim)
    distances = torch.norm(relative_positions, p=2, dim=-1, keepdim=True)  # (num_edges, 1)
    edge_features = torch.cat([relative_positions, distances], dim=-1)
    
    # print("Relative Pos: ", relative_positions.shape)
    # print("Distances: ", distances.shape)
    # print("Edge Features:  ", edge_features.shape)

    
    left_bottom = torch.clamp(left_bottom, min=connectivity_radius)
    top_right = torch.clamp(top_right, min=connectivity_radius)

    node_features = torch.cat([positions, left_bottom, top_right, velocities], dim=1)
    # print(node_features.shape)
    mean = node_features.mean(dim=0, keepdim=True)
    std = node_features.std(dim=0, keepdim=True)
    node_features = (node_features - mean) / std

    mean = edge_features.mean(dim=0, keepdim=True)
    std = edge_features.std(dim=0, keepdim=True)
    edge_features = (edge_features - mean) / std
    
    # print("Nodes: ", node_features.shape)
    # print("Edges: ", edge_features.shape)
    # print("Index: ", edge_index.shape)

    missing_nodes = set(range(node_features.shape[0])) - set(torch.unique(edge_index).tolist())
    # print(f"Missing nodes: {len(missing_nodes)}")
    #print("Edge Diff", set(torch.unique(edge_index[0]).tolist()) - set(torch.unique(edge_index[1]).tolist()))
    # print(node_features.shape, edge_index.shape, edge_features.shape)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        graph = Data(x=node_features.to(device), 
                edge_index=edge_index.to(device), 
                edge_attr=edge_features.to(device))
    else:
        graph = Data(x=node_features,
                edge_index=edge_index, 
                edge_attr=edge_features)

    return graph