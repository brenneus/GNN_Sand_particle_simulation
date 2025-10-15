import torch
import torch.nn as nn
import torch_scatter
from torch_geometric.nn import MessagePassing

class NodeEncoder(nn.Module):
    def __init__(self, input_dim, output_dim=128, hidden_dim=128):
        super(NodeEncoder, self).__init__()
        self.input_dim = input_dim # input dim should be num_features (12)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, nodes_t):
         # mask out position from nodes_t
        # print(nodes_t.shape)
        mask = torch.ones(nodes_t.shape).to(self.device)
        mask[:, 0:2] = 0 # index 0,1 is where our position feautres are, set to 0 in mask
        nodes_t = nodes_t * mask

        return self.mlp(nodes_t) 
        
class EdgeEncoder(nn.Module):
    def __init__(self, input_dim, output_dim=128, hidden_dim=128):
        super(EdgeEncoder, self).__init__()
        self.input_dim = input_dim # this should be the number of edge features (3)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
    
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, edge_features): # edge_features is [ (pi − pj), ||pi − pj||^2 ] 
        return self.mlp(edge_features) 
        

class Processor(MessagePassing):
    def __init__(self, input_dim=128, output_dim=128, hidden_dim=128):
        super(Processor, self).__init__()

        self.input_dim = input_dim # all 128
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
    
        self.mlp = nn.Sequential(
            nn.Linear(2*input_dim, hidden_dim), #2*input dim bc we concatenate edge features
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.agg_dim = None

    def forward(self, x, edge_index, edge_features): 
        # print("Edge Features 1: ", edge_features.shape)
        self.agg_dim = len(x)
        return self.propagate(edge_index=edge_index, x=x, edge_features=edge_features) # self.propagate() sends the messages
    
    def message(self, x_j, edge_features): # defines how we compute messages (we use MLP)
        # print("Edge Features 2: ", edge_features.shape)
        combined = torch.cat([x_j, edge_features], dim=-1) # concatenate the edge attributes
        # print("Messages: ", self.mlp(combined).shape)
        # print(x_j.shape)
        return self.mlp(combined)
    
    def aggregate(self, messages, index): # defines how messages are aggregated at target nodes
        # sum, avg, max?
        # print("Agg Mess: ", messages.shape)
        # print("Agg Idx: ", index.shape)
        # print("Unique: ", torch.unique(index).numel())
        # print(torch.mean(messages, dim=1).shape)
        #return torch.mean(messages, dim=1)
        
        # print("Aggregate:", torch_scatter.scatter(messages, index, dim=0, reduce="mean").shape)
        
        agged = torch_scatter.scatter(messages, index, dim=0, reduce="mean", dim_size=self.agg_dim)
        # print('agatha', agged.shape)
        return agged
    
    def update(self, messages, x): # defines how we combine messages with node features (we implement residuals)
        # print(messages.shape)
        # print(x.shape)
        # print((messages + x).shape)
        return messages + x


class Decoder(nn.Module):
    def __init__(self, input_dim=128, output_dim=2, hidden_dim=128):
        super(Decoder, self).__init__()
        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim # should be 2: x and y acceleration

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x): # operate on nodes
        return self.mlp(x)
    

class GNN(nn.Module):
    def __init__(self, num_node_features=16, num_edge_features=3, num_message_passing_steps=10, hidden_dim=128, output_dim=2):
        super(GNN, self).__init__()
        self.num_message_passing_steps = num_message_passing_steps

        # econder layers
        self.node_encoder = NodeEncoder(input_dim=num_node_features, output_dim=hidden_dim, hidden_dim=hidden_dim)
        self.edge_encoder = EdgeEncoder(input_dim=num_edge_features, output_dim=hidden_dim, hidden_dim=hidden_dim)

        # processer layers
        self.processor_layers = nn.ModuleList([Processor(input_dim=128) for _ in range(num_message_passing_steps)])

        # decoder layers
        self.decoder = Decoder(input_dim=hidden_dim, output_dim=output_dim, hidden_dim=hidden_dim)

    def forward(self, graph):
        # encode nodes and edges
        node_embeddings = self.node_encoder(graph.x)   
        edge_embeddings = self.edge_encoder(graph.edge_attr) 
        # print(f"Node embeddings: {node_embeddings.shape}")
        # print(f"Edge embeddings: {edge_embeddings.shape}")

        # perform message passing steps on graph embedding
        for processor in self.processor_layers:
            node_embeddings = processor(node_embeddings, graph.edge_index, edge_embeddings)

        # print(f"Node embeddings after processor: {node_embeddings.shape}")

        # decode latent graph to get acceleration prediction
        predicted_acceleration = self.decoder(node_embeddings)

        # print("predAccel", predicted_acceleration.shape)
        return predicted_acceleration