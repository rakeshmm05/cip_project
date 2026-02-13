import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class EdgeAwareGAT(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, temporal_dim=0):
        super().__init__()

        self.gat1 = GATv2Conv(node_dim, hidden_dim, heads=4, edge_dim=edge_dim)
        self.gat2 = GATv2Conv(hidden_dim * 4, hidden_dim, edge_dim=edge_dim)
        edge_input_dim = hidden_dim * 2 + edge_dim + temporal_dim
        self.edge_classifier = torch.nn.Sequential(
            torch.nn.Linear(edge_input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, edge_index, edge_attr, temporal=None):
        x = self.gat1(x, edge_index, edge_attr)
        x = F.elu(x)
        x = self.gat2(x, edge_index, edge_attr)
        x = F.elu(x)

        src, dst = edge_index
        edge_embeddings = torch.cat([x[src], x[dst]], dim=1)
        features = [edge_embeddings, edge_attr]
        if temporal is not None:
            features.append(temporal)
        edge_features = torch.cat(features, dim=1)

        logits = self.edge_classifier(edge_features).squeeze(-1)
        return logits
