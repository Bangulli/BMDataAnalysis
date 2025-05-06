import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, GATv2Conv
from torch_geometric.nn import global_mean_pool
from torch.nn import Sequential, ReLU
from torch_geometric.nn import NNConv

class SimplestGCNRegress(torch.nn.Module):
    def __init__(self, num_classes, num_node_features):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr, edge_weights = data.x, data.edge_index, data.edge_attr, data.edge_weights

        x = self.conv1(x, edge_index, edge_weight=edge_weights)

        return x
    
    def save(self, path):
        torch.save(self, path/'SimplestGCNRegress.pkl')
    
class GCNRegress(torch.nn.Module):
    def __init__(self, num_classes, num_node_features):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, num_node_features*2)
        self.conv2 = GCNConv(2*num_node_features, 3*num_node_features)
        self.conv3 = GCNConv(3*num_node_features, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr, edge_weights = data.x, data.edge_index, data.edge_attr, data.edge_weights

        x = self.conv1(x, edge_index, edge_weight=edge_weights)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight=edge_weights)
        x = F.relu(x)
        x = self.conv3(x, edge_index, edge_weight=edge_weights)

        return x
    
    def save(self, path):
        torch.save(self, path/'GCNRegress.pkl')

    

class TimeAwareGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.edge_mlp1 = Sequential(
            Linear(1, 16),
            ReLU(),
            Linear(16, 1 * 16)  # for nnconv1: 1 input channel → 16 output features
        )
        self.nnconv1 = NNConv(1, 16, self.edge_mlp1, aggr='mean')

        self.edge_mlp2 = Sequential(
            Linear(1, 32),
            ReLU(),
            Linear(32, 16 * 16)  # <-- FIX: for nnconv2: 16 input channels → 16 output features
        )
        self.nnconv2 = NNConv(16, 16, self.edge_mlp2, aggr='mean')

        self.lin = Linear(16, 1)  # final regression head

    def forward(self, batch):
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr

        x = self.nnconv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.nnconv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.lin(x)
        return x
    
    def save(self, path):
        torch.save(self, path/'TimeAwareGNN.pkl')