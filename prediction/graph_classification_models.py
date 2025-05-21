import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, GATv2Conv
from torch_geometric.nn import global_mean_pool
from torch.nn import Sequential, ReLU
from torch_geometric.nn import NNConv


class GCN(torch.nn.Module):
    def __init__(self, num_classes, num_node_features):
        super().__init__()
        self.num_classes=num_classes
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, 32)
        self.conv3 = GCNConv(32,64)
        self.fcn1 = Linear(64, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr, edge_weight = data.x, data.edge_index, data.edge_attr, data.edge_weights

        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv3(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = global_mean_pool(x, data.batch)

        x = self.fcn1(x)

        if self.num_classes == 1:
            return x.view(-1)  # Shape [batch_size], raw logits
        else:
            return x
    
    def save(self, path):
        torch.save(self, path/'GCN.pkl')
    
class SimplestGCN(torch.nn.Module):
    def __init__(self, num_classes, num_node_features):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, num_classes)
        self.num_classes = num_classes

    def forward(self, data):
        x, edge_index, edge_attr, edge_weights = data.x, data.edge_index, data.edge_attr, data.edge_weights
        x = self.conv1(x, edge_index, edge_weight=edge_weights)
        x = global_mean_pool(x, data.batch)
        if self.num_classes == 1:
            return x.view(-1)  # Shape [batch_size], raw logits
        else:
            return x
    
    def save(self, path):
        torch.save(self, path/'SimplestGCN.pkl')

    
class GAT(torch.nn.Module):
    """
    Random Bullshit GO!™
    """
    def __init__(self, num_classes, num_node_features):
        super().__init__()
        ## encoder
        self.gat1 = GATv2Conv(num_node_features, 16, edge_dim=1)
        self.gat2 = GATv2Conv(16, 32, edge_dim=1)
        self.gat3 = GATv2Conv(32, 64, edge_dim=1)
        self.gat4 = GATv2Conv(64, 128, edge_dim=1)
        self.gat5 = GATv2Conv(128, 256, edge_dim=1)
        # classifier head
        self.fcn1 = Linear(256, 128)
        self.fcn2 = Linear(128, 64)
        self.fcn3 = Linear(64, 32)
        self.fcn4 = Linear(32, 16)
        self.fcn5 = Linear(16, 8)
        self.fcn6 = Linear(8, num_classes)


    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # encoder pass
        x = self.gat1(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = self.gat2(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = self.gat3(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = self.gat4(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = self.gat5(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = global_mean_pool(x, data.batch)
        # classifier pass
        x = self.fcn1(x)
        x = F.relu(x)
        x = self.fcn2(x)
        x = F.relu(x)
        x = self.fcn3(x)
        x = F.relu(x)
        x = self.fcn4(x)
        x = F.relu(x)
        x = self.fcn5(x)
        x = F.relu(x)
        x = self.fcn6(x)
        return F.log_softmax(x, dim=1)
    
    def save(self, path):
        torch.save(self, path/'GAT.pkl')
    


