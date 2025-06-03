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

class SimpleGCN(torch.nn.Module):
    def __init__(self, num_classes, num_node_features):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, int(num_node_features/2))
        self.lin1 = Linear(int(num_node_features/2), num_classes)
        self.num_classes = num_classes

    def forward(self, data):
        x, edge_index, edge_attr, edge_weights = data.x, data.edge_index, data.edge_attr, data.edge_weights
        x = self.conv1(x, edge_index, edge_weight=edge_weights)
        x = global_mean_pool(x, data.batch)
        x = self.lin1(x)
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
        self.gat1 = GATv2Conv(num_node_features, num_classes, edge_dim=1)
        self.num_classes = num_classes


    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # encoder pass
        x = self.gat1(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = global_mean_pool(x, data.batch)
        if self.num_classes == 1:
            return x.view(-1)  # Shape [batch_size], raw logits
        else:
            return x
    def save(self, path):
        torch.save(self, path/'GAT.pkl')
    


class BigGAT(torch.nn.Module):
    """
    Random Bullshit GO!™
    """
    def __init__(self, num_classes, num_node_features):
        super().__init__()
        ## encoder
        if num_node_features != 1:
            self.gat1 = GATv2Conv(num_node_features, round(num_node_features/2), edge_dim=1)
            self.lin1 = Linear(round(num_node_features/2), num_classes)
        else:
            self.gat1 = GATv2Conv(num_node_features, 2, edge_dim=1)
            self.lin1 = Linear(2, num_classes)
        #self.gat2 = GATv2Conv(round(num_node_features/2), round(num_node_features/4), edge_dim=1)
       
        self.num_classes = num_classes


    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # encoder pass
        x = self.gat1(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        # x = self.gat2(x=x, edge_index=edge_index, edge_attr=edge_attr)
        # x = F.relu(x)
        x = global_mean_pool(x, data.batch)
        x = self.lin1(x)
        if self.num_classes == 1:
            return x.view(-1)  # Shape [batch_size], raw logits
        else:
            return x
    def save(self, path):
        torch.save(self, path/'GAT.pkl')