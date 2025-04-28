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
        self.conv1 = GCNConv(num_node_features, 16)
        self.fcn1 = Linear(16, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        #x = F.dropout(x, training=self.training)
        x = global_mean_pool(x, data.batch)
        x = self.fcn1(x)

        return F.log_softmax(x, dim=1)
    
class SimplestGCN(torch.nn.Module):
    def __init__(self, num_classes, num_node_features):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr, edge_weights = data.x, data.edge_index, data.edge_attr, data.edge_weights

        x = self.conv1(x, edge_index, edge_weights=edge_weights)
        #x = F.dropout(x, training=self.training)
        x = global_mean_pool(x, data.batch)
        return F.log_softmax(x, dim=1)
    
class SimplestGCNRegress(torch.nn.Module):
    def __init__(self, num_classes, num_node_features):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr, edge_weights = data.x, data.edge_index, data.edge_attr, data.edge_weights

        x = self.conv1(x, edge_index, edge_weight=edge_weights)

        return x
    
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
    
class GraphClassificationModel(torch.nn.Module):
    def __init__(self, base_model, pooling_function=global_mean_pool):
        super().__init__()
        self.base = base_model
        self.pooling_function = pooling_function

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = self.base(x, edge_index, edge_weight=edge_weight)
        x = self.pooling_function(x, data.batch)
        return F.log_softmax(x, dim=1)
    

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

