import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch.nn import Linear

# Define GCN model
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# Define GAT for fusion and classification
class MultimodalFusionGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes):
        super(MultimodalFusionGAT, self).__init__()
        self.gat = GATConv(in_channels, hidden_channels, concat=True)
        self.fc = Linear(hidden_channels * 3, num_classes)

    def forward(self, x1, x2, x3, edge_index):
        x1 = self.gat(x1, edge_index)
        x2 = self.gat(x2, edge_index)
        x3 = self.gat(x3, edge_index)
        x = torch.cat([x1, x2, x3], dim=1)
        return F.log_softmax(self.fc(x), dim=1)
