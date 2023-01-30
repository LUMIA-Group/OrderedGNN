import torch
from torch_geometric.nn import GATConv
import torch.nn.functional as F

from torch.nn import Module, ModuleList, Linear

class GAT(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        
        self.params = params
        self.conv_layers = ModuleList()

        for i in range(params['num_layers']):
            if i==0:
                self.conv_layers.append(GATConv(params['in_channel'], int(params['hidden_channel']/8), heads=8, dropout=params['dropout_rate'], concat=True))
            elif i==params['num_layers']-1:
                self.conv_layers.append(GATConv(params['hidden_channel'], params['out_channel'], heads=1, dropout=params['dropout_rate'], concat=False))
            else:
                self.conv_layers.append(GATConv(params['hidden_channel'], int(params['hidden_channel']/8), heads=8, dropout=params['dropout_rate'], concat=True))

    def forward(self, x, edge_index):
        for i in range(self.params['num_layers']):
            x = F.dropout(x, p=self.params['dropout_rate'], training=self.training)
            x = self.conv_layers[i](x, edge_index)
            if i != self.params['num_layers']-1:
                x = F.elu(x)
        
        return {'x':x}