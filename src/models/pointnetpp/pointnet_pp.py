import torch.nn as nn
import torch.nn.functional as F
from src.models.pointnetpp.set_abstraction_layer import get_set_abstraction_layer

class pointnet_pp_cls(nn.Module):
    def __init__(self, config):
        super(pointnet_pp_cls, self).__init__()
        n_sa_layers = len(config['sa_layers'])
        self.sa_layers = nn.ModuleList()
        for i in range(n_sa_layers):
            layer_config = config['sa_layers'][i]
            self.sa_layers.append(get_set_abstraction_layer(layer_config))
        n_mlp_layers = len(config['mlp_layers'])
        self.mlp_layers = nn.ModuleList()
        for i in range(n_mlp_layers):
            mlp_config = config['mlp_layers'][i]
            self.mlp_layers.append(nn.Linear(mlp_config['in_channel'], mlp_config['out_channel']))
            if 'bn' in mlp_config and mlp_config['bn']:
                self.mlp_layers.append(nn.BatchNorm1d(mlp_config['out_channel']))
            if 'dropout' in mlp_config:
                self.mlp_layers.append(nn.Dropout(mlp_config['dropout']))
        self.log_softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, xyz, normals=None):
        B, _, _ = xyz.shape
        features = normals
        for sa_layer in self.sa_layers:
            xyz, features = sa_layer(xyz, features)
        features = features.view(B, -1)
        for mlp_layer in self.mlp_layers:
            features = mlp_layer(features)
            if isinstance(mlp_layer, nn.BatchNorm1d):
                features = F.relu(features)
        features = self.log_softmax(features)
        
        return features
        