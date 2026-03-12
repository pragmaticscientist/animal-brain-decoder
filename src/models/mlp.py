import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        mlp_layers = config.get('mlp_layers', [])
        layers = []
        for i, layer_cfg in enumerate(mlp_layers):
            in_chan = layer_cfg["in_channel"]
            out_chan = layer_cfg["out_channel"]
            bn = layer_cfg.get("bn", False)
            dropout = layer_cfg.get("dropout", 0.0)

            layers.append(nn.Linear(in_chan, out_chan))
            layers.append(nn.ReLU())

            if bn:
                layers.append(nn.BatchNorm1d(out_chan))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

      # Classification vs regression head
        if out_chan > 1:
            self.log_softmax = nn.LogSoftmax(dim=-1)
        else:
            self.log_softmax = None

    def forward(self, x):
        # x already flattened: (B, features)
        x = self.mlp(x)
        
        if self.log_softmax is not None:
            return self.log_softmax(x)  # (B, num_classes)
        else:
            return x.squeeze(-1)  # (B,)