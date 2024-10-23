from torch import nn
from torch.nn import functional as F

from models.deepsvdd.base.base_net import BaseNet


class TabularNet(BaseNet):
    def __init__(self, input_dim, model_config=None):
        super().__init__()
        self.input_dim = input_dim
        self.rep_dim = model_config.model_parameters.hidden_layers[-1]
        self.model_config = model_config
        self.layers = nn.ModuleList()
        self.layers.append(
            nn.Linear(
                self.input_dim,
                self.model_config.model_parameters.hidden_layers[0],
            )
        )
        previous_size = self.model_config.model_parameters.hidden_layers[0]
        for hidden_size in model_config.model_parameters.hidden_layers[1:]:
            self.layers.append(nn.Linear(previous_size, hidden_size))
            previous_size = hidden_size

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        return x
