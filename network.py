import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    
    def __init__(self, sizes, activation=nn.ReLU, input_norm=None, output_layer_init=None, output_activation=None, seed=None):
        super().__init__()

        if seed is not None:
            self.seed = torch.manual_seed(seed)
        
        self.layers = []
        for i in range(len(sizes)-2):
            self.layers += [nn.Linear(sizes[i], sizes[i+1]), activation()]
        self.layers.append( nn.Linear(sizes[-2], sizes[-1]) )

        if output_layer_init is not None:
            self.layers[-1].weight.data.uniform_(-output_layer_init, output_layer_init)

        if output_activation is not None:
            self.layers.append(output_activation())
        
        self.layers = nn.ModuleList(self.layers)

        self.input_norm = input_norm
        
    def forward(self, x):
        
        if self.input_norm is not None:
            x = self.input_norm(x)

        for layer in self.layers:
            x = layer(x)
        
        return x


class Actor(MLP):
    pass


class Critic(MLP):

    def __init__(self, **kargs):
        super().__init__(**kargs)
    
    def forward(self, states, actions):

        if self.input_norm is not None:
            x = self.input_norm(states)
        else:
            x = states
        
        x = torch.cat((x, actions), dim=1)

        for layer in self.layers:
            x = layer(x)
        
        return x

