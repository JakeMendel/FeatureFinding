
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class Net(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 tied = True,
                 final_bias = False,
                 hidden_bias = False,
                 nonlinearity = F.relu,
                 unit_weights=False,
                 learnable_scale_factor = False,
                 standard_magnitude = False,
                 initial_scale_factor = 1.0,
                 initial_embed = None,
                 initial_bias = None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.nonlinearity = nonlinearity
        self.tied = tied
        self.final_bias = final_bias
        self.unit_weights = unit_weights
        self.learnable_scale_factor = learnable_scale_factor
        self.standard_magnitude = standard_magnitude


        # Define the input layer (embedding)
        self.embedding = nn.Linear(self.input_dim, self.hidden_dim, bias=hidden_bias)
        if initial_embed is not None:
            self.embedding.weight.data = initial_embed


        if self.standard_magnitude:
            # Normalize the weight to have unit norm for each row
            avg_norm = torch.norm(self.embedding.weight.data, p=2, dim = 0).mean()
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=0) * avg_norm
        if self.unit_weights:
            # Normalize the weight to have unit norm for each row
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=0)
        
        # Define the output layer (unembedding)
        if not tied:
            self.unembedding = nn.Linear(self.hidden_dim, self.input_dim, bias=final_bias)
            if initial_bias is not None:
                self.unembedding.bias.data = initial_bias
        # Tie the weights
        else:
            # self.unembedding.weight = torch.nn.Parameter(self.embedding.weight.transpose(0, 1))
            if self.final_bias:
                self.unembedding_bias = torch.Tensor(input_dim)
                std = np.sqrt(self.embedding.weight.data.size(1))
                self.unembedding_bias.data.uniform_(-1. / std, 1 / std)
                self.unembedding_bias = nn.Parameter(self.unembedding_bias)
                if initial_bias is not None:
                    self.unembedding_bias.data = initial_bias
            else:
                self.unembedding_bias = torch.zeros(input_dim)

        if self.learnable_scale_factor:
            self.scale_factor = nn.Parameter(torch.tensor(initial_scale_factor))
        else:
            self.scale_factor = initial_scale_factor

    def forward(self, x, hooked = False):
        if self.unit_weights:
            # Normalize the weight to have unit norm for each row
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=0)
        if self.standard_magnitude:
            avg_norm = torch.norm(self.embedding.weight.data, p=2, dim = 0).mean()
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=0)
            self.embedding.weight.data = self.embedding.weight.data * avg_norm
        # if self.tied:
        #     self.unembedding.weight.data = self.embedding.weight.data.transpose(0, 1)
        if hooked:
            activations = {}
            activations['res_pre'] = self.embedding(x)
            if self.tied:    
                activations['unembed_pre'] = F.linear(activations['res_pre'], self.embedding.weight.t(), self.unembedding_bias)
            else:
                activations['unembed_pre'] = self.unembedding(activations['res_pre'])
            activations['output'] = self.scale_factor * self.nonlinearity(activations['unembed_pre'])
            return activations['output'], activations
        else:
            x = self.embedding(x)
            x = F.linear(x, self.embedding.weight.t(), self.unembedding_bias) if self.tied else self.unembedding(x)
            x = self.nonlinearity(x)
            return self.scale_factor * x

class ResNet(Net):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 mlp_dim,
                 tied = True,
                 mlp_tied = True,
                 mlp_bias = False,
                 final_bias = False,
                 nonlinearity = F.relu,
                 n_mlps = 1):
        super().__init__(input_dim, hidden_dim, tied, final_bias, nonlinearity)

        mlp_ins = []
        mlp_outs = []
        for i in range(n_mlps):
            mlp_in = nn.Linear(hidden_dim, mlp_dim, bias = mlp_bias)
            mlp_out = nn.Linear(mlp_dim, hidden_dim, bias = mlp_bias)
            
            if mlp_tied:
                assert not mlp_bias
                mlp_out.weight = nn.Parameter(mlp_in.weight.transpose(0, 1))
            mlp_ins.append(mlp_in)
            mlp_outs.append(mlp_out)

        self.mlp_ins = nn.ModuleList(mlp_ins)
        self.mlp_outs = nn.ModuleList(mlp_outs)
        self.n_mlps = n_mlps

    def forward(self, x, hooked = False):
        if hooked:
            activations = {}
            activations['res_0'] = self.embedding(x)
            for i in range(1,self.n_mlps+1):
                activations[f'mlp_in_pre_{i}'] = self.mlp_ins[i-1](activations[f'res_{i-1}'])
                activations[f'mlp_in_post_{i}'] = self.nonlinearity(activations[f'mlp_in_pre_{i}'])
                activations[f'mlp_out_{i}'] = self.mlp_outs[i-1](activations[f'mlp_in_post_{i}'])
                activations[f'res_{i}'] = activations[f'res_{i-1}'] + activations[f'mlp_out_{i}']
            activations['unembed_pre'] = self.unembedding(activations[f'res_{self.n_mlps}'])
            activations['output'] = self.nonlinearity(activations['unembed_pre'])
            return activations['output'], activations

        else:
            x = self.embedding(x)
            for i in range(self.n_mlps):
                x = x + self.mlp_outs[i](self.nonlinearity(self.mlp_ins[i](x)))
            x = self.unembedding(x)
            x = self.nonlinearity(x)
            return x
