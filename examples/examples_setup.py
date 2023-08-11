#%%
import numpy as np
from scipy.optimize import minimize
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import clear_output
import time
import plotly.graph_objs as goa
import matplotlib as mpl
import torch.nn.functional as F
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import copy
from itertools import combinations
from torch.optim.lr_scheduler import _LRScheduler
from fancy_einsum import einsum
from scipy.spatial import ConvexHull
import os
from torch.autograd import grad
from typing import Dict, Callable, List, Any, Union, Optional, Tuple, Iterable
import plotly.express as px
import pandas as pd

#%%
mpl.style.use('seaborn-v0_8')
mpl.rcParams['figure.figsize'] = (15,10)
fontsize = 20
mpl.rcParams['font.size'] = fontsize
mpl.rcParams['xtick.labelsize'] = fontsize
mpl.rcParams['ytick.labelsize'] = fontsize
mpl.rcParams['legend.fontsize'] = fontsize
mpl.rcParams['axes.titlesize'] = fontsize
mpl.rcParams['axes.labelsize'] = fontsize
# %%
# Create synthetic dataset
class SyntheticDataset(Dataset):
    def __init__(self, num_samples, f):
        self.num_samples = num_samples
        self.f = f
        self.data = self.generate_data()
        
    def generate_data(self):
        data = torch.zeros((self.num_samples, self.f))
        for i in range(self.num_samples):
            index = torch.randint(0, self.f, (1,))
            data[i, index] = torch.rand(1)
        return data

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]

class SyntheticNormalised(Dataset):
    #Creates a dataset with f 1-hot vectors as the dataset.
    def __init__(self, f):
        self.f = f
        self.data = self.generate_data()
        
    def generate_data(self):
        return torch.eye(self.f)

    def __len__(self):
        return self.f

    def __getitem__(self, idx):
        return self.data[idx]

class SyntheticKHot(Dataset):
    def __init__(self, f, k):
        self.f = f
        self.k = k
        self.data = []

        # Create all possible combinations of f choose k
        for indices in combinations(range(f), k):
            vec = torch.zeros(f)
            vec[list(indices)] = 1
            self.data.append(vec)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data[idx]

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

#%%
def plot_weights(weight_matrix, jitter = 0.05, normalised = False, save = False, epoch = None):
    plt.figure(figsize=(8, 8))

    for i in range(weight_matrix.shape[0]):
        normalisation = (weight_matrix[i,0]**2 + weight_matrix[i,1]**2) **0.5 if normalised else 1 
        plt.arrow(0, 0, weight_matrix[i,0]/normalisation, weight_matrix[i,1]/normalisation, head_width=0.05, head_length=0.1, fc='blue', ec='blue')
        plt.text(weight_matrix[i,0]/normalisation + jitter * torch.randn(1), weight_matrix[i,1]/normalisation + jitter * torch.randn(1), f"{i}", color='red', fontsize=12)

    mins = -1.2 if normalised else weight_matrix.min()-0.5
    maxs = 1.2 if normalised else weight_matrix.max()+0.5
    plt.xlim(mins,maxs)
    plt.ylim(mins,maxs)
    plt.grid()
    plt.show()
    if save:
        assert epoch is not None
        plt.savefig(f"weights_{epoch}.png")
    plt.close()

def force_numpy(matrix):
    if isinstance(matrix,np.ndarray):
        return matrix
    elif isinstance(matrix, torch.Tensor):
        return matrix.cpu().detach().numpy()
    else:
        raise ValueError

def plot_weights_interactive(weights_history, store_rate=1, dotsize = 5, with_labels = True, to_label = None, plot_size = 800):

    for key, weight_list in weights_history.items():
        # Initialize figure for each weight list
        fig = go.Figure()
        weight_list = [force_numpy(weight_matrix) for weight_matrix in weight_list]
        max_value = np.max([np.abs(weight_matrix).max() for weight_matrix in weight_list])

        # Check if weights are scalars
        if weight_list[0].ndim == 0:
            plt.plot(weight_list)
            continue

        weight_shape = min(weight_list[0].shape)
        is_bias = True if len(weight_list[0].squeeze().shape) == 1 else False

        # Create a scatter plot for each weight matrix
        for i, weight_matrix in enumerate(weight_list):
            weight_matrix = weight_matrix.squeeze()
            if is_bias:
                new_matrix = np.zeros((weight_matrix.shape[0],2))
                new_matrix[:,0] = weight_matrix
                weight_matrix = new_matrix
            if weight_matrix.shape[1] > weight_matrix.shape[0]:
                weight_matrix = weight_matrix.T 

            x_values = weight_matrix[:, 0]
            y_values = weight_matrix[:, 1] if weight_shape > 1 else np.zeros(weight_matrix[:, 1].shape)
            z_values = weight_matrix[:, 2] if weight_shape == 3 else None
            labels = list(range(len(x_values))) if with_labels else ['' for _ in range(len(x_values))]
            if to_label is not None:
                labels = [label if label in to_label else '' for label in labels]

            if z_values is None:
                scatter = go.Scatter(x=x_values, y=y_values, mode='markers+text', text=labels,
                                     textposition='top center', marker=dict(size=dotsize), visible=False, name=f'Epoch {i * store_rate}')
            else:
                scatter = go.Scatter3d(x=x_values, y=y_values, z=z_values, mode='markers+text', text=labels,
                                       marker=dict(size=dotsize), visible=False, name=f'Epoch {i * store_rate}')

            fig.add_trace(scatter)

        fig.data[0].visible = True
        if z_values is not None:
                fig.update_layout(scene = dict(
                    xaxis=dict(range=[-max_value * 1.1,max_value * 1.1], title='X Value'),
                    yaxis=dict(range=[-max_value * 1.1,max_value * 1.1], title='Y Value'),
                    zaxis=dict(range=[-max_value * 1.1,max_value * 1.1], title='Z Value'),
                    aspectmode='cube'))
        else:
            fig.update_xaxes(title_text='X Value', range=[-max_value * 1.1, max_value * 1.1])
            fig.update_yaxes(title_text='Y Value', range=[-max_value * 1.1, max_value * 1.1])

        steps = []
        for i in range(len(weight_list)):
            step = dict(
                method='restyle',
                args=['visible', [False] * len(fig.data)],
                label=f'Epoch {i * store_rate}'
            )
            step['args'][1][i] = True  # Toggle i'th trace to "visible"
            steps.append(step)

        slider = dict(
            active=0,
            currentvalue={"prefix": f"{key} - "},
            pad={"t": 50},
            steps=steps
        )

        fig.update_layout(sliders=[slider], width=plot_size, height=plot_size)

        fig.show()
    
def plot_weights_static(weights_history, losses: dict, store_rate=1, dotsize=60, to_label: Optional[List[int]] = None, epochs_to_show=[],scale = 1.6, num_across = 3):
    A4_WIDTH = 8.3 * scale  # inches, proportionally bigger  # inches, proportionally bigger
    epochs_to_show = sorted(epochs_to_show)
    
    for key, weight_list in weights_history.items():
        nearest_epoch_idxs = [min(range(len(weight_list)), key=lambda x: abs(x*store_rate-epoch)) for epoch in epochs_to_show]
        weight_list = [force_numpy(weight_list[i]) for i in nearest_epoch_idxs]
        
        # Calculate the max and min value for the current key
        max_value = np.max([weight_matrix.max() for weight_matrix in weight_list])
        max_value = max(0.9*max_value, 1.1*max_value)
        min_value = np.min([weight_matrix.min() for weight_matrix in weight_list])
        min_value = min(0.9*min_value, 1.1*min_value)

        weight_shape = 1 if len(weight_list[0].shape) == 1 else min(weight_list[0].shape)
        is_bias = True if len(weight_list[0].squeeze().shape) == 1 else False

        num_epochs = len(epochs_to_show)
        rows = (num_epochs + num_across - 1) // num_across + 1  # +1 for the loss subplot

        fig = plt.figure(figsize=(A4_WIDTH, rows*A4_WIDTH/num_across))
        loss_ax = fig.add_subplot(rows, 1, 1)
        loss_ax.plot(list(losses.keys()), list(losses.values()), '-k', label='Loss')
        loss_ax.scatter(epochs_to_show, [losses[e] for e in epochs_to_show], color='red', s=round(250*scale), marker='+', label='Selected Epochs')
        loss_ax.set_xlabel('Epoch', fontsize=round(scale*9))
        loss_ax.set_ylabel('Loss', fontsize=round(scale*9))
        loss_ax.legend(fontsize=round(scale*9))
        loss_ax.grid(True)
        loss_ax.tick_params(axis='both', which='major', labelsize=round(scale*7))
    
        # Create a grid for the weight subplots
        grid = plt.GridSpec(rows, num_across, wspace=0.1*scale, hspace=0.15*scale)
        
        for ax_idx, epoch in enumerate(epochs_to_show):
            row_idx, col_idx = divmod(ax_idx, num_across)
            row_idx += 1  # Adjust for the loss subplot
            weight_matrix = weight_list[ax_idx].squeeze()
            
            if is_bias:
                new_matrix = np.zeros((weight_matrix.shape[0], 2))
                new_matrix[:, 0] = weight_matrix + 1
                weight_matrix = new_matrix
            if weight_matrix.shape[1] > weight_matrix.shape[0]:
                weight_matrix = weight_matrix.T

            x_values = weight_matrix[:, 0]
            y_values = weight_matrix[:, 1] if weight_shape > 1 else [0] * len(x_values)
            
            # Determine colors based on to_label
            colors = ['green' if i in to_label else 'blue' for i in range(len(x_values))] if to_label is not None else 'blue'
            
            ax = fig.add_subplot(grid[row_idx, col_idx])
            ax.scatter(x_values, y_values, s=dotsize*scale, marker='+', color=colors)
            
            if col_idx == 0:
                if weight_shape > 1:
                    ax.set_ylabel('Neuron 2', fontsize=round(scale*9))
            if row_idx == rows - 1:
                if weight_shape == 1:
                    ax.set_xlabel('Value', fontsize=round(scale*9))
                else:
                    ax.set_xlabel('Neuron 1', fontsize=round(scale*9))
            
            ax.tick_params(axis='both', which='major', labelsize=round(scale*7))
            ax.set_xlim([min_value+int(is_bias), max_value+int(is_bias)])
            if weight_shape > 1:
                ax.set_ylim([min_value, max_value])
            else:
                ax.set_ylim([-1,1])
            ax.set_title(f'Epoch {epoch}', fontsize=round(scale*8))

        fig.suptitle(key, fontsize=round(scale*12), y=1.05)
        fig.tight_layout()
        plt.show()


def get_activation_history(model_history, f, included_keys=None):
    out, activations = list(model_history.values())[0](torch.eye(f), hooked = True)
    if included_keys is None:
        activation_history = {k: [] for k in activations}
    else:
        assert all([k in activations for k in included_keys]), f'Valid keys are {activations.keys()}'
        activation_history = {k: [] for k in included_keys}
    for model in model_history.values():
        out, activations = model(torch.eye(f), hooked = True)
        for k in activation_history:
            activation_history[k].append(activations[k])
    return activation_history

def calculate_angles(tensor):
    assert tensor.shape[1] == 2, "Input tensor must be of shape (n, 2)"
    return torch.atan2(tensor[:, 1], tensor[:, 0])

def linearity(x):
    return x

#%%
class CustomScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, max_lr, decay_factor, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.decay_factor = decay_factor
        super(CustomScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # linear warmup
            return [base_lr + self.last_epoch * ((self.max_lr - base_lr) / self.warmup_steps) for base_lr in self.base_lrs]
        else:
            # exponential decay
            return [self.max_lr * (self.decay_factor ** (self.last_epoch - self.warmup_steps)) for _ in self.base_lrs]


def train(model, loader, criterion, optimizer, epochs, logging_loss, plot_rate, store_rate, scheduler = None, lr_print_rate = 0, dtype = torch.float32):
    weights_history = {k:[] for k in dict(model.named_parameters()).keys()}  # Store the weights here
    model_history = {} #store model here
    losses = []
    model = model.to(dtype=dtype)
    for epoch in tqdm(range(epochs)):
        total_loss = 0
        for batch in loader:
            batch = batch.to(dtype)
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        if logging_loss:
            losses.append(avg_loss)
            if plot_rate > 0:
                if (epoch + 1) % plot_rate == 0:
                    plt.figure(figsize=(5,5))
                    plt.plot(losses)
                    plt.show()
        if (epoch) % store_rate == 0:
            for k,v in dict(model.named_parameters()).items():
                weights_history[k].append(v.detach().numpy().copy())
            model_history[epoch] = copy.deepcopy(model)
        if scheduler is not None:
            scheduler.step()
        if lr_print_rate > 0:
            if (epoch % lr_print_rate) == 0:
                print(optimizer.param_groups[0]['lr'])
    return losses, weights_history, model_history  # Return the weights history

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#%%
def group_vectors(vectors, epsilon):
    # Store the groups of similar vectors here
    groups = []
    norms = []
    directions = []

    for v in vectors:
        # Normalize the current vector
        v_norm = v / np.linalg.norm(v)
        if np.linalg.norm(v) < 0.01:
            continue

        # This flag will tell us if the current vector has been added to any group
        added_to_group = False

        # Go through each existing group to check if this vector belongs there
        for i,group in enumerate(groups):
            # We use the first vector in the group as representative
            group_representative = group[0]
            group_representative_norm = group_representative / np.linalg.norm(group_representative)

            # Calculate the dot product between the normalized vectors
            dot_product = np.dot(v_norm, group_representative_norm)

            # Check if the dot product is close enough to 1 (indicating they are scalar multiples of each other)
            if np.abs(dot_product - 1) < epsilon:
                group.append(v)
                norms[i].append(v_norm)
                added_to_group = True
                break

        # If the current vector has not been added to any group, we create a new group for it
        if not added_to_group:
            groups.append([v])
            norms.append([v_norm])
    
    for norm in norms:
        arr = np.array(norm)
        directions.append(np.mean(arr,axis=0))

        

    return groups,directions

#%%
def visualize_matrices_with_slider(matrices, rate, const_colorbar=False, plot_size = 800):
    # Find global min and max if constant colorbar is requested
    if const_colorbar:
        global_min = np.min([np.min(matrix) for matrix in matrices])
        global_max = np.max([np.max(matrix) for matrix in matrices])

    # Create empty figure
    fig = go.Figure()

    # Add traces for each matrix
    for i, matrix in enumerate(matrices):
        # Create a heatmap for the matrix
        heatmap = go.Heatmap(
            z=matrix, 
            colorscale='magma', 
            showscale=True,
            zmin=global_min if const_colorbar else None,
            zmax=global_max if const_colorbar else None
        )
        fig.update_yaxes(autorange='reversed')
        # Add the heatmap to the figure, but only make it visible if it's the first one
        fig.add_trace(heatmap)
        fig.data[i].visible = (i == 0)
        fig.data[i].name = f'Epoch {i * rate}'
        
    # Create a slider
    steps = []
    for i in range(len(matrices)):
        step = dict(
            method="restyle",
            args=["visible", [False] * len(matrices)],
            label=f'Epoch {i * rate}'
        )
        step["args"][1][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Displaying: "},
        pad={"t": 50},
        steps=steps
    )]

    # Add the slider to the figure
    fig.update_layout(
        sliders=sliders,
        height = plot_size,
        width = plot_size
    )

    fig.show()


def generate_matrix_list(weights_history):
    n = len(weights_history['embedding.weight'])
    return [weights_history['unembedding.weight'][i] @ weights_history['embedding.weight'][i] for i in range(n)]

def np_gelu(matrix):
    return F.gelu(torch.tensor(matrix)).detach().numpy()

def nonlinearity_numpy(matrix, nonlinearity):
    return nonlinearity(torch.tensor(matrix)).detach().numpy()


def filter_to_convex_hull(points):
    # Convert points to numpy array
    points = np.array(points)
    
    # Calculate the convex hull of the points
    hull = ConvexHull(points)
    
    # Return only the points that are vertices of the hull
    hull_points = points[hull.vertices]
    
    # Convert back to python list before returning
    return hull_points.tolist()

def visualise_polyhedron(vertices, filled_faces = True, opacity = 1, with_labels = False):
    # convert vertices list to numpy array for convenience
    if vertices.shape[1] > vertices.shape[0]:
                vertices = vertices.T 
    
    # scipy's ConvexHull will give us the simplices (triangles) that form the polyhedron
    from scipy.spatial import ConvexHull
    hull = ConvexHull(vertices)
    
    # initialize 3D plot
    fig = go.Figure()
    
    # add each simplex as a triangular face
    # add each simplex as a triangular face
    for s in hull.simplices:
        # Ensure vertices are in counterclockwise order
        cross_product = np.cross(vertices[s[1]] - vertices[s[0]], vertices[s[2]] - vertices[s[0]])
        dot_product = np.dot(cross_product, hull.equations[s[0], :-1])
        
        if dot_product < 0:
            s = s[[0, 2, 1]]  # Swap the last two elements to change the order to counterclockwise
        
        s = np.append(s, s[0])  # Here we cycle back to the first coordinate for plotly
        if filled_faces:
            x = vertices[s, 0]
            y = vertices[s, 1]
            z = vertices[s, 2]
            fig.add_trace(go.Mesh3d(x=x, y=y, z=z, color='lightpink', opacity=opacity))
        fig.add_trace(go.Scatter3d(x=vertices[s, 0], y=vertices[s, 1], z=vertices[s, 2],
                                mode='lines',
                                line=dict(color='blue', width=2)))

    # add each vertex in the hull as a label
    if with_labels:
        for i in hull.vertices:
            fig.add_trace(go.Scatter3d(x=[vertices[i, 0]], y=[vertices[i, 1]], z=[vertices[i, 2]],
                                    mode='text',
                                    text=[str(i)],  # or other string labels
                                    textposition='top center'))

        
    # set the 3d scene parameters
    fig.update_layout(showlegend = False,
                      scene = dict(xaxis_title='X',
                                   yaxis_title='Y',
                                   zaxis_title='Z',
                                   aspectmode='auto'),
                      width=700,
                      margin=dict(r=20, l=10, b=10, t=10))
    fig.show()


def smallest_angle_vectors(list1, list2):
    result = {}
    
    for i, v in enumerate(list1):
        smallest_angle = None
        smallest_index = None
        
        for j, u in enumerate(list2):
            cosine_angle = np.dot(v, u) / (np.linalg.norm(v) * np.linalg.norm(u))
            angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # ensure value is in the valid domain for arccos
            
            if smallest_angle is None or angle < smallest_angle:
                smallest_angle = angle
                smallest_index = j
                
        result[i] = (smallest_index, np.degrees(smallest_angle))  # convert angle to degrees

    return result

def relu_plusone(x):
    return F.relu(x+1)

def gelu_plusone(x):
    return F.gelu(x+1)

def relu_minusone(x):
    return F.relu(x-1)

#%%

def pairwise_angles(vectors):
    n = len(vectors)
    angles = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            u = vectors[i]
            v = vectors[j]
            cos_angle = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
            angle = np.arccos(np.clip(cos_angle, -1, 1))  # Clipping to handle potential floating-point errors
            angles[i, j] = np.degrees(angle)  # Converting to degrees
            
    return angles

def calculate_hessian(model, inputs, targets, loss_func, loading_bar = False):
    # Compute the loss
    n = sum(p.numel() for p in model.parameters())
    n = 0
    for name, p in model.named_parameters():
        if model.tied:
            if name == 'unembedding.weight':
                continue
        n += p.numel()
    if 1000<n:
        print(f'Warning: this model has {n} parameters. Computing the Hessian take a long time or cause a memory overflow.')
    output = model(inputs)
    loss = loss_func(output, targets)

    # Compute the gradient of the loss with respect to the model parameters
    # grad_params = []
    # for name, params in model.named_parameters():
    #     if model.tied:
    #         if name == 'embedding.weight':
    #             continue
    #     gradients, = grad(loss, params, create_graph=True)
    #     grad_params.append(gradients)
    grad_params = grad(loss, model.parameters(), create_graph=True)
    grad_params = torch.concat(tuple([g.reshape(-1) for g in grad_params]))

    # Compute the Hessian
    hessian = []
    if loading_bar:
        for g in tqdm(grad_params):
            # grad2_params = []
            # for name, params in model.named_parameters():
            #     if model.tied:
            #         if name == 'embedding.weight':
            #             continue
            #     gradients, = grad(g, params, retain_graph=True)
            #     grad2_params.append(gradients)
            grad2_params = grad(g, model.parameters(), retain_graph = True)
            grad2_params = torch.concat(tuple([g.reshape(-1) for g in grad2_params]))
            hessian.extend(grad2_params)
    else:
        for g in grad_params:
            # grad2_params = []
            # for name, params in model.named_parameters():
            #     if model.tied:
            #         if name == 'embedding.weight':
            #             continue
            #     gradients, = grad(g, params, retain_graph=True)
            #     grad2_params.append(gradients)
            grad2_params = grad(g, model.parameters(), retain_graph = True)
            grad2_params = torch.concat(tuple([g.reshape(-1) for g in grad2_params]))
            hessian.extend(grad2_params)
    # Calculate the total number of parameters in the model

    # Reshape the Hessian components into a matrix of size (n, n)
    hessian_matrix = torch.cat([h.reshape(-1) for h in hessian]).reshape(n, n)

    return hessian_matrix

def calculate_hessians(
    checkpoints: Dict[int, Net],
    batch: torch.Tensor,
    target: torch.Tensor,
    loss_fn: Callable
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """
    Calculates the Hessian matrices for each checkpoint.

    :param checkpoints: Dictionary of checkpoints with keys being the epoch number and values being the model.
    :param batch: The batch of data used for calculating the Hessian.
    :param target: The target values used for calculating the Hessian.
    :param loss_fn: The loss function used to calculate the Hessian.
    :return: Dictionary of (epoch, hessian) pairs.
    """
    hessians_dict = {}
    eigenvalues_dict = {}
    for epoch, model in tqdm(checkpoints.items()):
        hessian = calculate_hessian(model, batch, target, loss_fn)
        hessians_dict[epoch] = hessian

        eigenvalues, _ = np.linalg.eig(hessian)
        # Use the real part of the eigenvalues
        eigenvalues = eigenvalues.real

        eigenvalues_dict[epoch] = eigenvalues
    return hessians_dict, eigenvalues_dict

def hist_eigenvalues(
    eigenvalues_dict: Dict[int, torch.Tensor],
    bins: int
) -> None:
    """
    Plots an interactive histogram of the eigenvalues of the Hessian matrix at different epochs.

    :param checkpoints: Dictionary of checkpoints with keys being the epoch number and values being the model.
    :param batch: The batch of data used for calculating the Hessian.
    :param target: The target values used for calculating the Hessian.
    :param loss_fn: The loss function used to calculate the Hessian.
    :param bins: The number of bins to be used in the histogram.
    """
    # Initialize lists to store the data
    eigenvalues_data = []
    
    # Variables to store the min and max values for the x and y axes
    min_eigenvalue = float('inf')
    max_eigenvalue = float('-inf')
    max_frequency = 0
    epochs = eigenvalues_dict.keys()
    # Iterate over the checkpoints
    for epoch, eigenvalues in tqdm(eigenvalues_dict.items()):
        
        # Update the min and max eigenvalue
        min_eigenvalue = min(min_eigenvalue, eigenvalues.min().item())
        max_eigenvalue = max(max_eigenvalue, eigenvalues.max().item())
        
        # Compute the histogram
        hist, bin_edges = np.histogram(eigenvalues, bins=bins)
        
        # Update the max frequency
        max_frequency = max(max_frequency, hist.max())
        
        # Store the histogram data
        eigenvalues_data.append((epoch, hist, bin_edges[:-1]))

    # Create figure
    fig = go.Figure()

    # Add traces for each epoch
    for epoch, hist, bin_values in eigenvalues_data:
        fig.add_trace(go.Bar(x=bin_values, y=hist, visible=False))

    # Make the first trace visible
    fig.data[0].visible = True

    # Create slider
    steps = []
    for i, epoch in enumerate(epochs):
        step = dict(
            args=[{"visible": [False] * len(fig.data)}],
            label=str(epoch),
            method="restyle",
        )
        step["args"][0]["visible"][i] = True
        steps.append(step)

    slider = dict(
        active=0,
        yanchor="top",
        xanchor="left",
        currentvalue=dict(font=dict(size=16), prefix="Epoch: ", xanchor="right"),
        pad=dict(b=10, t=40),
        steps=steps,
    )

    # Set axis properties
    fig.update_layout(
        xaxis=dict(range=[min_eigenvalue, max_eigenvalue]),
        yaxis=dict(range=[0, max_frequency]),
        sliders=[slider],
        width=800,  # Width of the image
        height=700  # Height of the image
    )
    # Show the figure
    fig.show()

def eigenvalues_in_range(eigenvalues_dict: Dict[int, torch.Tensor],
                         min_val: float = float('-inf'),
                         max_val: float = float('inf')
                         )-> Dict[int, int]:
    """
    Counts the number of eigenvalues that fall within the specified range [min_val, max_val] for each epoch.

    Args:
        eigenvalues_dict (Dict[int, torch.Tensor]): A dictionary where the key represents the epoch and the value
            is a 1D torch tensor containing the eigenvalues of the hessian of the loss at that epoch.
        min_val (float, optional): The minimum value of the range to count eigenvalues within. Defaults to -inf.
        max_val (float, optional): The maximum value of the range to count eigenvalues within. Defaults to inf.

    Returns:
        Dict[int, int]: A dictionary where the key represents the epoch and the value is the count of eigenvalues
            within the specified range for that epoch.
    """
    count_dict = {}
    for epoch, eigenvalues in eigenvalues_dict.items():
        count = ((eigenvalues >= min_val) & (eigenvalues <= max_val)).sum().item()
        count_dict[epoch] = count
        
    return count_dict

def plot_eigenvalues_in_range(eigenvalues_dict: Dict[int, torch.Tensor], ranges: [(float, float)]):
    """
    Plots the count of eigenvalues within specified ranges across different epochs.

    Args:
        eigenvalues_dict (Dict[int, torch.Tensor]): A dictionary where the key represents the epoch and the value
            is a 1D torch tensor containing the eigenvalues of the hessian of the loss at that epoch.
        ranges (List[Tuple[float, float]]): A list of tuples, each containing a pair (min_val, max_val) to use
            in the calculation of eigenvalues_in_range function.
    """
    # Create an empty DataFrame to store the data for plotting
    plot_data = pd.DataFrame(columns=['Epoch', 'Number of Eigenvalues in Range', 'Range', 'min_val', 'max_val'])

    # Iterate through the ranges and calculate the count of eigenvalues for each epoch
    for min_val, max_val in ranges:
        count_dict = eigenvalues_in_range(eigenvalues_dict, min_val, max_val)

        # Create a label for the range
        if min_val == float('-inf'):
            label = f'eigenvalue < {max_val}'
        elif max_val == float('inf'):
            label = f'{min_val} < eigenvalue'
        else:
            label = f'{min_val} < eigenvalue < {max_val}'

        # Append the data to the DataFrame
        for epoch, count in count_dict.items():
            plot_data = plot_data.append({
                'Epoch': epoch,
                'Number of Eigenvalues in Range': count,
                'Range': label,
                'min_val': min_val,
                'max_val': max_val
            }, ignore_index=True)

    # Create the interactive plot using Plotly
    fig = px.line(plot_data, x='Epoch', y='Number of Eigenvalues in Range', color='Range', 
                  hover_data=['min_val', 'max_val', 'Number of Eigenvalues in Range', 'Epoch'],
                  title='Eigenvalue Counts Across Epochs')
    
    fig.update_layout(xaxis_title='Epoch', yaxis_title='Number of Eigenvalues in Range')
    fig.show()

def plot_histograms(data: dict[str, List], bins: int = 100, xlabel: str = 'Value', ylabel: str = 'Frequency', size: Tuple[int,int] = (800,700)) -> None:
    # Create a DataFrame from the dictionary
    df = pd.DataFrame.from_dict(
        {
            'value': [item for sublist in data.values() for item in sublist],
            'key': [key for key, sublist in data.items() for _ in sublist]
        }
    )

    # Find the minimum and maximum values across all the data
    min_value = df['value'].min()
    max_value = df['value'].max()

    # Determine the maximum frequency (count) across all histograms
    max_frequency = 0
    for values in data.values():
        counts, _ = np.histogram(values, bins=bins, range=(min_value, max_value))
        max_frequency = max(max_frequency, counts.max())

    # Create a histogram plot using Plotly Express
    fig = px.histogram(
        df,
        x="value",
        animation_frame="key",
        nbins=bins,
        labels={'value': xlabel},
        title='Histogram',
        height=size[1],
        width=size[0]
    )

    # Update y-axis label and set x and y ranges
    fig.update_yaxes(title_text=ylabel, range=[0, max_frequency])
    fig.update_xaxes(range=[min_value, max_value])

    # Show the plot
    fig.show()


def visualise_activation_space(final_layer: Callable, xlim: int = 3, ylim: int = 3, n_points: int = 200):

    # Creating a grid of points between -xlim to xlim and -ylim to ylim.
    x = np.linspace(-xlim, xlim, n_points)
    y = np.linspace(-ylim, ylim, n_points)
    xx, yy = np.meshgrid(x, y)
    xx_flat, yy_flat = xx.ravel(), yy.ravel()

    classifications = []

    for i in range(len(xx_flat)):
        activations = torch.tensor([[xx_flat[i], yy_flat[i]]], dtype=torch.float32)
        output = final_layer(activations)
        classification = torch.argmax(output).item()
        classifications.append(classification)

    classifications = np.array(classifications).reshape(xx.shape)

    # Plotting using plotly
    fig = go.Figure(data=go.Contour(z=classifications, x=x, y=y, colorscale='Viridis'))

    fig.update_layout(title="Hidden Activation Space Visualization",
                      xaxis_title="X Activation",
                      yaxis_title="Y Activation",
                      legend_title="Classifications",
                      coloraxis_colorbar_title="Class",
                      height = 700, width = 700*xlim/ylim)

    fig.show()

def final_layer(model: Net) -> Callable:
        def out(x):
            if model.tied:
                return F.linear(x, model.embedding.weight.t(), model.unembedding_bias)
            else:
                return model.unembedding(x)
        return out


def final_layer_dict(model_history: dict[str, Net]) -> dict[str, Callable]:
    final_layers = {}
    for epoch, model in model_history.items():
        def out(x, model=model):  # This is to capture the current model in the loop
            if model.tied:
                return F.linear(x, model.embedding.weight.t(), model.unembedding_bias)
            else:
                return model.unembedding(x)
        final_layers[epoch] = out
    return final_layers

def visualise_activation_space_history(final_layers: dict[str, Callable], xlim: int = 3, ylim: int = 3, n_points: int = 200):
    x = np.linspace(-xlim, xlim, n_points)
    y = np.linspace(-ylim, ylim, n_points)
    xx, yy = np.meshgrid(x, y)
    xx_flat, yy_flat = xx.ravel(), yy.ravel()

    # Store contour data for each epoch
    contours = []
    epochs = list(final_layers.keys())
    for final_layer in tqdm(final_layers.values()):
        classifications = []
        for i in range(len(xx_flat)):
            activations = torch.tensor([[xx_flat[i], yy_flat[i]]], dtype=torch.float32)
            output = final_layer(activations)
            classification = torch.argmax(output).item()
            classifications.append(classification)

        classifications = np.array(classifications).reshape(xx.shape)
        contours.append(classifications)

    # Generate the plotly figure with a slider
    fig_data = []
    for i, epoch in tqdm(enumerate(epochs)):
        visible = [False] * len(epochs)
        visible[i] = True
        fig_data.append(go.Contour(z=contours[i], x=x, y=y, colorscale='Viridis', visible=visible[i]))

    steps = []
    for i, epoch in enumerate(epochs):
        step = dict(
            args=[{"visible": [epoch == e for e in epochs]}],
            method="restyle",
            label=str(epoch)
        )
        steps.append(step)

    sliders = [dict(
        active=0,
        yanchor="top",
        xanchor="left",
        currentvalue={"font": {"size": 20}, "prefix": "Epoch:", "visible": True, "xanchor": "right"},
        pad={"b": 10, "t": 50},
        steps=steps
    )]

    layout = go.Layout(
        title="Hidden Activation Space Visualization",
        xaxis_title="X Activation",
        yaxis_title="Y Activation",
        sliders=sliders,
        height=700, width=700*xlim/ylim
    )

    fig = go.Figure(data=fig_data, layout=layout)
    fig.show()
