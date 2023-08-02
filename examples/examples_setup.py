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
    def __init__(self, input_dim, hidden_dim, tied = True, final_bias = False, hidden_bias = False,nonlinearity = F.relu, unit_weights=False, learnable_scale_factor = False, standard_magnitude = False, initial_scale_factor = 1.0, initial_embed = None, initial_bias = None):
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

        # Define the output layer (unembedding)
        self.unembedding = nn.Linear(self.hidden_dim, self.input_dim, bias=final_bias)
        if initial_bias is not None:
            self.unembedding.bias.data = initial_bias

        if self.standard_magnitude:
            # Normalize the weight to have unit norm for each row
            avg_norm = torch.norm(self.embedding.weight.data, p=2, dim = 0).mean()
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=0) * avg_norm
        if self.unit_weights:
            # Normalize the weight to have unit norm for each row
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=0)
        # Tie the weights
        if tied:
            self.unembedding.weight = torch.nn.Parameter(self.embedding.weight.transpose(0, 1))

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
        if self.tied:
            self.unembedding.weight.data = self.embedding.weight.data.transpose(0, 1)
        if hooked:
            activations = {}
            activations['res_pre'] = self.embedding(x)
            activations['unembed_pre'] = self.unembedding(activations['res_pre'])
            activations['output'] = self.scale_factor * self.nonlinearity(activations['unembed_pre'])
            return activations['output'], activations
        else:
            x = self.embedding(x)
            x = self.unembedding(x)
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


def train(model, loader, criterion, optimizer, epochs, logging_loss, plot_rate, store_rate, scheduler = None, lr_print_rate = 0):
    weights_history = {k:[v.detach().numpy().copy()] for k,v in dict(model.named_parameters()).items()}  # Store the weights here
    model_history = {} #store model here
    losses = []
    for epoch in tqdm(range(epochs)):
        total_loss = 0
        for batch in loader:
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
        if (epoch + 1) % store_rate == 0:
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

def calculate_hessian(model, inputs, targets, loss_func):
    # Compute the loss
    n = sum(p.numel() for p in model.parameters())
    if 1000<n:
        print(f'Warning: this model has {n} parameters. Computing the Hessian take a long time or cause a memory overflow.')
    output = model(inputs)
    loss = loss_func(output, targets)

    # Compute the gradient of the loss with respect to the model parameters
    grad_params = grad(loss, model.parameters(), create_graph=True)

    # Compute the Hessian
    hessian = []
    for g in tqdm(grad_params):
        grad2_params = grad(g, model.parameters(), retain_graph=True)
        hessian.extend(grad2_params)

    # Calculate the total number of parameters in the model

    # Reshape the Hessian components into a matrix of size (n, n)
    hessian_matrix = torch.cat([h.reshape(-1) for h in hessian]).reshape(n, n)

    return hessian_matrix
