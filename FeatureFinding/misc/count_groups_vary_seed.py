import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import torch.nn.functional as F
import random
import copy
from itertools import combinations
from torch.optim.lr_scheduler import _LRScheduler
import os
import json 

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
    def __init__(self, input_dim, hidden_dim, tied = True, final_bias = False, hidden_bias = False,nonlinearity = F.relu, unit_weights=False, with_scale_factor = False, standard_magnitude = False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.nonlinearity = nonlinearity
        self.tied = tied
        self.final_bias = final_bias
        self.unit_weights = unit_weights
        self.with_scale_factor = with_scale_factor
        self.standard_magnitude = standard_magnitude


        # Define the input layer (embedding)
        self.embedding = nn.Linear(self.input_dim, self.hidden_dim, bias=hidden_bias)

        # Define the output layer (unembedding)
        self.unembedding = nn.Linear(self.hidden_dim, self.input_dim, bias=final_bias)

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

        if self.with_scale_factor:
            self.scale_factor = nn.Parameter(torch.tensor(1.0))
        else:
            self.scale_factor = 1.0

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


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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


def train(model, loader, criterion, optimizer, epochs, scheduler = None):
    for epoch in tqdm(range(epochs)):
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if scheduler is not None:
            scheduler.step()

# your chosen seed
def training_run(chosen_seed,
                 f = 50,
                 k = 1,
                 n = 2,
                 nonlinearity = F.gelu,
                 tied = True,
                 final_bias = False,
                 hidden_bias = False,
                 unit_weights = False,
                 with_scale_factor = False,
                 standard_magnitude = False,
                 epochs = 40000,
                 max_lr = 2,
                 initial_lr = 0.001,
                 warmup_frac = 0.05,
                 final_lr = 0.2
                 ):

    set_seed(chosen_seed)
    #Scheduler params
    decay_factor=(final_lr/max_lr)**(1/(epochs * (1-warmup_frac)))
    warmup_steps = int(epochs * warmup_frac)

    # Instantiate synthetic dataset
    dataset = SyntheticKHot(f,k)
    batch_size = len(dataset) #Full batch gradient descent
    loader = DataLoader(dataset, batch_size=batch_size, shuffle = True, num_workers=0)

    #Define the Loss function
    criterion = nn.MSELoss()

    # Instantiate the model
    model = Net(f, n,
                tied = tied,
                final_bias = final_bias,
                hidden_bias = hidden_bias,
                nonlinearity=nonlinearity,
                unit_weights=unit_weights,
                with_scale_factor=with_scale_factor,
                standard_magnitude=standard_magnitude)

    # Define loss function and optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr)

    #Define a learning rate schedule
    scheduler = CustomScheduler(optimizer, warmup_steps, max_lr, decay_factor)

    # Train the model
    train(model, loader, criterion, optimizer, epochs, scheduler)

    groups, directions = group_vectors(model.unembedding.weight.data.detach().numpy(),0.00001)
    return len(groups)


num_groups = []
for seed in range(1,10001):
    num = training_run(seed)
    num_groups.append(num)


# Saving to a json file
with open('num_groups.json', 'w') as f:
    json.dump(num_groups, f)