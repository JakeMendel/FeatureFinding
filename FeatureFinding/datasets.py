from torch.utils.data import Dataset
import torch
from itertools import combinations


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
