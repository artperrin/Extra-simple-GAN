import torch
from torch.utils.data import Dataset

class Samples(Dataset):
    def __init__(self, data_sampler) -> None:
        self.dataset = list(data_sampler)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        data = self.dataset[index]
        return torch.tensor(data)