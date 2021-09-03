import torch
from torch.utils.data import Dataset


class Samples(Dataset):
    """Samples build a Dataset object for pytorch"""

    def __init__(self, data_sampler) -> None:
        """__init__ initialize the dataset

        Parameters
        ----------
        data_sampler : generator
            generator to create samples
        """
        self.dataset = list(data_sampler)

    def __len__(self):
        """__len__ get the dataset's length

        Returns
        -------
        int
            dataset's length
        """
        return len(self.dataset)

    def __getitem__(self, index):
        """__getitem__ gets the item corresponding to the given index

        Parameters
        ----------
        index : int
            index of the item to be retrieved

        Returns
        -------
        torch.Tensor
            tensor representing the item corresponding to the given index
        """
        data = self.dataset[index]
        return torch.tensor(data)
