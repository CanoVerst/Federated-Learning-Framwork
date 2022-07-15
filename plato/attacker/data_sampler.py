
import random
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import SubsetRandomSampler

from plato.attacker.cifar100 import CIFAR100_DataSource
from plato.attacker.purchase import Purchase_DataSource
import plato.datasources.registry as data_registry

class IndependentSampler():
    """Create a data sampler for each client to use a randomly divided partition of the
    dataset."""

    def __init__(self, datasource, total_clients, partition_size, client_index, random_seed):
        assert client_index > 0

        self.random_seed = random_seed
        self.data_set_size = datasource.num_train_examples()

        dataset_indices = list(range(self.data_set_size))
        np.random.seed(self.random_seed)
        np.random.shuffle(dataset_indices)

        # partition_size = 500
        # total_clients = 100
        total_size = partition_size * total_clients

        # add extra samples to make it evenly divisible, if needed
        if len(dataset_indices) < total_size:
            while len(dataset_indices) < total_size:
                dataset_indices += dataset_indices[:(total_size - len(dataset_indices))]
        else:
            dataset_indices = dataset_indices[:total_size]
        assert len(dataset_indices) == total_size

        # Compute the indices of data in the subset for this client
        self.subset_indices = dataset_indices[int(client_index) - 1:total_size:total_clients]

    def get(self):
        """Obtains an instance of the sampler. """
        gen = torch.Generator()
        gen.manual_seed(self.random_seed)
        version = torch.__version__.split(".")
        if int(version[0]) <= 1 and int(version[1]) <= 5:
            return SubsetRandomSampler(self.subset_indices)
        return SubsetRandomSampler(self.subset_indices, generator=gen)

    def trainset_size(self):
        """Returns the length of the dataset after sampling. """
        return len(self.subset_indices)

class AttackDataLoader():
    def __init__(self, total_client, partition_size, random_seed) -> None:
        self.datasource = data_registry.get()
        assert not self.datasource is None

        self.total_client = total_client
        self.partition_size = partition_size
        self.random_seed = random_seed
    
    def get_train_loader(self, client_id):
        sampler = IndependentSampler(self.datasource, self.total_client, self.partition_size, 
                                        client_id, self.random_seed)
        sampler_inst = sampler.get()
        data_loader = torch.utils.data.DataLoader(dataset = self.datasource.trainset,
                                                  shuffle = False,
                                                  batch_size = sampler.trainset_size(),
                                                  sampler = sampler_inst)
        return data_loader

    def get_test_loader(self):
        return torch.utils.data.DataLoader(dataset = self.datasource.testset,
                                            shuffle = False,
                                            batch_size = 64)
    
    def get_shadow_loader(self):
        test_size = self.datasource.num_test_examples()
        trai_size = self.datasource.num_train_examples()
        random_sampler = SubsetRandomSampler(random.sample(range(trai_size), test_size))
        return torch.utils.data.DataLoader(dataset = self.datasource.trainset,
                                            shuffle = False,
                                            batch_size = 64,
                                            sampler = random_sampler)